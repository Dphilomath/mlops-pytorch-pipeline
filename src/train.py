import json
import sys
from pathlib import Path
import torch
import torch.nn as nn
import yaml
import datetime
import logging
from src.dataset import get_dataloaders
from src.model import get_model

logger = logging.getLogger("training")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Train one epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(loader):
            logger.debug(json.dumps({"event":"batch_progress","batch":batch_idx+1,"total_batches":len(loader),"loss":loss.item(),"timestamp":datetime.datetime.utcnow().isoformat()+"Z"}))
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate the model on validation data and return average loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def main():
    """Main entry point for training the model according to the supplied config."""
    config_path = Path("/app/configs/training_config.yaml")
    if not config_path.exists():
        config_path = Path("configs/training_config.yaml")
    config = load_config(str(config_path))
    # Use all available CPU cores for PyTorch linear algebra ops
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(
        architecture=config["model"]["architecture"],  num_classes=config["model"]["num_classes"],
    ).to(device)
    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
    )
    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(config["training"]["epochs"]):
        logger.info(json.dumps({"event":"epoch_start","epoch":epoch+1,"total_epochs":config["training"]["epochs"],"timestamp":datetime.datetime.utcnow().isoformat()+"Z"}))
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        log_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
            "timestamp": datetime.datetime.utcnow().isoformat()+"Z"
        }
        logger.info(json.dumps(log_entry))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_path = checkpoint_dir / config["output"]["model_name"]
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, save_path)
            logger.info(json.dumps({"event":"checkpoint_saved","path":str(save_path),"timestamp":datetime.datetime.utcnow().isoformat()+"Z"}))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(json.dumps({"event":"early_stopping","epoch":epoch+1,"timestamp":datetime.datetime.utcnow().isoformat()+"Z"}))
                break
    logger.info(json.dumps({"event":"training_complete","best_val_loss":round(best_val_loss,4),"timestamp":datetime.datetime.utcnow().isoformat()+"Z"}))

if __name__ == "__main__":
    main()
