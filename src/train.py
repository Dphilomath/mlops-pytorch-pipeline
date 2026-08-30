import json
import sys
from pathlib import Path
import torch
import torch.nn as nn
import yaml
import datetime
import logging
import os
from src.dataset import get_dataloaders
from src.model import get_model
import mlflow
import mlflow.pytorch

logger = logging.getLogger("training")
# Determine log level from environment (default INFO)
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
numeric_level = getattr(logging, log_level_str, logging.INFO)
logger.setLevel(numeric_level)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(message)s")
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
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(loader):
            progress_data = {
                "event": "batch_progress",
                "batch": batch_idx + 1,
                "total_batches": len(loader),
                "loss": round(loss.item(), 4),
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
            print(json.dumps(progress_data), flush=True)
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

    # Configure MLflow tracking
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    try:
        mlflow.set_experiment("cifar10-optimization")
    except Exception:
        import time

        time.sleep(2)
        mlflow.set_experiment("cifar10-optimization")

    # Parallelize linear algebra operations across available CPU cores
    num_threads = int(os.getenv("TORCH_NUM_THREADS", "4"))
    torch.set_num_threads(num_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    strategy = config["training"].get("strategy", "transfer_learning")
    architecture = config["model"]["architecture"]
    num_classes = config["model"]["num_classes"]

    model = get_model(
        architecture=architecture,
        num_classes=num_classes,
        strategy=strategy,
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        batch_size=config["training"]["batch_size"],
        strategy=strategy,
    )

    # Configure optimizer based on training config
    opt_type = config["training"].get("optimizer", "adam").lower()
    lr = config["training"]["learning_rate"]
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if opt_type == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
    else:
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=lr,
        )

    # Configure learning rate scheduler
    sched_type = config["training"].get("scheduler", "none").lower()
    if sched_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["training"]["epochs"],
        )
    else:
        scheduler = None

    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]
    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model_name = config["output"]["model_name"]

    # Start MLflow run
    run_name = (
        f"{architecture}_{strategy}_{datetime.datetime.now().strftime('%m%d_%H%M')}"
    )
    with mlflow.start_run(run_name=run_name):
        # Log training hyperparameters
        mlflow.log_params(
            {
                "architecture": architecture,
                "num_classes": num_classes,
                "strategy": strategy,
                "optimizer": opt_type,
                "scheduler": sched_type,
                "epochs": config["training"]["epochs"],
                "batch_size": config["training"]["batch_size"],
                "learning_rate": lr,
                "early_stopping_patience": patience,
            }
        )

        for epoch in range(config["training"]["epochs"]):
            logger.info(
                json.dumps(
                    {
                        "event": "epoch_start",
                        "epoch": epoch + 1,
                        "total_epochs": config["training"]["epochs"],
                        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                    }
                )
            )
            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            if scheduler:
                scheduler.step()

            # Log epoch metrics to MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch + 1)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch + 1)
            mlflow.log_metric("val_loss", val_loss, step=epoch + 1)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch + 1)
            if scheduler:
                mlflow.log_metric("lr", scheduler.get_last_lr()[0], step=epoch + 1)
            else:
                mlflow.log_metric("lr", lr, step=epoch + 1)

            log_entry = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "train_accuracy": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_accuracy": round(val_acc, 4),
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
            logger.info(json.dumps(log_entry))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                patience_counter = 0
                save_path = checkpoint_dir / model_name
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                    },
                    save_path,
                )
                logger.info(
                    json.dumps(
                        {
                            "event": "checkpoint_saved",
                            "path": str(save_path),
                            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
                        }
                    )
                )
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(
                        json.dumps(
                            {
                                "event": "early_stopping",
                                "epoch": epoch + 1,
                                "timestamp": datetime.datetime.utcnow().isoformat()
                                + "Z",
                            }
                        )
                    )
                    break

        # Load the best saved checkpoint to log/register the correct model version
        best_path = checkpoint_dir / model_name
        if best_path.exists():
            checkpoint = torch.load(best_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])

        # Log the PyTorch model inside the MLflow run and register it
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="cifar10-resnet18",
        )

        mlflow.log_metrics(
            {
                "best_val_loss": best_val_loss,
                "best_val_accuracy": best_val_acc,
            }
        )

    logger.info(
        json.dumps(
            {
                "event": "training_complete",
                "best_val_loss": round(best_val_loss, 4),
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            }
        )
    )


if __name__ == "__main__":
    main()
