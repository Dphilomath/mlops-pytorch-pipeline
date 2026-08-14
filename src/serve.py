import yaml
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
import json
import datetime
import io
import sys
import logging
from torchvision import transforms
from PIL import Image
from src.model import get_model

app = FastAPI()

logger = logging.getLogger("serving")
# Determine log level from environment (default INFO)
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
numeric_level = getattr(logging, log_level_str, logging.INFO)
logger.setLevel(numeric_level)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Load serving configuration
config_path = os.getenv("CONFIG_PATH", "/app/configs/serving_config.yaml")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Serving config not found at {config_path}")
with open(config_path) as f:
    cfg = yaml.safe_load(f)

# Build model and load checkpoint
model = get_model(
    architecture=cfg["model"]["architecture"],
    num_classes=cfg["model"]["num_classes"],
)

# Determine checkpoint location
checkpoint_dir = cfg["output"]["checkpoint_dir"]
model_file = os.path.join(checkpoint_dir, cfg["output"]["model_name"])
import time

# Wait for the model checkpoint to become available (up to 5 minutes)
max_wait = int(os.getenv("MODEL_WAIT_SECONDS", "300"))
waited = 0
while not os.path.isfile(model_file):
    if waited >= max_wait:
        raise FileNotFoundError(f"Model checkpoint not found at {model_file} after waiting {max_wait}s")
    time.sleep(5)
    waited += 5
    logger.info(json.dumps({"event":"waiting_for_checkpoint","elapsed_seconds":waited,"timestamp":datetime.datetime.utcnow().isoformat()+"Z"}))

# Load the checkpoint once it exists
state = torch.load(model_file, map_location=torch.device("cpu"))
model.load_state_dict(state["model_state_dict"])
model.eval()
logger.info(json.dumps({"event":"model_loaded","path":model_file,"timestamp":datetime.datetime.utcnow().isoformat()+"Z"}))

# Image preprocessing (same as training)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
    ),
])

@app.get("/health")
def health():
    """Health check endpoint returning a simple JSON status."""
    return {"status": "healthy"}

class PredictionResponse(BaseModel):
    predicted_class: int
    probability: float
    class_name: str  # human‑readable CIFAR‑10 label

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Receive an image file, run inference, and return predicted class & probability.
    Logs a JSON event `prediction_requested` with the filename (if available).
    """
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        prob, pred = probs.max(dim=1)
    logger.info(json.dumps({"event":"prediction_requested","filename":file.filename or "<unknown>","timestamp":datetime.datetime.utcnow().isoformat()+"Z"}))
    # CIFAR‑10 class labels (same order as training)
    CIFAR10_LABELS = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
    return PredictionResponse(
        predicted_class=pred.item(),
        probability=prob.item(),
        class_name=CIFAR10_LABELS[pred.item()],
    )
