import yaml
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import torch
from torchvision import transforms
from PIL import Image
from src.model import get_model

app = FastAPI()

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
checkpoint_dir = cfg["output"]["checkpoint_dir"]
model_file = os.path.join(checkpoint_dir, cfg["output"]["model_name"])
if not os.path.isfile(model_file):
    raise FileNotFoundError(f"Model checkpoint not found at {model_file}")
state = torch.load(model_file, map_location=torch.device("cpu"))
model.load_state_dict(state["model_state_dict"])
model.eval()

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
    return {"status": "healthy"}

class PredictionResponse(BaseModel):
    predicted_class: int
    probability: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
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
    return PredictionResponse(predicted_class=pred.item(), probability=prob.item())
