import os
from pathlib import Path
import torch

# --- debug info (useful when running inside Webots) ---
print("evaluate.py running from CWD:", os.getcwd())

# Compute project root relative to this file:
# file: .../app/models/evaluate.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # parents[0]=models, [1]=app, [2]=project root
MODEL_PATH = PROJECT_ROOT / "app" / "robot_model.pth"


import torch
from app.models.models.cnn_model import ObstacleAvoidanceCNN
from app.models.utils.transforms import get_transforms
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ObstacleAvoidanceCNN().to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
except Exception as e:
    pass
transform = get_transforms()

def predict(img_path):
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        angle = model(x).item()
    return angle

if __name__ == "__main__":
    angle = predict("sample.jpg")
    print("Predicted steering angle:", angle)
