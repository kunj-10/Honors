import os
from pathlib import Path
import torch

# --- debug info (useful when running inside Webots) ---
print("evaluate.py running from CWD:", os.getcwd())

# Compute project root relative to this file:
# file: .../app/models/evaluate.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # parents[0]=models, [1]=app, [2]=project root
MODEL_PATH = PROJECT_ROOT / "app" / "robot_model.pth"



import cv2
import torch
from app.models.models.cnn_model import ObstacleAvoidanceCNN
from app.models.utils.transforms import get_transforms
from app.models.utils.controller import SteeringController

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ObstacleAvoidanceCNN().to(device)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
except Exception as e:
    pass

transform = get_transforms()
controller = SteeringController()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    x = transform(Image.fromarray(img)).unsqueeze(0).to(device)
    with torch.no_grad():
        angle = model(x).item()

    left, right = controller.compute_wheel_speeds(angle)
    print("Wheel speeds:", left, right)

    cv2.imshow("Robot View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
