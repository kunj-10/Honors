import torch
from models.cnn_model import ObstacleAvoidanceCNN
from utils.transforms import get_transforms
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ObstacleAvoidanceCNN().to(device)
model.load_state_dict(torch.load("robot_model.pth", map_location=device))
model.eval()
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
