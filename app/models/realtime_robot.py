import cv2
import torch
from models.cnn_model import ObstacleAvoidanceCNN
from utils.transforms import get_transforms
from utils.controller import SteeringController

device = "cuda" if torch.cuda.is_available() else "cpu"
model = ObstacleAvoidanceCNN().to(device)
model.load_state_dict(torch.load("robot_model.pth", map_location=device))
model.eval()

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
