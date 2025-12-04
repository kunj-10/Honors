import torch
from torch.utils.data import DataLoader
from models.cnn_model import ObstacleAvoidanceCNN
from utils.dataset import SteeringDataset
from utils.transforms import get_transforms
from torch.optim import Adam
import torch.nn as nn

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ObstacleAvoidanceCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    train_set = SteeringDataset("data/train", transform=get_transforms())
    val_set = SteeringDataset("data/val", transform=get_transforms())
    train_loader = DataLoader(train_set, 32, shuffle=True)
    val_loader = DataLoader(val_set, 32)

    for epoch in range(20):
        model.train()
        for img, angle in train_loader:
            img, angle = img.to(device), angle.to(device).unsqueeze(1)
            pred = model(img)
            loss = criterion(pred, angle)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss = {loss.item():.4f}")

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for img, angle in val_loader:
                img, angle = img.to(device), angle.to(device).unsqueeze(1)
                pred = model(img)
                val_loss += criterion(pred, angle).item()
            print(f"Validation loss: {val_loss/len(val_loader):.4f}")

    torch.save(model.state_dict(), "robot_model.pth")
    print("Model saved.")

if __name__ == "__main__":
    train()
