import torch
import torch.nn as nn
import torch.nn.functional as F

class ObstacleAvoidanceCNN(nn.Module):
    """
    Monocular image → steering command regression network.
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2), nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 512), nn.ReLU(),
            nn.Linear(512, 1)   # Output: steering angle
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
