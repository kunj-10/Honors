import os
from torch.utils.data import Dataset
from PIL import Image

class SteeringDataset(Dataset):
    """
    Dataset format:
    image.jpg , steering_angle
    stored in labels.txt
    """

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform

        with open(os.path.join(root, "labels.txt")) as f:
            self.items = [line.strip().split(",") for line in f]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, angle = self.items[idx]
        img = Image.open(os.path.join(self.root, img_path)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, float(angle)
