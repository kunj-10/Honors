import os
import glob
import argparse
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import csv
import math

class ObstacleAvoidanceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class SteeringDataset(Dataset):
    """
    Expects: data/test/labels.csv with rows: filename,angle
    and images in the same folder (or use full paths in CSV).
    """
    def __init__(self, root, img_size=64):
        self.root = root
        self.items = []
        csv_path = os.path.join(root, "labels.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"labels.csv not found in {root}")
        with open(csv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if "," in line:
                    fname, angle = line.split(",", 1)
                else:
                    parts = line.split()
                    fname, angle = parts[0], parts[1]
                img_path = os.path.join(root, fname)
                if not os.path.exists(img_path):
                    if os.path.exists(fname):
                        img_path = fname
                    else:
                        print(f"Warning: {img_path} not found — skipping")
                        continue
                self.items.append((img_path, float(angle)))

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, ang = self.items[idx]
        img = Image.open(p).convert("RGB")
        img = self.transform(img)
        return img, torch.tensor(ang, dtype=torch.float32), p

def compute_metrics(preds, targets, tol=0.1):
    """
    preds, targets: 1D lists or tensors of same length
    tol: tolerance (units = same as steering angles). A sample is correct if |pred - target| <= tol
    Returns: dict {mae, rmse, within_tol_percent}
    """
    import numpy as np
    preds = np.array(preds, dtype=float)
    targets = np.array(targets, dtype=float)
    errs = preds - targets
    mae = float(np.mean(np.abs(errs)))
    rmse = float(np.sqrt(np.mean(errs**2)))
    within = float((np.abs(errs) <= tol).mean() * 100.0)
    return {"MAE": mae, "RMSE": rmse, f"Within_{tol}": within}

def evaluate(model_paths, test_root="data/test", batch_size=32, tol=0.1, device=None, img_size=64, save_preds="predictions.csv"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = SteeringDataset(test_root, img_size=img_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    results = []
    for model_path in model_paths:
        print(f"\nLoading checkpoint: {model_path}")
        model = ObstacleAvoidanceCNN().to(device)
        state = torch.load(model_path, map_location=device)
        if isinstance(state, dict) and any(k.startswith('module.') for k in state.keys()):
            new_state = {k.replace('module.', ''): v for k, v in state.items()}
            model.load_state_dict(new_state)
        elif isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        model.eval()

        all_preds = []
        all_targets = []
        all_files = []

        criterion = nn.MSELoss()
        total_loss = 0.0
        batches = 0

        with torch.no_grad():
            for imgs, angles, paths in loader:
                imgs = imgs.to(device)
                angles = angles.to(device).unsqueeze(1)
                preds = model(imgs).cpu().squeeze(1).numpy()
                angles_np = angles.cpu().squeeze(1).numpy()
                all_preds.extend(preds.tolist())
                all_targets.extend(angles_np.tolist())
                all_files.extend(paths)
                loss = criterion(torch.tensor(preds), torch.tensor(angles_np)).item()
                total_loss += loss
                batches += 1

        avg_loss = total_loss / max(1, batches)
        metrics = compute_metrics(all_preds, all_targets, tol=tol)
        print(f"Checkpoint: {os.path.basename(model_path)}  AvgLoss:{avg_loss:.6f}  MAE:{metrics['MAE']:.6f}  RMSE:{metrics['RMSE']:.6f}  Within{tol}:{metrics[f'Within_{tol}']:.2f}%")

        out_csv = os.path.splitext(save_preds)[0] + "_" + os.path.splitext(os.path.basename(model_path))[0] + ".csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image","target","prediction","abs_error"])
            for fn, t, p in zip(all_files, all_targets, all_preds):
                writer.writerow([fn, f"{t:.6f}", f"{p:.6f}", f"{abs(p-t):.6f}"])
        print("Predictions written to:", out_csv)

        results.append({
            "checkpoint": model_path,
            "avg_loss": avg_loss,
            **metrics
        })

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", nargs="+", help="Path(s) to .pth checkpoint(s). You can pass multiple to evaluate each.", required=True)
    parser.add_argument("--test_root", "-t", default="data/test", help="Folder containing images and labels.csv")
    parser.add_argument("--batch_size", "-b", type=int, default=32)
    parser.add_argument("--tol", type=float, default=0.1, help="Tolerance for accuracy (|pred-target| <= tol)")
    parser.add_argument("--img_size", type=int, default=64, help="Image resize (must match training)")
    parser.add_argument("--save_preds", default="predictions.csv", help="Base filename for saved predictions")
    args = parser.parse_args()

    model_paths = []
    for pat in args.model:
        model_paths += sorted(glob.glob(pat))
    if not model_paths:
        raise SystemExit("No checkpoints found for given pattern(s)")

    results = evaluate(model_paths, test_root=args.test_root, batch_size=args.batch_size, tol=args.tol, img_size=args.img_size, save_preds=args.save_preds)
    print("\nSummary:")
    for r in results:
        print(r)
