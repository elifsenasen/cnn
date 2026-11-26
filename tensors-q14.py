import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from MNISTConvNet import MNISTConvNet, device

script_dir = Path(__file__).resolve().parent
ROOT = script_dir / "mnist-varres"


class VarResNet(nn.Module):
    def __init__(self, num_classes=10, pool='max', channels=1, depth=64):   
        super().__init__()
        assert pool in ('max', 'mean')
        self.pool = pool
        self.features = nn.Sequential(
            nn.Conv2d(channels, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, depth, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(depth, num_classes)       

    def forward(self, x):
        x = self.features(x)                           
        if self.pool == 'max':
            x = x.amax(dim=(2, 3))                    
        else:
            x = x.mean(dim=(2, 3))                    
        return self.fc(x)                            


def load_torch_tensor(files):
    images, labels = defaultdict(list), defaultdict(list)
    for f in files:
        img = np.array(Image.open(f).convert('L'))    
        res = img.shape[0]                             
        images[res].append(img)
        labels[res].append(int(f.parent.name))         
    return {res: (torch.from_numpy(np.stack(images[res])).float().unsqueeze(1) / 255.0,
                  torch.tensor(labels[res], dtype=torch.long))
            for res in sorted(images)}


def load_by_resolution(root=ROOT, seed=0, n_val=10000):
    train_files = sorted((root / 'train').glob('*/*.png'))
    test_files = sorted((root / 'test').glob('*/*.png'))

    perm = torch.randperm(len(train_files), generator=torch.Generator().manual_seed(seed))
    val_files = [train_files[i] for i in perm[:n_val]]
    tr_files = [train_files[i] for i in perm[n_val:]]

    return load_torch_tensor(tr_files), load_torch_tensor(val_files), load_torch_tensor(test_files)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate(model, data, batch_size=1000):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for res, (X, y) in data.items():                   
        for i in range(0, len(X), batch_size):
            images, labels = X[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
            outputs = model(images)
            total_loss += F.cross_entropy(outputs, labels, reduction='sum').item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            n += len(labels)
    return total_loss / n, correct / n


def train(model, train_data, val_data, optimizer, epochs=8, batch_size=16):
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    n_train = sum(len(X) for X, _ in train_data.values())

    for epoch in range(epochs):
        model.train()
        batches = []
        for res, (X, _) in train_data.items():
            perm = torch.randperm(len(X))
            for i in range(0, len(X), batch_size):
                batches.append((res, perm[i:i + batch_size]))

        
        total_loss = 0.0
        for j in torch.randperm(len(batches)).tolist():
            res, idx = batches[j]
            X, y = train_data[res]
            images, labels = X[idx].to(device), y[idx].to(device)

            optimizer.zero_grad()
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        val_loss, val_acc = evaluate(model, val_data)
        history['train_loss'].append(total_loss / n_train)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss / n_train:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=64)
    parser.add_argument("--pool", choices=["max", "mean"], default="max")
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    train_data, val_data, test_data = load_by_resolution()
    for name, data in [('train', train_data), ('val', val_data), ('test', test_data)]:
        print(name, {res: tuple(X.shape) for res, (X, _) in data.items()})

    torch.manual_seed(0)
    model = VarResNet(pool=args.pool, depth=args.depth).to(device)
    print("parameters:", count_parameters(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = train(model, train_data, val_data, optimizer, epochs=args.epochs)

    with open(f"q14_{args.pool}_N{args.depth}.json", "w") as f:
        json.dump(history, f, indent=2)
    
    #question 15
    # print("fixed (Q8/Q12):", count_parameters(MNISTConvNet()))
    # for N in [64, 80, 81, 82]:
    #     print(f"variable, N={N}:", count_parameters(VarResNet(depth=N)))