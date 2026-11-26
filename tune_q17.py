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

parent_root = Path(__file__).resolve().parent
root = parent_root / "mnist-varres"

N = 81
POOL = 'max'


def load(files):
    native, resized, labels = defaultdict(list), defaultdict(list), defaultdict(list)
    for path in files:
        img = Image.open(path).convert('L')
        arr = np.array(img)
        res = arr.shape[0]
        native[res].append(arr)
        resized[res].append(np.array(img.resize((28, 28), Image.BILINEAR)))
        labels[res].append(int(path.parent.name))

    def to_tensors(groups):
        return {res: (torch.from_numpy(np.stack(groups[res])).float().unsqueeze(1) / 255.0,
                      torch.tensor(labels[res], dtype=torch.long))
                for res in sorted(groups)}
    return to_tensors(native), to_tensors(resized)


def load_data(root, n_val=10000, seed=0):
    train_files = sorted((root / 'train').glob('*/*.png'))
    test_files = sorted((root / 'test').glob('*/*.png'))
    perm = torch.randperm(len(train_files), generator=torch.Generator().manual_seed(seed))
    val_files = [train_files[i] for i in perm[:n_val]]
    tr_files = [train_files[i] for i in perm[n_val:]]

    (tr_n, tr_r), (va_n, va_r), (te_n, te_r) = load(tr_files), load(val_files), load(test_files)
    return {'var': (tr_n, va_n, te_n), 'fixed': (tr_r, va_r, te_r)}


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
        x = x.amax(dim=(2, 3)) if self.pool == 'max' else x.mean(dim=(2, 3))
        return self.fc(x)


def make_model(name):
    return MNISTConvNet() if name == 'fixed' else VarResNet(depth=N, pool=POOL)


@torch.no_grad()
def evaluate(model, data, batch_size=1000):
    model.eval()
    total_loss, total_correct, n, per_res = 0.0, 0, 0, {}
    for res, (X, y) in data.items():
        correct = 0
        for i in range(0, len(X), batch_size):
            images, labels = X[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
            outputs = model(images)
            total_loss += F.cross_entropy(outputs, labels, reduction='sum').item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
        per_res[res] = correct / len(X)
        total_correct += correct
        n += len(X)
    return total_loss / n, total_correct / n, per_res


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

        val_loss, val_acc, _ = evaluate(model, val_data)
        history['train_loss'].append(total_loss / n_train)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss / n_train:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    return history


def run(name, data, lr, epochs):
    train_data, val_data, _ = data[name]
    if name == 'fixed':
        train_data = {28: (torch.cat([X for X, _ in train_data.values()]),
                           torch.cat([y for _, y in train_data.values()]))}
    torch.manual_seed(0)
    model = make_model(name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history = train(model, train_data, val_data, optimizer, epochs)
    return model, history


def tune(name, lrs, epochs):
    data = load_data(root)
    results = {}
    for lr in lrs:
        print(f"\n= {name}, lr = {lr} =")
        _, results[lr] = run(name, data, lr, epochs)

    print(f"\n{name}: lr       best val acc   best epoch")
    for lr, h in results.items():
        best = max(range(epochs), key=lambda i: h['val_acc'][i])
        print(f"       {lr:<8} {h['val_acc'][best]:.4f}         {best + 1}")

    with open(f"q17_tune_{name}.json", "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)


def test(settings):
    data = load_data(root)
    out = {}
    for name, (lr, epochs) in settings.items():
        print(f"\n= {name}, lr = {lr}, {epochs} epochs =")
        model, _ = run(name, data, lr, epochs)
        test_loss, test_acc, per_res = evaluate(model, data[name][2])
        out[name] = {'lr': lr, 'epochs': epochs,
                     'parameters': sum(p.numel() for p in model.parameters()),
                     'test_loss': test_loss, 'test_acc': test_acc,
                     'test_acc_per_res': {str(r): a for r, a in per_res.items()}}

    print("\nmodel   params   test loss   test acc   acc 32x32   acc 48x48   acc 64x64")
    for name, r in out.items():
        pr = r['test_acc_per_res']
        print(f"{name:<7} {r['parameters']:<8} {r['test_loss']:.4f}      {r['test_acc']:.4f}     "
              f"{pr['32']:.4f}      {pr['48']:.4f}      {pr['64']:.4f}")

    with open("q17_test.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["tune", "test"])
    parser.add_argument("--model", choices=["fixed", "var"])
    parser.add_argument("--lrs", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fixed-lr", type=float, default=1e-3)
    parser.add_argument("--fixed-epochs", type=int, default=12)
    parser.add_argument("--var-lr", type=float, default=1e-3)
    parser.add_argument("--var-epochs", type=int, default=7)
    args = parser.parse_args()

    if args.task == "tune":
        assert args.model, "use --model fixed or --model var"
        tune(args.model, args.lrs, args.epochs)
    else:
        test({'fixed': (args.fixed_lr, args.fixed_epochs), 'var': (args.var_lr, args.var_epochs)})