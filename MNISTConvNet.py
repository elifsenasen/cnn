import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.transforms import ToTensor


device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
lr_list=[1e-4, 3e-4, 1e-3, 3e-3]
class MNISTConvNet(nn.Module):
    def __init__(self, in_channels=1, c1=16, c2=32, c3=64, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(c3 * 3 * 3, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        return self.classifier(x)
    
def load_mnist(train_size=50000):
    transform = ToTensor()
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)

    #loading into tensors
    X_all, y_all = next(iter(train_loader))
    X_test, y_test = next(iter(test_loader))

    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(len(X_all), generator=g)
    train_size_idx, val_size_idx = perm[:train_size], perm[train_size:]

    return (X_all[train_size_idx], y_all[train_size_idx]), (X_all[val_size_idx], y_all[val_size_idx]), (X_test, y_test)

@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss = F.cross_entropy(logits, y_batch, reduction='sum')
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def train(model, train_data, val_loader, optimizer, device, epochs=5, batch_size=16):
    X_train, y_train = train_data
    length_x = len(X_train)
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(length_x)
        running_loss = 0.0
        for i in range(0, length_x, batch_size):
            idx = perm[i:i + batch_size]
            X_batch, y_batch = X_train[idx].to(device), y_train[idx].to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = F.cross_entropy(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(idx)

        val_loss, val_acc = evaluate_model(model, val_loader, device)
        history['train_loss'].append(running_loss / length_x)  # FIX 4
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {running_loss / length_x:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    return history

def question7():
    torch.manual_seed(0)
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_mnist()
    print(X_tr.shape, X_va.shape, X_te.shape) 

    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_va, y_va), batch_size=1000)

    model = MNISTConvNet().to(device)            
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)   
    train(model, (X_tr, y_tr), val_loader, optimizer, device, epochs=5)

def tune():
    import json
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_mnist()
    val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_va, y_va), batch_size=1000)

    results = {}
    for lr in lr_list:
        print(f"\n= lr = {lr} =")
        torch.manual_seed(0)                     
        model = MNISTConvNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        history = train(model, (X_tr, y_tr), val_loader, optimizer, device, epochs=8)
        results[lr] = history

   
    print("\nlr       best val acc   best epoch   final val acc")
    for lr, h in results.items():
        best = max(range(len(h['val_acc'])), key=lambda i: h['val_acc'][i])
        print(f"{lr:<8} {h['val_acc'][best]:.4f}         {best + 1}            {h['val_acc'][-1]:.4f}")

    with open("q8_results.json", "w") as f:       
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["q7", "tune"], help="which experiment to run")
    args = parser.parse_args()

    tasks = {"q7": question7, "tune": tune}
    tasks[args.task]()

