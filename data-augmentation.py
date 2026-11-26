import argparse                                                 
import json
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms as T

from MNISTConvNet import load_mnist, evaluate_model, MNISTConvNet, device

best_lr = 1e-3
baseline_epochs = 6                                             

AUGMENTATIONS = {
    'none':      [],
    'rotation':  [T.RandomRotation(10)],
    'translate': [T.RandomAffine(degrees=0, translate=(0.1, 0.1))],
    'affine':    [T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))],
}
criterion = torch.nn.CrossEntropyLoss()


class TransformedDataset(Dataset):
    def __init__(self, original_dataset, indices, transform):
        self.original_dataset = original_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, label = self.original_dataset[int(self.indices[idx])]
        img = self.transform(img)
        return img, label

def make_train_loader(aug_type, batch_size=16, shuffle=True):
    raw_train = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    perm = torch.randperm(len(raw_train), generator=torch.Generator().manual_seed(0))
    transform = T.Compose(AUGMENTATIONS[aug_type] + [T.ToTensor()])
    train_set = TransformedDataset(raw_train, perm[:50000], transform)
    return DataLoader(train_set, batch_size=batch_size, shuffle=shuffle)

def train_model(aug_type, model, criterion, optimizer, val_loader, num_epochs=8):
    loader = make_train_loader(aug_type)
    history = {'val_loss': [], 'val_acc': []}
    for epoch in range(num_epochs):
        model.train()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        val_loss, val_acc = evaluate_model(model, val_loader, device)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        print(f"{aug_type} | Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    return model, history                                        

def run_experiment(aug_type, val_loader, num_epochs=8):
    torch.manual_seed(0)                                         
    model = MNISTConvNet().to(device)                            
    optimizer = torch.optim.Adam(model.parameters(), lr=best_lr)
    return train_model(aug_type, model, criterion, optimizer, val_loader, num_epochs)   

def compare(epochs):
    _, (X_va, y_va), _ = load_mnist()
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=1000)

    results = {}
    for name in AUGMENTATIONS:
        print(f"\n= {name} =")
        _, results[name] = run_experiment(name, val_loader, epochs)   

    print("\naugmentation   best val acc   best epoch   final val acc")
    for name, h in results.items():
        best = max(range(epochs), key=lambda i: h['val_acc'][i])
        print(f"{name:<14} {h['val_acc'][best]:.4f}         {best + 1}            {h['val_acc'][-1]:.4f}")

    with open("q9_results.json", "w") as f:
        json.dump(results, f, indent=2)

def test(aug_name, aug_epochs):                                  
    _, (X_va, y_va), (X_te, y_te) = load_mnist()                 
    test_loader = DataLoader(TensorDataset(X_te, y_te), batch_size=1000)
    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=1000)
    out = {}
    for name, epochs in [('none', baseline_epochs), (aug_name, aug_epochs)]:   
        print(f"\n= {name}, {epochs} epochs =")
        model, _ = run_experiment(name, val_loader, epochs)      
        test_loss, test_acc = evaluate_model(model, test_loader, device)
        out[name] = {'epochs': epochs, 'test_loss': test_loss, 'test_acc': test_acc}

    print("\nmodel          epochs   test loss   test acc")
    for name, r in out.items():
        print(f"{name:<14} {r['epochs']:<8} {r['test_loss']:.4f}      {r['test_acc']:.4f}")

    with open("q9_test_results.json", "w") as f:
        json.dump(out, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["compare", "test"])
    parser.add_argument("--epochs", type=int, default=8, help="epochs per run in 'compare'")
    parser.add_argument("--aug", choices=list(AUGMENTATIONS), default="affine", help="augmentation for 'test'")
    parser.add_argument("--aug-epochs", type=int, default=8, help="epochs for the augmented model in 'test'")
    args = parser.parse_args()

    if args.task == "compare":
        compare(args.epochs)
    else:
        test(args.aug, args.aug_epochs)