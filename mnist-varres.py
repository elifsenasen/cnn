import json
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from MNISTConvNet import MNISTConvNet, train, device

ROOT = 'mnist-varres'
LR = 1e-3
EPOCHS = 8

def load_data(n_val=10000):
    transform = T.Compose([
        T.Grayscale(num_output_channels=1),        
        T.Resize((28, 28)),
        T.ToTensor(),
    ])
    train_dataset = ImageFolder(root=f'{ROOT}/train', transform=transform)
    test_dataset = ImageFolder(root=f'{ROOT}/test', transform=transform)

    #one tensor
    X_all, y_all = next(iter(DataLoader(train_dataset, batch_size=len(train_dataset))))
    X_test, y_test = next(iter(DataLoader(test_dataset, batch_size=len(test_dataset))))

    perm = torch.randperm(len(X_all), generator=torch.Generator().manual_seed(0))
    tr_idx, va_idx = perm[n_val:], perm[:n_val]
    return (X_all[tr_idx], y_all[tr_idx]), (X_all[va_idx], y_all[va_idx]), (X_test, y_test)

if __name__ == "__main__":
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_data()
    print(X_tr.shape, X_va.shape, X_te.shape)     

    val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=1000)

    torch.manual_seed(0)
    model = MNISTConvNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    history = train(model, (X_tr, y_tr), val_loader, optimizer, device, epochs=EPOCHS)   

    best = max(range(EPOCHS), key=lambda i: history['val_acc'][i])                       
    print(f"\nbest val acc {history['val_acc'][best]:.4f} at epoch {best + 1}")

    with open("q12_results.json", "w") as f:
        json.dump(history, f, indent=2)