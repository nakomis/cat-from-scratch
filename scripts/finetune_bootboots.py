import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from train import CatDogCNN

DATA_DIR = os.path.expanduser("~/repos/nakomis/bootboots/local-training/data_multiclass")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS_FROZEN = 5    # train only the new head, feature layers frozen
EPOCHS_UNFROZEN = 10 # then unfreeze everything with a lower lr
LR_HEAD = 1e-3
LR_FULL = 1e-4

train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class BootBootsDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), label
        except Exception:
            return None


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return torch.zeros(0), torch.zeros(0, dtype=torch.long)
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels)


def run_epoch(model, loader, criterion, optimiser, device, train):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for images, labels in loader:
            if images.size(0) == 0:
                continue
            images, labels = images.to(device), labels.to(device)
            if train:
                optimiser.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                loss.backward()
                optimiser.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)
    return total_loss / total, 100 * correct / total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backbone",
        default=os.path.join(ROOT, "models", "best_oxford_cats.pt"),
        help="Pretrained weights to start from (default: best_oxford_cats.pt)",
    )
    args = parser.parse_args()

    from logger import setup_logging
    setup_logging()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Backbone: {args.backbone}")

    # --- Data ---
    train_base = ImageFolder(os.path.join(DATA_DIR, "training"))
    val_base   = ImageFolder(os.path.join(DATA_DIR, "validation"))
    classes = train_base.classes
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")
    for c in classes:
        n_train = len([s for s in train_base.samples if train_base.classes[s[1]] == c])
        n_val   = len([s for s in val_base.samples   if val_base.classes[s[1]] == c])
        print(f"  {c}: {n_train} train, {n_val} val")

    train_set = BootBootsDataset(train_base.samples, train_transforms)
    val_set   = BootBootsDataset(val_base.samples,   val_transforms)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, collate_fn=collate_skip_none)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, collate_fn=collate_skip_none)

    # Weighted loss to handle class imbalance (Wolf: 50, NoCat: 470)
    class_counts = [len([s for s in train_base.samples if s[1] == i]) for i in range(num_classes)]
    weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float).to(device)
    weights = weights / weights.sum() * num_classes  # normalise
    criterion = nn.CrossEntropyLoss(weight=weights)

    # --- Model: load pretrained backbone, swap the head ---
    model = CatDogCNN()
    model.load_state_dict(torch.load(args.backbone, map_location="cpu"))

    # Replace the 2-class output with num_classes
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,}")

    best_val_acc = 0.0

    # -----------------------------------------------------------------------
    # Phase 1: freeze feature layers, train head only
    # -----------------------------------------------------------------------
    print(f"\n--- Phase 1: frozen features ({EPOCHS_FROZEN} epochs) ---")
    for p in model.features.parameters():
        p.requires_grad = False

    optimiser = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)

    for epoch in range(1, EPOCHS_FROZEN + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimiser, device, train=True)
        val_loss, val_acc     = run_epoch(model, val_loader,   criterion, optimiser, device, train=False)
        print(f"Epoch {epoch:02d}/{EPOCHS_FROZEN}  train loss: {train_loss:.4f}  acc: {train_acc:.1f}%  |  val loss: {val_loss:.4f}  acc: {val_acc:.1f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(ROOT, "models", "best_bootboots.pt"))
            print(f"  → saved new best ({val_acc:.1f}%)")

    # -----------------------------------------------------------------------
    # Phase 2: unfreeze everything, fine-tune end-to-end at lower lr
    # -----------------------------------------------------------------------
    print(f"\n--- Phase 2: full fine-tune ({EPOCHS_UNFROZEN} epochs) ---")
    for p in model.features.parameters():
        p.requires_grad = True

    optimiser = optim.Adam(model.parameters(), lr=LR_FULL)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS_UNFROZEN)

    for epoch in range(1, EPOCHS_UNFROZEN + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimiser, device, train=True)
        val_loss, val_acc     = run_epoch(model, val_loader,   criterion, optimiser, device, train=False)
        scheduler.step()
        print(f"Epoch {epoch:02d}/{EPOCHS_UNFROZEN}  train loss: {train_loss:.4f}  acc: {train_acc:.1f}%  |  val loss: {val_loss:.4f}  acc: {val_acc:.1f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(ROOT, "models", "best_bootboots.pt"))
            print(f"  → saved new best ({val_acc:.1f}%)")

    print(f"\nFine-tuning complete. Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Classes: {classes}")
