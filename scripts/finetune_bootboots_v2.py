"""
Fine-tune the CatCNNv2 Oxford backbone on the BootBoots 7-class dataset.

Usage:
    python finetune_bootboots_v2.py
    python finetune_bootboots_v2.py --backbone best_oxford_v2.pt
    python finetune_bootboots_v2.py --frozen-epochs 10 --full-epochs 30
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

from model_v2 import CatCNNv2

DATA_DIR        = os.path.expanduser("~/repos/nakomis/bootboots/local-training/data_multiclass")
ROOT            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_SIZE        = 128
BATCH_SIZE      = 32
FROZEN_EPOCHS   = 10   # head-only warm-up
FULL_EPOCHS     = 30   # end-to-end fine-tune
LR_HEAD         = 1e-3
LR_FULL         = 5e-5  # low enough not to destroy the pretrained features
EXCLUDE_CLASSES = {"Wolf"}  # stuffed toy — not a real cat; too few images, features won't transfer

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
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
    from logger import setup_logging

    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default=os.path.join(ROOT, "models", "best_oxford_v2.pt"),
                        help="Pretrained CatCNNv2 weights (default: best_oxford_v2.pt)")
    parser.add_argument("--frozen-epochs", type=int, default=FROZEN_EPOCHS)
    parser.add_argument("--full-epochs",   type=int, default=FULL_EPOCHS)
    args = parser.parse_args()
    setup_logging()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Backbone: {args.backbone}")

    # --- Data ---
    train_folder = ImageFolder(os.path.join(DATA_DIR, "training"))
    val_folder   = ImageFolder(os.path.join(DATA_DIR, "validation"))

    # Build a remapped class list excluding Wolf (and any other excluded classes)
    classes     = [c for c in train_folder.classes if c not in EXCLUDE_CLASSES]
    num_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    # Remap samples, dropping excluded classes
    def remap(folder):
        return [
            (path, class_to_idx[folder.classes[lbl]])
            for path, lbl in folder.samples
            if folder.classes[lbl] not in EXCLUDE_CLASSES
        ]

    train_samples = remap(train_folder)
    val_samples   = remap(val_folder)

    print(f"\nClasses ({num_classes}): {classes}")
    for i, c in enumerate(classes):
        n_train = sum(1 for _, lbl in train_samples if lbl == i)
        n_val   = sum(1 for _, lbl in val_samples   if lbl == i)
        print(f"  {c}: {n_train} train, {n_val} val")

    train_set = BootBootsDataset(train_samples, train_transforms)
    val_set   = BootBootsDataset(val_samples,   val_transforms)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, collate_fn=collate_skip_none)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, collate_fn=collate_skip_none)

    # Weighted loss — inverse frequency, normalised
    class_counts = [sum(1 for _, lbl in train_samples if lbl == i)
                    for i in range(num_classes)]
    weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float)
    weights = (weights / weights.sum() * num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # --- Model ---
    # Load backbone trained on 12 Oxford breeds, swap head for 7 BootBoots classes
    model = CatCNNv2(num_classes=12)
    model.load_state_dict(torch.load(args.backbone, map_location="cpu"))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes),
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {total_params:,}")

    best_val_acc = 0.0
    save_path = os.path.join(ROOT, "models", "best_bootboots_v2.pt")

    # -----------------------------------------------------------------------
    # Phase 1: freeze backbone, warm up the new head
    # -----------------------------------------------------------------------
    print(f"\n--- Phase 1: frozen backbone ({args.frozen_epochs} epochs) ---")
    for p in model.stem.parameters():   p.requires_grad = False
    for p in model.stage1.parameters(): p.requires_grad = False
    for p in model.stage2.parameters(): p.requires_grad = False
    for p in model.stage3.parameters(): p.requires_grad = False
    for p in model.stage4.parameters(): p.requires_grad = False

    optimiser = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD
    )

    for epoch in range(1, args.frozen_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimiser, device, train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimiser, device, train=False)
        print(f"Epoch {epoch:02d}/{args.frozen_epochs}  "
              f"train loss: {train_loss:.4f}  acc: {train_acc:.1f}%  |  "
              f"val loss: {val_loss:.4f}  acc: {val_acc:.1f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  → saved new best ({val_acc:.1f}%)")

    # -----------------------------------------------------------------------
    # Phase 2: unfreeze everything, fine-tune end-to-end at low LR
    # -----------------------------------------------------------------------
    print(f"\n--- Phase 2: full fine-tune ({args.full_epochs} epochs) ---")
    for p in model.parameters():
        p.requires_grad = True

    optimiser = optim.Adam(model.parameters(), lr=LR_FULL, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.full_epochs)

    for epoch in range(1, args.full_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimiser, device, train=True)
        val_loss,   val_acc   = run_epoch(model, val_loader,   criterion, optimiser, device, train=False)
        scheduler.step()
        print(f"Epoch {epoch:02d}/{args.full_epochs}  "
              f"train loss: {train_loss:.4f}  acc: {train_acc:.1f}%  |  "
              f"val loss: {val_loss:.4f}  acc: {val_acc:.1f}%")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  → saved new best ({val_acc:.1f}%)")

    print(f"\nFine-tuning complete. Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Classes: {classes}")
    print(f"Model saved to {save_path}")

    # Auto-generate plot
    import subprocess
    log_files = sorted(
        [os.path.join("logs", f) for f in os.listdir("logs")
         if f.startswith("finetune_bootboots_v2_")],
        key=os.path.getmtime,
    )
    if log_files:
        try:
            subprocess.run([sys.executable, "plot_training.py", log_files[-1]],
                           check=True, capture_output=True)
            print("Plot saved alongside log.")
        except Exception as e:
            print(f"Plot generation failed: {e}")
