"""
Train CatCNNv2 on Oxford cat breeds (+ optional extra dataset directories).

Usage:
    python train_oxford_v2.py
    python scripts/train_oxford_v2.py --extra-data data/kaggle_cats/images/
    python scripts/train_oxford_v2.py --epochs 60 --extra-data data/kaggle_cats/images/
"""

import argparse
import os
import re
import sys
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image

from model_v2 import CatCNNv2, mixup_batch

ROOT              = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OXFORD_IMAGES     = os.path.join(ROOT, "data", "oxford", "images")
OXFORD_ANNOT      = os.path.join(ROOT, "data", "oxford", "annotations", "xmls")
IMG_SIZE          = 128
BATCH_SIZE        = 64
DEFAULT_EPOCHS    = 40
LR                = 1e-3
VAL_SPLIT         = 0.2
HEAD_PAD          = 0.2
MIXUP_ALPHA       = 0.3

# Oxford cat breeds: filenames starting with a capital letter
CAT_BREEDS = sorted({
    re.match(r"^(.+)_\d+\.jpg$", f).group(1)
    for f in os.listdir(OXFORD_IMAGES)
    if f.endswith(".jpg") and re.match(r"^[A-Z]", f) and re.match(r"^(.+)_\d+\.jpg$", f)
})

# ---------------------------------------------------------------------------
# Transforms — RandomResizedCrop for train (much more variety than Resize)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_head_box(stem):
    xml_path = os.path.join(OXFORD_ANNOT, stem + ".xml")
    if not os.path.exists(xml_path):
        return None
    try:
        root = ET.parse(xml_path).getroot()
        box = root.find(".//bndbox")
        return (int(box.find("xmin").text), int(box.find("ymin").text),
                int(box.find("xmax").text), int(box.find("ymax").text))
    except Exception:
        return None


def crop_head(img, box, pad=HEAD_PAD):
    w, h = img.size
    xmin, ymin, xmax, ymax = box
    bw, bh = xmax - xmin, ymax - ymin
    return img.crop((
        max(0, xmin - int(bw * pad)), max(0, ymin - int(bh * pad)),
        min(w, xmax + int(bw * pad)), min(h, ymax + int(bh * pad)),
    ))


class CatBreedDataset(Dataset):
    """samples: list of (path, stem_or_None, label)"""
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, stem, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            if stem:
                box = load_head_box(stem)
                if box:
                    img = crop_head(img, box)
            return self.transform(img), label
        except Exception:
            return None


def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    if not batch:
        return torch.zeros(0), torch.zeros(0, dtype=torch.long)
    images, labels = zip(*batch)
    return torch.stack(images), torch.tensor(labels)


def build_oxford_samples(breed_to_idx):
    samples = []
    for fname in os.listdir(OXFORD_IMAGES):
        if not fname.endswith(".jpg"):
            continue
        m = re.match(r"^(.+)_\d+\.jpg$", fname)
        if m and m.group(1) in breed_to_idx:
            samples.append((
                os.path.join(OXFORD_IMAGES, fname),
                fname[:-4],
                breed_to_idx[m.group(1)],
            ))
    return samples


def normalise_breed_name(name):
    """Normalise folder names to match Oxford convention (spaces→underscores).
    Also handles Kaggle quirks like 'Sphynx - Hairless Cat' → 'Sphynx'.
    """
    name = name.replace(" - Hairless Cat", "")  # Kaggle Sphynx variant
    return name.replace(" ", "_")


def build_extra_samples(extra_dir, breed_to_idx):
    """
    Load images from an ImageFolder-style directory tree:
        extra_dir/BreedName/image.jpg
    Normalises folder names to Oxford convention before matching.
    """
    samples = []
    for raw_breed in os.listdir(extra_dir):
        breed = normalise_breed_name(raw_breed)
        if breed not in breed_to_idx:
            continue
        breed_dir = os.path.join(extra_dir, raw_breed)
        if not os.path.isdir(breed_dir):
            continue
        for fname in os.listdir(breed_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                samples.append((
                    os.path.join(breed_dir, fname),
                    None,   # no Oxford XML annotations for extra data
                    breed_to_idx[breed],
                ))
    return samples


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_epoch(model, loader, criterion, optimiser, device, num_classes, train, use_mixup):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        if images.size(0) == 0:
            continue
        images, labels = images.to(device), labels.to(device)

        if train:
            optimiser.zero_grad()

        if train and use_mixup:
            images, soft_labels = mixup_batch(images, labels, num_classes, MIXUP_ALPHA)
            outputs = model(images)
            # Soft cross-entropy: -sum(soft_labels * log_softmax(outputs))
            loss = -(soft_labels * F.log_softmax(outputs, dim=1)).sum(dim=1).mean()
        else:
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
    parser.add_argument("--extra-data", default=None,
                        help="Path to an ImageFolder-style directory of extra cat breed images")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--no-mixup", action="store_true")
    parser.add_argument("--resume", default=None,
                        help="Path to checkpoint to resume from (e.g. best_oxford_v2.pt)")
    parser.add_argument("--resume-epoch", type=int, default=0,
                        help="Epoch number the checkpoint was saved at (for scheduler state)")
    args = parser.parse_args()
    setup_logging()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Cat breeds ({len(CAT_BREEDS)}): {CAT_BREEDS}")

    breed_to_idx = {b: i for i, b in enumerate(CAT_BREEDS)}

    # Build sample lists
    oxford_samples = build_oxford_samples(breed_to_idx)
    extra_samples  = build_extra_samples(args.extra_data, breed_to_idx) if args.extra_data else []
    all_samples    = oxford_samples + extra_samples

    print(f"\nOxford samples: {len(oxford_samples)}")
    if extra_samples:
        print(f"Extra samples:  {len(extra_samples)}")
    print(f"Total:          {len(all_samples)}")

    # Stratified 80/20 split (Oxford only for val — we own those labels)
    train_samples, val_samples = [], []
    for idx in range(len(CAT_BREEDS)):
        class_ox = [s for s in oxford_samples if s[2] == idx]
        torch.manual_seed(42 + idx)
        perm  = torch.randperm(len(class_ox)).tolist()
        n_val = max(1, int(len(class_ox) * VAL_SPLIT))
        val_samples   += [class_ox[i] for i in perm[:n_val]]
        train_samples += [class_ox[i] for i in perm[n_val:]]

    # All extra data goes to train
    train_samples += extra_samples

    print(f"Train: {len(train_samples)}  Val: {len(val_samples)}")
    for breed, idx in breed_to_idx.items():
        n = sum(1 for s in train_samples if s[2] == idx)
        print(f"  {breed}: {n} train")

    train_set = CatBreedDataset(train_samples, train_transforms)
    val_set   = CatBreedDataset(val_samples,   val_transforms)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, collate_fn=collate_skip_none)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, collate_fn=collate_skip_none)

    model = CatCNNv2(num_classes=len(CAT_BREEDS)).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nParameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    # T_max covers the full intended run so LR decays correctly to the end
    total_epochs = args.resume_epoch + args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=total_epochs)

    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        # Fast-forward the scheduler to where we left off
        for _ in range(args.resume_epoch):
            scheduler.step()
        print(f"Resumed from {args.resume} at epoch {args.resume_epoch}")

    use_mixup = not args.no_mixup
    print(f"Mixup: {'on' if use_mixup else 'off'}  Epochs: {args.epochs}  Total: {total_epochs}")

    best_val_acc = 0.0
    save_path = os.path.join(ROOT, "models", "best_oxford_v2.pt")

    for epoch in range(args.resume_epoch + 1, total_epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimiser,
                                          device, len(CAT_BREEDS), train=True, use_mixup=use_mixup)
        val_loss, val_acc     = run_epoch(model, val_loader,   criterion, optimiser,
                                          device, len(CAT_BREEDS), train=False, use_mixup=False)
        scheduler.step()
        print(
            f"Epoch {epoch:02d}/{total_epochs}  "
            f"train loss: {train_loss:.4f}  acc: {train_acc:.1f}%  |  "
            f"val loss: {val_loss:.4f}  acc: {val_acc:.1f}%"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  → saved new best ({val_acc:.1f}%)")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Model saved to {save_path}")

    # Auto-generate plot from the log file written by setup_logging()
    import subprocess
    log_files = sorted(
        [os.path.join("logs", f) for f in os.listdir("logs") if f.startswith("train_oxford_v2_")],
        key=os.path.getmtime,
    )
    if log_files:
        latest_log = log_files[-1]
        try:
            subprocess.run([sys.executable, "plot_training.py", latest_log],
                           check=True, capture_output=True)
            print("Plot saved alongside log.")
        except Exception as e:
            print(f"Plot generation failed: {e}")
