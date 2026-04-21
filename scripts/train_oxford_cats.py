import os
import re
import xml.etree.ElementTree as ET
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from train import CatDogCNN

ROOT            = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR      = os.path.join(ROOT, "data", "oxford", "images")
ANNOTATIONS_DIR = os.path.join(ROOT, "data", "oxford", "annotations", "xmls")
IMG_SIZE   = 128
BATCH_SIZE = 64
EPOCHS     = 20
LR         = 1e-3
VAL_SPLIT  = 0.2
HEAD_PAD   = 0.2  # expand the bounding box by 20% on each side for context

# Cat breeds start with a capital letter; dog breeds are all lowercase.
# Filenames are {Breed_Name}_{number}.jpg — extract breed as everything before _\d+.jpg
CAT_BREEDS = sorted({
    re.match(r"^(.+)_\d+\.jpg$", f).group(1)
    for f in os.listdir(IMAGES_DIR)
    if f.endswith(".jpg") and re.match(r"^[A-Z]", f) and re.match(r"^(.+)_\d+\.jpg$", f)
})


def load_head_box(stem):
    """Return (xmin, ymin, xmax, ymax) from the PASCAL VOC XML, or None if missing."""
    xml_path = os.path.join(ANNOTATIONS_DIR, stem + ".xml")
    if not os.path.exists(xml_path):
        return None
    try:
        root = ET.parse(xml_path).getroot()
        box = root.find(".//bndbox")
        return (
            int(box.find("xmin").text),
            int(box.find("ymin").text),
            int(box.find("xmax").text),
            int(box.find("ymax").text),
        )
    except Exception:
        return None


def crop_head(img, box, pad=HEAD_PAD):
    """Crop to the head bounding box with proportional padding."""
    w, h = img.size
    xmin, ymin, xmax, ymax = box
    bw, bh = xmax - xmin, ymax - ymin
    xmin = max(0, xmin - int(bw * pad))
    ymin = max(0, ymin - int(bh * pad))
    xmax = min(w, xmax + int(bw * pad))
    ymax = min(h, ymax + int(bh * pad))
    return img.crop((xmin, ymin, xmax, ymax))


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


class OxfordCatDataset(Dataset):
    def __init__(self, samples, transform):
        # samples: list of (path, stem, label)
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, stem, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
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
    setup_logging()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Cat breeds ({len(CAT_BREEDS)}): {CAT_BREEDS}")

    breed_to_idx = {b: i for i, b in enumerate(CAT_BREEDS)}
    all_samples = []
    for fname in os.listdir(IMAGES_DIR):
        if not fname.endswith(".jpg"):
            continue
        m = re.match(r"^(.+)_(\d+)\.jpg$", fname)
        if m and m.group(1) in breed_to_idx:
            stem = fname[:-4]  # strip .jpg
            all_samples.append((
                os.path.join(IMAGES_DIR, fname),
                stem,
                breed_to_idx[m.group(1)],
            ))

    n_with_box = sum(1 for _, stem, _ in all_samples if load_head_box(stem) is not None)
    print(f"Total cat images: {len(all_samples)}  ({n_with_box} with head bounding box)")
    for breed, idx in breed_to_idx.items():
        n = sum(1 for _, _, l in all_samples if l == idx)
        print(f"  {breed}: {n}")

    # Stratified 80/20 train/val split
    train_samples, val_samples = [], []
    for idx in range(len(CAT_BREEDS)):
        class_samples = [s for s in all_samples if s[2] == idx]
        torch.manual_seed(42 + idx)
        perm = torch.randperm(len(class_samples)).tolist()
        n_val = max(1, int(len(class_samples) * VAL_SPLIT))
        val_samples   += [class_samples[i] for i in perm[:n_val]]
        train_samples += [class_samples[i] for i in perm[n_val:]]

    print(f"\nTrain: {len(train_samples)}  Val: {len(val_samples)}")

    train_set = OxfordCatDataset(train_samples, train_transforms)
    val_set   = OxfordCatDataset(val_samples,   val_transforms)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, collate_fn=collate_skip_none)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, collate_fn=collate_skip_none)

    model = CatDogCNN()
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(CAT_BREEDS))
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=EPOCHS)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimiser, device, train=True)
        val_loss, val_acc     = run_epoch(model, val_loader,   criterion, optimiser, device, train=False)
        scheduler.step()
        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"train loss: {train_loss:.4f}  acc: {train_acc:.1f}%  |  "
            f"val loss: {val_loss:.4f}  acc: {val_acc:.1f}%"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(ROOT, "models", "best_oxford_cats.pt"))
            print(f"  → saved new best ({val_acc:.1f}%)")

    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Backbone saved to models/best_oxford_cats.pt — use this for BootBoots fine-tuning.")
