# cat-from-scratch

Training a CNN image classifier entirely from scratch — no pretrained weights — on a MacBook Pro with Apple Silicon, then fine-tuning it to recognise individual cats.

**Blog post:** [How to Train a Cat Recogniser From Scratch on a MacBook](https://blog.nakomis.com/2026-04-21-cat-recogniser-from-scratch) *(coming soon)*

---

## What we built

A ResNet-style convolutional neural network trained on the Oxford-IIIT Pet dataset (12 cat breeds) supplemented with Kaggle cat breed images (~21,000 images total), then fine-tuned on a small dataset of six specific cats.

**Final results:**

| Model | Task | Val accuracy |
|-------|------|-------------|
| CatCNNv2 (from scratch) | Oxford 12 breeds | 88.5% |
| CatCNNv2 fine-tuned | BootBoots 6 cats | **89.6%** |
| EfficientNetV2-S (ImageNet pretrained) | BootBoots 6 cats | 94.6% |

From scratch gets within 5 points of a model pretrained on 1.2 million images.

---

## Architecture

`CatCNNv2` — a from-scratch ResNet-style CNN:
- 4 stages of residual blocks: 64 → 128 → 256 → 512 channels
- Global Average Pooling (replaces flatten, dramatically reduces overfitting)
- 17.8M trainable parameters
- Trained on Apple Silicon MPS backend via PyTorch

Key training techniques: Mixup data augmentation, RandomResizedCrop, cosine annealing LR, weighted CrossEntropyLoss, head bounding-box crops from Oxford annotations.

---

## Repository structure

```
scripts/          Training and utility scripts
  model_v2.py           CatCNNv2 architecture + Mixup
  train_oxford_v2.py    Main Oxford+Kaggle training script
  finetune_bootboots_v2.py   Fine-tune on custom cats (from-scratch backbone)
  finetune_bootboots_imagenet.py  Fine-tune EfficientNetV2-S for comparison
  live_plot.py          Live-updating training chart
  plot_training.py      Generate accuracy plots from log files
  show_mixup.py         Visualise Mixup data augmentation
  logger.py             Tee stdout to timestamped log files

data/             Datasets (see data/README.md for download instructions)
  bootboots-training-data.tar.gz   BootBoots cat images (Git LFS, 145MB)

models/           Saved model weights (gitignored — train to regenerate)
logs/             Training logs (gitignored)
plots/            Training plots (gitignored)
```

---

## Getting started

### Prerequisites

- Python 3.11 (via [asdf](https://asdf-vm.com/) or any other version manager)
- [Git LFS](https://git-lfs.com/) — required to download the BootBoots training data

```bash
git lfs install   # one-time setup
git clone https://github.com/nakomis/cat-from-scratch
cd cat-from-scratch
git lfs pull      # downloads bootboots-training-data.tar.gz (~145MB)
```

### Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Download datasets

See [data/README.md](data/README.md) for instructions to download the Oxford-IIIT Pet and Kaggle cat breed datasets.

### Train

```bash
# Oxford breeds only (12 classes)
python scripts/train_oxford_v2.py

# Oxford + Kaggle extra data (recommended)
python scripts/train_oxford_v2.py --extra-data data/kaggle_cats/images/

# Resume from a checkpoint
python scripts/train_oxford_v2.py \
  --extra-data data/kaggle_cats/images/ \
  --resume models/best_oxford_v2.pt \
  --resume-epoch 50 \
  --epochs 90
```

### Fine-tune on your own cats

Extract the BootBoots training data and point the fine-tune script at it:

```bash
cd data && tar -xzf bootboots-training-data.tar.gz && cd ..

python scripts/finetune_bootboots_v2.py \
  --backbone models/best_oxford_v2.pt
```

The dataset follows an ImageFolder structure:
```
data_multiclass/
  training/
    CatName/  image1.jpg  image2.jpg  ...
  validation/
    CatName/  ...
```

### Monitor training

```bash
# In a separate terminal — auto-picks the latest log
python scripts/live_plot.py --interval 30
```

---

## BootBoots training data

`data/bootboots-training-data.tar.gz` (tracked via [Git LFS](https://git-lfs.com/)) contains labelled images of six cats: Boots, Chi, Kappa, Mu, Tau, and a NoCat class. ~2,900 images total across training, validation, and manual test splits.

To use your own cats instead, build an ImageFolder-style directory tree and pass it to the fine-tune script.

---

## Licence

[CC0 1.0 Universal](LICENCE) — public domain. Do what you like.
