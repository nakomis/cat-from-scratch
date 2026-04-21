# Datasets

The dataset archives and expanded directories are excluded from git (see `.gitignore`).
Download and expand them here before running any training scripts.

---

## Oxford-IIIT Pet Dataset (`oxford/`)

**Download:** https://www.robots.ox.ac.uk/~vgg/data/pets/

Download the two archives and expand them into this directory:

```bash
cd data/
curl -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
curl -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xzf images.tar.gz
tar -xzf annotations.tar.gz
```

Expected structure after expanding:
```
data/oxford/
  images/          # .jpg files, e.g. Abyssinian_1.jpg
  annotations/
    xmls/          # PASCAL VOC bounding boxes, e.g. Abyssinian_1.xml
    trimaps/
    list.txt
    trainval.txt
    test.txt
```

---

## Kaggle Cat Breeds Dataset (`kaggle_cats/`)

**Download:** https://www.kaggle.com/datasets/ma7555/cat-breeds-dataset

Requires a Kaggle account. Download via the Kaggle CLI:

```bash
pip install kaggle
kaggle datasets download ma7555/cat-breeds-dataset -p data/
cd data/
unzip cat-breeds-dataset.zip -d kaggle_cats/
```

Expected structure after expanding:
```
data/kaggle_cats/
  images/
    Abyssinian/
    Bengal/
    ...
```

Used as supplementary training data alongside Oxford. Pass to the training script with:
```bash
python scripts/train_oxford_v2.py --extra-data data/kaggle_cats/images/
```

---

## MS Cats vs Dogs (`PetImages/`)

**Download:** https://www.microsoft.com/en-us/download/details.aspx?id=54765

Download `kagglecatsanddogs_5340.zip` and expand here:

```bash
cd data/
unzip kagglecatsanddogs_5340.zip
```

Expected structure:
```
data/PetImages/
  Cat/   # ~12,000 .jpg files
  Dog/   # ~12,000 .jpg files
```

Used by `scripts/train.py` for the initial cats vs dogs classifier.
