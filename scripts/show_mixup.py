"""
Show a Mixup example: blend a Maine Coon and a British Shorthair at 70/30.
"""

import os
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from PIL import Image

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(ROOT, "data", "oxford", "images")

def load_random(breed):
    files = [f for f in os.listdir(IMAGES_DIR)
             if f.startswith(breed + "_") and f.endswith(".jpg")]
    path = os.path.join(IMAGES_DIR, random.choice(files))
    return Image.open(path).convert("RGB").resize((256, 256))

random.seed(1)
maine  = load_random("Maine_Coon")
brit   = load_random("British_Shorthair")

lam = 0.7
mixed = Image.fromarray(
    (np.array(maine) * lam + np.array(brit) * (1 - lam)).astype(np.uint8)
)

fig = plt.figure(figsize=(12, 4))
gs  = gridspec.GridSpec(1, 4, width_ratios=[1, 0.15, 1, 1])

ax1 = fig.add_subplot(gs[0])
ax1.imshow(maine)
ax1.set_title("Maine Coon\n(λ = 0.7)", fontsize=12)
ax1.axis("off")

ax_plus = fig.add_subplot(gs[1])
ax_plus.text(0.5, 0.5, "+", fontsize=28, ha="center", va="center")
ax_plus.axis("off")

ax2 = fig.add_subplot(gs[2])
ax2.imshow(brit)
ax2.set_title("British Shorthair\n(1−λ = 0.3)", fontsize=12)
ax2.axis("off")

ax3 = fig.add_subplot(gs[3])
ax3.imshow(mixed)
ax3.set_title("Mixup result\n(label: 70% Maine Coon,\n30% British Shorthair)", fontsize=12)
ax3.axis("off")

os.makedirs("plots", exist_ok=True)
plt.suptitle("Mixup data augmentation", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("plots/mixup_example.png", dpi=150, bbox_inches="tight")
print("Saved to plots/mixup_example.png")
plt.show()
