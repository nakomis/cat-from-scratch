"""
Parse training log output and plot accuracy vs epoch.

Usage:
    python plot_training.py <log_file> [<log_file2> ...]
    python plot_training.py training.log
    cat training.log | python plot_training.py -
"""

import os
import re
import sys
import argparse
import matplotlib.pyplot as plt


def parse_log(text):
    """Extract (epoch, train_acc, val_acc) tuples from training output."""
    pattern = re.compile(
        r"Epoch\s+(\d+)/\d+.*?acc:\s+([\d.]+)%.*?acc:\s+([\d.]+)%"
    )
    results = []
    for m in pattern.finditer(text):
        results.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
    return results


def load(path):
    if path == "-":
        return sys.stdin.read()
    with open(path) as f:
        return f.read()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", help="Log file paths (use - for stdin)")
    parser.add_argument("--output", default=None,
                        help="Output PNG path (default: alongside first log file)")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(10, 6))
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, path in enumerate(args.logs):
        data = parse_log(load(path))
        if not data:
            print(f"No epoch data found in {path}", file=sys.stderr)
            continue

        epochs     = [d[0] for d in data]
        train_accs = [d[1] for d in data]
        val_accs   = [d[2] for d in data]
        label      = path if path != "-" else "stdin"
        c          = colours[i % len(colours)]

        ax.plot(epochs, train_accs, linestyle="--", color=c, alpha=0.7,
                label=f"{label} — train")
        ax.plot(epochs, val_accs,   linestyle="-",  color=c,
                label=f"{label} — val")

        best_val = max(val_accs)
        best_ep  = epochs[val_accs.index(best_val)]
        ax.annotate(f"{best_val:.1f}%",
                    xy=(best_ep, best_val),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=9, color=c)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training vs Validation Accuracy")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=100/12, color="grey", linestyle=":", alpha=0.5, label="random chance (8.3%)")

    plt.tight_layout()
    if args.output:
        out = args.output
    elif args.logs[0] != "-":
        os.makedirs("plots", exist_ok=True)
        base = os.path.splitext(os.path.basename(args.logs[0]))[0]
        out = os.path.join("plots", base + "_plot.png")
    else:
        os.makedirs("plots", exist_ok=True)
        out = os.path.join("plots", "training_plot.png")
    plt.savefig(out, dpi=150)
    print(f"Saved to {out}")
    plt.show()
