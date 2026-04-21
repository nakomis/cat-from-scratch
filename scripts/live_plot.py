"""
Live-updating training plot. Reads the latest log file and refreshes
the chart as new epochs are written.

Usage:
    python live_plot.py                  # auto-picks latest log in logs/
    python live_plot.py logs/my_run.log  # specific file
    python live_plot.py --interval 30    # refresh every 30s (default: 60)
"""

import argparse
import os
import re
import sys
import matplotlib.pyplot as plt
import matplotlib.animation as animation

LOG_DIR = "logs"


def latest_log():
    files = [
        os.path.join(LOG_DIR, f)
        for f in os.listdir(LOG_DIR)
        if f.endswith(".log")
    ]
    if not files:
        sys.exit(f"No log files found in {LOG_DIR}/")
    return max(files, key=os.path.getmtime)


def parse_log(path):
    pattern = re.compile(
        r"Epoch\s+(\d+)/(\d+).*?acc:\s+([\d.]+)%.*?acc:\s+([\d.]+)%"
    )
    epochs, train_accs, val_accs, total_epochs = [], [], [], None
    try:
        with open(path) as f:
            for m in pattern.finditer(f.read()):
                epochs.append(int(m.group(1)))
                total_epochs = int(m.group(2))
                train_accs.append(float(m.group(3)))
                val_accs.append(float(m.group(4)))
    except Exception:
        pass
    return epochs, train_accs, val_accs, total_epochs


def make_plot(log_path, interval_ms):
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.canvas.manager.set_window_title(f"Live: {os.path.basename(log_path)}")

    train_line, = ax.plot([], [], "b--", alpha=0.7, label="train acc")
    val_line,   = ax.plot([], [], "b-",          label="val acc")
    best_dot,   = ax.plot([], [], "r*", markersize=12, label="best val")
    random_line = ax.axhline(y=100/12, color="grey", linestyle=":",
                             alpha=0.5, label="random (8.3%)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    title = ax.set_title("")
    best_ann = ax.annotate("", xy=(0, 0), xytext=(5, 5),
                           textcoords="offset points", fontsize=9, color="red")

    def update(_frame):
        epochs, train_accs, val_accs, total = parse_log(log_path)
        if not epochs:
            title.set_text(f"{os.path.basename(log_path)} — waiting for data…")
            return train_line, val_line, best_dot, title, best_ann

        train_line.set_data(epochs, train_accs)
        val_line.set_data(epochs, val_accs)

        best_val = max(val_accs)
        best_ep  = epochs[val_accs.index(best_val)]
        best_dot.set_data([best_ep], [best_val])
        best_ann.set_position((best_ep, best_val))
        best_ann.set_text(f"{best_val:.1f}%")
        best_ann.xy = (best_ep, best_val)

        ax.set_xlim(0, max(total or max(epochs), max(epochs)) + 1)
        progress = f"{epochs[-1]}/{total}" if total else str(epochs[-1])
        title.set_text(
            f"{os.path.basename(log_path)} — epoch {progress}  |  "
            f"best val: {best_val:.1f}%  train: {train_accs[-1]:.1f}%"
        )
        return train_line, val_line, best_dot, title, best_ann

    ani = animation.FuncAnimation(
        fig, update, interval=interval_ms, blit=False, cache_frame_data=False
    )
    update(None)  # draw immediately without waiting for first interval
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", default=None,
                        help="Log file to watch (default: latest in logs/)")
    parser.add_argument("--interval", type=int, default=60,
                        help="Refresh interval in seconds (default: 60)")
    args = parser.parse_args()

    log_path = args.log or latest_log()
    print(f"Watching: {log_path}  (refreshing every {args.interval}s)")
    make_plot(log_path, args.interval * 1000)
