"""
Drop-in stdout tee: writes to both the terminal and a log file.

Usage in any training script — add near the top of if __name__ == "__main__":
    from logger import setup_logging
    setup_logging("logs/my_run.log")  # or pass None to auto-name
"""

import os
import sys
from datetime import datetime


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_logging(path=None, log_dir="logs"):
    if path is None:
        os.makedirs(log_dir, exist_ok=True)
        caller = os.path.splitext(os.path.basename(sys.argv[0]))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"{caller}_{timestamp}.log")
    else:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    log_file = open(path, "w", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    print(f"Logging to {path}")
    return path
