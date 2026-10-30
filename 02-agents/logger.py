"""JSON-lines step log: one line per tool call and per model response."""
import json
import pathlib
import time
from datetime import datetime, timezone

LOG_DIR = pathlib.Path("logs")


class StepLogger:
    def __init__(self, run_name: str):
        LOG_DIR.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.path = LOG_DIR / f"{stamp}_{run_name}.jsonl"
        self.step = 0

    def log(self, **event):
        self.step += 1
        event = {"step": self.step, "ts": time.time(), **event}
        with self.path.open("a") as f:
            f.write(json.dumps(event, default=str) + "\n")
        print(f"  step {self.step:>3} {event.get('kind'):>9} {event.get('tool', '')}")
