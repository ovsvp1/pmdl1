"""Run data engineering and model engineering immediately and every 5 minutes."""
import subprocess
import time
from datetime import datetime

INTERVAL_SECONDS = 5 * 60


def run_pipeline():
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Starting pipeline run", flush=True)
    subprocess.run(["python", "code/datasets/data_pipeline.py"], check=True)
    subprocess.run(["python", "code/models/train.py"], check=True)
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Pipeline run complete", flush=True)


if __name__ == "__main__":
    while True:
        started = time.monotonic()
        try:
            run_pipeline()
        except subprocess.CalledProcessError as exc:
            print(f"Pipeline failed with exit code {exc.returncode}", flush=True)
        elapsed = time.monotonic() - started
        time.sleep(max(0, INTERVAL_SECONDS - elapsed))
