"""Run the complete MLOps pipeline automatically every five minutes."""

import subprocess
import time
from datetime import datetime
from pathlib import Path

import schedule

PROJECT_DIR = Path(__file__).resolve().parent
PIPELINE_SCRIPT = PROJECT_DIR / "run_pipeline.sh"


def run_pipeline() -> None:
    """Execute all three pipeline stages through run_pipeline.sh."""
    started_at = datetime.now().isoformat(timespec="seconds")
    print(f"[{started_at}] Starting scheduled pipeline run", flush=True)
    try:
        subprocess.run(
            ["bash", str(PIPELINE_SCRIPT)],
            cwd=PROJECT_DIR,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"Pipeline failed with exit code {exc.returncode}", flush=True)
    else:
        print("Scheduled pipeline run completed successfully", flush=True)


if __name__ == "__main__":
    # Run immediately so the user does not have to wait five minutes after start.
    run_pipeline()

    # Then trigger another complete run every five minutes while this process lives.
    schedule.every(5).minutes.do(run_pipeline)
    print("Scheduler is running. Next pipeline run is scheduled in 5 minutes.", flush=True)

    while True:
        schedule.run_pending()
        time.sleep(1)
