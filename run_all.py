import subprocess
import time
from datetime import datetime

LOG_PATH = "pipeline_log.txt"

def log(message):
    print(message)
    with open(LOG_PATH, "a") as f:
        f.write(message + "\n")

def run_step(description, command):
    log(f"\n[START] {description}")
    start_time = time.time()
    start_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"Start time: {start_dt}")

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        log(f"[ERROR] {description} failed: {e}")
        exit(1)

    end_time = time.time()
    end_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed = end_time - start_time

    log(f"End time: {end_dt}")
    log(f"Duration: {elapsed:.2f} seconds")
    log(f"[DONE] {description}")

if __name__ == "__main__":
    with open(LOG_PATH, "w") as f:
        f.write(f"[PIPELINE START] {datetime.now()}\n")

    run_step("Generating random walks for train set", ["python", "save_walk_as_npz.py", "--dataset", "train"])
    run_step("Generating random walks for test set", ["python", "save_walk_as_npz.py", "--dataset", "test"])
    run_step("Training proxy model", ["python", "train_proxy.py"])
    run_step("Running attack on point clouds", ["python", "attack_pc.py"])

    log(f"\n[PIPELINE COMPLETE] {datetime.now()}")
