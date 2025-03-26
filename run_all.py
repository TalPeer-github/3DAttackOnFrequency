import argparse
import json
import os
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

def update_config(path, updates):
    with open(path, "r") as f:
        config = json.load(f)

    for key, value in updates.items():
        if "__" in key:
            outer, inner = key.split("__")
            config[outer][inner] = value
        else:
            config[key] = value

    with open(path, "w") as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Proxy training overrides
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--scheduler_step_size", type=int)

    # Attack config overrides
    parser.add_argument("--walk_len", type=int)
    parser.add_argument("--attacking_weight", type=float)

    # Walk generation overrides
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--num_walks", type=int)
    parser.add_argument("--k_neighbors", type=int)

    args = parser.parse_args()

    with open(LOG_PATH, "w") as f:
        f.write(f"[PIPELINE START] {datetime.now()}\n")

    # --- update walks_creating.json ---
    walk_updates = {}
    if args.seq_len: walk_updates["seq_len"] = args.seq_len
    if args.num_walks: walk_updates["num_walks_per_sample"] = args.num_walks
    if args.k_neighbors: walk_updates["k_neighbors"] = args.k_neighbors
    update_config("configs/walks_creating.json", walk_updates)

    # --- update proxy_params.json ---
    proxy_updates = {}
    if args.num_epochs: proxy_updates["args__num_epochs"] = args.num_epochs
    if args.lr: proxy_updates["args__lr"] = args.lr
    if args.weight_decay: proxy_updates["args__weight_decay"] = args.weight_decay
    if args.scheduler_step_size: proxy_updates["args__scheduler_step_size"] = args.scheduler_step_size
    update_config("configs/proxy_params.json", proxy_updates)

    # --- update attack_config.json ---
    attack_updates = {}
    if args.walk_len: attack_updates["walk_len"] = args.walk_len
    if args.attacking_weight: attack_updates["attacking_weight"] = args.attacking_weight
    update_config("configs/attack_config.json", attack_updates)

    # --- run pipeline ---
    run_step("Generating random walks for train set", ["python", "save_walk_as_npz.py", "--dataset", "train"])
    run_step("Generating random walks for test set", ["python", "save_walk_as_npz.py", "--dataset", "test"])
    run_step("Training proxy model", ["python", "train_proxy.py"])
    run_step("Running attack on point clouds", ["python", "attack_pc.py"])

    log(f"\n[PIPELINE COMPLETE] {datetime.now()}")
