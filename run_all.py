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

    # CloudWalker training overrides
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--scheduler_step_size", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--batch_size", type=int)

    # Walk generation overrides
    parser.add_argument("--seq_len", type=int)
    parser.add_argument("--num_walks", type=int)
    parser.add_argument("--k_neighbors", type=int)
    
    # Attack overrides
    parser.add_argument("--attack_epsilon", type=float, help="Maximum perturbation magnitude")
    parser.add_argument("--attack_iterations", type=int, help="Number of attack iterations")
    parser.add_argument("--attack_model_id", type=str, help="Specific model ID to attack")
    parser.add_argument("--attack_num_samples", type=int, help="Number of samples to attack")
    parser.add_argument("--visualize_attack", action="store_true", help="Visualize attack results")
    
    # Pipeline control
    parser.add_argument("--skip_walk_generation", action="store_true", help="Skip random walk generation stage")
    parser.add_argument("--skip_training", action="store_true", help="Skip model training stage")
    parser.add_argument("--plot_only", action="store_true", help="Only generate plots from existing training")
    parser.add_argument("--evaluate_only", action="store_true", help="Only run evaluation")
    parser.add_argument("--attack_only", action="store_true", help="Only run adversarial attack")
    parser.add_argument("--model_path", type=str, help="Path to specific model checkpoint for evaluation/attack")

    args = parser.parse_args()

    with open(LOG_PATH, "w") as f:
        f.write(f"[PIPELINE START] {datetime.now()}\n")

    # --- update walks_creating.json ---
    if not args.evaluate_only and not args.plot_only and not args.attack_only:
        walk_updates = {}
        if args.seq_len: walk_updates["seq_len"] = args.seq_len
        if args.num_walks: walk_updates["num_walks_per_sample"] = args.num_walks
        if args.k_neighbors: walk_updates["k_neighbors"] = args.k_neighbors
        if walk_updates:
            update_config("configs/walks_creating.json", walk_updates)

    # --- update cloudwalker_params.json ---
    if not args.evaluate_only and not args.plot_only and not args.attack_only:
        cloudwalker_updates = {}
        if args.num_epochs: cloudwalker_updates["args__num_epochs"] = args.num_epochs
        if args.lr: cloudwalker_updates["args__lr"] = args.lr
        if args.weight_decay: cloudwalker_updates["args__weight_decay"] = args.weight_decay
        if args.scheduler_step_size: cloudwalker_updates["args__scheduler_step_size"] = args.scheduler_step_size
        if args.dropout: cloudwalker_updates["dropout"] = args.dropout
        if args.batch_size: cloudwalker_updates["args__batch_size"] = args.batch_size
        if cloudwalker_updates:
            update_config("configs/cloudwalker_params.json", cloudwalker_updates)

    # --- update attack_config.json ---
    if args.model_path or args.attack_only or args.attack_epsilon or args.attack_iterations or args.visualize_attack:
        attack_updates = {}
        if args.model_path: attack_updates["checkpoint_path"] = args.model_path
        if args.attack_epsilon: attack_updates["attack_epsilon"] = args.attack_epsilon
        if args.attack_iterations: attack_updates["attack_iterations"] = args.attack_iterations
        if args.visualize_attack: attack_updates["visualize_attacks"] = True
        if attack_updates:
            update_config("configs/attack_config.json", attack_updates)

    # --- plot only ---
    if args.plot_only:
        run_step("Generating training plots", ["python", "train_cloudwalker.py", "--plot-only"])
        log(f"\n[PIPELINE COMPLETE] {datetime.now()}")
        exit(0)

    # --- attack only ---
    if args.attack_only:
        attack_cmd = ["python", "attack_single_pc.py", "--config", "configs/attack_config.json"]
        if args.attack_model_id:
            attack_cmd.extend(["--id", args.attack_model_id])
        if args.attack_num_samples:
            attack_cmd.extend(["--num_samples", str(args.attack_num_samples)])
        if args.visualize_attack:
            attack_cmd.append("--visualize")
        
        run_step("Running adversarial attack on CloudWalker", attack_cmd)
        log(f"\n[PIPELINE COMPLETE] {datetime.now()}")
        exit(0)

    # --- run pipeline ---
    if not args.skip_walk_generation and not args.evaluate_only:
        run_step("Generating random walks for train set", ["python", "save_walk_as_npz.py", "--dataset", "train"])
        run_step("Generating random walks for test set", ["python", "save_walk_as_npz.py", "--dataset", "test"])
    
    if not args.skip_training and not args.evaluate_only:
        run_step("Training CloudWalker model", ["python", "train_cloudwalker.py"])
    
    # --- evaluation ---
    if not args.skip_training or args.evaluate_only:
        eval_cmd = ["python", "evaluate_cloudwalker.py"]
        if args.model_path:
            eval_cmd.extend(["--model_path", args.model_path])
        
        run_step("Evaluating CloudWalker model", eval_cmd)
    
    # --- adversarial attack (unless explicitly skipped) ---
    attack_cmd = ["python", "attack_single_pc.py", "--config", "configs/attack_config.json"]
    if args.attack_model_id:
        attack_cmd.extend(["--id", args.attack_model_id])
    elif args.attack_num_samples:
        attack_cmd.extend(["--num_samples", str(args.attack_num_samples)])
    if args.visualize_attack:
        attack_cmd.append("--visualize")
        
    run_step("Running adversarial attack on CloudWalker", attack_cmd)

    log(f"\n[PIPELINE COMPLETE] {datetime.now()}")
