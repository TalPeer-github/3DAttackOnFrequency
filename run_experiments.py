# run_experiments.py
import os
import json
from shutil import copyfile
from argparse import ArgumentParser

CONFIGS_DIR = "config"

# Default configurations
DEFAULT_CONFIGS = {
    "attack_config.json": {
        "trained_model": "saved_checkpoints",
        "checkpoint_path": "saved_checkpoints/checkpoint.pth",
        "walk_npz_root": "pre_created_walks/test",
        "num_classes": 40,
        "walk_len": 600,
        "attacking_weight": 1.0,
        "step_size": 0.01,
        "max_iter": 50,
        "max_label_diff": 0.01,
        "use_norm_layer": "BatchNorm",
        "output_dir": "attacks",
        "layer_sizes": {
            "fc1": 64, "fc2": 128, "fc3": 256,
            "gru1": 1024, "gru2": 1024, "gru3": 512,
            "fc4": 512, "fc5": 128
        },
        "logdir": "saved_checkpoints"
    },
    "proxy_params.json": {
        "train_walk_npz_path": "pre_created_walks/train",
        "test_walk_npz_path": "pre_created_walks/test",
        "logdir": "saved_checkpoints",
        "num_classes": 40,
        "batch_size": 4,
        "num_epochs": 10,
        "learning_rate": 0.0005,
        "use_norm_layer": "BatchNorm",
        "layer_sizes": {
            "fc1": 64, "fc2": 128, "fc3": 256,
            "gru1": 1024, "gru2": 1024, "gru3": 512,
            "fc4": 512, "fc5": 128
        }
    },
    "walks_creation.json": {
        "train_path": "datasets_processed/modelnet40_normal_resampled/train",
        "test_path": "datasets_processed/modelnet40_normal_resampled/test",
        "save_path": "pre_created_walks",
        "seq_len": 600,
        "batch_size": 1,
        "num_epochs": 600,
        "num_walks_per_sample": 32,
        "k_neighbors": 10
    }
}


def print_step(message):
    print(f"\n[INFO] {message}")

def update_json_config(config_name, updates):
    """
    Updates JSON config by merging defaults with provided arguments.
    If the config file does not exist, it is created with default values.
    """
    config_path = os.path.join(CONFIGS_DIR, config_name)

    # Ensure config directory exists
    if not os.path.exists(CONFIGS_DIR):
        os.makedirs(CONFIGS_DIR)

    # Load default values if file does not exist
    config = DEFAULT_CONFIGS.get(config_name, {}).copy()
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config.update(json.load(f))
    
    config.update(updates)

    # Write the updated configuration back to the file
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print_step(f"Updated {config_name} with {updates}")

def run_walk_creation(args):
    update_json_config("walks_creation.json", vars(args))
    print_step("Running walk creation flow...")
    os.system("python src/walks/walks_creator.py --config config/walks_creation.json")

def run_proxy_training(args):
    print_step("Running imitation (proxy) network training...")
    update_json_config("proxy_params.json", vars(args))
    os.system("python src/proxy_network/train_proxy.py --config config/proxy_params.json")

def run_attack(args):
    update_json_config("attack_config.json", vars(args))
    print_step("Running point cloud attack...")
    os.system("python src/attack/attack_runner.py --config config/attack_config.json")

def main():
    parser = ArgumentParser()
    parser.add_argument("--walk_len", type=int, help="Length of walks", default=None)
    parser.add_argument("--num_epochs", type=int, help="Number of epochs", default=None)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=None)
    parser.add_argument("--learning_rate", type=float, help="Learning rate", default=None)
    parser.add_argument("--max_iter", type=int, help="Max iterations for attack", default=None)
    parser.add_argument("--execute", type=str, choices=["walks", "proxy", "attack", "all"], default="all",
                    help="Select which part of the pipeline to run")
    args = parser.parse_args()

    print_step("Starting full experiment flow")
    run_walk_creation(args)
    run_proxy_training(args)
    run_attack(args)
    print_step("Experiment pipeline completed.")

if __name__ == "__main__":
    main()

