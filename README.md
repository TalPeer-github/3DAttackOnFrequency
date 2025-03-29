# CloudWalker: 3D Point Cloud Shape Analysis using Random Walks

This project implements the CloudWalker approach for 3D point cloud shape analysis using random walks, based on the paper ["CloudWalker: 3D Point Cloud Learning by Random Walks for Shape Analysis"](https://arxiv.org/abs/2112.01050) (SMI 2022).

## Introduction

CloudWalker is a novel method for analyzing 3D point clouds by performing random walks over their implicit neighborhood structure. By learning from these sequential walks, the model can effectively capture both local and global structural information of 3D shapes for tasks like classification and retrieval.

![CloudWalker Architecture](https://i.imgur.com/dZ4jJ1n.png)

## Pipeline Overview

Our implementation consists of the following steps:

1. **Random Walk Generation**: Creates random walks on point clouds using k-nearest neighbors
2. **CloudWalker Training**: Trains the GRU-based CloudWalker network on the generated walks
3. **Evaluation**: Tests the trained model on classification tasks
4. **Adversarial Attacks**: Implements gradient-based adversarial attacks on the point clouds

## Project Structure

```
├── cloudwalker_arch.py    # CloudWalker architecture implementation
├── train_cloudwalker.py   # Training script for CloudWalker
├── evaluate_cloudwalker.py # Evaluation script
├── attack_single_pc.py    # Adversarial attack on a single point cloud
├── attack_pc.py           # Script to attack multiple point clouds
├── dataset.py             # Dataset loaders for point clouds and walks
├── save_walk_as_npz.py    # Script to generate and save random walks
├── run_all.py             # Pipeline orchestration 
├── configs/               # Configuration files
│   ├── cloudwalker_params.json  # CloudWalker parameters
│   ├── attack_config.json       # Adversarial attack parameters
│   └── walks_creating.json      # Walk generation parameters
├── pre_created_walks/     # Directory for storing pre-generated walks
├── datasets_processed/    # Processed point cloud datasets
├── attacks/               # Directory for adversarial attack results
├── saved_checkpoints/     # Directory for model checkpoints
│   └── cloudwalker/       # CloudWalker model checkpoints and plots
│       └── plots/         # Training progress visualizations
└── requirements.txt       # Project dependencies
```

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/cloudwalker.git
cd cloudwalker
```

2. Create a conda environment and install dependencies:
```
conda create -n cloudwalker python=3.8
conda activate cloudwalker
pip install -r requirements.txt
```

## Usage

### 1. Generate Random Walks

To generate random walks for both train and test sets:

```
python save_walk_as_npz.py --dataset train
python save_walk_as_npz.py --dataset test
```

### 2. Train CloudWalker Model

To train the CloudWalker model on the generated walks:

```
python train_cloudwalker.py
```

Training progress will be automatically tracked and plotted. The plots and model checkpoints will be saved in the `saved_checkpoints/cloudwalker` directory.

### 3. Visualize Training Progress

If you want to visualize the training progress from an existing training run without retraining:

```
python train_cloudwalker.py --plot-only
```

This will generate loss curves, accuracy curves, and a combined visualization.

### 4. Evaluate the Model

To evaluate the trained model:

```
python evaluate_cloudwalker.py
```

By default, the evaluation will use the best model checkpoint. You can specify a different checkpoint with:

```
python evaluate_cloudwalker.py --model_path path/to/checkpoint.pth
```

### 5. Run Adversarial Attacks

To attack a single point cloud:

```
python attack_single_pc.py --id airplane_0521
```

Optional parameters:
- `--visualize`: Generate visualizations of the attack
- `--epsilon <value>`: Set maximum perturbation magnitude
- `--iterations <value>`: Set number of attack iterations

To attack multiple point clouds:

```
python attack_pc.py
```

### Running the Full Pipeline

You can also run the complete pipeline using:

```
python run_all.py
```

With options to customize the process:

```
python run_all.py --skip_walk_generation --attack_only --attack_model_id airplane_0521 --visualize_attack
```

## Implementation Details

- **Random Walk Generation**: We use a KD-Tree structure to find k-nearest neighbors for each point and then perform random walks over this implicit graph.
- **Network Architecture**: The CloudWalker network consists of:
  - Feature extraction layers: Input → 64 → 128 → 256
  - Sequential processing: GRU layers (1024 → 1024 → 512)
  - Post-processing FC layers: 512 → 128
  - Classification layer: 128 → num_classes
- **Training**: 
  - We train on a subset of ModelNet40 categories (specified in configs/walks_creating.json)
  - Cross-Entropy loss and Adam optimizer (lr=0.0005)
  - StepLR scheduler that reduces learning rate by a factor of 0.7 every 10 steps
  - 10 epochs with batch size of 16
  - Model checkpoints saved every 5 epochs and when best validation accuracy is achieved
  - Detailed training metrics (loss and accuracy) are logged and visualized with matplotlib
- **Adversarial Attack**:
  - Gradient-based adversarial attack on point cloud walks
  - Directly perturbs the original point cloud coordinates
  - Smart mapping of perturbations to walk points using direct mapping or KD-Tree
  - Constrained by L∞ norm to preserve the structure of the point cloud
  - Visualizations of the original and perturbed point clouds

## Citation

If you find this implementation useful, please cite the original paper:

```
@article{mesika2021cloudwalker,
  title={CloudWalker: 3D Point Cloud Learning by Random Walks for Shape Analysis},
  author={Mesika, Adi and Ben-Shabat, Yizhak and Tal, Ayellet},
  journal={arXiv preprint arXiv:2112.01050},
  year={2021}
}
```

## Acknowledgments

This implementation is based on the paper by Adi Mesika, Yizhak Ben-Shabat, and Ayellet Tal from Technion - Israel Institute of Technology. 