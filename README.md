# S3GLC

## Overview

S3GLC is a graph clustering method that combines spectral subspace learning with graph contrastive learning for unsupervised graph representation learning and clustering.

## Dependencies

The required packages and versions are listed in `env.txt`. Key dependencies include:

- **PyTorch**: 1.12.1+cu116
- **PyTorch Geometric**: 2.6.1
- **NumPy**: 1.24.4
- **scikit-learn**: 1.1.2
- **scipy**: 1.10.1
- **ogb**: 1.3.6 (for OGB datasets)

For a complete list of dependencies, please refer to `env.txt`.

### Installation

```bash
# Install PyTorch (CUDA 11.6)
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# Install PyTorch Geometric
pip install torch-geometric==2.6.1

# Install other dependencies
pip install numpy scipy scikit-learn ogb tqdm matplotlib
```

## Dataset Preparation

The code supports datasets from:
- **TUDataset**: ENZYMES, AIDS, BZR, MUTAG, PTC_MR, PTC_MM, DD, COLLAB etc.
- **OGB**: PPA (ogbg-ppa)
- **Custom**: [REDDIT-12K](https://networkrepository.com/REDDIT-MULTI-12K.php)
  
For **TUDataset** and **OGB**, it download automaticly when your run the code, for REDDIT-12K, you can download it from the website.

## Usage



### Key Parameters

- `--DS`: Dataset name (e.g., ENZYMES, AIDS, BZR, MUTAG, DD, PTC_MR, PTC_MM, COLLAB, REDDIT-12K, PPA)
- `--lr`: Learning rate
- `--hidden-dim`: Hidden dimension size
- `--num-gc-layers`: Number of graph convolution layers
- `--batch-size`: Batch size for training
- `--epochs`: Number of training epochs
- `--iter`: Number of iterations (runs)
- `--seed`: Random seed (default: 42)
- `--r`: Graph encoding dimension (default: 10)
- `--d`: Number of leading eigenvectors to use when fusing views (default: None, uses all)
- `--gpu`: GPU device number (optional)


## Best Hyperparameters

The following hyperparameters have been tested and work well for each dataset:

### Small Datasets

| Dataset | Learning Rate | Hidden Dim | Batch Size | r | d | 
|---------|--------------|------------|------------|---|----|
| ENZYMES | 0.002 | 16 | 512 | 16 | 8 | 
| AIDS | 0.001 | 16 | 256 | 10 | 5 | 
| BZR | 0.0008 | 16 | 512 | 6 | 4 | 
| MUTAG | 0.0001 | 16 | 512 | 8 | 4 |
| PTC_MR | 0.001 | 16 | 512 | 5 | 3 | 
| DD | 0.002 | 16 | 512 | 12 | 4 | 
| PTC_MM | 0.002 | 16 | 256 | 5 | 3 | 

### Large Datasets

| Dataset | Learning Rate | Hidden Dim | Batch Size | r | d | 
|---------|--------------|------------|------------|---|----|
| COLLAB | 0.001 | 32 | 4096 | 40 | 20 | 
| REDDIT-12K | 0.001 | 32 | 4096 | 70 | 20 | 
| PPA | 0.001 | 32 | 4096 | 60 | 20 | 

### Example Commands

**ENZYMES:**
```bash
python main.py --lr 0.002 --hidden-dim 16 --num-gc-layers 4 --batch-size 512 \
  --epochs 20 --iter 5 --seed 42 --DS ENZYMES --r 16 --d 8 --gpu 1
```

**AIDS:**
```bash
python main.py --lr 0.001 --hidden-dim 16 --num-gc-layers 4 --batch-size 256 \
  --epochs 20 --iter 5 --seed 42 --DS AIDS --r 10 --d 5 --gpu 1 --e 2
```

**COLLAB:**
```bash
srun -p short --gres=gpu:a100:1 python main.py --lr 0.001 --hidden-dim 32 \
  --num-gc-layers 4 --batch-size 4096 --epochs 20 --iter 5 --seed 42 \
  --DS COLLAB --r 40 --d 20
```

## Running All Experiments

You can run all experiments using the provided script:

```bash
bash run.sh
```

This will execute all the commands listed in `run.sh` sequentially.

## Output

Results are saved in the `./result/` directory:
- `{DATASET}_result.txt`: Contains ACC, NMI, ARI metrics for each run

Each result file includes:
- `r`: Graph encoding dimension used
- `d`: Number of leading eigenvectors used
- `ACC_MEAN`: Mean accuracy with standard deviation
- `NMI_MEAN`: Mean normalized mutual information with standard deviation
- `ARI_MEAN`: Mean adjusted rand index with standard deviation

## Code Structure

```
S3GLC/
├── main.py              # Main training script
├── arguments.py         # Argument parser
├── run.sh              # Batch execution script
├── model/
│   ├── model.py        # S3GLC model implementation
│   ├── gin.py          # GIN encoder
│   └── Graph_encoder.py # Graph encoding utilities
├── loss/
│   └── losses.py       # Loss functions
├── Utils/
│   ├── aug.py          # Data augmentation
│   ├── evaluate_embedding.py # Evaluation metrics
│   └── preprocess.py   # Dataset preprocessing
└── load_reddit12k.py   # REDDIT-12K dataset loader
```




