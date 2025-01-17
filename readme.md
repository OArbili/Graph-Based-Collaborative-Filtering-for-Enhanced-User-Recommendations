# Enhanced Graph-Based Collaborative Filtering

This repository contains the implementation of an enhanced graph neural network (GNN) approach for recommender systems, based on UltraGCN. The method introduces novel improvements in weight initialization and learning optimization to enhance recommendation accuracy and efficiency.

## Overview

The model improves upon UltraGCN by introducing:
- A novel weight initialization method incorporating behavioral patterns
- An improved negative sampling strategy
- An adaptive loss function considering item popularity
- A model-agnostic approach compatible with various GNN architectures

## Requirements

```bash
torch>=1.7.0
numpy>=1.19.2
scipy>=1.6.0
configparser>=5.0.0
tensorboard>=2.4.0
tqdm>=4.50.0
scikit-learn>=0.24.0
```

You can install all requirements using:
```bash
pip install -r requirements.txt
```

## Project Structure

```
├── src/
│   ├── data.py          # Data loading and preprocessing
│   ├── model.py         # Model architecture implementation
│   ├── preprocessing.py # Data preprocessing utilities
│   ├── run.py          # Main training script
│   ├── train.py        # Training loop implementation
│   └── utils.py        # Utility functions
├── configs/            # Configuration files
├── data/              # Dataset directory
│   ├── AmazonBooks_m1/
│   │   ├── train.txt
│   │   └── test.txt
│   └── Gowalla_m1/
│       ├── train.txt
│       └── test.txt
└── logs/              # Training logs
```

## Data Acquisition

The datasets used in this project can be downloaded from:

1. Amazon-Book dataset:
   - Source: [Amazon review dataset](https://cseweb.ucsd.edu/~jmcauley/datasets.html#amazon_reviews)
   - Download the Books category data

2. Gowalla dataset:
   - Source: [Stanford Gowalla dataset](https://snap.stanford.edu/data/loc-gowalla.html)

After downloading, preprocess the data according to the format below.

## Data Format

### Training and Test Files
Each line represents a user and their interactions:
```
user_id item_id1 item_id2 item_id3 ...
```

## Datasets

The code supports two benchmark datasets:

1. Amazon-Book
   - Users: 52,643
   - Items: 91,599
   - Interactions: 2,984,108
   - Density: 0.00062

2. Gowalla
   - Users: 29,858
   - Items: 40,981
   - Interactions: 1,027,370
   - Density: 0.00084

## Usage

### Training

To train the model:

```bash
python src/run.py --config_file ./configs/amazonbooks_m1-100-pre.ini
```

Command line arguments:
- `--config_file`: Path to configuration file (required)
- `--weights`: Enable/disable pre-initialized weights (default: True)

### Configuration

Key parameters in the configuration file:

Model Parameters:
- `embedding_dim`: Dimension of user/item embeddings
- `ii_neighbor_num`: Number of neighbors for item-item graph
- `model_save_path`: Path to save trained model
- `max_epoch`: Maximum training epochs
- `enable_tensorboard`: Enable TensorBoard logging
- `initial_weight`: Initial weight value

Training Parameters:
- `dataset`: Dataset name
- `train_file_path`: Path to training file
- `gpu`: GPU device ID
- `learning_rate`: Initial learning rate
- `batch_size`: Training batch size
- `early_stop_epoch`: Early stopping patience
- `negative_num`: Number of negative samples
- `negative_weight`: Weight for negative samples
- `gamma`: Regularization parameter
- `lambda`: Loss function parameter

Testing Parameters:
- `test_batch_size`: Batch size for testing
- `topk`: Number of top items for evaluation
- `test_file_path`: Path to test file

## Performance

The model achieves state-of-the-art performance on both datasets:

Amazon-Book:
- Recall@20: 0.0694 (1.91% improvement)
- NDCG@20: 0.0568 (2.16% improvement)

Gowalla:
- Recall@20: 0.1881 (1.02% improvement)
- NDCG@20: 0.1595 (0.95% improvement)

## Training Details

The model employs several optimization strategies:

1. Weight Initialization:
   - Uses behavioral patterns between users and items
   - Incorporates both direct and indirect interactions
   - Applies PCA for dimension reduction

2. Negative Sampling:
   - Employs a neighbor-based selection mechanism
   - Uses threshold-based filtering
   - Considers interaction likelihood

3. Loss Function:
   - Implements a triplet weighted loss
   - Adaptively adjusts weights based on item popularity
   - Integrates model confidence in predictions

## Monitoring

Training progress can be monitored through:
- TensorBoard logs (if enabled)
- Training logs in the logs directory
- Model checkpoints saved according to config

## License

This project is licensed under the MIT License.

## Acknowledgments

This implementation is based on the UltraGCN architecture, with significant modifications to enhance recommendation performance through improved weight initialization and learning optimization.