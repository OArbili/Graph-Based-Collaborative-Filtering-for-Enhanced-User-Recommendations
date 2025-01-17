# Import required libraries
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
import torch.utils.data as data
import scipy.sparse as sp
import os
import gc
import configparser
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from train import *
from utils import *
from data import *
from model import *

print("started")

if __name__ == "__main__":
    # Initialize argument parser for command line inputs
    parser = argparse.ArgumentParser()
    # Config file path argument for model parameters and settings
    parser.add_argument('--config_file', type=str, help='config file path')
    # Boolean flag for using pre-initialized weights based on collaborative patterns
    parser.add_argument('--weights', type=bool, help='config file path', default=True)
    args = parser.parse_args()

    # Setup logging to file
    old_stdout = sys.stdout
    name = args.config_file.split("/")[-1][:-4]
    log_file = open(f"./logs/{name}.log","w")
    #sys.stdout = log_file

    print('###################### MultiAspectGraph ######################')
    print(args.config_file)

    # Load and prepare all necessary data and parameters
    # This includes:
    # - constraint matrices for user-item interactions
    # - ii_constraint_mat for item-item relationships
    # - ii_neighbor_mat for neighborhood information
    # - train/test data loaders
    # - interaction masks
    # - ground truth for testing
    # - pre-computed embeddings (rec_i_64, rec_u_64) for initialization
    params, constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, \
    test_loader, mask, test_ground_truth_list, interacted_items, train_mat, \
    rec_i_64, rec_u_64 = data_param_prepare(args.config_file)

    print('Load Configuration OK, show them below')
    print('Configuration:')
    print(params)

    # Initialize the MultiAspectGraph model
    # - Uses constraint matrices for loss computation
    # - Incorporates pre-trained embeddings if weights=True
    # - Based on UltraGCN architecture with additional behavioral pattern integration
    mag = MultiAspectGraph(params, constraint_mat, ii_constraint_mat, 
                          ii_neighbor_mat, rec_i_64, rec_u_64, args.weights)
    
    # Optional: Load pre-trained model weights
    # mag.load_state_dict(torch.load("ultragcn_gowalla_0.pt"))

    # Move model to specified device (CPU/GPU)
    mag = mag.to(params['device'])

    # Initialize Adam optimizer with learning rate from params
    optimizer = torch.optim.Adam(mag.parameters(), lr=params['lr'])

    # Train the model using:
    # - train_loader: batched training data
    # - test_loader: batched test data
    # - mask: interaction mask for negative sampling
    # - test_ground_truth_list: for evaluation
    # - interacted_items: user interaction history
    # - params: model parameters
    # - train_mat: training interaction matrix
    train(mag, optimizer, train_loader, test_loader, mask, 
          test_ground_truth_list, interacted_items, params, train_mat)

    print('END')

    # Restore original stdout and close log file
    sys.stdout = old_stdout
    log_file.close()