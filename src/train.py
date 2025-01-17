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
from utils import *

# Function to evaluate a single batch of predictions
# Args:
#   X: Tuple containing (sorted_predictions, ground_truth)
#   k: Number of top items to consider (e.g., top-20)
# Returns:
#   Precision, Recall, and NDCG metrics for the batch
def test_one_batch(X, k):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    ret = RecallPrecision_ATk(groundTrue, r, k)
    return ret['precision'], ret['recall'], NDCGatK_r(groundTrue,r,k)

# Function to create binary relevance labels for predictions
# Args:
#   test_data: Ground truth items for users
#   pred_data: Predicted top-k items for users
# Returns:
#   Binary matrix where 1 indicates the predicted item is in ground truth
def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# Main training function implementing the enhanced methodology from the paper
# Args:
#   model: The MultiAspectGraph model instance
#   optimizer: Adam optimizer
#   train_loader: DataLoader for training data with positive samples
#   test_loader: DataLoader for test data
#   mask: Interaction mask for masking seen items during testing
#   test_ground_truth_list: List of ground truth items for each test user
#   interacted_items: Dictionary of user-item interactions
#   params: Dictionary of model parameters
#   train_mat: Training interaction matrix
def train(model, optimizer, train_loader, test_loader, mask, test_ground_truth_list, interacted_items, params,train_mat): 
    
    train_mat = train_mat.toarray().astype(bool).astype(int)
    device = params['device']
    print(device)
    print(f"Using device: {device}")
    best_epoch, best_recall, best_ndcg = 0, 0, 0
    early_stop_count = 0
    early_stop = False

    # Calculate total number of training batches
    batches = len(train_loader.dataset) // params['batch_size']
    if len(train_loader.dataset) % params['batch_size'] != 0:
        batches += 1
    print('Total training batches = {}'.format(batches))
    
    # Initialize TensorBoard writer if enabled
    if params['enable_tensorboard']:
        writer = SummaryWriter()

    # Setup learning rate scheduler with plateau detection
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.75, 
                                                                       patience=25, threshold=0.0001, threshold_mode='rel', cooldown=0, 
                                                                       min_lr=0.00001, eps=1e-08, verbose='deprecated')
    
    # Main training loop
    for epoch in range(params['max_epoch']):
        train_loader.dataset.set_epoch(epoch+1)
        model.train().cuda()
        start_time = time.time()

        # Training iteration over batches
        for batch, (users, pos_items, neg_items) in enumerate(tqdm(train_loader)):
            users = users.cuda()
            pos_items = pos_items.cuda()
            neg_items = neg_items.cuda()

            model.zero_grad()
            loss = model(users, pos_items, neg_items,epoch)
            if params['enable_tensorboard']:
                writer.add_scalar("Loss/train_batch", loss, batches * epoch + batch)
            loss.backward()
            optimizer.step()
        
        train_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
        if params['enable_tensorboard']:
            writer.add_scalar("Loss/train_epoch", loss, epoch)

        # Evaluation phase
        need_test = True
        if epoch < 50 and epoch % 5 != 0:
            need_test = False
            
        if need_test:
            start_time = time.time()
            F1_score, Precision, Recall, NDCG = test(model, test_loader, test_ground_truth_list, mask, params['topk'], params['user_num'])
            if params['enable_tensorboard']:
                writer.add_scalar('Results/recall@20', Recall, epoch)
                writer.add_scalar('Results/ndcg@20', NDCG, epoch)
            test_time = time.strftime("%H: %M: %S", time.gmtime(time.time() - start_time))
            
            print('The time for epoch {} is: train time = {}, test time = {}'.format(epoch, train_time, test_time))
            print("Loss = {:.5f}, F1-score: {:5f} \t Precision: {:.5f}\t Recall: {:.5f}\tNDCG: {:.5f}".format(loss.item(), F1_score, Precision, Recall, NDCG))
            
            # Learning rate adjustment after epoch 100
            if epoch >100:
                scheduler.step(Recall)

            # Save best model and handle early stopping
            if Recall > best_recall:
                best_recall, best_ndcg, best_epoch = Recall, NDCG, epoch
                early_stop_count = 0
                torch.save(model.state_dict(), params['model_save_path'])
            else:
                early_stop_count += 1
                if early_stop_count == params['early_stop_epoch']:
                    early_stop = True
        
        if early_stop:
            print('##########################################')
            print('Early stop is triggered at {} epochs.'.format(epoch))
            print('Results:')
            print('best epoch = {}, best recall = {}, best ndcg = {}'.format(best_epoch, best_recall, best_ndcg))
            print('The best model is saved at {}'.format(params['model_save_path']))
            break

    writer.flush()
    print('Training end!')

# Test function for model evaluation
# Args:
#   model: Trained model to evaluate
#   test_loader: DataLoader for test data
#   test_ground_truth_list: List of ground truth items for each test user
#   mask: Interaction mask for seen items
#   topk: Number of top items to recommend
#   n_user: Total number of users
# Returns:
#   F1 score, Precision, Recall, and NDCG metrics
def test(model, test_loader, test_ground_truth_list, mask, topk, n_user):
    users_list = []
    rating_list = []
    groundTrue_list = []

    with torch.no_grad():
        model.eval()
        for idx, batch_users in enumerate(test_loader):
            
            batch_users = batch_users.to(model.get_device())
            rating = model.test_foward(batch_users) 
            rating = rating.cpu()
            rating += mask[batch_users.cpu()]
            
            _, rating_K = torch.topk(rating.float(), k=topk)
            rating_list.append(rating_K)

            groundTrue_list.append([test_ground_truth_list[u.cpu()] for u in batch_users])

    X = zip(rating_list, groundTrue_list)
    Recall, Precision, NDCG = 0, 0, 0

    for i, x in enumerate(X):
        precision, recall, ndcg = test_one_batch(x, topk)
        Recall += recall
        Precision += precision
        NDCG += ndcg
        
    Precision /= n_user
    Recall /= n_user
    NDCG /= n_user
    F1_score = 2 * (Precision * Recall) / (Precision + Recall)

    return F1_score, Precision, Recall, NDCG