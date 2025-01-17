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
from sklearn.decomposition import PCA
from torch.utils.data import Dataset

class gcnDataset(Dataset):
    def __init__(self, train_data,params,train_mat,slop = 50):
        self.train_data = train_data
        self.params = params
        self.train_mat = train_mat.toarray()
        self.e = 1
        self.negative_num = self.params['negative_num']

    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, idx):
        pos = self.train_data[idx]
        neg_candidates = np.where(self.train_mat[pos[0]]==0)[0]
        neg_items = np.random.choice(neg_candidates, (self.negative_num), replace = True)
        neg_items = torch.from_numpy(neg_items)
        return torch.tensor(pos[0]),torch.tensor(pos[1]), neg_items
    
    def set_epoch(self,e):
        self.e = e
    


def data_param_prepare(config_file):

    config = configparser.ConfigParser()
    config.read(config_file)

    params = {}

    embedding_dim = config.getint('Model', 'embedding_dim')
    params['embedding_dim'] = embedding_dim
    ii_neighbor_num = config.getint('Model', 'ii_neighbor_num')
    params['ii_neighbor_num'] = ii_neighbor_num
    model_save_path = config['Model']['model_save_path']
    params['model_save_path'] = model_save_path
    max_epoch = config.getint('Model', 'max_epoch')
    params['max_epoch'] = max_epoch

    params['enable_tensorboard'] = config.getboolean('Model', 'enable_tensorboard')
    
    initial_weight = config.getfloat('Model', 'initial_weight')
    params['initial_weight'] = initial_weight

    dataset = config['Training']['dataset']
    params['dataset'] = dataset
    train_file_path = config['Training']['train_file_path']
    gpu = config['Training']['gpu']
    params['gpu'] = gpu
    device = torch.device('cuda')
    params['device'] = device
    lr = config.getfloat('Training', 'learning_rate')
    params['lr'] = lr
    batch_size = config.getint('Training', 'batch_size')
    params['batch_size'] = batch_size
    early_stop_epoch = config.getint('Training', 'early_stop_epoch')
    params['early_stop_epoch'] = early_stop_epoch
    w1 = config.getfloat('Training', 'w1')
    w2 = config.getfloat('Training', 'w2')
    w3 = config.getfloat('Training', 'w3')
    w4 = config.getfloat('Training', 'w4')
    params['w1'] = w1
    params['w2'] = w2
    params['w3'] = w3
    params['w4'] = w4
    negative_num = config.getint('Training', 'negative_num')
    negative_weight = config.getfloat('Training', 'negative_weight')
    params['negative_num'] = negative_num
    params['negative_weight'] = negative_weight

    gamma = config.getfloat('Training', 'gamma')
    params['gamma'] = gamma
    lambda_ = config.getfloat('Training', 'lambda')
    params['lambda'] = lambda_
    sampling_sift_pos = config.getboolean('Training', 'sampling_sift_pos')
    params['sampling_sift_pos'] = sampling_sift_pos
    
    test_batch_size = config.getint('Testing', 'test_batch_size')
    params['test_batch_size'] = test_batch_size
    topk = config.getint('Testing', 'topk') 
    params['topk'] = topk

    test_file_path = config['Testing']['test_file_path']

    # dataset processing
    train_data, test_data, train_mat, user_num, item_num, constraint_mat = load_data(train_file_path, test_file_path)
    train_data_datasets = gcnDataset(train_data,params,train_mat)
    
    train_loader = data.DataLoader(train_data_datasets, batch_size=batch_size, shuffle = True, num_workers=1)
    test_loader = data.DataLoader(list(range(user_num)), batch_size=test_batch_size, shuffle=False, num_workers=1)

    params['user_num'] = user_num
    params['item_num'] = item_num

    mask = torch.zeros(user_num, item_num)
    interacted_items = [[] for _ in range(user_num)]
    for (u, i) in train_data:
        mask[u][i] = -np.inf
        interacted_items[u].append(i)

    # test user-item interaction, which is ground truth
    test_ground_truth_list = [[] for _ in range(user_num)]
    for (u, i) in test_data:
        test_ground_truth_list[u].append(i)

    # Compute \Omega to extend UltraGCN to the item-item co-occurrence graph
    ii_cons_mat_path = 'prebuilds/' + dataset + '_ii_constraint_mat'
    ii_neigh_mat_path = 'prebuilds/' + dataset + '_ii_neighbor_mat'
    
    rec_i_64_path = 'prebuilds/' + dataset + '_rec_i_64'
    rec_u_64_path = 'prebuilds/' + dataset + '_rec_u_64'
    
    if os.path.exists(ii_cons_mat_path):
        ii_constraint_mat = pload(ii_cons_mat_path)
        ii_neighbor_mat = pload(ii_neigh_mat_path)
        # rec_i_64 = pload(rec_i_64_path)
        # rec_u_64 = pload(rec_u_64_path)
        rec_i_64 = np.load(rec_i_64_path)
        rec_u_64 = np.load(rec_u_64_path)
    else:
        ii_neighbor_mat, ii_constraint_mat,rec_i_64,rec_u_64 = get_ii_constraint_mat(train_mat, ii_neighbor_num,params=params)
        pstore(ii_neighbor_mat, ii_neigh_mat_path)
        pstore(ii_constraint_mat, ii_cons_mat_path)
        pstore(rec_i_64, rec_i_64_path)
        pstore(rec_u_64, rec_u_64_path)

    return params, constraint_mat, ii_constraint_mat, ii_neighbor_mat, train_loader, test_loader, mask, test_ground_truth_list, interacted_items,train_mat,rec_i_64,rec_u_64


def get_ii_constraint_mat(train_mat, num_neighbors, ii_diagonal_zero = False,params=None):
    print('Computing \\Omega for the item-item graph... ')
    A = train_mat.T.dot(train_mat)	# I * I
    print(1)
    n_items = A.shape[0]
    res_mat = torch.zeros((n_items, num_neighbors))
    res_sim_mat = torch.zeros((n_items, num_neighbors))
    if ii_diagonal_zero:
        A[range(n_items), range(n_items)] = 0
    items_D = np.sum(A, axis = 0).reshape(-1)
    users_D = np.sum(A, axis = 1).reshape(-1)
    
    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)
    print(2)
    temp = beta_uD.dot(beta_iD)
    print(3)
    all_ii_constraint_mat = torch.from_numpy(temp)
    print(4)
    for i in range(n_items):
        row = all_ii_constraint_mat[i] * torch.from_numpy(A.getrow(i).toarray()[0])
        row_sims, row_idxs = torch.topk(row, num_neighbors)
        res_mat[i] = row_idxs
        res_sim_mat[i] = row_sims
        if i % 15000 == 0:
            print('i-i constraint matrix {} ok'.format(i))

    print('Computation \\Omega OK!')
    return res_mat.long(), res_sim_mat.float(),res_mat.long(), res_sim_mat.float()

def sapling(M,projection):
    if projection == 0:
        N = M.shape[1]
        k = np.sum(M, axis = 1)
        CO = np.dot(M,M.T)
        B = np.nan_to_num((1-(CO*(1-CO/k)+(k-CO.T).T*(1-(k-CO.T).T/(N-k))).T/(k*(1-k/N))).T*np.sign(((CO*N/k).T/k).T-1))
    else:
        N = M.shape[0]
        k = np.sum(M, axis = 0)
        CO = np.dot(M.T,M)
        B = np.nan_to_num((1-(CO*(1-CO/k)+(k-CO.T).T*(1-(k-CO.T).T/(N-k))).T/(k*(1-k/N))).T*np.sign(((CO*N/k).T/k).T-1))
    return B
  
def load_data(train_file, test_file):
    trainUniqueUsers, trainItem, trainUser = [], [], []
    testUniqueUsers, testItem, testUser = [], [], []
    n_user, m_item = 0, 0
    trainDataSize, testDataSize = 0, 0
    with open(train_file, 'r') as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                items = [int(i) for i in l[1:]]
                uid = int(l[0])
                trainUniqueUsers.append(uid)
                trainUser.extend([uid] * len(items))
                trainItem.extend(items)
                m_item = max(m_item, max(items))
                n_user = max(n_user, uid)
                trainDataSize += len(items)
    trainUniqueUsers = np.array(trainUniqueUsers)
    trainUser = np.array(trainUser)
    trainItem = np.array(trainItem)

    with open(test_file) as f:
        for l in f.readlines():
            if len(l) > 0:
                l = l.strip('\n').split(' ')
                try:
                    items = [int(i) for i in l[1:]]
                except:
                    items = []
                uid = int(l[0])
                testUniqueUsers.append(uid)
                testUser.extend([uid] * len(items))
                testItem.extend(items)
                try:
                    m_item = max(m_item, max(items))
                except:
                    m_item = m_item
                n_user = max(n_user, uid)
                testDataSize += len(items)

    train_data = []
    test_data = []

    n_user += 1
    m_item += 1

    for i in range(len(trainUser)):
        train_data.append([trainUser[i], trainItem[i]])
    for i in range(len(testUser)):
        test_data.append([testUser[i], testItem[i]])
    train_mat = sp.dok_matrix((n_user, m_item), dtype=np.float32)

    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    # construct degree matrix for graphmf

    items_D = np.sum(train_mat, axis = 0).reshape(-1)
    users_D = np.sum(train_mat, axis = 1).reshape(-1)

    beta_uD = (np.sqrt(users_D + 1) / users_D).reshape(-1, 1)
    beta_iD = (1 / np.sqrt(items_D + 1)).reshape(1, -1)

    constraint_mat = {"beta_uD": torch.from_numpy(beta_uD).reshape(-1),
                      "beta_iD": torch.from_numpy(beta_iD).reshape(-1)}

    return train_data, test_data, train_mat, n_user, m_item, constraint_mat


def pload(path):
	with open(path, 'rb') as f:
		res = pickle.load(f)
	print('load path = {} object'.format(path))
	return res

def pstore(x, path):
	with open(path, 'wb') as f:
		pickle.dump(x, f)
	print('store object in path = {} ok'.format(path))


def Sampling(pos_train_data, item_num, neg_ratio, interacted_items, sampling_sift_pos,train_mat):
    Neg = []
    for pos in pos_train_data[0]:
        neg_candidates = np.where(train_mat[pos]==0)[0] #np.arange(item_num)
        neg_items = np.random.choice(neg_candidates, (neg_ratio), replace = True)
        Neg.append(neg_items)
    neg_items = torch.from_numpy(np.array(Neg))
    return pos_train_data[0], pos_train_data[1], neg_items	# users, pos_items, neg_items