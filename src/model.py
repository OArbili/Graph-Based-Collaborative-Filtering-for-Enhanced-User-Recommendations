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

# Multi Aspect Control Graph Recommendation System


class MultiAspectGraph(nn.Module):
    def __init__(self, params, constraint_mat, ii_constraint_mat, ii_neighbor_mat,rec_i_64,rec_u_64,load_weights=False):
        super(MultiAspectGraph, self).__init__()
        self.user_num = params['user_num']
        self.item_num = params['item_num']
        self.embedding_dim = params['embedding_dim']
        self.w1 = params['w1']
        self.w2 = params['w2']
        self.w3 = params['w3']
        self.w4 = params['w4']
        self.load_weights = load_weights
        
        self.weight1 = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.weight2 = nn.Parameter(torch.tensor(1.0, requires_grad=True))
        self.weight3 = nn.Parameter(torch.tensor(1.0, requires_grad=True))

        self.negative_weight = params['negative_weight']
        self.gamma = params['gamma']
        self.lambda_ = params['lambda']


        self.user_embeds = nn.Embedding(self.user_num, self.embedding_dim)
        self.item_embeds = nn.Embedding(self.item_num, self.embedding_dim)
        


        self.constraint_mat = constraint_mat
        self.ii_constraint_mat = ii_constraint_mat
        self.ii_neighbor_mat = ii_neighbor_mat

        self.rec_i_64 = rec_i_64
        self.rec_u_64 = rec_u_64

        
        self.initial_weight = params['initial_weight']
        self.initial_weights()

    def initial_weights(self):
        nn.init.normal_(self.user_embeds.weight, std=self.initial_weight)
        nn.init.normal_(self.item_embeds.weight, std=self.initial_weight)
        if self.load_weights:
            print("start load weight")
            print(self.user_embeds.weight.shape)
            print(self.item_embeds.weight.shape)

            weight_user_embeds = self.rec_u_64
            weight_item_embeds= self.rec_i_64

            self.user_embeds.weight = torch.nn.Parameter(torch.from_numpy(weight_user_embeds).cuda().float())
            self.item_embeds.weight = torch.nn.Parameter(torch.from_numpy(weight_item_embeds).cuda().float())
            print("end load weight")

    def get_omegas(self, users, pos_items, neg_items):
        device = self.get_device()
        if self.w2 > 0:
            pos_weight = torch.mul(self.constraint_mat['beta_uD'][users.cpu()].cuda(), self.constraint_mat['beta_iD'][pos_items.cpu()].cuda()).cuda()
            pos_weight = self.w1 + self.w2 * pos_weight
        else:
            pos_weight = self.w1 * torch.ones(len(pos_items)).cuda()
        
        # users = (users * self.item_num).unsqueeze(0)
        if self.w4 > 0:
            neg_weight = torch.mul(torch.repeat_interleave(self.constraint_mat['beta_uD'][users.cpu()], neg_items.size(1)), self.constraint_mat['beta_iD'][neg_items.cpu().flatten()]).cuda()
            neg_weight = self.w3 + self.w4 * neg_weight
        else:
            neg_weight = self.w3 * torch.ones(neg_items.size(0) * neg_items.size(1)).cuda()


        weight = torch.cat((pos_weight, neg_weight))
        return weight

    def cal_loss_L(self, users, pos_items, neg_items, omega_weight,epoch):
        device = self.get_device()
        user_embeds = self.user_embeds(users)
        pos_embeds = self.item_embeds(pos_items)
        neg_embeds = self.item_embeds(neg_items)
        
      
        pos_scores = (user_embeds * pos_embeds).sum(dim=-1) # batch_size
        user_embeds = user_embeds.unsqueeze(1)
        neg_scores = (user_embeds * neg_embeds).sum(dim=-1) # batch_size * negative_num
        
        neg_labels = torch.zeros(neg_scores.detach().cpu().size()).cuda()
        neg_loss = F.binary_cross_entropy_with_logits(neg_scores, neg_labels, weight = omega_weight[len(pos_scores):].view(neg_scores.size()), reduction='none').mean(dim = -1)
        
        pos_labels = torch.ones(pos_scores.detach().cpu().size()).cuda()
        pos_loss = F.binary_cross_entropy_with_logits(pos_scores, pos_labels, weight = omega_weight[:len(pos_scores)], reduction='none')
        
        gamma = 4
        loss = pos_loss + neg_loss * self.negative_weight
        weight = torch.exp(gamma * (pos_scores - neg_scores[:,0])).detach().cpu().numpy().mean()
        g_loss = torch.sum(torch.nn.functional.softplus(neg_scores[:,0] - pos_scores,beta=weight))

        loss = loss.sum()
        w = self.clac_loss_w(epoch)
        return w*loss + (1-w)*g_loss

    def cal_loss_I(self, users, pos_items):
        device = self.get_device()
        neighbor_embeds = self.item_embeds(self.ii_neighbor_mat[pos_items.cpu()].cuda())    # len(pos_items) * num_neighbors * dim
        sim_scores = self.ii_constraint_mat[pos_items.cpu()].cuda()     # len(pos_items) * num_neighbors
        
        user_embeds = self.user_embeds(users)
        user_embeds = user_embeds.unsqueeze(1)
        
        loss = -sim_scores * (user_embeds * neighbor_embeds).sum(dim=-1).sigmoid().log()
      
        return loss.sum()

    def norm_loss(self):
        loss = 0.0
        for parameter in self.parameters():
            loss += torch.sum(parameter ** 2)
        return loss / 2
    
    def clac_loss_w(self,epoch,max_epoch = 30):
        w = min(1,epoch / max_epoch)
        return w

    def forward(self, users, pos_items, neg_items,epoch):
    
        omega_weight = self.get_omegas(users, pos_items, neg_items)
        
        loss = self.cal_loss_L(users, pos_items, neg_items, omega_weight,epoch)
        loss += self.gamma * self.norm_loss()
        loss += self.lambda_ * self.cal_loss_I(users, pos_items)

        return loss

    def test_foward(self, users):
        items = torch.arange(self.item_num).to(users.device)
        user_embeds = self.user_embeds(users)
        item_embeds = self.item_embeds(items)
         
        return user_embeds.mm(item_embeds.t())

    def get_device(self):
        return self.user_embeds.weight.device
