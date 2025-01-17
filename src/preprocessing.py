import numpy as np
# import utils
import time

def read_data(data):
    if data == "AmazonBooks_m1":
        Nu = 52643
        Ni = 91599
    elif data == "Gowalla_m1":
        Nu = 29858
        Ni = 40981
        
    f = open('./data/{}/train.txt'.format(data), 'r')
    lines = f.readlines()
    lines = [l.strip('\n\r') for l in lines]
    lines = [l.split(' ') for l in lines]
    train = [l[1:] for l in lines]
    for i in range(len(train)):
        if train[i] == [""]:
            train[i] = []
    train = [[int(x) for x in i] for i in train]


    f = open('./data/{}/test.txt'.format(data), 'r')
    lines = f.readlines()
    lines = [l.strip('\n\r') for l in lines]
    lines = [l.split(' ') for l in lines]
    test = [l[1:] for l in lines]
    for i in range(len(test)):
        if test[i] == [""]:
            test[i] = []
    test = [[int(x) for x in i] for i in test]
    
    M_train = np.zeros([Nu,Ni])
    for u in range(Nu):
        if len(train[u]) != 0:
            for i in range(len(train[u])):
                M_train[u,int(train[u][i])] = 1
                
    return Nu, Ni, M_train, train, test


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

t1 = time.time()

# data = "AmazonBooks_m1"
# N_users, N_items, M, train, test = utils.read_data(data)
# M = M.astype(np.float32)

# print("measuring similarity of users...")
# B = sapling(M,0)

# print("measuring user-based recommendations...")
# rec_u = np.nan_to_num(np.dot(B,M).T/np.sum(abs(B), axis = 1)).T 
# np.save("rec_u.npy", rec_u)

# print("measuring similarity of items...")
# B = sapling(M,1)

# print("measuring item-based recommendations...")
# rec_i = np.nan_to_num(np.dot(M,B)/np.sum(abs(B), axis = 0))
# np.save("rec_i.npy", rec_i)
