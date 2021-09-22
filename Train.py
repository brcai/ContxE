# -*- coding: utf-8 -*-
import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'

import model_contxe as KGE
from Dataset import KnowledgeGraph
from Dataset_YG import KnowledgeGraphYG
from DatasetAKG import KnowledgeGraphAKG


import torch
import numpy as np
from time import time
from sklearn.utils import shuffle as skshuffle
import os

def mean_rank(rank):
    m_r = 0
    N = len(rank)
    for i in rank:
        m_r = m_r + i / N

    return m_r


def mrr(rank):
    mrr = 0
    N = len(rank)
    for i in rank:
        mrr = mrr + 1 / i / N

    return mrr


def hit_N(rank, N):
    hit = 0
    for i in rank:
        if i <= N:
            hit = hit + 1

    hit = hit / len(rank)

    return hit

def get_minibatches(X, mb_size, shuffle=True):
    X_shuff = X.copy()
    if shuffle:
        X_shuff = skshuffle(X_shuff)

    for i in range(0, X_shuff.shape[0], mb_size):
        yield X_shuff[i:i + mb_size]


def sample_negatives(X, C, kg):
    M = X.shape[0]
    X_corr = X
    for i in range(C-1):
        X_corr = np.concatenate((X_corr,X),0)
    X_corr[:,1]=torch.randint(kg.n_entity,[int(M*C)]) 

    return X_corr


def train(data_dir='yago',
          dim=500,
          batch=512,
          lr=0.1,
          max_epoch=5000,
          min_epoch=250,
          gamma1=1,
          gamma2=1,
          L = 'L1',
          negsample_num=10,
          lossname = 'logloss',
          cmin = 0.001,
          cuda_able = False,
          rev_set = 0,
          temp = 0.5,
          gran = 7,
          count = 300,
          cuda_idx = 0
          ):

    randseed = 9999
    np.random.seed(randseed)
    torch.manual_seed(randseed)

    model_path = "./saved/ContxE_"+data_dir+"_lr"+str(lr)+"_loss"+lossname+"gamma1"+str(gamma1)

    """
    Data Loading
    """
    if data_dir == 'gdelt':
        n_day = 366
        kg = KnowledgeGraph(data_dir=data_dir,gran=gran,rev_set = rev_set)
    elif data_dir=='icews14':
        n_day = 365
        kg = KnowledgeGraph(data_dir=data_dir,gran=gran,rev_set = rev_set)
    elif data_dir == 'icews05-15':
        n_day = 4017
        kg = KnowledgeGraph(data_dir=data_dir,gran=gran,rev_set = rev_set)      
    elif data_dir == 'yago1830':
        n_day = 190
        kg = KnowledgeGraphYG(data_dir=data_dir, rev_set = rev_set) 
    elif data_dir == 'AKG':
        n_day = 5
        kg = KnowledgeGraphAKG(data_dir=data_dir,gran=gran,rev_set = rev_set)  

    """
    Create a model
    """

    model = KGE.ContxE(kg, embedding_dim=dim, batch_size=batch, learning_rate=lr, gamma1=gamma1, gamma2=gamma2, L=L, gran=gran, n_day=n_day, gpu=cuda_able,cuda_idx=cuda_idx)
    solver = torch.optim.Adagrad(model.parameters(), model.learning_rate)
    optimizer = 'Adagrad'
    

    train_pos = np.array(kg.training_triples)
    validation_pos = np.array(kg.validation_triples)
    test_pos = np.array(kg.test_triples)
        
    losses = []
    mrr_std = 0
    C = negsample_num
    patience = 0

    start = time()
    
    """
    Training Process
    """
    for epoch in range(max_epoch):
        print('Epoch-{}'.format(epoch + 1))
        print('————————————————')
        it = 0
        train_triple = list(get_minibatches(train_pos, batch, shuffle=True))
        start = time()
        for iter_triple in train_triple:
            if iter_triple.shape[0] < batch:
                break
            
            iter_neg = sample_negatives(iter_triple, C, kg)
            pos_score = model.forward(iter_triple)
            neg_score = model.forward(iter_neg)
                
            if lossname == 'logloss':
                loss = model.log_rank_loss(pos_score, neg_score,temp=temp)
            elif lossname == "doubleloss":
                loss = model.double_rank_loss(pos_score, neg_score,temp=temp)
            else:
                loss = model.rank_loss(pos_score, neg_score)
            losses.append(loss.item())

            solver.zero_grad()
            loss.backward()
            solver.step()

            model.normalize_embeddings()

            it += 1
        end = time()
        print('Iter-{}; loss: {:.4f};time per batch:{:.4f}s'.format(it, loss.item(), end - start))
        """
        Evaluation for Link Prediction
        """
        if epoch % 5 == 0 and epoch != 0:
            with torch.no_grad():
                print("validation: ")

                rank = model.rank_simple(test_pos,kg.test_facts,kg,rev_set=rev_set)

                m_rank = mean_rank(rank)
                mean_rr = mrr(rank)
                hit_1 = hit_N(rank, 1)
                hit_3 = hit_N(rank, 3)
                hit_5 = hit_N(rank, 5)
                hit_10 = hit_N(rank, 10)
                print('test result:')
                print('Mean Rank: {:.0f}'.format(m_rank))
                print('Mean RR: {:.4f}'.format(mean_rr))
                print('Hit@1: {:.4f}'.format(hit_1))
                print('Hit@3: {:.4f}'.format(hit_3))
                print('Hit@5: {:.4f}'.format(hit_5))
                print('Hit@10: {:.4f}'.format(hit_10))

                if hit_1 > 0.5:
                    torch.save(model.state_dict(), model_path+"_hits1"+str(hit_1))
    #fp.close()