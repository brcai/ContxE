# -*- coding: utf-8 -*-
import os
#os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.nn import functional as F
from torch.autograd import Variable
from numpy.random import RandomState
import random
        
class ContxE(nn.Module):
	def __init__(self, kg, embedding_dim, batch_size, learning_rate, L, gran, gamma1, gamma2, n_day, window=5, gpu=False, cuda_idx=0):
		super(ContxE, self).__init__()
		self.gpu = gpu
		self.kg = kg
		self.embedding_dim = embedding_dim
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.gamma1 = gamma1
		self.gamma2 = gamma2
		self.n_day = n_day
		self.gran = gran
		self.cuda_idx = cuda_idx

		self.base_time = 0
		self.dim = embedding_dim
		self.maxtime = n_day
		self.device = "cpu" 
		if gpu==True:
			self.device = "cuda"+":"+str(cuda_idx)
		self.alpha = 0.01
		self.window = window

		print(self.device)

		self.sigmoid = nn.Sigmoid()
		self.relu = nn.ReLU()
		self.beta = 1.0

		self.L = L
		# Nets
		self.emb_E_real = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
		self.emb_E_img = torch.nn.Embedding(self.kg.n_entity, self.embedding_dim, padding_idx=0)
		self.emb_R_real = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
		self.emb_R_img = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim, padding_idx=0)
		self.time_base_emb, self.time_inc_emb, self.time_emb = self._init_time_emb()
		#self.matrix_left = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim*self.embedding_dim, padding_idx=0)
		#self.matrix_right = torch.nn.Embedding(self.kg.n_relation, self.embedding_dim*self.embedding_dim, padding_idx=0)

		# Initialization
		r = 6 / np.sqrt(self.embedding_dim)
		self.emb_E_real.weight.data.uniform_(-r, r)
		self.emb_E_img.weight.data.uniform_(-r, r)
		self.emb_R_real.weight.data.uniform_(-r, r)
		self.emb_R_img.weight.data.uniform_(-r, r)
		#self.time_emb.weight.data.uniform_(-r, r)
		#self.emb_T_img.weight.data.uniform_(-r, r)

		if self.gpu:
			self.cuda(device=self.cuda_idx)

	def _init_time_emb(self):
		time_base_emb = nn.Embedding(num_embeddings=1,
										embedding_dim=self.dim)
		time_inc_emb = nn.Embedding(num_embeddings=1,
										embedding_dim=self.dim)
		uniform_range = 6 / np.sqrt(self.dim)
		time_base_emb.weight.data.uniform_(-uniform_range, uniform_range)
		time_inc_emb.weight.data.uniform_(-uniform_range, uniform_range)
		time_emb = nn.Embedding(num_embeddings=self.maxtime+1,
										embedding_dim=self.dim)

		time_ts = torch.zeros(self.maxtime+2, self.dim).to(self.device)
		idx = torch.tensor([0])
		time_base_ts = time_base_emb(idx)
		time_base_ts = time_base_ts.to(self.device)
		time_inc_ts = time_inc_emb(idx)
		time_inc_ts = time_inc_ts.to(self.device)
		for i in range(self.maxtime+1):
			inc = torch.tensor([i]).to(self.device)
			time_ts[i] = time_base_ts + self.alpha*inc*time_inc_ts
			#eps = Variable(torch.randn_like(time_inc_ts), requires_grad=False)
			#time_ts[i] = time_base_ts + self.alpha*inc*time_inc_ts + eps
		time_emb.weight.data = time_ts
		return time_base_emb, time_inc_emb, time_emb

	def _get_time_emb(self, time):
		tmp_time = time.reshape(-1,1).repeat(1,self.window)
		tmp = (torch.zeros(tmp_time.size(), dtype=torch.int64) - 1).to(self.device)
		tmp_int = torch.tensor([self.window - i - 1 for i in range(self.window)]).to(self.device)
		tmp_int = tmp_int.repeat(time.size()[0],1)
		tmp_time_new = torch.max(tmp_time - tmp_int, tmp)
		max_val = (torch.zeros(tmp_time.size(), dtype=torch.int64) + self.n_day).to(self.device)
		tmp_time_new = torch.where(tmp_time_new != -1, tmp_time_new, max_val)
		time_emb = torch.zeros(time.size()[0], self.window, self.dim).to(self.device)
		for i in range(self.window):
			time_emb[:,i] = self.time_emb(tmp_time_new[:,i])
		return time_emb


	'''
	def forward(self, X):
		h_i, t_i, r_i, d_i = X[:, 0].astype(np.int64), X[:, 1].astype(np.int64), X[:, 2].astype(np.int64), X[:, 3].astype(np.int64)//self.gran

		if self.gpu:
			h_i = Variable(torch.from_numpy(h_i).cuda(device=self.cuda_idx))
			t_i = Variable(torch.from_numpy(t_i).cuda(device=self.cuda_idx))
			r_i = Variable(torch.from_numpy(r_i).cuda(device=self.cuda_idx))
			d_i = Variable(torch.from_numpy(d_i).cuda(device=self.cuda_idx))
		else:
			h_i = Variable(torch.from_numpy(h_i))
			t_i = Variable(torch.from_numpy(t_i))
			r_i = Variable(torch.from_numpy(r_i))
			d_i = Variable(torch.from_numpy(d_i))

		pi = 3.14159265358979323846

		d_list = self._get_time_emb(d_i)
		#d_list = self.time_emb(d_i)

		d_img = torch.sin(d_list)#/(6 / np.sqrt(self.embedding_dim)/pi))

		d_real = torch.cos(d_list)

       
		h_real = self.emb_E_real(h_i).view(-1, 1, self.embedding_dim) *d_real-\
					self.emb_E_img(h_i).view(-1, 1, self.embedding_dim) *d_img

		t_real = self.emb_E_real(t_i).view(-1, 1, self.embedding_dim) *d_real-\
					self.emb_E_img(t_i).view(-1, 1, self.embedding_dim)*d_img



		h_img = self.emb_E_real(h_i).view(-1, 1, self.embedding_dim) *d_img+\
					self.emb_E_img(h_i).view(-1, 1, self.embedding_dim) *d_real

		t_img = self.emb_E_real(t_i).view(-1, 1, self.embedding_dim) *d_img+\
				self.emb_E_img(t_i).view(-1, 1, self.embedding_dim) *d_real

		r_img = self.emb_R_img(r_i).view(-1, self.embedding_dim)

		r_real = self.emb_R_real(r_i).view(-1, self.embedding_dim)


		a_1 = torch.matmul(r_real.reshape(r_real.size()[0],1,-1), h_real.permute(0,2,1)).reshape(h_real.size()[0],-1,1)
		a_2 = torch.matmul(r_img.reshape(r_img.size()[0],1,-1), h_img.permute(0,2,1)).reshape(h_img.size()[0],-1,1)
		a_3 = torch.matmul(r_real.reshape(r_real.size()[0],1,-1), h_img.permute(0,2,1)).reshape(h_img.size()[0],-1,1)
		a_4 = torch.matmul(r_img.reshape(r_img.size()[0],1,-1), h_real.permute(0,2,1)).reshape(h_real.size()[0],-1,1)

		a_12 = torch.abs(a_1 - a_2)**2
		a_34 = torch.abs(a_3 + a_4)**2

		s_attention = F.softmax(torch.sqrt(a_12+a_34),dim=1)

		b_1 = torch.matmul(r_real.reshape(r_real.size()[0],1,-1), t_real.permute(0,2,1)).reshape(t_real.size()[0],-1,1)
		b_2 = torch.matmul(r_img.reshape(r_img.size()[0],1,-1), t_img.permute(0,2,1)).reshape(t_img.size()[0],-1,1)
		b_3 = torch.matmul(r_real.reshape(r_real.size()[0],1,-1), t_img.permute(0,2,1)).reshape(t_img.size()[0],-1,1)
		b_4 = torch.matmul(r_img.reshape(r_img.size()[0],1,-1), t_real.permute(0,2,1)).reshape(t_real.size()[0],-1,1)

		b_12 = torch.abs(b_1 - b_2)**2
		b_34 = torch.abs(b_3 + b_4)**2

		o_attention = F.softmax(torch.sqrt(b_12+b_34),dim=1)
		
		hy_real = torch.sum(s_attention*h_real, 1)
		hy_img = torch.sum(s_attention*h_img, 1)

		ty_real = torch.sum(o_attention*t_real, 1)
		ty_img = torch.sum(o_attention*t_img, 1)
        
		l1 = True

		if l1:
			out_real = torch.sum(torch.abs(hy_real + r_real - ty_real), 1)
			out_img = torch.sum(torch.abs(hy_img + r_img + ty_img), 1)
			out = out_real + out_img
		else:
			out_real = torch.sum((hy_real + r_real - ty_real)**2, 1)
			out_img = torch.sum((hy_img + r_img + ty_img)**2, 1)
			out = out_real + out_img
			out = torch.sqrt(out_img + out_real)

		return out
    '''

	def normalize_embeddings(self):
		self.emb_E_real.weight.data.renorm_(p=2, dim=0, maxnorm=1)
		self.emb_E_img.weight.data.renorm_(p=2, dim=0, maxnorm=1)

	def forward(self, X):
		h_i, t_i, r_i, d_i = X[:, 0].astype(np.int64), X[:, 1].astype(np.int64), X[:, 2].astype(np.int64), X[:, 3].astype(np.int64)

		if self.gpu:
			h_i = Variable(torch.from_numpy(h_i).cuda(device=self.cuda_idx))
			t_i = Variable(torch.from_numpy(t_i).cuda(device=self.cuda_idx))
			r_i = Variable(torch.from_numpy(r_i).cuda(device=self.cuda_idx))
			d_i = Variable(torch.from_numpy(d_i).cuda(device=self.cuda_idx))
		else:
			h_i = Variable(torch.from_numpy(h_i))
			t_i = Variable(torch.from_numpy(t_i))
			r_i = Variable(torch.from_numpy(r_i))
			d_i = Variable(torch.from_numpy(d_i))
		
		pi = 3.14159265358979323846

		d_list = self._get_time_emb(d_i)
		#d_list = self.time_emb(d_i)

		d_img = torch.sin(d_list)

		d_real = torch.cos(d_list)
				
		h_real = self.emb_E_real(h_i).view(-1, 1, self.embedding_dim) *d_real-\
					self.emb_E_img(h_i).view(-1, 1, self.embedding_dim) *d_img

		t_real = self.emb_E_real(t_i).view(-1, 1, self.embedding_dim) *d_real-\
					self.emb_E_img(t_i).view(-1, 1, self.embedding_dim)*d_img

		h_img = self.emb_E_real(h_i).view(-1, 1, self.embedding_dim) *d_img+\
					self.emb_E_img(h_i).view(-1, 1, self.embedding_dim) *d_real

		t_img = self.emb_E_real(t_i).view(-1, 1, self.embedding_dim) *d_img+\
				self.emb_E_img(t_i).view(-1, 1, self.embedding_dim) *d_real

		r_img = self.emb_R_img(r_i).view(-1, self.embedding_dim)

		r_real = self.emb_R_real(r_i).view(-1, self.embedding_dim)

		a_real = F.softmax(torch.matmul(r_real.reshape(r_real.size()[0],1,-1), h_real.permute(0,2,1)).reshape(h_real.size()[0],-1,1),dim=1)
		a_img = F.softmax(torch.matmul(r_img.reshape(r_img.size()[0],1,-1), h_img.permute(0,2,1)).reshape(h_real.size()[0],-1,1),dim=1)

		b_real = F.softmax(torch.matmul(r_real.reshape(r_real.size()[0],1,-1), t_real.permute(0,2,1)).reshape(t_real.size()[0],-1,1),dim=1)
		b_img = F.softmax(torch.matmul(r_img.reshape(r_img.size()[0],1,-1), t_img.permute(0,2,1)).reshape(t_real.size()[0],-1,1),dim=1)

		y_real = torch.sum(a_real*h_real, 1)
		y_img = torch.sum(a_img*h_img, 1)

		z_real = torch.sum(b_real*t_real, 1)
		z_img = torch.sum(b_img*t_img, 1)
        
		out_real = torch.sum(torch.abs(y_real + r_real - z_real), 1)
		out_img = torch.sum(torch.abs(y_img + r_img + z_img), 1)
		out = out_real + out_img

		return out


	def log_rank_loss(self, y_pos, y_neg, temp=0):
		M = y_pos.size(0)
		N = y_neg.size(0)
		y_pos = self.gamma1-y_pos
		y_neg = self.gamma1-y_neg
		C = int(N / M)
		y_neg = y_neg.view(C, -1).transpose(0, 1)
		p = F.softmax(temp * y_neg)
		loss_pos = torch.sum(F.softplus(-1 * y_pos))
		loss_neg = torch.sum(p * F.softplus(y_neg))
		loss = (loss_pos + loss_neg) / 2 / M
		if self.gpu:
			loss = loss.cuda(device=self.cuda_idx)
		return loss


	def rank_loss(self, y_pos, y_neg):
		M = y_pos.size(0)
		N = y_neg.size(0)
		C = int(N / M)
		y_pos = y_pos.repeat(C)
		if self.gpu:
			target = Variable(torch.from_numpy(-np.ones(N, dtype=np.float32))).cuda(device=self.cuda_idx)
		else:
			target = Variable(torch.from_numpy(-np.ones(N, dtype=np.float32))).cpu()
		loss = nn.MarginRankingLoss(margin=self.gamma1)
		loss = loss(y_pos, y_neg, target)
		return loss

	def double_rank_loss(self, y_pos, y_neg):
		M = y_pos.size(0)
		N = y_neg.size(0)
		C = int(N / M)
		y_pos = y_pos.repeat(C)
		if self.gpu:
			target = Variable(torch.from_numpy(-np.ones(N, dtype=np.float32))).cuda(device=self.cuda_idx)
			pads = Variable(torch.from_numpy(np.zeros(N, dtype=np.float32))).cuda(device=self.cuda_idx)
		else:
			target = Variable(torch.from_numpy(-np.ones(N, dtype=np.float32))).cpu()
			pads = Variable(torch.from_numpy(np.zeros(N, dtype=np.float32))).cuda(device=self.cuda_idx)
		loss_pos = nn.MarginRankingLoss(margin=self.gamma1)
		loss_pos = loss_pos(y_pos, pads, target)
		loss_neg = nn.MarginRankingLoss(margin=self.gamma2)
		loss_neg = loss_pos(pads, y_neg, target)
		loss = loss_pos + loss_neg
		return loss


	def rank_simple(self, X, facts, kg, rev_set=0):
		rank = []
		rev_set=0
		with torch.no_grad():
			for triple, fact in zip(X, facts):
				if random.randint(0,1):
					X_i = np.ones([self.kg.n_entity, 4])
					for i in range(0, self.kg.n_entity):
						X_i[i, 0] = triple[0]
						X_i[i, 1] = i
						X_i[i, 2] = triple[2]
						X_i[i, 3] = triple[3]
					i_score = self.forward(X_i)
					if self.gpu:
						i_score = i_score.cuda(device=self.cuda_idx)
					filter_out = kg.to_skip_final['rhs'][(fact[0], fact[2],fact[3], fact[4])]                            
					target = i_score[int(triple[1])].clone()
					i_score[filter_out]=1e6 
					rank_triple=torch.sum((i_score < target).float()).cpu().item()+1
					rank.append(rank_triple)
				else:
					X_i = np.ones([self.kg.n_entity, 4])
					for i in range(0, self.kg.n_entity):
						X_i[i, 0] = i
						X_i[i, 1] = triple[1]
						X_i[i, 2] = triple[2]
						X_i[i, 3] = triple[3]
					i_score = self.forward(X_i)
					if self.gpu:
						i_score = i_score.cuda(device=self.cuda_idx)
					filter_out = kg.to_skip_final['lhs'][(fact[1], fact[2],fact[3], fact[4])]       
					target = i_score[int(triple[0])].clone()
					i_score[filter_out]=1e6
					rank_triple=torch.sum((i_score < target).float()).cpu().item()+1
					rank.append(rank_triple)
		return rank


	def predict(self, triple, types):
		with torch.no_grad():
			X_i = np.ones([self.kg.n_entity, 4])
			for i in range(0, self.kg.n_entity):
				X_i[i, 0] = triple[0]
				X_i[i, 1] = i
				X_i[i, 2] = triple[1]
				X_i[i, 3] = triple[2] - self.kg.start_date
			i_score = self.forward(X_i)
			if self.gpu:
				i_score = i_score.cuda(device=self.cuda_idx)
			i_score[triple[0]] = 1e6 
			idx = 0
			rank = []
			while idx < 10:
				loc = torch.argmin(i_score)
				if types[loc][2] == 0 and triple[1] == 0:
					rank.append([loc,types[loc][1]])
					idx += 1
				elif types[loc][2] == 3 and triple[1] == 1:
					rank.append([loc,types[loc][1]])
					idx += 1
				i_score[loc] = 1e6

		return rank
