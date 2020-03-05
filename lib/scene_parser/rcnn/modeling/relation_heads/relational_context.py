import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(GCNLayer, self).__init__()
		self.linear =  nn.Linear(in_dim, out_dim)

	def forward(self, input, adj):
		return torch.matmul(adj, self.linear(input)) # Nx3x1024 -> (linear) -> Nx3x1024 -> (matmul) -> Nx3x1024

class GCN(nn.Module):
	def __init__(self, in_dim, hid_dim, out_dim):
		super(GCN, self).__init__()
		self.gc1 = GCNLayer(in_dim, hid_dim)
		self.gc2 = GCNLayer(hid_dim, out_dim)
		self.relu = nn.ReLU()
		# self.dropout = nn.Dropout(0.5)

	def forward(self, x, adj):
		adj = adj.to(x.device)
		out = self.gc1(x, adj) # Nx3x1024
		out = self.relu(out) # Nx3x1024
		# if self.training:
		# 	out = self.dropout(out) # Nx3x1024
		out = self.gc2(out, adj) # Nx3x1024
		out = x + out # Nx3x1024; add self
		return out # Nx3x1024

class RelationalContext(nn.Module):
	'''
	Relational context aggregation module
	'''
	def __init__(self, dim=1024):
		super(RelationalContext, self).__init__()
		self.gcn = GCN(in_dim=dim, hid_dim=dim, out_dim=dim)

	def forward(self, subj_emb, obj_emb, bg_emb): # Nx1024
		N = subj_emb.size(0) # N
		features = torch.stack((subj_emb, obj_emb, bg_emb), dim=1) # Nx3x1024
		adj = 1 - torch.eye(3) # ones(3x3) diagonal(0); each indicates subj, obj, background
		adj_N = torch.stack([adj for _ in range(N)], dim=0) # Nx3x3

		relation_ctx = self.gcn(features, adj_N) # Nx3x1024
		#relation_ctx = torch.sum(relation_ctx, dim=1) # Nx1024

		return relation_ctx # Nx1024