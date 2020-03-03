import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(GCNLayer, self).__init__()
		self.out_dim = out_dim
		self.weight = nn.Parameter(torch.FloatTensor(in_dim, out_dim)) # 1024x1024
		self.bias = nn.Parameter(torch.FloatTensor(out_dim)) # 1024
		self.reset_parameters()

	def reset_parameters(self):
		stdv = 1. / math.sqrt(self.out_dim) # self.weight.size(1)
		self.weight.data.uniform_(-stdv, stdv)
		self.bias.data.uniform_(-stdv, stdv)

	def forward(self, input, adj):
		output = torch.mm(adj, torch.mm(input, self.weight)) # (3x3) x ((3x1024) x (1024x1024)) = 3x1024
		return output + self.bias # 3x1024

class GCN(nn.Module):
	def __init__(self, in_dim, hid_dim, out_dim):
		super(GCN, self).__init__()
		self.gc1 = GCNLayer(in_dim, hid_dim)
		self.gc2 = GCNLayer(hid_dim, out_dim)
		self.relu = nn.ReLU()
		self.dropout = nn.Dropout(0.5)

	def forward(self, x, adj):
		adj = adj.to(x.device)
		out = self.gc1(x, adj) # 3x1024
		out = self.relu(out) # 3x1024
		if self.training:
			out = self.dropout(out) # 3x1024
		out = self.gc2(out, adj) # 3x1024
		out = x + out # 3x1024 + 3x1024 = 3x1024; add self
		out = torch.sum(out, dim=0) # 1024
		return out # 1024

class GlocalContext(nn.Module):
	'''
	Glocal context aggregation module
	'''
	def __init__(self, dim=1024):
		super(GlocalContext, self).__init__()
		self.gcn = GCN(in_dim=dim, hid_dim=dim, out_dim=dim)

	def forward(self, subj_emb, obj_emb, bg_emb):
		features = torch.stack((subj_emb, obj_emb, bg_emb), dim=1) # Nx3x1024
		adj = 1 - torch.eye(3) # ones(3x3) diagonal(0); each indicates subj, obj, background
		
		glocal_features = []
		for feature in features: # 3x1024
			feature = self.gcn(feature, adj) # 1024
			glocal_features.append(feature)

		glocal_features = torch.stack(glocal_features, dim=0) # Nx1024
		return glocal_features # Nx1024