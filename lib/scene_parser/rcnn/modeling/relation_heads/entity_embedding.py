import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearEmbedding(nn.Module):
	'''
	Basic Embedding (using 2-layer MLP)
	'''
	def __init__(self, in_channels, hid_channels, out_channels, pool='max'):
		super(LinearEmbedding, self).__init__()
		self.fg_emb = nn.Sequential(
			nn.Linear(in_channels, hid_channels),
			nn.ReLU(True),
			nn.Linear(hid_channels, out_channels)
		)

		self.bg_emb = nn.Sequential(
			nn.Linear(in_channels, hid_channels),
			nn.ReLU(True),
			nn.Linear(hid_channels, out_channels)
		)

		if pool == 'avg':
			self.pool = nn.AdaptiveAvgPool2d(1)
		elif pool == 'max':
			self.pool = nn.AdaptiveMaxPool2d(1)

	def forward(self, subj, obj, bg):
		subj, obj, bg = map(self.pool, (subj, obj, bg))
		subj, obj, bg = subj.squeeze(), obj.squeeze(), bg.squeeze()

		subj = self.fg_emb(subj)
		obj = self.fg_emb(obj)
		bg = self.bg_emb(bg)
		
		return subj, obj, bg

class ConvEmbedding(nn.Module):
	'''
	Convolutional Embedding
	'''
	def __init__(self, in_channels, hid_channels, out_channels, kernel_size=14):
		'''
		'kernel_size' are set by manual
		'''
		super(ConvEmbedding, self).__init__()
		self.fg_emb = nn.Sequential(
			nn.Conv2d(in_channels, hid_channels, kernel_size=(8, 8)),
			nn.ReLU(),
			nn.Conv2d(hid_channels, out_channels, kernel_size=(7, 7))
		)

		self.bg_emb = nn.Sequential(
			nn.Conv2d(in_channels, hid_channels, kernel_size=(8, 8)),
			nn.ReLU(),
			nn.Conv2d(hid_channels, out_channels, kernel_size=(7, 7))
		)

		'''
		self.fg_emb = nn.Sequential(
			nn.Conv2d(in_channels, hid_channels, kernel_size=(kernel_size, kernel_size)),
			nn.ReLU(),
			nn.Conv2d(hid_channels, out_channels, kernel_size=(1, 1))
		)

		self.bg_emb = nn.Sequential(
			nn.Conv2d(in_channels, hid_channels, kernel_size=(kernel_size, kernel_size)),
			nn.ReLU(),
			nn.Conv2d(hid_channels, out_channels, kernel_size=(1, 1))
		)
		'''
	def forward(self, subj, obj, bg):
		subj = self.fg_emb(subj)
		obj = self.fg_emb(obj)
		bg = self.bg_emb(bg)
		
		subj, obj, bg = subj.squeeze(), obj.squeeze(), bg.squeeze()
		return subj, obj, bg

class EntityEmbedding(nn.Module):
	'''
	Embed foreground(subject, object) and background
	'''
	def __init__(self, in_channels, hid_channels, out_channels, mode=None):
		super(EntityEmbedding, self).__init__()
		self.mode = mode

		if mode == 'linear':
			self.linear_emb = LinearEmbedding(in_channels, hid_channels, out_channels)
		elif mode == 'conv':
			self.conv_emb = ConvEmbedding(in_channels, hid_channels, out_channels)

	def forward(self, subj, obj, bg):
		if self.mode == 'linear':
			return self.linear_emb(subj, obj, bg)
		if self.mode == 'conv':
			return self.conv_emb(subj, obj, bg)

def entity_embedding(in_channels, hid_channels, out_channels, mode=None):
	return EntityEmbedding(in_channels, hid_channels, out_channels, mode)