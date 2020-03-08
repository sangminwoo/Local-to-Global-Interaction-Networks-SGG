import torch
import torch.nn as nn
import torch.nn.functional as F

class Interaction(nn.Module):
	'''
	relation network
	'''
	def __init__(self, x):
		super(Interaction, self).__init__()
		self.mlp1 = nn.Linear(2048, 1024)
		self.mlp2 = nn.Linear(1024, 1024)
		self.mlp3 = nn.Linear(1024, 1024)
		self.mlp4 = nn.Linear(1024, 1024)
		self.relu = nn.ReLU()

	def forward(self, subj, obj, bg):
		self.mlp1(torch.cat((subj, obj), 0))
		self.mlp1


		subj_mlp = self.fg_mlp(subj)
		obj_mlp = self.fg_mlp(obj)
		bg_mlp = self.bg_mlp(bg)

		torch.cat((subj, obj, bg))