import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.scene_parser.rcnn.modeling.relation_heads.relpn.multi_head_att import MultiHeadAttention

class InstanceEmbedding(nn.Module):
	def __init__(self, subj_dim, obj_dim, bg_dim, hid_dim, out_dim):
		super(InstanceEmbedding, self).__init__()
		self.subj_emb = nn.Sequential(
            nn.Linear(subj_dim, hid_dim),
            nn.ReLU(True),
            nn.Linear(hid_dim, out_dim)
        )
		self.obj_emb = nn.Sequential(
		    nn.Linear(obj_dim, hid_dim),
		    nn.ReLU(True),
		    nn.Linear(hid_dim, out_dim)
		)
		self.bg_emb = nn.Sequential(
		    nn.Linear(bg_dim, hid_dim),
		    nn.ReLU(True),
		    nn.Linear(hid_dim, out_dim)
		)

		self.subj_mha = MultiHeadAttention(8, 1024)
		self.obj_mha = MultiHeadAttention(8, 1024)
		self.bg_mha = MultiHeadAttention(8, 1024)

	def forward(self, subj, obj, bg):
		subj_emb = self.subj_emb(subj)
		subj_emb = self.subj_mha(subj_emb, subj_emb, subj_emb).squeeze(1)
		obj_emb = self.obj_emb(obj)
		obj_emb = self.obj_mha(obj_emb, obj_emb, obj_emb).squeeze(1)
		bg_emb = self.bg_emb(bg)
		bg_emb = self.bg_mha(bg_emb, bg_emb, bg_emb).squeeze(1)
		return subj_emb, obj_emb, bg_emb

class RelationalEmbedding(nn.Module):
	def __init__(self, subj_dim, obj_dim, bg_dim, hid_dim, out_dim):
		super(RelationalEmbedding, self).__init__()
		self.rel_so = nn.Sequential(
			nn.Linear(subj_dim + obj_dim, hid_dim),
			nn.ReLU(True),
			nn.Linear(hid_dim, hid_dim)
		)
		self.rel_sob = nn.Sequential(
			nn.Linear(hid_dim + bg_dim, out_dim),
			nn.ReLU(True),
			nn.Linear(out_dim, out_dim)
		)

		self.so_mha = MultiHeadAttention(num_heads=8, d_model=hid_dim, dropout=0.1)
		self.sob_mha = MultiHeadAttention(num_heads=8, d_model=out_dim, dropout=0.1)

	def forward(self, subj, obj, bg):
		so = torch.cat((subj, obj), dim=1) # NxD*2
		rel_so = self.rel_so(so) # NxD*2 -> NxH
		rel_so =self.so_mha(rel_so, rel_so, rel_so).squeeze(1) # NxH

		sob1 = torch.cat((rel_so, bg), dim=1) # NxH+D
		sob2 = torch.cat((bg, rel_so), dim=1) # NxD+H
		rel_sob = self.rel_sob(sob1) + self.rel_sob(sob2) # NxH+D -> NxO
		rel_sob = self.sob_mha(rel_sob, rel_sob, rel_sob).squeeze(1) # NxO

		return rel_sob # NxO