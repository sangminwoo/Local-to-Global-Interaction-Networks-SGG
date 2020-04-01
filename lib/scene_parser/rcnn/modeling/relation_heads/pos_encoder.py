import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionEncoder(nn.Module):
	def __init__(self, kernel_size=3):
		super(PositionEncoder, self).__init__()
		self.pos_encoder = nn.Sequential(
			nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2),
			nn.Sigmoid()
		)

	def forward(self, x):
		x_max = torch.max(x, dim=1)[0].unsqueeze(1)
		x_avg = torch.mean(x, dim=1).unsqueeze(1)
		x_cat = torch.cat((x_max, x_avg), dim=1) # Nx2x14x14

		pos_mask = self.pos_encoder(x_cat) # Nx1x14x14
		return pos_mask # Nx1x14x14

class PositionEncoderV2(nn.Module):
	def __init__(self, in_dim, hid_dim, out_dim):
		super(PositionEncoderV2, self).__init__()
		self.pos_encoder = nn.Sequential(
			nn.Linear(in_dim, out_dim),
			nn.ReLU(True),
			nn.Linear(out_dim, out_dim),
			nn.Sigmoid()
		)

	def forward(self, x):
		out = self.pos_encoder(x)
		return out

class UnionBoxEncoder(nn.Module):
	def __init__(self, in_dim, hid_dim, num_classes=151):
		super(UnionBoxEncoder, self).__init__()
		self.multi_class = nn.Sequential(
			nn.Linear(in_dim, hid_dim),
			nn.ReLU(True),
			nn.Linear(hid_dim, num_classes),
			nn.Sigmoid()
		)
	def forward(self, x):
		x = self.multi_class(x) # Nx1024 -> Nx151
		return x