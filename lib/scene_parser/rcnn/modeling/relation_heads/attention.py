import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelGate(nn.Module):
	'''
	Channel Attention
	'''
	def __init__(self, in_channels, reduction_ratio=16):
		super(ChannelGate, self).__init__()
		self.in_channels = in_channels
		self.mlp = nn.Sequential(
				nn.Linear(in_channels, in_channels//reduction_ratio),
				nn.ReLU(inplace=True),
				nn.Linear(in_channels//reduction_ratio, in_channels)
			)

	def forward(self, x):
		'''
		Arguments
			x: Tensor[N, C, H, W]
		'''
		maxpool = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3))).squeeze() # N,C,H,W -> N,C,1,1 -> N,C
		avgpool = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3))).squeeze() # N,C,H,W -> N,C,1,1 -> N,C
		
		# self.mlp = self.mlp.to(x.device)
		channel_att = F.sigmoid(self.mlp(maxpool) + self.mlp(avgpool)).unsqueeze(2).unsqueeze(3) # N,C -> N,C,1 -> N,C,1,1
		
		return x * channel_att # N,C,H,W

class SpatialGate(nn.Module):
	'''
	Spatial Attention 1x1 conv channel-wise pooling for dimensionality reduction
	'''
	def __init__(self, in_channels, reduction_ratio=16):
		super(SpatialGate, self).__init__()
		self.spatial = nn.Sequential(
				# nn.Conv2d(in_channels, 1, kernel_size=3, padding=1),
				nn.Conv2d(in_channels, in_channels//reduction_ratio, kernel_size=1),
				nn.BatchNorm2d(in_channels//reduction_ratio, eps=1e-5, momentum=0.01, affine=True),
				nn.ReLU(inplace=True),
				nn.Conv2d(in_channels//reduction_ratio, 1, kernel_size=1),
				nn.Sigmoid()
			)

	def forward(self, x):
		'''
		Arguments
			x: Tensor[N, C, H, W]
		'''
		# self.spatial = self.spatial.to(x.device)
		spatial_att = self.spatial(x) # N,1,H,W
		return x * spatial_att # N,C,H,W

class AttentionGate(nn.Module):
	'''
	Filtering the number of channels based on channel & spatial attention gate
	'''
	def __init__(self, in_channels, reduction_ratio=16):
		super(AttentionGate, self).__init__()
		self.channel_att = ChannelGate(in_channels, reduction_ratio) # 2048 x 128
		self.spatial_att = SpatialGate(in_channels, reduction_ratio)

	def forward(self, x):
		att = self.channel_att(x)
		att = self.spatial_att(att)
		return att

class SpatialMaskGate(nn.Module):
	'''
	Spatial Mask Attention let model to attend more on objects conditioned on the object mask
	'''
	def __init__(self, in_channels, reduction_ratio=16):
		super(SpatialMaskGate, self).__init__()
		self.mask = nn.Sequential(
				nn.Conv2d(in_channels, in_channels//reduction_ratio, kernel_size=1), # NxRxHxW
				nn.ReLU(inplace=True),
				nn.Conv2d(in_channels//reduction_ratio, 1, kernel_size=1), # Nx1xHxW
				nn.Sigmoid()
			)
		self.maskloss = nn.BCELoss()

	def forward(self, x, binary_mask):
		binary_mask = binary_mask.to(x.device)
		mask_att = self.mask(x)
		mask_loss = self.maskloss(mask_att, binary_mask)
		return x * mask_att, mask_loss

class BinaryAttention(nn.Module):
	'''
	Filtering the number of channels based on channel & spatial attention gate
	'''
	def __init__(self, in_channels, reduction_ratio=16):
		super(BinaryAttention, self).__init__()
		self.channel_att = ChannelGate(in_channels, reduction_ratio) # 2048 x 128
		self.spatial_att = SpatialGate(in_channels, reduction_ratio)
		self.mask_att = SpatialMaskGate(in_channels, reduction_ratio)

	def forward(self, x, mode='MUL', mask=None):
		assert mode in ['SUM', 'MUL', 'MAX']
		assert mask is not None, 'Binary mask must given when using mask attention'
		
		channel_att = self.channel_att(x)
		spatial_att = self.spatial_att(x)
		mask_att, mask_loss = self.mask_att(x, binary_mask)
		
		if mode == 'SUM':
			return channel_att + mask_att, mask_loss
		elif mode == 'MUL':
			return channel_att * mask_att, mask_loss
		elif mode == 'MAX':
			'''TODO'''
			return

class MultiHeadAttention(nn.Module):
	'''
	Multi-Head Attention num_heads=8, reduction_ratio=4
	has same parameters(131,072) with Single-Head Attention reduction_ratio=32
	'''
	def __init__(self, in_channels, num_heads=8, reduction_ratio=8): 
		super(MultiHeadAttention, self).__init__()
		self.in_channels = in_channels
		self.num_heads = num_heads
		self.channels_per_head = in_channels // num_heads # 2048/8=256

		self.multi_head_att = nn.ModuleList(
				[ChannelGate(self.channels_per_head, reduction_ratio) for _ in range(num_heads)] # 256 x 32
			)
	
	def forward(self, x):
		#self.multi_head_att = self.multi_head_att.to(x.device)
		out = []
		for i in range(self.num_heads):
			x_i = x[:, self.channels_per_head*i:self.channels_per_head*(i+1), :, :]
			multi_head_att = self.multi_head_att[i](x_i)
			out.append(x_i)

		out = torch.cat(out, dim=1)
		return out #out.to(x.device)