import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLocalBlockND(nn.Module):
	def __init__(self, in_channels, reduction_ratio=2, maxpool=True, use_bn=True, dim=2):
		super(NonLocalBlock, self).__init__()
		assert dim in [1, 2, 3], 'dim within 1 to 3'

		self.in_channels = in_channels
		if in_channels >= reduction_ratio:
			self.inter_channels = in_channels // reduction_ratio
		else:
			self.inter_channels = 1

		if dim == 1:
			conv = nn.Conv1d
			maxpool = nn.MaxPool1d(kernel_size=(2))
			batchnorm = nn.BatchNorm1d(self.in_channels)
		elif dim == 2:
			conv = nn.Conv2d
			maxpool = nn.MaxPool2d(kernel_size=(2, 2))
			batchnorm = nn.BatchNorm2d(self.in_channels)

		self.g = conv(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
		self.theta = conv(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)
		self.phi = conv(self.in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0)

		if maxpool: # maxpool
			self.g = nn.Sequential(self.g, maxpool)
			self.phi = nn.Sequential(self.phi, maxpool)

		if use_bn:
			self.W = nn.Sequential(
				conv(self.inter_channels, self.in_channels, kernel_size=1, stride=1, padding=0),
				batchnorm
			)
			nn.init.constant_(self.W[1].weight, 0)
			nn.init.constant_(self.W[1].bias, 0)
		else:
			self.W = conv
			nn.init.constant_(self.W.weight, 0)
			nn.init.constant_(self.W.bias, 0)

	def forward(self, x): # 2x1024x14x14
		batch_size = x.size(0)

		g_x = self.g(x).view(batch_size, self.inter_channels, -1) # 2x1024x14x14 -> 2x512x14x14 -> 2x512x196
		g_x = g_x.permute(0, 2, 1) # 2x196x512

		theta_x = self.theta(x).view(batch_size, self.inter_channels, -1) # 2x1024x14x14 -> 2x512x14x14 -> 2x512x196
		theta_x = theta_x.permute(0, 2, 1) # 2x196x512

		phi_x = self.phi(x).view(batch_size, self.inter_channels, -1) # 2x1024x14x14 -> 2x512x14x14 ->2x512x196

		f = torch.matmul(theta_x, phi_x) # 2x196x196
		f_div_C = F.softmax(f, dim=-1) # 2x196x196

		y = torch.matmul(f_div_C, g_x) # 2x196x512
		y = y.permute(0, 2, 1).contiguous() # 2x512x196
		y = y.view(batch_size, self.inter_channels, *x.size()[2:]) # 2x512x14x14
		W_y = self.W(y) # 2x1024x14x14
		z = W_y + x # self-connection; 2x1024x14x14

		return z # 2x1024x14x14

def nonlocal_block1d(in_channels, reduction_ratio=2, maxpool=True, use_bn=True):
	return NonLocalBlock(in_channels, reduction_ratio, maxpool, use_bn, dim=1)

def nonlocal_block2d(in_channels, reduction_ratio=2, maxpool=True, use_bn=True):
	return NonLocalBlock(in_channels, reduction_ratio, maxpool, use_bn, dim=2) 

if __name__ == '__main__':
    for (maxpool_, use_bn_) in [(True, True), (False, False), (True, False), (False, True)]:
        img = torch.zeros(2, 3, 20, 20)
        net = NonLocalBlock(3, maxpool=maxpool_, use_bn=use_bn_)
        out = net(img)
        print(out.size())