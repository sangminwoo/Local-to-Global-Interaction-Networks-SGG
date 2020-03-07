import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLocalBlock(nn.Module):
	def __init__(self, in_channels, reduction_ratio=2, maxpool=True, use_bn=True):
		super(NonLocalBlock, self).__init__()

		self.in_channels = in_channels
		if in_channels >= reduction_ratio:
			self.inter_channels = in_channels // reduction_ratio
		else:
			self.inter_channels = 1

		self.g = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)

		if use_bn:
			self.W = nn.Sequential(
				nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1),
				nn.BatchNorm2d(self.in_channels)
			)
			nn.init.constant_(self.W[1].weight, 0)
			nn.init.constant_(self.W[1].bias, 0)
		else:
			self.W = nn.Conv2d(self.inter_channels, self.in_channels, kernel_size=1)
			nn.init.constant_(self.W.weight, 0)
			nn.init.constant_(self.W.bias, 0)


		self.theta = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)
		self.phi = nn.Conv2d(self.in_channels, self.inter_channels, kernel_size=1)

		if maxpool: # maxpool
			self.g = nn.Sequential(
				self.g,
				nn.MaxPool2d(kernel_size=(2, 2))
			)
			self.phi = nn.Sequential(
				self.phi,
				nn.MaxPool2d(kernel_size=(2, 2))
			)

	def forward(self, x): # 2x3x20x20
		batch_size = x.size(0)

		g_x = self.g(x).view(batch_size, self.inter_channels, -1) # 2x1x100
		g_x = g_x.permute(0, 2, 1) # 2x100x1

		theta_x = self.theta(x).view(batch_size, self.inter_channels, -1) # 2x1x400
		theta_x = theta_x.permute(0, 2, 1) # 2x400x1

		phi_x = self.phi(x).view(batch_size, self.inter_channels, -1) # 2x1x100

		f = torch.matmul(theta_x, phi_x) # 2x400x100
		f_div_C = F.softmax(f, dim=-1) # 2x400x100

		y = torch.matmul(f_div_C, g_x) # 2x400x1
		y = y.permute(0, 2, 1).contiguous() # 2x1x400
		y = y.view(batch_size, self.inter_channels, x.size(2), x.size(3)) # 2x1x20x20
		W_y = self.W(y) # 2x3x20x20
		z = W_y + x # self-connection; 2x3x20x20

		return z # 2x3x20x20

def nonlocal_block(in_channels, reduction_ratio=2, maxpool=True, use_bn=True):
	return NonLocalBlock(in_channels, reduction_ratio, maxpool, use_bn) 

if __name__ == '__main__':
    for (maxpool_, use_bn_) in [(True, True), (False, False), (True, False), (False, True)]:
        img = torch.zeros(2, 3, 20, 20)
        net = NonLocalBlock(3, maxpool=maxpool_, use_bn=use_bn_)
        out = net(img)
        print(out.size())