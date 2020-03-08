import torch
import torch.nn as nn
import torch.nn.functional as F

class BottleNeck(nn.Module):
	expansion = 4

	def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class FPN(nn.Module):
	def __init__(self, block, num_blocks):
		super(FPN, self).__init__()
		self.in_planes = 64

		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn1 = nn.BatchNorm2d(64)

		# Bottom-Up layers
		self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 128m)

	def _make_layer(self, block, planes, num_blocks, stride):



class PyramidGCN(nn.Module):
	'''
	Feature Pyramid GCN
	'''
	def __init__(self):
		super(PyramidGCN, self).__init__()


def FPN101():
	return FPN(BottleNeck, [2,2,2,2])

if __name__=='__main__':
	model = FPN101()
	fms = model(Variable(torch.randn(1,3,600,900)))
	for fm in fms:
		print(fm.size())