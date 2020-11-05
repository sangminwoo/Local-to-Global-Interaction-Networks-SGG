import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

############### Split (Self-Attention) ################
class ScaledDotProductAttention(nn.Module):
    def __init__(self, num_heads, d_k, dropout):
        super(ScaledDotProductAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_k
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, att_mask=None):
        scores = torch.matmul(q, k.transpose(-1, -2)) /  np.sqrt(self.d_k) # (100x8x1x16)x(100x8x16x1) = 100x8x1x1
        if att_mask is not None:
            att_mask = att_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1) # 100x8x1x1
            scores.masked_fill_(att_mask == 0, -1e9)

        scores = self.softmax(scores) # 100x8x1x1

        if self.dropout and self.training:
            scores = self.dropout(scores) # 100x8x1x1

        output = torch.matmul(scores, v) # (100x8x1x1)x(100x8x1x16) = 100x8x1x16
        return output # 100x8x1x16

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads # 8
        self.d_model = d_model # 128
        self.d_k = d_model // num_heads # 16
        self.d_v = d_model // num_heads # 16

        self.W_q = nn.Linear(d_model, self.d_k * num_heads) # 128-128
        self.W_k = nn.Linear(d_model, self.d_k * num_heads) # 128-128
        self.W_v = nn.Linear(d_model, self.d_v * num_heads) # 128-128
        self.mlp = nn.Linear(self.d_v * num_heads, d_model) # 128-128
        self.dropout = nn.Dropout(dropout)

        self.attention = ScaledDotProductAttention(self.num_heads, self.d_k, dropout)

    def forward(self, q, k, v, att_mask=None):
        '''
         q: [batch_size x len_q x d_model]
         k: [batch_size x len_k x d_model]
         v: [batch_size x len_v x d_model]
        '''
        N = q.size(0) # 100
        q = self.W_q(q).view(N, -1, self.num_heads, self.d_k).transpose(1,2) # 100x128 -> 100x1x8x8 -> 100x8x1x16
        k = self.W_k(k).view(N, -1, self.num_heads, self.d_k).transpose(1,2) # 100x8x1x16
        v = self.W_v(v).view(N, -1, self.num_heads, self.d_v).transpose(1,2) # 100x8x1x16

        scores = self.attention(q, k, v, att_mask) # 100x8x1x16
        concat = scores.transpose(1,2).contiguous().view(N, -1, self.d_model) # # 100x1x8x16 -> 100x1x128
        output = self.mlp(concat) # 100x1x128
        return output # 100x1x128

############### Split (CBAM) ################
class ChannelGate(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.in_channels = in_channels
        self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels//reduction_ratio),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels//reduction_ratio, in_channels)
            )

    def forward(self, x):
        maxpool = F.max_pool2d(x, kernel_size=(x.size(2), x.size(3))).squeeze() # N,C,H,W -> N,C,1,1 -> N,C
        avgpool = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3))).squeeze() # N,C,H,W -> N,C,1,1 -> N,C
        
        # self.mlp = self.mlp.to(x.device)
        channel_att = torch.sigmoid(self.mlp(maxpool) + self.mlp(avgpool)).unsqueeze(2).unsqueeze(3) # N,C -> N,C,1 -> N,C,1,1
        
        return x * channel_att # N,C,H,W


class SpatialGate(nn.Module):
    def __init__(self, kernel_size):
        super(SpatialGate, self).__init__()
        self.spatial = nn.Sequential(
                nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2),
                nn.Sigmoid()
            )

    def forward(self, x):
        x_max = torch.max(x, dim=1)[0].unsqueeze(1)
        x_avg = torch.mean(x, dim=1).unsqueeze(1)
        x_cat = torch.cat((x_max, x_avg), dim=1)
        
        spatial_att = self.spatial(x_cat) # N,1,H,W
        return x * spatial_att # N,C,H,W

class AttentionGate(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=3):
        super(AttentionGate, self).__init__()
        self.channel_att = ChannelGate(in_channels, reduction_ratio) # 2048 x 128
        self.spatial_att = SpatialGate(kernel_size)

    def forward(self, x):
        att = self.channel_att(x)
        att = self.spatial_att(att)
        return att

############### Split (Axis-wise Attention) ################
class AWAttention(nn.Module):
    def __init__(self, channels, height, width, dim):
        super(AWAttention, self).__init__()
        self.channels = channels
        self.f = nn.Sequential(
                nn.Linear(channels, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, channels)
            )
        self.g = nn.Sequential(
                nn.Linear(height, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, height)
            )
        self.h = nn.Sequential(
                nn.Linear(width, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, width)
            )
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def _pool_hxw(self, x, pool='avg'):
        if pool == 'avg':
            return self.avgpool(x).squeeze()
        elif pool == 'max':
            return self.maxpool(x).squeeze()

    def _pool_wxc(self, x, pool='avg'):
        if pool == 'avg':
            x = torch.mean(x, 1) # NxCxHxW -> NxHxW
            x = torch.mean(x, 2) # NxHxW -> NxH
        elif pool == 'max':
            x = torch.max(x, 1) # NxCxHxW -> NxHxW
            x = torch.max(x, 2) # NxHxW -> NxH
        return x

    def _pool_hxc(self, x, pool='avg'):
        if pool == 'avg':
            x = torch.mean(x, 1) # NxCxHxW -> NxHxW
            x = torch.mean(x, 1) # NxHxW -> NxW
        elif pool == 'max':
            x = torch.max(x, 1) # NxCxHxW -> NxHxW
            x = torch.max(x, 1) # NxHxW -> NxW
        return x

    def forward(self, x):
        f_out = torch.sigmoid(self.f(self._pool_hxw(x))).contiguous().view(x.shape[0],-1,1,1) # channel-axis
        g_out = torch.sigmoid(self.g(self._pool_wxc(x))).contiguous().view(x.shape[0],1,-1,1) # height-axis
        h_out = torch.sigmoid(self.h(self._pool_hxc(x))).contiguous().view(x.shape[0],1,1,-1) # width-axis
        return (f_out * g_out * h_out) * x

############### Split (Non-local)  ################
class NonLocalBlock(nn.Module):
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

    def forward(self, x): # NxCxHxW
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1) # NxCxHxW -> NxC'xHxW -> NxC'xHW
        g_x = g_x.permute(0, 2, 1) # NxHWxC'

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1) # NxCxHxW -> NxC'xHxW -> NxC'xHW
        theta_x = theta_x.permute(0, 2, 1) # NxHWxC'

        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1) # NxCxHxW -> NxC'xHxW ->NxC'xHW

        f = torch.matmul(theta_x, phi_x) # NxHWxHW
        f_div_C = F.softmax(f, dim=-1) # NxHWxHW

        y = torch.matmul(f_div_C, g_x) # NxHWxC'
        y = y.permute(0, 2, 1).contiguous() # NxC'xHW
        y = y.view(batch_size, self.inter_channels, *x.size()[2:]) # NxC'xHxW
        W_y = self.W(y) # NxCxHxW
        z = W_y + x # self-connection; NxCxHxW

        return z # NxCxHxW