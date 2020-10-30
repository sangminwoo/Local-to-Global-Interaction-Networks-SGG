import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

############### Cut (MHA) ################
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
        self.out = nn.Linear(self.d_v * num_heads, d_model) # 128-128
        self.dropout = nn.Dropout(dropout)

        self.attention = ScaledDotProductAttention(self.num_heads, self.d_k, dropout)

    def forward(self, q, k, v, att_mask=None):
        '''
         q: [batch_size x len_q x d_model]
         k: [batch_size x len_k x d_model]
         v: [batch_size x len_k x d_model]
        '''
        N = q.size(0) # 100
        q = self.W_q(q).view(N, -1, self.num_heads, self.d_k).transpose(1,2) # 100x128 -> 100x1x8x8 -> 100x8x1x16
        k = self.W_k(k).view(N, -1, self.num_heads, self.d_k).transpose(1,2) # 100x8x1x16
        v = self.W_v(v).view(N, -1, self.num_heads, self.d_v).transpose(1,2) # 100x8x1x16

        scores = self.attention(q, k, v, att_mask) # 100x8x1x16
        concat = scores.transpose(1,2).contiguous().view(N, -1, self.d_model) # # 100x1x8x16 -> 100x1x128
        output = self.out(concat) # 100x1x128
        return output # 100x1x128

############### Split (Attention) ################
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

############### Entity-Interact ################
class RelationalEmbedding(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(RelationalEmbedding, self).__init__()
        self.rel_embedding = nn.Sequential(
            nn.Linear(3 * in_dim, hid_dim),
            nn.ReLU(True),
            nn.Linear(hid_dim, out_dim)
        )

    def forward(self, subj, obj, bg):
        sob = torch.cat((subj, obj, bg), dim=1) # NxD*3
        sbo = torch.cat((subj, bg, obj), dim=1) # NxD*3
        bso = torch.cat((bg, subj, obj), dim=1) # NxD*3

        rel_sob = self.rel_embedding(sob) # NxD*3 -> NxO
        rel_sbo = self.rel_embedding(sbo) # NxD*3 -> NxO
        rel_bso = self.rel_embedding(bso) # NxD*3 -> NxO

        rel_emb = rel_sob + rel_sbo + rel_bso # NxO

        return rel_emb # NxO

############### Graph-Interact ################
class GraphConvolution(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GraphConvolution, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Linear(in_dim, out_dim, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, norm=True):
        '''
        input: KxC
        adj: KxK
        '''
        XW = self.linear(input) # (KxC) x (CxC) = KxC
        AXW = torch.mm(adj, XW) # (KxK) x (KxC) = KxC
        # normalize
        if norm:
            output = AXW / adj.sum(1).view(-1, 1)
        else:
            output = AXW
        return output
class GCN(nn.Module):
    def __init__(self, dim, attention=False):
        super(GCN, self).__init__()
        self.attention = attention

        self.gc1 = GraphConvolution(dim, dim)
        self.gc2 = GraphConvolution(dim, dim)
        self.gc3 = GraphConvolution(dim, dim)
        self.gc4 = GraphConvolution(dim, dim)

        if self.attention:
            self.mha = MultiHeadAttention(num_heads=8, d_model=dim)

    def forward(self, h0, adj, residual=False):
        h1 = F.relu(self.gc1(h0, adj))
        h2 = F.relu(self.gc2(h1, adj))
        if residual:
            h2 += h0

        h3 = F.relu(self.gc3(h2, adj))
        h4 = self.gc4(h3, adj)
        if residual:
            h4 += h2

        if self.attention:
            h4 = self.mha(h4, h4, h4).squeeze()

        return h4