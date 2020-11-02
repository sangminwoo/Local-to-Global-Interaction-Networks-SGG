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

        return F.relu(output)

class GCN(nn.Module):
    def __init__(self, num_layers, dim, dropout=0.5, residual=True):
        super(GCN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual

        self.gcn_layers = [GraphConvolution(dim, dim) for _ in range(num_layers)]

        for i, gcn_layer in enumerate(self.gcn_layers):
            self.add_module('gcn_layer_{}'.format(i), gcn_layer)

    def forward(self, x, adj):
        residual = x
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, adj)
            if i != self.num_layers-1:
                x = F.dropout(x, self.dropout, training=self.training)
            if self.residual:
                x += residual
                residual = x

        return x

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, alpha=0.2, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_dim), Wh.shape: (N, out_dim)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0] # number of nodes
        
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        
        return all_combinations_matrix.view(N, N, 2 * self.out_dim)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim) + ')'

class GAT(nn.Module):
    def __init__(self, num_layers, dim, dropout=0.6, residual=True, num_heads=8):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual

        self.gat_layer = [GraphAttentionLayer(dim, dim//num_heads, dropout=dropout) for _ in range(num_heads)]

        for i, att_head in enumerate(self.gat_layer):
            self.add_module('gat_head_{}'.format(i), att_head)

    def forward(self, x, adj):
        x = torch.cat([att_head(x, adj) for att_head in self.gat_layer], dim=1)
           
        return x

# class GAT(nn.Module):
#     def __init__(self, num_layers, dim, dropout=0.6, residual=True, alpha=0.2, num_heads=8):
#         super(GAT, self).__init__()
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.residual = residual

#         self.gat_layers = [[
#             GraphAttentionLayer(dim, dim//num_heads, dropout=dropout, alpha=alpha, concat=True) for _ in range(num_heads)]
#             for _ in range(num_layers)]

#         for i, gat_layer in enumerate(self.gat_layers):
#             for j, att_head in enumerate(gat_layer):
#                 self.add_module('gat_layer_{}_head_{}'.format(i, j), att_head)

#     def forward(self, x, adj):
#         residual = x
#         for i, gat_layer in enumerate(self.gat_layers):
#             x = torch.cat([att_head(x, adj) for att_head in gat_layer], dim=1)
#             if i != self.num_layers-1:
#                 x = F.dropout(x, self.dropout, training=self.training)
#             if self.residual:
#                 x += residual
#                 residual = x
#         return x