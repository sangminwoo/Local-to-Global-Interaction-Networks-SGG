import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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