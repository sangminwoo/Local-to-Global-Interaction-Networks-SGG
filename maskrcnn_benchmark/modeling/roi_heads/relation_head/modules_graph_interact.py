import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

############### Graph-Interact (GCN) ################
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

############### Graph-Interact (GAT) ################
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
    def __init__(self, dim, num_heads=8, concat=True, dropout=0.6, alpha=0.2):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.concat = concat

        out_dim = dim//num_heads if self.concat else dim
        self.gat_layer = [GraphAttentionLayer(dim, out_dim, dropout=dropout, alpha=alpha, concat=concat) for _ in range(num_heads)]

        for i, att_head in enumerate(self.gat_layer):
            self.add_module('gat_head_{}'.format(i), att_head)

    def forward(self, x, adj):
        if self.concat:
                out = torch.cat([att_head(x, adj) for att_head in self.gat_layer], dim=1)
        else:
            summ = 0
            for att_head in self.gat_layer:
                summ += att_head(x, adj)
            out = summ / self.num_heads
     
        return out

############### Graph-Interact (AGAIN) ################
class AttentionalGraphInteractLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, concat=False):
        super(AttentionalGraphInteractLayer, self).__init__()
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.concat = concat

        self.f = nn.Linear(in_dim, in_dim, bias=True)
        self.g = nn.Linear(in_dim, in_dim, bias=True)
        self.W = nn.Linear(in_dim, out_dim, bias=True)

    def forward(self, x, adj):
        f = self.f(x)
        g = self.g(x)

        e = F.relu(torch.mm(f, g.t()))
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)

        x_W = self.W(x)
        a_x_W = torch.mm(attention, x_W)

        if self.concat:
            return F.relu(a_x_W)
        else:
            return a_x_W

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim) + ')'

class AGAIN(nn.Module):
    def __init__(self, num_layers, dim, num_heads=8, concat=True, residual=True, dropout=0.1):
        super(AGAIN, self).__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.concat = concat
        self.residual = residual

        out_dim = dim//num_heads if self.concat else dim
        self.again_layers = [
                [AttentionalGraphInteractLayer(dim, out_dim, dropout=dropout, concat=concat) \
            for _ in range(num_heads)]
        for _ in range(num_layers)]

        for i, again_layer in enumerate(self.again_layers):
            for j, att_head in enumerate(again_layer):
                self.add_module('again_layer_{}_head_{}'.format(i, j), att_head)

    def forward(self, x, adj):
        residual = x
        for again_layer in self.again_layers:

            if self.concat:
                x = torch.cat([att_head(x, adj) for att_head in again_layer], dim=1)
            else:
                summ = 0
                for att_head in again_layer:
                    summ += att_head(x, adj)
                x = summ / self.num_heads
            
            if self.residual:
                x += residual
                residual = x
        return x