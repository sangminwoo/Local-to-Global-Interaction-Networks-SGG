import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_adjacency_mat(proposals, rel_pair_idxs, edge2edge=True):
    device = rel_pair_idxs[0].device
    n2n_offset = 0
    e2e_offset = 0
    bboxes = torch.cat([proposal.bbox for proposal in proposals], 0)
    num_nodes = bboxes.shape[0] # 20
    pair_idxs = torch.cat(rel_pair_idxs, dim=0)
    num_edges = pair_idxs.shape[0]
    node_to_node = torch.zeros(num_nodes, num_nodes, device=device)
    edge_to_edge = torch.zeros(num_edges, num_edges, device=device)

    for proposal, pair_idxs_per_image in zip(proposals, rel_pair_idxs):
        node_to_node[n2n_offset+pair_idxs_per_image[:, 0].view(-1, 1),
                     n2n_offset+pair_idxs_per_image[:, 1].view(-1, 1)] = 1 # 1024x1024
        node_to_node[n2n_offset+pair_idxs_per_image[:, 1].view(-1, 1),
                     n2n_offset+pair_idxs_per_image[:, 0].view(-1, 1)] = 1
        n2n_offset += len(proposal.bbox)

        if edge2edge:
            # consider edge-to-edge propagation only when they are in the opposite direction
            e2e_idxs_per_image = torch.tensor([[[i,j], [j,i]] for i in range(len(pair_idxs_per_image)) for j in range(i+1, len(pair_idxs_per_image)) \
                if pair_idxs_per_image[i,0] == pair_idxs_per_image[j,1] and pair_idxs_per_image[i,1] == pair_idxs_per_image[j,0]], device=device)
            e2e_idxs_per_image = e2e_idxs_per_image.contiguous().view(-1, 2)
            if len(e2e_idxs_per_image) > 0:
                edge_to_edge[e2e_offset+e2e_idxs_per_image[:, 0].view(-1, 1),
                             e2e_offset+e2e_idxs_per_image[:, 1].view(-1, 1)] = 1
            e2e_offset += len(pair_idxs_per_image)

    node_to_edge = torch.zeros(num_nodes, num_edges, device=device)
    node_to_edge.scatter_(0, (pair_idxs[:, 0].view(1, -1)), 1)
    node_to_edge.scatter_(0, (pair_idxs[:, 1].view(1, -1)), 1)
    edge_to_node = node_to_edge.t()

    n2n_n2e = torch.cat((node_to_node, node_to_edge), dim=1)
    e2n_e2e = torch.cat((edge_to_node, edge_to_edge), dim=1)
    adj = torch.cat((n2n_n2e, e2n_e2e), dim=0)
    adj = adj + torch.eye(len(adj), device=device) # regarding self-connection
    return adj

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

        self.gcn_layers = nn.ModuleList([GraphConvolution(dim, dim) for _ in range(num_layers)])

        # for i, gcn_layer in enumerate(self.gcn_layers):
        #     self.add_module('gcn_layer_{}'.format(i), gcn_layer)

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
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.dropout = dropout
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
        self.gat_layer = nn.ModuleList([GraphAttentionLayer(dim, out_dim, dropout=dropout, alpha=alpha, concat=concat) for _ in range(num_heads)])

        # for i, att_head in enumerate(self.gat_layer):
        #     self.add_module('gat_head_{}'.format(i), att_head)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        if self.concat:
                out = torch.cat([att_head(x, adj) for att_head in self.gat_layer], dim=1)
        else:
            summ = 0
            for att_head in self.gat_layer:
                summ += att_head(x, adj)
            out = summ / self.num_heads
     
        return out

############### Graph-Interact (Sparse-GAT) ################
class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b

class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)

class SpGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, alpha=0.2, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(in_dim, out_dim)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)
                
        self.a = nn.Parameter(torch.zeros(size=(1, 2*out_dim)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, h, adj):
        device = 'cuda' if h.is_cuda else 'cpu'
        N = h.shape[0]
        adj = adj.nonzero().t() # adj: 2 x E (where E is the numer of edges)

        Wh = torch.mm(h, self.W) # Wh: N x out (where N is the number of nodes)
        assert not torch.isnan(Wh).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((Wh[adj[0, :], :], Wh[adj[1, :], :]), dim=1).t() # edge_h: 2*out x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze())) # edge_e: E
        assert not torch.isnan(edge_e).any()

        e_rowsum = self.special_spmm(adj, edge_e, torch.Size([N, N]), torch.ones(size=(N,1), device=device)) # e_rowsum: N x 1

        edge_e = self.dropout(edge_e) # edge_e: E

        h_prime = self.special_spmm(adj, edge_e, torch.Size([N, N]), Wh) # h_prime: N x out
        assert not torch.isnan(h_prime).any()
        
        h_prime = h_prime.div(e_rowsum) # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_dim) + ' -> ' + str(self.out_dim) + ')'

class SpGAT(nn.Module):
    def __init__(self, dim, num_heads=8, concat=True, dropout=0.6, alpha=0.2):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout
        self.concat = concat

        out_dim = dim//num_heads if self.concat else dim
        self.attentions = [SpGraphAttentionLayer(dim, 
                                                 out_dim, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(num_heads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
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
        self.dropout = dropout

        out_dim = dim//num_heads if self.concat else dim
        self.again_layers = nn.ModuleList([
                nn.ModuleList([AttentionalGraphInteractLayer(dim, out_dim, dropout=dropout, concat=concat) \
            for _ in range(num_heads)])
        for _ in range(num_layers)])

        # for i, again_layer in enumerate(self.again_layers):
        #     for j, att_head in enumerate(again_layer):
        #         self.add_module('again_layer_{}_head_{}'.format(i, j), att_head)  

    def forward(self, x, adj):
        residual = x
        x = F.dropout(x, self.dropout, training=self.training)
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