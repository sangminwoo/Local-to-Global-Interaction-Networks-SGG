import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from maskrcnn_benchmark.modeling.utils import cat
from .utils_motifs import to_onehot
from .roi_relation_feature_extractors import make_roi_relation_feature_extractor
# from .roi_relation_predictors import make_roi_relation_predictor
from .modules_utils import Anchor, masking, CoordConv, RelationalEmbedding
from .modules_attention import MultiHeadAttention, CBAM, NonLocalBlock, AWAttention
from .modules_graph_interact import get_adjacency_mat, GCN, GAT, SpGAT, AGAIN

class CSINet(nn.Module):
    def __init__(self, cfg, in_channels):
        super(CSINet, self).__init__()
        self.cfg = cfg
        self.obj_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.rel_classes = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else: 
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.mask_size = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        self.dim = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        self.out_dim = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        in_dim = self.obj_classes if self.mode == 'predcls' \
        else cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM+self.obj_classes+4
        self.obj_embedding = nn.Sequential(
            nn.Linear(in_dim, self.dim),
            nn.ReLU(True),
            nn.Linear(self.dim, self.dim),
        )

        # if self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.USE_COORD_CONV:
        #     self.coord_conv = CoordConv(
        #         in_channels=self.dim, out_channels=self.dim, kernel_size=3,
        #         stride=1, padding=1, dilation=1, groups=1, bias=True, with_r=False
        #     )

        if self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.REDUCE_DIM:
            self.out_dim = self.dim//4
            self.reduce_dim = nn.Sequential(
                nn.Conv2d(in_channels=self.dim, out_channels=self.dim//2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(self.dim//2),
                nn.ReLU(True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Conv2d(in_channels=self.dim//2, out_channels=self.out_dim, kernel_size=3, stride=1, padding=1)
            )
            resolution = resolution//2+1

        self.instance_emb = nn.ModuleList([
                                nn.Sequential(
                                    nn.Linear(self.out_dim*resolution*resolution, self.dim),
                                    nn.ReLU(True),
                                    nn.Linear(self.dim, self.dim)
                                ) for _ in range(3)
                            ])

        if self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.ATT_TYPE == "self_att":
            self.att = MultiHeadAttention(num_heads=8, d_model=self.dim*3) if self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.ATT_ALL_AT_ONCE \
                else nn.ModuleList([MultiHeadAttention(num_heads=8, d_model=self.dim) for _ in range(3)])
        elif self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.ATT_TYPE == "cbam":
            self.att = CBAM(in_channels=self.out_dim*3, reduction_ratio=8, kernel_size=3) if self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.ATT_ALL_AT_ONCE \
                else nn.ModuleList([CBAM(in_channels=self.out_dim, reduction_ratio=8, kernel_size=3) for _ in range(3)])
        elif self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.ATT_TYPE == "non_local":
            self.att = NonLocalBlock(in_channels=self.out_dim, reduction_ratio=2, subsample=True, use_bn=True, dim=3) if self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.ATT_ALL_AT_ONCE \
                else nn.ModuleList([NonLocalBlock(in_channels=self.out_dim, reduction_ratio=2, use_bn=True, dim=2) for _ in range(3)])
        elif self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.ATT_TYPE == "awa":
            self.att = AWAttention(channels=self.out_dim*3, height=resolution*3, width=resolution*3, hidden_dim=self.dim*3, pool='avg', residual=True) if self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.ATT_ALL_AT_ONCE \
                else nn.ModuleList([AWAttention(channels=self.out_dim, height=resolution, width=resolution, hidden_dim=self.out_dim, pool='avg', residual=True) for _ in range(3)])
        
        self.compose = RelationalEmbedding(in_dim=self.dim, hid_dim=self.dim, out_dim=self.dim)

        self.edge2edge = cfg.MODEL.ROI_RELATION_HEAD.CSINET.EDGE2EDGE
        if self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.GRAPH_INTERACT_MODULE == "gcn":
            self.graph_interact = GCN(num_layers=cfg.MODEL.ROI_RELATION_HEAD.CSINET.NUM_GIN_LAYERS, dim=self.dim, dropout=0.1, residual=True)
        elif self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.GRAPH_INTERACT_MODULE == 'gat':
            self.graph_interact = GAT(dim=self.dim, num_heads=8, concat=True, dropout=0.1)
        elif self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.GRAPH_INTERACT_MODULE == 'again':
            self.graph_interact = AGAIN(num_layers=cfg.MODEL.ROI_RELATION_HEAD.CSINET.NUM_GIN_LAYERS, dim=self.dim, num_heads=8, concat=True, residual=False, dropout=0.1)
        elif self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.GRAPH_INTERACT_MODULE == 'self_att':
            self.graph_interact = MultiHeadAttention(num_heads=8, d_model=self.dim)

        self.obj_predictor = nn.Linear(self.dim, self.obj_classes)
        self.rel_predictor = nn.Linear(self.dim, self.rel_classes)

        self.bi_rel_anchor = Anchor()
        self.l2_loss = nn.PairwiseDistance(p=2)
        self.l1_loss = nn.L1Loss()
        self.cosine_loss = nn.CosineEmbeddingLoss()

    def _reduce_dim(self, rel_features):
        out = []
        for inst in rel_features:
            out.append(self.reduce_dim(inst))
        return out

    def _att(self, rel_features):
        if self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.ATT_ALL_AT_ONCE:
            return self.att.forward_xyz(*rel_features)

        out = []
        for inst, att in zip(rel_features, self.att):
            if self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.ATT_TYPE == "self_att":
                N,C,H,W = inst.shape
                inst = inst.contiguous().view(N,-1,C) # NxHWxC
                out.append(att(inst, inst, inst).contiguous().view(N,H,W,C)) # Nx1xHWxC -> NxHxWxC
            else:
                out.append(att(inst))
        return out

    def _flatten(self, rel_features):
        out = []
        for inst in rel_features:
            out.append(inst.contiguous().view(inst.shape[0], -1))
        return out

    def _emb(self, rel_features):
        out = []
        for inst, emb in zip(rel_features, self.instance_emb):
            out.append(emb(inst))
        return out

    def _pool(self, rel_features):
        out= []
        for inst in rel_features:
            out.append(self.avgpool(inst).squeeze())
        return out

    def _compose(self, rel_features, compose_type='half_permute'):
        assert compose_type in ['no_permute', 'half_permute', 'full_permute']
        if compose_type == 'no_permute':
            out = self.compose.forward_no_permute(*rel_features)
        elif compose_type == 'half_permute':
            out = self.compose(*rel_features)
        elif compose_type == 'full_permute':
            out = self.compose.forward_full_permute(*rel_features)
        return out

    def forward(self, roi_features, proposals, rel_features, rel_pair_idxs, logger=None):
        '''
        roi_features: tensor(20x4096) (where, 20=9+11)
        proposals: [BoxList(num_boxes=9, image_width=800, image_height=600, mode=xyxy),
                    BoxList(num_boxes=11, image_width=800, image_height=600, mode=xyxy)]
        union_features: tensor(182x4096) (where, 182=72+110)
        rel_pair_idxs: list[tensor(72x2), tensor(110x2)]
            N: number of objects
            M: number of possible pairs (=N*(N-1))
            batch size = 2
        '''
        # 1. init object feats (vis+sem+spa)
        if self.mode == 'predcls':
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            obj_dists = to_onehot(obj_labels, self.obj_classes)
            obj_features = self.obj_embedding(obj_dists)
        else:
            obj_dists = torch.cat([proposal.get_field("predict_logits") for proposal in proposals], 0) # 20x151 "predict_logits"
            bboxes = torch.cat([proposal.bbox for proposal in proposals], 0) # 20x4
            obj_features = torch.cat((roi_features, obj_dists, bboxes), dim=1) # Nx(4096+151+4)
            obj_features = self.obj_embedding(obj_features)

        # not required. use when OOM occurs.
        if self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.REDUCE_DIM:
            rel_features = self._reduce_dim(rel_features)

        # 2. (local) attention within sbj, obj, bg 
        if self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.USE_ATT:
            rel_features = self._att(rel_features)

        if self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.FLATTEN:
            rel_features = self._flatten(rel_features)
            rel_features = self._emb(rel_features)
        else:
            rel_features = self._pool(rel_features)

        # 3. compose
        if self.cfg.MODEL.ROI_RELATION_HEAD.POOL_SBJ_OBJ or self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.USE_MASKING:
            rel_features = self._compose(rel_features, compose_type=self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.COMPOSE_TYPE)
        else:
            assert len(rel_features) == 1, "using union feature only"
            rel_features = rel_features[0]

        # 4. (global) context aggregation via graph-interaction networks
        if self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.USE_GIN:
            if self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.GRAPH_INTERACT_MODULE == "self_att":
                obj_features, rel_features = self.graph_interact.forward_graph(proposals, rel_pair_idxs, obj_features, rel_features)
            else:
                adj = get_adjacency_mat(proposals, rel_pair_idxs, edge2edge=self.edge2edge)

                n = obj_features.size(0)
                feats = torch.cat((obj_features, rel_features), dim=0) # (n+m)xC

                graph_out = self.graph_interact(feats, adj)
                obj_features = graph_out[:n] # nxC
                rel_features = graph_out[n:] # mxC
            
        # 5. predict obj & rel dist
        if self.mode != 'predcls':
            obj_dists = self.obj_predictor(obj_features) # nx151

        rel_dists = self.rel_predictor(rel_features) # mx51

        
        
        # atrract & repel
        # if self.training:
        #     if self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.USE_BI_ATT_REP_LOSS:
        #         for rel_pair_idx in rel_pair_idxs:
        #             num_rels = rel_pair_idx.shape[0]

        #             # bi_rel_idxs
        #             all_rel_h2t = rel_pair_idx.repeat(num_rels,1)
        #             rel_pair_idx_rev = torch.cat([rel_pair_idx[:,1].view(-1,1), rel_pair_idx[:,0].view(-1,1)], dim=1)
        #             all_rel_t2h = rel_pair_idx_rev.repeat_interleave(num_rels,0)
        #             rel_bi_pairs = (all_rel_h2t == all_rel_t2h).all(1)
        #             rel_bi_pairs = rel_bi_pairs.view(num_rels, num_rels)
        #             bi_rel_idxs = torch.nonzero(torch.triu(rel_bi_pairs)) # num_bi_rels x 2 (e.g. [0,1])
        #             num_bi_rels = len(bi_rel_idxs)

        #             if num_bi_rels > 0:
        #                 bi_rels = rel_pair_idx[bi_rel_idxs[:, 0]] # num_bi_rels x 2 (e.g., [126, 28])
        #                 bi_rel_dists = rel_dists[bi_rels] # num_bi_rels x 2 x 51

        #                 bi_rel_preds = bi_rel_dists.max(-1)[1] # num_bi_rels x 2

        #                 for (h2t_idx, t2h_idx), (h2t_label, t2h_label) in zip(bi_rels, bi_rel_preds):
        #                     if h2t_label == t2h_label:
        #                         self.bi_rel_anchor.update(key=h2t_label, pos=rel_dists[h2t_idx], neg=-rel_dists[t2h_idx])
        #                     else:
        #                         self.bi_rel_anchor.update(key=h2t_label, pos=rel_dists[h2t_idx], neg=rel_dists[t2h_idx])
        #                         self.bi_rel_anchor.update(key=t2h_label, pos=rel_dists[t2h_idx], neg=rel_dists[h2t_idx])

        #                     # h2t_loss = torch.sum(self.l2_loss(self.bi_rel_anchor[h2t_label].to(rel_dists.device).unsqueeze(0), rel_dists[h2t_idx].unsqueeze(0), p=2))
        #                     # t2h_loss = torch.sum(self.l2_loss(self.bi_rel_anchor[t2h_label].to(rel_dists.device).unsqueeze(0), rel_dists[t2h_idx].unsqueeze(0), p=2))

        #                     # h2t_loss = torch.sum(self.l1_loss(self.bi_rel_anchor[h2t_label].to(rel_dists.device).unsqueeze(0), rel_dists[h2t_idx].unsqueeze(0)))
        #                     # t2h_loss = torch.sum(self.l1_loss(self.bi_rel_anchor[t2h_label].to(rel_dists.device).unsqueeze(0), rel_dists[t2h_idx].unsqueeze(0)))

        #                     h2t_loss = torch.sum(self.cosine_loss(self.bi_rel_anchor[h2t_label].to(rel_dists.device).unsqueeze(0), rel_dists[h2t_idx].unsqueeze(0), torch.tensor(1, device=rel_dists.device)))
        #                     t2h_loss = torch.sum(self.cosine_loss(self.bi_rel_anchor[t2h_label].to(rel_dists.device).unsqueeze(0), rel_dists[t2h_idx].unsqueeze(0), torch.tensor(1, device=rel_dists.device)))

        #                     loss += (h2t_loss + t2h_loss)
        
        #      # repulsive loss 
        #     if self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.USE_REPULSIVE_LOSS:
        #         for rel_pair_idx in rel_pair_idxs:
        #             num_rels = rel_pair_idx.shape[0]

        #             # bi_rel_idxs
        #             all_rel_h2t = rel_pair_idx.repeat(num_rels,1)
        #             rel_pair_idx_rev = torch.cat([rel_pair_idx[:,1].view(-1,1), rel_pair_idx[:,0].view(-1,1)], dim=1)
        #             all_rel_t2h = rel_pair_idx_rev.repeat_interleave(num_rels,0)
        #             rel_bi_pairs = (all_rel_h2t == all_rel_t2h).all(1)
        #             rel_bi_pairs = rel_bi_pairs.view(num_rels, num_rels)
        #             bi_rel_idxs = torch.nonzero(torch.triu(rel_bi_pairs))
        #             num_bi_rels = len(bi_rel_idxs)

        #             if num_bi_rels > 0:
        #                 bi_rels = rel_pair_idx[bi_rel_idxs[:, 0]]
        #                 bi_rel_dists = rel_dists[bi_rels] # num_bi_rels x 2 x dim

        #                 pdist = F.pairwise_distance(bi_rel_dists[:,0,:], bi_rel_dists[:,1,:], p=2)
        #                 margin = torch.tensor(self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.MARGIN, device=pdist.device)
        #                 min_loss = torch.tensor(0., device=pdist.device)
        #                 loss += torch.max(min_loss, margin-pdist)[0]

        return rel_features, obj_dists, rel_dists #, loss
