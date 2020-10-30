import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from maskrcnn_benchmark.modeling.utils import cat
from .utils_motifs import to_onehot
from .roi_relation_feature_extractors import make_roi_relation_feature_extractor
# from .roi_relation_predictors import make_roi_relation_predictor
from .utils_csinet import AttentionGate, RelationalEmbedding, GCN

class CSINet(nn.Module):
    def __init__(self, cfg, in_channels):
        super(CSINet, self).__init__()
        self.cfg = cfg
        self.obj_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.rel_classes = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.mask_size = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        self.dim = 256
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pred_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)

        self.obj_embedding = nn.Sequential(
            nn.Linear(4096+151+4, self.dim),
            nn.ReLU(True),
            nn.Linear(self.dim, self.dim),
        )

        self.sbj_att = AttentionGate(in_channels=self.dim, reduction_ratio=4)
        self.obj_att = AttentionGate(in_channels=self.dim, reduction_ratio=4)
        self.bg_att = AttentionGate(in_channels=self.dim, reduction_ratio=4)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.compose = RelationalEmbedding(in_dim=self.dim, hid_dim=self.dim, out_dim=self.dim)

        self.gcn = GCN(self.dim, attention=False)

        self.obj_predictor = nn.Linear(self.dim, self.obj_classes)
        self.rel_predictor = nn.Linear(self.dim, self.rel_classes)

    def _masking(self, union_features, proposals, rel_pair_idxs, mask_size=14): # proposal_pairs, proposals_union):
        bboxes = torch.cat([proposal.bbox for proposal in proposals], 0) # 20x4
        sbj_boxes = bboxes[torch.cat(rel_pair_idxs, dim=0)[:, 0]]
        obj_boxes = bboxes[torch.cat(rel_pair_idxs, dim=0)[:, 1]]
        pair_boxes = torch.cat([sbj_boxes, obj_boxes], dim=1)
        union_boxes = torch.cat((
            torch.min(sbj_boxes[:,:2], obj_boxes[:,:2]),
            torch.max(sbj_boxes[:,2:], obj_boxes[:,2:])
            ),dim=1)

        x = pair_boxes[:,[0,2,4,6]]-union_boxes[:,0].view(-1, 1).repeat(1, 4) # Nx4
        y = pair_boxes[:,[1,3,5,7]]-union_boxes[:,1].view(-1, 1).repeat(1, 4) # Nx4

        num_pairs = pair_boxes.shape[0]
        x_rescale_factor = (mask_size / torch.max(x[:,1], x[:,3])).view(-1, 1) # Nx1
        y_rescale_factor = (mask_size / torch.max(y[:,1], y[:,3])).view(-1, 1) # Nx1

        x_pooled = (x * x_rescale_factor).round().long() # Nx4
        y_pooled = (y * y_rescale_factor).round().long() # Nx4
        xy_pooled = torch.stack((x_pooled, y_pooled), dim=2) # Nx4x2

        subj_xy = xy_pooled[:, :2, :].reshape(num_pairs, 4) # Nx4
        obj_xy = xy_pooled[:, 2:, :].reshape(num_pairs, 4) # Nx4

        sbj_mask = torch.zeros(num_pairs, mask_size, mask_size).cuda() # Nx14x14
        obj_mask = torch.zeros(num_pairs, mask_size, mask_size).cuda() # Nx14x14

        for i in range(num_pairs):
            sbj_mask[i, subj_xy[i,0]:subj_xy[i,2], subj_xy[i,1]:subj_xy[i,3]] = 1
            obj_mask[i, obj_xy[i,0]:obj_xy[i,2], obj_xy[i,1]:obj_xy[i,3]] = 1

        sbj_mask = sbj_mask.view(num_pairs, 1, mask_size, mask_size) # Nx1x14x14
        obj_mask = obj_mask.view(num_pairs, 1, mask_size, mask_size) # Nx1x14x14

        bg_mask = torch.ones(num_pairs, 1, mask_size, mask_size).cuda() # Nx1x14x14
        bg_mask = bg_mask - sbj_mask - obj_mask # Nx1x14x14
        bg_mask[bg_mask < 0] = 0 # Nx1x14x14

        return union_features*sbj_mask, union_features*obj_mask, union_features*bg_mask # Nx1x14x14, Nx1x14x14, Nx1x14x14

    def _adjacency(self, proposals, rel_pair_idxs):
        device = rel_pair_idxs[0].device
        offset = 0
        bboxes = torch.cat([proposal.bbox for proposal in proposals], 0) # 20x4
        num_bboxes = bboxes.shape[0] # 20
        obj_to_obj = torch.zeros(num_bboxes, num_bboxes) # 20x20

        for proposal, pair_idx in zip(proposals, rel_pair_idxs):
            obj_to_obj[offset+pair_idx[:, 0].view(-1, 1), offset+pair_idx[:, 1].view(-1, 1)] = 1 # 1024x1024
            offset += len(proposal.bbox)

        pair_idxs = torch.cat(rel_pair_idxs, dim=0) # 182x2
        obj_to_rel = torch.zeros(num_bboxes, pair_idxs.shape[0]).to(device) # 20x182
        obj_to_rel.scatter_(0, (pair_idxs[:, 0].view(1, -1)), 1)
        obj_to_rel.scatter_(0, (pair_idxs[:, 1].view(1, -1)), 1)

        obj_to_obj = obj_to_obj.to(device)
        top = torch.cat((obj_to_obj, obj_to_rel), dim=1)
        zero = torch.zeros(pair_idxs.shape[0], pair_idxs.shape[0]).to(device)
        bot = torch.cat((obj_to_rel.t(), zero), dim=1)
        adj = torch.cat((top, bot), dim=0)
        adj = adj + torch.eye(len(adj)).to(device) # regarding self-connection
        return adj

    def forward(self, roi_features, proposals, union_features, rel_pair_idxs, logger=None):
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
            obj_logits = to_onehot(obj_labels, self.obj_classes)
            # obj_logits = torch.eye(self.obj_classes, device=obj_labels.device)[obj_labels]
        else:
            obj_logits = torch.cat([proposal.get_field("predict_logits") for proposal in proposals], 0) # 20x151
        bboxes = torch.cat([proposal.bbox for proposal in proposals], 0) # 20x4
        obj_feats = torch.cat((roi_features, obj_logits, bboxes), dim=1) # Nx(4096+151+4)
        obj_feats = self.obj_embedding(obj_feats)
        
        # 2. split by mask and attend
        sbj, obj, bg = self._masking(union_features, proposals, rel_pair_idxs, mask_size=self.mask_size)
        sbj = self.sbj_att(sbj)
        obj = self.obj_att(obj)
        bg = self.bg_att(bg)
        
        sbj = self.avgpool(sbj).squeeze()
        obj = self.avgpool(obj).squeeze()
        bg = self.avgpool(bg).squeeze()

        # 3. compose
        rel_feats = self.compose(sbj, obj, bg)
        
        # 4. context aggregation via gcn
        adj = self._adjacency(proposals, rel_pair_idxs)

        n = obj_feats.size(0)
        feats = torch.cat((obj_feats, rel_feats), dim=0) # (n+m)xC
        gcn_out = self.gcn(feats, adj, residual=True) # (n+m)xC
        obj_feats = gcn_out[:n] # nxC
        rel_feats = gcn_out[n:] # mxC
        
        # 5. predict obj & rel dist
        if self.mode == 'predcls':
            obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
            obj_dists = to_onehot(obj_labels, self.obj_classes)
            # # obj_logits = torch.eye(self.obj_classes, device=obj_labels.device)[obj_labels]
        else:
            obj_dists = self.obj_predictor(obj_feats) # nx150

        rel_dists = self.rel_predictor(rel_feats) # mx50
        
        return obj_dists, rel_dists