import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

from .grcnn_agcn import _GraphConvolutionLayer_Collect, _GraphConvolutionLayer_Update
from maskrcnn_benchmark.modeling.utils import cat
from .utils_motifs import to_onehot

class GRCNN(nn.Module):
    def __init__(self, cfg, in_channels):
        super(GRCNN, self).__init__()
        self.cfg = cfg
        self.dim = 256
        self.feat_update_step = cfg.MODEL.ROI_RELATION_HEAD.GRCNN.FEATURE_UPDATE_STEP
        self.score_update_step = cfg.MODEL.ROI_RELATION_HEAD.GRCNN.SCORE_UPDATE_STEP
        self.obj_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.rel_classes = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else: 
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.obj_embedding = nn.Sequential(
            nn.Linear(self.cfg.MODEL.ROI_BOX_HEAD.MLP_HEAD_DIM, self.dim),
            nn.ReLU(True),
            nn.Linear(self.dim, self.dim),
        )
        self.rel_embedding = nn.Sequential(
            nn.Linear(2048, self.dim),
            nn.ReLU(True),
            nn.Linear(self.dim, self.dim),
        )

        if self.feat_update_step > 0:
            self.gcn_collect_feat = _GraphConvolutionLayer_Collect(self.dim, self.dim)
            self.gcn_update_feat = _GraphConvolutionLayer_Update(self.dim, self.dim)

        if self.score_update_step > 0:
            self.gcn_collect_score = _GraphConvolutionLayer_Collect(self.obj_classes, self.rel_classes)
            self.gcn_update_score = _GraphConvolutionLayer_Update(self.obj_classes, self.rel_classes)

        self.obj_predictor = nn.Linear(self.dim, self.obj_classes)
        self.rel_predictor = nn.Linear(self.dim, self.rel_classes)

    def _get_map_idxs(self, proposals, rel_pair_idxs):
        rel_inds = []
        offset = 0
        obj_num = sum([len(proposal) for proposal in proposals])
        obj_obj_map = torch.FloatTensor(obj_num, obj_num).fill_(0)
        for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):
            obj_obj_map_i = (1 - torch.eye(len(proposal))).float()
            obj_obj_map[offset:offset + len(proposal), offset:offset + len(proposal)] = obj_obj_map_i
            rel_pair_idx += offset
            offset += len(proposal)
            rel_inds.append(rel_pair_idx)

        rel_inds = torch.cat(rel_inds, 0)

        subj_pred_map = rel_inds.new(obj_num, rel_inds.shape[0]).fill_(0).float().detach()
        obj_pred_map = rel_inds.new(obj_num, rel_inds.shape[0]).fill_(0).float().detach()

        subj_pred_map.scatter_(0, (rel_inds[:, 0].contiguous().view(1, -1)), 1)
        obj_pred_map.scatter_(0, (rel_inds[:, 1].contiguous().view(1, -1)), 1)
        obj_obj_map = obj_obj_map.type_as(obj_pred_map)

        return rel_inds, obj_obj_map, subj_pred_map, obj_pred_map

    def forward(self, roi_features, proposals, rel_features, rel_pair_idxs, logger):
        rel_inds, obj_obj_map, subj_pred_map, obj_pred_map = self._get_map_idxs(proposals, rel_pair_idxs)
        obj_feats = self.obj_embedding(roi_features)

        if self.mode == 'predcls':
            obj_labels = cat([proposal.get_field("labels").detach() for proposal in proposals], dim=0)
            obj_class_logits = to_onehot(obj_labels, self.obj_classes)
        else:
            obj_class_logits = torch.cat([proposal.get_field("predict_logits").detach() for proposal in proposals], 0)

        pred_feats = self.rel_embedding(rel_features)

        '''feature level agcn'''
        obj_features = [obj_feats]
        pred_features = [pred_feats]

        for t in range(self.feat_update_step):
            # message from other objects
            source_obj = self.gcn_collect_feat(obj_features[t], obj_features[t], obj_obj_map, 4)

            source_rel_sub = self.gcn_collect_feat(obj_features[t], pred_features[t], subj_pred_map, 0)
            source_rel_obj = self.gcn_collect_feat(obj_features[t], pred_features[t], obj_pred_map, 1)
            source2obj_all = (source_obj + source_rel_sub + source_rel_obj) / 3
            obj_features.append(self.gcn_update_feat(obj_features[t], source2obj_all, 0))

            '''update predicate logits'''
            source_obj_sub = self.gcn_collect_feat(pred_features[t], obj_features[t], subj_pred_map.t(), 2)
            source_obj_obj = self.gcn_collect_feat(pred_features[t], obj_features[t], obj_pred_map.t(), 3)
            source2rel_all = (source_obj_sub + source_obj_obj) / 2
            pred_features.append(self.gcn_update_feat(pred_features[t], source2rel_all, 1))

        if self.mode != 'predcls':
            obj_class_logits = self.obj_predictor(obj_features[-1])
        pred_class_logits = self.rel_predictor(pred_features[-1])

        '''score level agcn'''
        obj_scores = [obj_class_logits]
        pred_scores = [pred_class_logits]

        for t in range(self.score_update_step):
            '''update object logits'''
            # message from other objects
            source_obj = self.gcn_collect_score(obj_scores[t], obj_scores[t], obj_obj_map, 4)

            #essage from predicate
            source_rel_sub = self.gcn_collect_score(obj_scores[t], pred_scores[t], subj_pred_map, 0)
            source_rel_obj = self.gcn_collect_score(obj_scores[t], pred_scores[t], obj_pred_map, 1)
            source2obj_all = (source_obj + source_rel_sub + source_rel_obj) / 3
            obj_scores.append(self.gcn_update_score(obj_scores[t], source2obj_all, 0))

            '''update predicate logits'''
            source_obj_sub = self.gcn_collect_score(pred_scores[t], obj_scores[t], subj_pred_map.t(), 2)
            source_obj_obj = self.gcn_collect_score(pred_scores[t], obj_scores[t], obj_pred_map.t(), 3)
            source2rel_all = (source_obj_sub + source_obj_obj) / 2
            pred_scores.append(self.gcn_update_score(pred_scores[t], source2rel_all, 1))

        obj_dists = obj_scores[-1]
        rel_dists = pred_scores[-1]

        return obj_dists, rel_dists