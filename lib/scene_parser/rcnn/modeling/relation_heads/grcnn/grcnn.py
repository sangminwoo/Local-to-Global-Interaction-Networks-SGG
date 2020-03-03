# Graph R-CNN for scene graph generation
# Reimnplemetned by Jianwei Yang (jw2yang@gatech.edu)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from ..roi_relation_feature_extractors import make_roi_relation_feature_extractor
from ..roi_relation_box_feature_extractors import make_roi_relation_box_feature_extractor
from ..roi_relation_box_predictors import make_roi_relation_box_predictor
from ..roi_relation_predictors import make_roi_relation_predictor
from .agcn.agcn import _GraphConvolutionLayer_Collect, _GraphConvolutionLayer_Update
from ..spatial_rel_embedding import make_spatial_relation_feature_extractor
from ..attention import AttentionGate, MultiHeadAttention

class GRCNN(nn.Module):
	# def __init__(self, fea_size, dropout=False, gate_width=1, use_kernel_function=False):
    def __init__(self, cfg, in_channels):
        super(GRCNN, self).__init__()
        self.cfg = cfg
        self.dim = 1024
        self.feat_update_step = cfg.MODEL.ROI_RELATION_HEAD.GRCNN_FEATURE_UPDATE_STEP # 2
        self.score_update_step = cfg.MODEL.ROI_RELATION_HEAD.GRCNN_SCORE_UPDATE_STEP # 2
        num_classes_obj = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES # 151
        num_classes_pred = cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES # 51
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # self.obj_feature_extractor = make_roi_relation_box_feature_extractor(cfg, in_channels)
        self.pred_feature_extractor = make_roi_relation_feature_extractor(cfg, in_channels)
        self.spatial_feature_extractor = make_spatial_relation_feature_extractor(
            xy_hidden=32, spatial_hidden=256, num_dots=16, reduction_ratio=8
        )

        # Visual(2048), Word(300), Spatial(212)
        self.obj_embedding = nn.Sequential(
            nn.Linear(self.pred_feature_extractor.out_channels, self.dim), # 2048, 1024
            nn.ReLU(True),
            nn.Linear(self.dim, self.dim), # 1024, 1024
        )

        # Visual(2048), Word(300), Spatial()
        self.rel_embedding = nn.Sequential(
            nn.Linear(self.pred_feature_extractor.out_channels, self.dim), # ?, 1024
            nn.ReLU(True),
            nn.Linear(self.dim, self.dim), # 1024, 1024
        )

        self.attention = AttentionGate(
            in_channels=self.pred_feature_extractor.out_channels,
            reduction_ratio=16
        )

        if self.feat_update_step > 0: # 2
            self.gcn_collect_feat = _GraphConvolutionLayer_Collect(self.dim, self.dim) # 1024, 1024
            self.gcn_update_feat = _GraphConvolutionLayer_Update(self.dim, self.dim)  # 1024, 1024

        if self.score_update_step > 0: # 2
            self.gcn_collect_score = _GraphConvolutionLayer_Collect(num_classes_obj, num_classes_pred) # 151, 51
            self.gcn_update_score = _GraphConvolutionLayer_Update(num_classes_obj, num_classes_pred) # 151, 51

        self.obj_predictor = make_roi_relation_box_predictor(cfg, self.dim) # 1024
        self.pred_predictor = make_roi_relation_predictor(cfg, self.dim) # 1024

    def _get_map_idxs(self, proposals, proposal_pairs):
        '''
        proposals: list(4 x BoxList)
            [BoxList(num_boxes=36, image_width=1024, image_height=681, mode=xyxy),
            BoxList(num_boxes=21, image_width=1024, image_height=768, mode=xyxy),
            BoxList(num_boxes=56, image_width=1024, image_height=679, mode=xyxy),
            BoxList(num_boxes=20, image_width=1024, image_height=681, mode=xyxy)]

        proposal_pairs: list(4 x BoxPairList)
            [BoxPairList(num_boxes=256, image_width=800, image_height=800, mode=xyxy),
            BoxPairList(num_boxes=256, image_width=679, image_height=1024, mode=xyxy),
            BoxPairList(num_boxes=90, image_width=768, image_height=1024, mode=xyxy),
            BoxPairList(num_boxes=256, image_width=681, image_height=1024, mode=xyxy)]
        '''
        rel_inds = []
        offset = 0
        obj_num = sum([len(proposal) for proposal in proposals]) # 36 + 32+ 56 + 20 = 144
        obj_obj_map = torch.FloatTensor(obj_num, obj_num).fill_(0) # zeros(144x144)
        for proposal, proposal_pair in zip(proposals, proposal_pairs):
            rel_ind_i = proposal_pair.get_field("idx_pairs").detach() # Kx2
            obj_obj_map_i = (1 - torch.eye(len(proposal))).float() # ones(36x36) (diagonal 0)
            obj_obj_map[offset:offset + len(proposal), offset:offset + len(proposal)] = obj_obj_map_i 
            rel_ind_i += offset # size: Kx2, (idx_subj, idx_obj) + 0
            offset += len(proposal) # 36
            rel_inds.append(rel_ind_i) # (idx_subj, idx_obj)

        rel_inds = torch.cat(rel_inds, 0) # Nx2

        subj_pred_map = rel_inds.new(obj_num, rel_inds.shape[0]).fill_(0).float().detach() # 144xN
        obj_pred_map = rel_inds.new(obj_num, rel_inds.shape[0]).fill_(0).float().detach() # 144xN

        subj_pred_map.scatter_(0, (rel_inds[:, 0].contiguous().view(1, -1)), 1) # 
        obj_pred_map.scatter_(0, (rel_inds[:, 1].contiguous().view(1, -1)), 1)
        #print(f'obj_obj_map.type(): {obj_obj_map.type()}, obj_pred_map.type(): {obj_pred_map.type()}')
        obj_obj_map = obj_obj_map.type_as(obj_pred_map)
        #obj_obj_map = obj_obj_map.to(obj_pred_map.device)

        return rel_inds, obj_obj_map, subj_pred_map, obj_pred_map

    def forward(self, features, proposals, proposal_pairs):
        rel_inds, obj_obj_map, subj_pred_map, obj_pred_map = self._get_map_idxs(proposals, proposal_pairs)
        
        # Object features
        x_obj = torch.cat([proposal.get_field("features").detach() for proposal in proposals], 0) # Kx2048x1x1
        obj_class_logits = torch.cat([proposal.get_field("logits").detach() for proposal in proposals], 0)
        # x_obj = self.avgpool(self.obj_feature_extractor(features, proposals))
        
        x_obj = x_obj.view(x_obj.size(0), -1) # ?x2048
        x_obj = self.obj_embedding(x_obj) # ?x2048 -> ?x1024

        # Predicate features
        x_pred, _ = self.pred_feature_extractor(features, proposals, proposal_pairs) # 1024x2048x7x7
        #x_pred = self.avgpool(x_pred) # 1024x2048x1x1
        
        # Attention Gate for predicate
        # x_pred = self.attention(x_pred)

        # Spatial feature for predicate
        # spatial_embeds = self.spatial_feature_extractor(proposal_pairs) # 1024x256
        # spatial_embeds = spatial_embeds.to(x_pred.device)

        #x_pred = x_pred.view(x_pred.size(0), -1) # 1024x2048
        # x_pred = torch.cat((x_pred, spatial_embeds), dim=1)
        #x_pred = self.rel_embedding(x_pred) # 1024x2048 ->  1024x1024
        
        # x_pred = torch.mm(x_pred, spatial_embeds) #  1024x1024 x 1024x256 = 1024x256
        
        # word_embedding = 
        # spatial_embeds


        '''feature level agcn'''
        obj_feats = [x_obj]
        pred_feats = [x_pred]

        for t in range(self.feat_update_step): # 2
            # message from other objects
            source_obj = self.gcn_collect_feat(obj_feats[t], obj_feats[t], obj_obj_map, 4) # from obj to obj
            source_rel_sub = self.gcn_collect_feat(obj_feats[t], pred_feats[t], subj_pred_map, 0) # from rel to obj (subject)
            source_rel_obj = self.gcn_collect_feat(obj_feats[t], pred_feats[t], obj_pred_map, 1) # from rel to obj (object)
            source2obj_all = (source_obj + source_rel_sub + source_rel_obj) / 3
            obj_feats.append(self.gcn_update_feat(obj_feats[t], source2obj_all, 0)) # obj from others

            '''update predicate logits'''
            source_obj_sub = self.gcn_collect_feat(pred_feats[t], obj_feats[t], subj_pred_map.t(), 2) # from obj (subject) to rel
            source_obj_obj = self.gcn_collect_feat(pred_feats[t], obj_feats[t], obj_pred_map.t(), 3) # from obj (object) to rel
            source2rel_all = (source_obj_sub + source_obj_obj) / 2
            pred_feats.append(self.gcn_update_feat(pred_feats[t], source2rel_all, 1)) # rel from others

        obj_class_logits = self.obj_predictor(obj_feats[-1].unsqueeze(2).unsqueeze(3)) # Nx51
        pred_class_logits = self.pred_predictor(pred_feats[-1].unsqueeze(2).unsqueeze(3)) # Nx151

        '''score level agcn'''
        obj_scores = [obj_class_logits]
        pred_scores = [pred_class_logits]

        for t in range(self.score_update_step):
            '''update object logits'''
            # message from other objects
            source_obj = self.gcn_collect_score(obj_scores[t], obj_scores[t], obj_obj_map, 4)

            # message from predicate
            source_rel_sub = self.gcn_collect_score(obj_scores[t], pred_scores[t], subj_pred_map, 0)
            source_rel_obj = self.gcn_collect_score(obj_scores[t], pred_scores[t], obj_pred_map, 1)
            source2obj_all = (source_obj + source_rel_sub + source_rel_obj) / 3
            obj_scores.append(self.gcn_update_score(obj_scores[t], source2obj_all, 0))

            '''update predicate logits'''
            source_obj_sub = self.gcn_collect_score(pred_scores[t], obj_scores[t], subj_pred_map.t(), 2)
            source_obj_obj = self.gcn_collect_score(pred_scores[t], obj_scores[t], obj_pred_map.t(), 3)
            source2rel_all = (source_obj_sub + source_obj_obj) / 2
            pred_scores.append(self.gcn_update_score(pred_scores[t], source2rel_all, 1))

        obj_class_logits = obj_scores[-1]
        pred_class_logits = pred_scores[-1]

        if obj_class_logits is None:
            logits = torch.cat([proposal.get_field("logits") for proposal in proposals], 0)
            obj_class_labels = logits[:, 1:].max(1)[1] + 1
        else:
            obj_class_labels = obj_class_logits[:, 1:].max(1)[1] + 1

        return (x_pred), obj_class_logits, pred_class_logits, obj_class_labels, rel_inds

def build_grcnn_model(cfg, in_channels):
    return GRCNN(cfg, in_channels)
