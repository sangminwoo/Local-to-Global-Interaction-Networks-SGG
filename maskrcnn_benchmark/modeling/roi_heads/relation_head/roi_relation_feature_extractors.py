# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from maskrcnn_benchmark.modeling import registry
from maskrcnn_benchmark.modeling.backbone import resnet
from maskrcnn_benchmark.modeling.poolers import Pooler
from maskrcnn_benchmark.modeling.make_layers import group_norm
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_union, boxlist_intersection
from maskrcnn_benchmark.modeling.roi_heads.box_head.roi_box_feature_extractors import make_roi_box_feature_extractor
from maskrcnn_benchmark.modeling.roi_heads.attribute_head.roi_attribute_feature_extractors import make_roi_attribute_feature_extractor
from .utils_motifs import to_onehot

@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("RelationFeatureExtractor")
class RelationFeatureExtractor(nn.Module):
    """
    Heads for Motifs for relation triplet classification
    """
    def __init__(self, cfg, in_channels):
        super(RelationFeatureExtractor, self).__init__()
        self.cfg = cfg.clone()
        # should corresponding to tail_feature_map function in neural-motifs
        self.resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pool_all_levels = cfg.MODEL.ROI_RELATION_HEAD.POOLING_ALL_LEVELS
        
        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else: 
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        if cfg.MODEL.ATTRIBUTE_ON:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, half_out=True, cat_all_levels=pool_all_levels)
            self.att_feature_extractor = make_roi_attribute_feature_extractor(cfg, in_channels, half_out=True, cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels * 2
        else:
            self.feature_extractor = make_roi_box_feature_extractor(cfg, in_channels, cat_all_levels=pool_all_levels)
            self.out_channels = self.feature_extractor.out_channels

        # separete spatial
        self.separate_spatial = self.cfg.MODEL.ROI_RELATION_HEAD.CAUSAL.SEPARATE_SPATIAL
        if self.separate_spatial:
            input_size = self.feature_extractor.resize_channels
            out_dim = self.feature_extractor.out_channels
            self.spatial_fc = nn.Sequential(*[make_fc(input_size, out_dim//2), nn.ReLU(inplace=True),
                                              make_fc(out_dim//2, out_dim), nn.ReLU(inplace=True),
                                            ])

        assert not(self.cfg.MODEL.ROI_RELATION_HEAD.POOL_SBJ_OBJ and self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.USE_MASKING),\
            "pool_sbj_obj and masking should not be used at once!"
        self.use_sbj_obj = self.cfg.MODEL.ROI_RELATION_HEAD.POOL_SBJ_OBJ or self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.USE_MASKING 
        # union rectangle size
        self.rect_size = self.resolution if self.cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "CSIPredictor" \
        and self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.USE_MASKING else self.resolution * 4 -1
        rect_input = 1 if self.use_sbj_obj else 2
        self.rect_conv = nn.Sequential(
            nn.Conv2d(rect_input, in_channels //2, kernel_size=7, stride=2, padding=3, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels//2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels, momentum=0.01),
        )
        self.obj_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.logit_conv = nn.Sequential(
            nn.Conv2d(self.obj_classes, in_channels //2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels//2, momentum=0.01),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(in_channels, momentum=0.01),
        )


    def forward(self, x, proposals, rel_pair_idxs=None):
        device = x[0].device
        # for pooling visual feat
        union_proposals = []
        head_proposals = []
        tail_proposals = []
        
        # spatial feat
        union_rect_inputs = []
        head_rect_inputs = []
        tail_rect_inputs = []
        
        # semantic feat
        union_logits = []
        head_logits = []
        tail_logits = []
        for proposal, rel_pair_idx in zip(proposals, rel_pair_idxs):
            head_proposal = proposal[rel_pair_idx[:, 0]]
            tail_proposal = proposal[rel_pair_idx[:, 1]]
            union_proposal = boxlist_union(head_proposal, tail_proposal)
            union_proposals.append(union_proposal)

            if self.mode == 'predcls':
                head_labels = proposal[rel_pair_idx[:, 0]].get_field("labels")
                head_logit = to_onehot(head_labels, self.obj_classes)
                tail_labels = proposal[rel_pair_idx[:, 1]].get_field("labels")
                tail_logit = to_onehot(tail_labels, self.obj_classes)
            else:
                head_logit = torch.cat([proposal[rel_pair_idx[:, 0]].get_field("pred_scores") for proposal in proposals], 0) # "predict_logits"
                tail_logit = torch.cat([proposal[rel_pair_idx[:, 1]].get_field("pred_scores") for proposal in proposals], 0) # "predict_logits"

            union_logit = head_logit + tail_logit
            union_logits.append(union_logit)

            if self.use_sbj_obj:
                head_proposals.append(head_proposal)
                tail_proposals.append(tail_proposal)

                head_logits.append(head_logit)
                tail_logits.append(tail_logit)

            # use range to construct rectangle, sized (rect_size, rect_size)
            num_rel = len(rel_pair_idx)
            dummy_x_range = torch.arange(self.rect_size, device=device).view(1, 1, -1).expand(num_rel, self.rect_size, self.rect_size)
            dummy_y_range = torch.arange(self.rect_size, device=device).view(1, -1, 1).expand(num_rel, self.rect_size, self.rect_size)
            # resize bbox to the scale rect_size
            # before) BoxList(num_boxes=90, image_width=800, image_height=600, mode=xyxy)
            # after) BoxList(num_boxes=90, image_width=27, image_height=27, mode=xyxy)
            head_proposal = head_proposal.resize((self.rect_size, self.rect_size))
            tail_proposal = tail_proposal.resize((self.rect_size, self.rect_size))

            head_rect = ((dummy_x_range >= head_proposal.bbox[:,0].floor().view(-1,1,1).long()) & \
                        (dummy_x_range <= head_proposal.bbox[:,2].ceil().view(-1,1,1).long()) & \
                        (dummy_y_range >= head_proposal.bbox[:,1].floor().view(-1,1,1).long()) & \
                        (dummy_y_range <= head_proposal.bbox[:,3].ceil().view(-1,1,1).long())).float()
            tail_rect = ((dummy_x_range >= tail_proposal.bbox[:,0].floor().view(-1,1,1).long()) & \
                        (dummy_x_range <= tail_proposal.bbox[:,2].ceil().view(-1,1,1).long()) & \
                        (dummy_y_range >= tail_proposal.bbox[:,1].floor().view(-1,1,1).long()) & \
                        (dummy_y_range <= tail_proposal.bbox[:,3].ceil().view(-1,1,1).long())).float()

        if self.use_sbj_obj:
            head_rect_input = head_rect.unsqueeze(1)
            tail_rect_input = tail_rect.unsqueeze(1)
            union_rect_input = head_rect_input + tail_rect_input

            union_rect_inputs.append(union_rect_input)
            head_rect_inputs.append(head_rect_input)
            tail_rect_inputs.append(tail_rect_input)

            union_logits = torch.cat(union_logits, dim=0) # Nx151
            head_logits = torch.cat(head_logits, dim=0) # Nx151
            tail_logits = torch.cat(tail_logits, dim=0) # Nx151
        else:
            union_rect_input = torch.stack((head_rect, tail_rect), dim=1) # (num_rel, 2, rect_size, rect_size)
            union_rect_inputs.append(union_rect_input)
            union_logits = torch.cat(union_logits, dim=0) # Nx151
     
        # rectangle feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        union_rect_inputs = torch.cat(union_rect_inputs, dim=0)
        union_rect_features = self.rect_conv(union_rect_inputs)
        union_logit_inputs = union_logits.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.resolution, self.resolution) # Nx151x1x1 -> Nx151x7x7
        union_logit_features = self.logit_conv(union_logit_inputs)

        if self.use_sbj_obj:
            head_rect_inputs = torch.cat(head_rect_inputs, dim=0)
            tail_rect_inputs = torch.cat(tail_rect_inputs, dim=0)
            head_rect_features = self.rect_conv(head_rect_inputs)
            tail_rect_features = self.rect_conv(tail_rect_inputs)
            
            head_logit_inputs = head_logits.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.resolution, self.resolution)
            tail_logit_inputs = tail_logits.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.resolution, self.resolution)
            head_logit_features = self.logit_conv(head_logit_inputs)
            tail_logit_features = self.logit_conv(tail_logit_inputs)

        if self.cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR == "CSIPredictor":
            if self.cfg.MODEL.ROI_RELATION_HEAD.POOL_SBJ_OBJ:
                union_features = self.feature_extractor.pooler(x, union_proposals) # NxCx7x7
                head_features = self.feature_extractor.pooler(x, head_proposals)
                tail_features = self.feature_extractor.pooler(x, tail_proposals)

                # add spatial & semantics 
                union_features = union_features + union_rect_features + union_logit_features
                head_features = head_features + head_rect_features + head_logit_features
                tail_features = tail_features + tail_rect_features + tail_logit_features

                return head_features, tail_features, union_features

            elif self.cfg.MODEL.ROI_RELATION_HEAD.CSINET.USE_MASKING: # always requires spatial
                union_features = self.feature_extractor.pooler(x, union_proposals) # NxCxHxW
                head_features = union_features * head_rect_inputs # NxCxHxW x Nx1xHxW
                tail_features = union_features * tail_rect_inputs
                background_features = union_features * union_rect_inputs

                return head_features, tail_features, background_features

            else:
                union_features = self.feature_extractor.pooler(x, union_proposals)
                return [union_features]

        # union visual feature. size (total_num_rel, in_channels, POOLER_RESOLUTION, POOLER_RESOLUTION)
        union_vis_features = self.feature_extractor.pooler(x, union_proposals)

        # merge two parts
        if self.separate_spatial:
            region_features = self.feature_extractor.forward_without_pool(union_vis_features)
            spatial_features = self.spatial_fc(union_rect_features.view(union_rect_features.size(0), -1))
            union_features = (region_features, spatial_features)
        else:
            union_features = union_vis_features + union_rect_features
            union_features = self.feature_extractor.forward_without_pool(union_features) # (total_num_rel, out_channels)

        if self.cfg.MODEL.ATTRIBUTE_ON:
            union_att_features = self.att_feature_extractor.pooler(x, union_proposals)
            union_features_att = union_att_features + union_rect_features
            union_features_att = self.att_feature_extractor.forward_without_pool(union_features_att)
            union_features = torch.cat((union_features, union_features_att), dim=-1)
            
        return union_features

def make_roi_relation_feature_extractor(cfg, in_channels):
    func = registry.ROI_RELATION_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
