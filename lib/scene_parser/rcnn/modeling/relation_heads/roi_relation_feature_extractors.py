# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lib.scene_parser.rcnn.modeling import registry
from lib.scene_parser.rcnn.modeling.backbone import resnet
from lib.scene_parser.rcnn.modeling.poolers import Pooler
from lib.scene_parser.rcnn.modeling.make_layers import group_norm
from lib.scene_parser.rcnn.modeling.make_layers import make_fc
from .sparse_targets import _get_tensor_from_boxlist, _get_rel_inds

from lib.scene_parser.rcnn.structures.bounding_box import BoxList
# from lib.scene_parser.rcnn.modeling.relation_heads.spatial_rel_embedding import SpatialRelEmbedding
from lib.scene_parser.rcnn.modeling.relation_heads.attention import AttentionGate, BinaryAttention, MultiHeadAttention
from lib.scene_parser.rcnn.modeling.relation_heads.relational_context import RelationalContext
from lib.scene_parser.rcnn.modeling.relation_heads.entity_embedding import EntityEmbedding

@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIRelationFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION # 14
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES # (1.0 / 16,)= 0.0625
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO # 0
        self.pooler = Pooler(
            output_size=(resolution, resolution), # (14, 14)
            scales=scales, # (1.0 / 16,)
            sampling_ratio=sampling_ratio, # 0
        )

        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        self.head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC, # BottleneckWithFixedBatchNorm
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS, # 1
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP, # 64
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1, # True
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS, # 256 -> 128
            dilation=config.MODEL.RESNETS.RES5_DILATION # 1
        )

        self.out_channels = self.head.out_channels # 2048
        
        self.avgpool = nn.AdaptiveAvgPool2d(resolution) # (14, 14)
        self.maxpool = nn.AdaptiveMaxPool2d(resolution) # (14, 14)
        
        self.att = AttentionGate(in_channels=1024, reduction_ratio=16)
        self.att2048 = AttentionGate(in_channels=2048, reduction_ratio=32)
        self.conv7x7 = nn.Conv2d(2048, 2048, kernel_size=7)
        self.binary_att = BinaryAttention(in_channels=1024, reduction_ratio=16)

        self.entity_emb = EntityEmbedding(in_channels=1024, hid_channels=1024, out_channels=1024, mode='conv')
        self.rel_ctx = RelationalContext(dim=1024)

    def _object_mask(self, proposal_pairs, proposals_union):
        box_pairs = torch.cat([pair_boxes.bbox for pair_boxes in proposal_pairs], dim=0) # Nx8
        box_unions = torch.cat([union_boxes.bbox for union_boxes in proposals_union], dim=0) # Nx4
        rescaled_ux = box_unions[:,[0,2]]-box_unions[:,0].view(-1, 1).repeat(1, 2) # Nx2
        rescaled_uy = box_unions[:,[1,3]]-box_unions[:,1].view(-1, 1).repeat(1, 2) # Nx2
        union_xy = torch.cat((rescaled_ux[:, 1].view(-1,1), rescaled_uy[:, 1].view(-1, 1)), dim=1) # Nx2
        union_xy = union_xy.round().int() # Nx2

        rescaled_x = box_pairs[:,[0,2,4,6]]-box_unions[:,0].view(-1, 1).repeat(1, 4) # Nx4
        rescaled_y = box_pairs[:,[1,3,5,7]]-box_unions[:,1].view(-1, 1).repeat(1, 4) # Nx4
        rescaled_x = rescaled_x.round().int() # Nx4  
        rescaled_y = rescaled_y.round().int() # Nx4

        objectness = []
        for i, (x, y, u_xy) in enumerate(zip(rescaled_x, rescaled_y, union_xy)):
            objectness.append(torch.zeros(u_xy[0], u_xy[1]))
            
            objectness[i][x[0]:x[1], y[0]:y[1]] = 1
            objectness[i][x[2]:x[3], y[2]:y[3]] = 1
            
            objectness[i] = self.maxpool(objectness[i].unsqueeze(0)) # 1x14x14

        objectness = torch.stack(objectness, dim=0) # Nx1x14x14
        return objectness

    def _graph_mask(self, proposal_pairs, proposals_union):
        box_pairs = torch.cat([pair_boxes.bbox for pair_boxes in proposal_pairs], dim=0) # Nx8
        box_unions = torch.cat([union_boxes.bbox for union_boxes in proposals_union], dim=0) # Nx4
        rescaled_ux = box_unions[:,[0,2]]-box_unions[:,0].view(-1, 1).repeat(1, 2) # Nx2
        rescaled_uy = box_unions[:,[1,3]]-box_unions[:,1].view(-1, 1).repeat(1, 2) # Nx2
        union_xy = torch.cat((rescaled_ux[:, 1].view(-1,1), rescaled_uy[:, 1].view(-1, 1)), dim=1) # Nx2
        union_xy = union_xy.round().int() # Nx2

        rescaled_x = box_pairs[:,[0,2,4,6]]-box_unions[:,0].view(-1, 1).repeat(1, 4) # Nx4
        rescaled_y = box_pairs[:,[1,3,5,7]]-box_unions[:,1].view(-1, 1).repeat(1, 4) # Nx4
        rescaled_x = rescaled_x.round().int() # Nx4  
        rescaled_y = rescaled_y.round().int() # Nx4

        subjectness = []; objectness = []; background = []
        for i, (x, y, u_xy) in enumerate(zip(rescaled_x, rescaled_y, union_xy)):
            subjectness.append(torch.zeros(u_xy[0], u_xy[1])) # zeros(ux x uy)
            objectness.append(torch.zeros(u_xy[0], u_xy[1])) # zeros(ux x uy)
            background.append(torch.ones(u_xy[0], u_xy[1])) # ones(ux x uy)
            
            subjectness[i][x[0]:x[1], y[0]:y[1]] = 1 # ux x uy
            objectness[i][x[2]:x[3], y[2]:y[3]] = 1 # ux x uy
            background[i] = background[i] - subjectness[i] - objectness[i] # ux x uy

            subjectness[i] = self.avgpool(subjectness[i].unsqueeze(0)) # 1x14x14
            objectness[i] = self.avgpool(objectness[i].unsqueeze(0)) # 1x14x14
            background[i] = self.avgpool(background[i].unsqueeze(0)) # 1x14x14

        subjectness = torch.stack(subjectness, dim=0) # Nx1x14x14
        objectness = torch.stack(objectness, dim=0) # Nx1x14x14
        background = torch.stack(background, dim=0) # Nx1x14x14
        return subjectness, objectness, background # Nx1x14x14, Nx1x14x14, Nx1x14x14

    ''' separate box features'''
    def _sep_box_feats(self, x, proposal_pairs):
        '''
        Arguments
            x (list[Tensor]): feature maps
            proposal_pairs (list[BoxPairList]): box pairs list
        '''
        # ret = [proposal_pair.copy_with_separate() for proposal_pair in proposal_pairs] 
        # rel_features = [proposal_pair.relativity_embedding() for proposal_pair in proposal_pairs] 
        # rel_features = torch.stack(rel_features, 0) 

        # proposal1 = []; proposal2 = []
        # for i in range(len(ret)):
        #     proposal1.append(ret[i][0]) 
        #     proposal2.append(ret[i][1]) 

        bbox_list1 = []
        bbox_list2 = []
        #rel_features = []
        for proposal_pair in proposal_pairs:
            bbox1, bbox2 = proposal_pair.copy_with_separate() # (BoxList1, BoxList2), (BoxList3, BoxList4), ...
            bbox_list1.append(bbox1) # list(BoxList1, BoxList3, ...)
            bbox_list2.append(bbox2) # list(BoxList2, BoxList4, ...)
            #rel_features.append(proposal_pair.relativity_embedding()) # list(Tensor[256x2x64], ...)
        #rel_features = torch.stack(rel_features, 0) # Tensor[4/GPUx256x2x64]

        # import pdb; pdb.set_trace()
        x1 = self.pooler(x, bbox_list1) 
        x2 = self.pooler(x, bbox_list2) # 1024x1024x14x14
        '''
        TODO: ADD, AVG, CONCAT
            x_add = x1 + x2
            x_avg = (x1 + x2) / 2
            x_concat = torch.cat((x1, x2), dim=1)
        TODO: MERGE then CONV, CONV then MERGE
        '''
        x1 = self.head(x1) # 1024x2048x7x7
        x2 = self.head(x2) # 1024x2048x7x7
        x = torch.cat((x1, x2), dim=1) # 1024x4096x7x7

        # x = attention(x.permute(0,2,3,1)) # 1024x7x7x4096 -> 1024x7x7x2048
        # x = x.permute(0,3,1,2) # 1024x7x7x2048 -> 1024x2048x7x7
        return x # 1024x4096x7x7

    # def _relativity_embedding(self, proposal_pairs):
    #     rel_features = []
    #     for proposal_pair in proposal_pairs:
    #         rel_features.append(proposal_pair.relativity_embedding()) # list(Tensor[256x2x64], ...)
    #     rel_features = torch.stack(rel_features, 0) # Tensor[4/GPUx256x2x64]

    # def _attention_filter(self, x):
    #     '''
    #     Arguments
    #         x (list[Tensor]): feature maps (NxCxHxW)
    #     '''

    #     att_filter = AttentionFilter(in_channels=x.size(1), reduction_ratio=16)
    #     x = att_filter(x)
    #     return x

    def _union_box_feats(self, x, proposal_pairs):
        '''
        x: list(1 x Tensor(4x1024x48x64))
        
        proposal_pairs: list(4 x BoxPairList)
            [BoxPairList(num_boxes=256, image_width=800, image_height=800, mode=xyxy),
            BoxPairList(num_boxes=256, image_width=679, image_height=1024, mode=xyxy),
            BoxPairList(num_boxes=90, image_width=768, image_height=1024, mode=xyxy),
            BoxPairList(num_boxes=256, image_width=681, image_height=1024, mode=xyxy)]

        proposal_pairs[0].bbox: Tensor(256x8)
            tensor([[   1.4859,  572.2100, 1023.0000,  ...,  517.2621,  110.0586,  584.8137],
                    [  77.8399,  517.2621,  110.0586,  ...,   45.9913,  869.6760,  617.0351],
                    [  11.7182,  568.7908, 1007.4652,  ...,  517.2621,  110.0586,  584.8137],
                    ...,
                    [   0.0000,   66.5250,  978.3199,  ...,  318.7549, 1004.7848,  535.4465],
                    [ 277.1177,  493.0235,  672.6985,  ...,   66.5250,  978.3199,  390.9438],
                    [   0.0000,   66.5250,  978.3199,  ...,  136.2595,  282.8971,  175.8052]])
        '''
        proposals_union = [proposal_pair.copy_with_union() for proposal_pair in proposal_pairs]
        '''
        proposals_union: list(4 x BoxList)
            [BoxList(num_boxes=256, image_width=1024, image_height=681, mode=xyxy),
            BoxList(num_boxes=256, image_width=1024, image_height=768, mode=xyxy),
            BoxList(num_boxes=256, image_width=1024, image_height=679, mode=xyxy),
            BoxList(num_boxes=256, image_width=1024, image_height=681, mode=xyxy)]
            
        proposals_union[0].bbox: Tensor(256x4)
            tensor([[185.7104,  15.6538, 947.6257, 466.6268],
                    [  4.9850, 235.4452, 947.6257, 596.2276],
                    [185.7104,  15.6538, 953.4727, 533.4866],
                    ...,
                    [  1.9670, 245.0121, 236.0204, 584.1255],
                    [175.3459, 419.3000, 947.6257, 591.8372],
                    [571.0229, 286.1477, 841.9486, 539.7220]])
        '''
        x_union = self.pooler(x, proposals_union) # x_union: Tensor(858x1024x14x14)
        x = self.head(x_union) # x: Tensor(858x2048x7x7)
        x = self.att2048(x)
        x = self.conv7x7(x).squeeze()
        return x

    def _object_mask_att(self, x, proposal_pairs):
        '''
        Make Network only attend to objects
        '''
        proposals_union = [proposal_pair.copy_with_union() for proposal_pair in proposal_pairs]

        x_union = self.pooler(x, proposals_union) # x_union: Tensor(858x1024x14x14)
        object_mask = self._object_mask(proposal_pairs, proposals_union) # (Nx1x14x14)
        x_att, maskloss = self.binary_att(x_union, object_mask) # Nx1024x14x14

        x = self.head(x_att) # x: Tensor(858x2048x7x7)
        return x, maskloss

    def _graph_mask_att(self, x, proposal_pairs):
        '''
        Multi-Head Attention for each instances & Sum and Embed
        '''
        proposals_union = [proposal_pair.copy_with_union() for proposal_pair in proposal_pairs]

        x_union = self.pooler(x, proposals_union) # x_union: Tensor(858x1024x14x14)
        
        subject_mask, object_mask, background_mask = self._graph_mask(proposal_pairs, proposals_union) # Nx1x14x14
        x_subject = x_union * subject_mask.to(x_union.device) # Nx1024x14x14
        x_object = x_union * object_mask.to(x_union.device) # Nx1024x14x14
        x_background = x_union * background_mask.to(x_union.device) # Nx1024x14x14
        
        subj_att = self.att(x_subject) # Nx1024x14x14
        obj_att = self.att(x_object) # Nx1024x14x14
        bg_att = self.att(x_background) # Nx1024x14x14

        x_att = subj_att + obj_att + bg_att # Nx1024x14x14

        x = self.head(x_att) # Nx2048x7x7

        return x # # Nx2048x7x7

    def _graph_mask_gcn(self, x, proposal_pairs):
        '''
        Multi-Head Attention for each instances & Embed each instances and aggregate with gcn
        '''
        proposals_union = [proposal_pair.copy_with_union() for proposal_pair in proposal_pairs]

        x_union = self.pooler(x, proposals_union) # x_union: Tensor(858x1024x14x14)

        subject_mask, object_mask, background_mask = self._graph_mask(proposal_pairs, proposals_union) # Nx1x14x14
        x_subject = x_union * subject_mask.to(x_union.device) # Nx1024x14x14
        x_object = x_union * object_mask.to(x_union.device) # Nx1024x14x14
        x_background = x_union * background_mask.to(x_union.device) # Nx1024x14x14
        
        subj_att = self.att(x_subject) # Nx1024x14x14
        obj_att = self.att(x_object) # Nx1024x14x14
        bg_att = self.att(x_background) # Nx1024x14x14

        subj_emb, obj_emb, bg_emb = self.entity_emb(subj_att, obj_att, bg_att) # Nx1024x14x14 -> Nx1024x1x1 -> Nx1024 -> Nx1024
        x = self.rel_ctx(subj_emb, obj_emb, bg_emb) # Nx1024

        # x = self.head(x_att) # x: Tensor(858x2048x7x7)
        return x

    def forward(self, x, proposals, proposal_pairs):

        # acquire tensor format per batch data
        # bboxes, cls_prob (N, k)
        # im_inds: (N,1), img ind for each roi in the batch
        obj_box_priors, obj_labels, im_inds \
            = _get_tensor_from_boxlist(proposals, 'labels')

        # get index in the proposal pairs
        _, proposal_idx_pairs, im_inds_pairs = _get_tensor_from_boxlist(
            proposal_pairs, 'idx_pairs')

        rel_inds = _get_rel_inds(im_inds, im_inds_pairs, proposal_idx_pairs)

        x = self._union_box_feats(x, proposal_pairs)
        # x = self._graph_mask_att(x, proposal_pairs)
        # x = self._graph_mask_gcn(x, proposal_pairs)

        return x, rel_inds # x, spatial_rel, rel_inds # 1024x2048x7x7, 1024x256


@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("FPN2MLPRelationFeatureExtractor")
class FPN2MLPFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPN2MLPFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_RELATION_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_RELATION_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_RELATION_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        input_size = in_channels * resolution ** 2
        representation_size = cfg.MODEL.ROI_RELATION_HEAD.MLP_HEAD_DIM
        use_gn = cfg.MODEL.ROI_RELATION_HEAD.USE_GN
        self.pooler = pooler
        self.fc6 = make_fc(input_size, representation_size, use_gn)
        self.fc7 = make_fc(representation_size, representation_size, use_gn)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        return x


@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("FPNXconv1fcRelationFeatureExtractor")
class FPNXconv1fcFeatureExtractor(nn.Module):
    """
    Heads for FPN for classification
    """

    def __init__(self, cfg, in_channels):
        super(FPNXconv1fcFeatureExtractor, self).__init__()

        resolution = cfg.MODEL.ROI_RELATION_HEAD.POOLER_RESOLUTION
        scales = cfg.MODEL.ROI_RELATION_HEAD.POOLER_SCALES
        sampling_ratio = cfg.MODEL.ROI_RELATION_HEAD.POOLER_SAMPLING_RATIO
        pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        self.pooler = pooler

        use_gn = cfg.MODEL.ROI_RELATION_HEAD.USE_GN
        conv_head_dim = cfg.MODEL.ROI_RELATION_HEAD.CONV_HEAD_DIM
        num_stacked_convs = cfg.MODEL.ROI_RELATION_HEAD.NUM_STACKED_CONVS
        dilation = cfg.MODEL.ROI_RELATION_HEAD.DILATION

        xconvs = []
        for ix in range(num_stacked_convs):
            xconvs.append(
                nn.Conv2d(
                    in_channels,
                    conv_head_dim,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation,
                    bias=False if use_gn else True
                )
            )
            in_channels = conv_head_dim
            if use_gn:
                xconvs.append(group_norm(in_channels))
            xconvs.append(nn.ReLU(inplace=True))

        self.add_module("xconvs", nn.Sequential(*xconvs))
        for modules in [self.xconvs,]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    if not use_gn:
                        torch.nn.init.constant_(l.bias, 0)

        input_size = conv_head_dim * resolution ** 2
        representation_size = cfg.MODEL.ROI_RELATION_HEAD.MLP_HEAD_DIM
        self.fc6 = make_fc(input_size, representation_size, use_gn=False)
        self.out_channels = representation_size

    def forward(self, x, proposals):
        x = self.pooler(x, proposals)
        x = self.xconvs(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc6(x))
        return x


def make_roi_relation_feature_extractor(cfg, in_channels):
    func = registry.ROI_RELATION_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
