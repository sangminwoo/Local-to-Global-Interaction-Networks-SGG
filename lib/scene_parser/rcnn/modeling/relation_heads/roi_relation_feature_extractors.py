# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
from torch.nn import functional as F

from lib.scene_parser.rcnn.modeling import registry
from lib.scene_parser.rcnn.modeling.backbone import resnet
from lib.scene_parser.rcnn.modeling.poolers import Pooler
from lib.scene_parser.rcnn.modeling.make_layers import group_norm
from lib.scene_parser.rcnn.modeling.make_layers import make_fc
from .sparse_targets import _get_tensor_from_boxlist, _get_rel_inds
from lib.scene_parser.rcnn.modeling.relation_heads.attention import InstanceAttention
from lib.scene_parser.rcnn.modeling.relation_heads.relpn.multi_head_att import MultiHeadAttention
from lib.scene_parser.rcnn.modeling.relation_heads.rel_embed import InstanceConvolution, InstanceEmbedding, RelationalEmbedding, RelationalEmbeddingMultiClass
from lib.scene_parser.rcnn.modeling.relation_heads.pos_encoder import PositionEncoder, UnionBoxEncoder, PositionEncoderV2

@registry.ROI_RELATION_FEATURE_EXTRACTORS.register("ResNet50Conv5ROIRelationFeatureExtractor")
class ResNet50Conv5ROIFeatureExtractor(nn.Module):
    def __init__(self, config, in_channels):
        super(ResNet50Conv5ROIFeatureExtractor, self).__init__()

        resolution = config.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        scales = config.MODEL.ROI_BOX_HEAD.POOLER_SCALES
        sampling_ratio = config.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.pooler = Pooler(
            output_size=(resolution, resolution),
            scales=scales,
            sampling_ratio=sampling_ratio,
        )
        stage = resnet.StageSpec(index=4, block_count=3, return_features=False)
        head = resnet.ResNetHead(
            block_module=config.MODEL.RESNETS.TRANS_FUNC,
            stages=(stage,),
            num_groups=config.MODEL.RESNETS.NUM_GROUPS,
            width_per_group=config.MODEL.RESNETS.WIDTH_PER_GROUP,
            stride_in_1x1=config.MODEL.RESNETS.STRIDE_IN_1X1,
            stride_init=None,
            res2_out_channels=config.MODEL.RESNETS.RES2_OUT_CHANNELS,
            dilation=config.MODEL.RESNETS.RES5_DILATION
        )

        self.head = nn.Sequential(
            head,
            nn.Conv2d(2048, 1024, kernel_size=3, stride=1, padding=1), # 7x7 -> 7x7
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(3,3), padding=1), # 7x7 -> 3x3
            nn.Conv2d(1024, 1024, kernel_size=3), # 3x3 -> 1x1
        )
        self.embed = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(resolution) # (14, 14)
        self.maxpool = nn.AdaptiveMaxPool2d(resolution) # (14, 14)

        self.instance_att_512 = InstanceAttention(in_channels=512, reduction_ratio=16, kernel_size=3)
        self.instance_att_1024 = InstanceAttention(in_channels=1024, reduction_ratio=64, kernel_size=3)
        self.instance_att_2048 = InstanceAttention(in_channels=2048, reduction_ratio=128, kernel_size=3)

        self.instance_conv = InstanceConvolution(config, in_dim=2048, hid_dim=1024, out_dim=1024)

        self.instance_embedding = InstanceEmbedding(in_dim=1024, hid_dim=512, out_dim=1024)
        # self.rel_embedding = RelationalEmbedding(
        #     in_dim=1024, so_hid_dim=1024, so_out_dim=1024, sob_hid_dim=1024, out_dim=1024
        # )
        self.rel_embedding = RelationalEmbeddingMultiClass(
            in_dim=1024, so_hid_dim=1024, so_out_dim=1024, sob_hid_dim=1024, out_dim=1024, num_classes=151
        )
        self.out_channels = 2048

        self.position_encoder = PositionEncoder(kernel_size=3)
        self.union_box_encoder = UnionBoxEncoder(in_dim=1024, hid_dim=512, num_classes=151)
        self.position_encoder_v2 = PositionEncoderV2(in_dim=1024, hid_dim=196, out_dim=196)

        self.subj_conv = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), # 7x7 -> 7x7
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(3,3), padding=1), # 7x7 -> 3x3
            nn.Conv2d(512, 1024, kernel_size=3), # 3x3 -> 1x1
        )
        self.obj_conv = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), # 7x7 -> 7x7
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(3,3), padding=1), # 7x7 -> 3x3
            nn.Conv2d(512, 1024, kernel_size=3), # 3x3 -> 1x1
        )
        self.bg_conv = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1), # 7x7 -> 7x7
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(3,3), padding=1), # 7x7 -> 3x3
            nn.Conv2d(512, 1024, kernel_size=3), # 3x3 -> 1x1
        )

    def _union_box_feats(self, x, proposal_pairs):
        proposals_union = [proposal_pair.copy_with_union() for proposal_pair in proposal_pairs]
        x_union = self.pooler(x, proposals_union) # Nx1024x14x14
        x = self.head(x_union).squeeze() # Nx1024x14x14 -> Nx2048x7x7 -> Nx1024x1x1 -> Nx1024
        x = self.embed(x) # Nx1024 -> Nx1024
        return x # Nx1024

    def _graph_mask(self, proposal_pairs, proposals_union, pool='avg'):
        assert pool in ['avg', 'max'], "pooling method should be 'avg' or 'max'"

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

        subjectness = []; objectness = []
        for i, (x, y, u_xy) in enumerate(zip(rescaled_x, rescaled_y, union_xy)):
            subjectness.append(torch.zeros(u_xy[0], u_xy[1])) # zeros(ux x uy)
            objectness.append(torch.zeros(u_xy[0], u_xy[1])) # zeros(ux x uy)
            
            subjectness[i][x[0]:x[1], y[0]:y[1]] = 1 # ux x uy
            objectness[i][x[2]:x[3], y[2]:y[3]] = 1 # ux x uy
            
            if pool == 'avg':
                subjectness[i] = self.avgpool(subjectness[i].unsqueeze(0)) # 1x14x14
                objectness[i] = self.avgpool(objectness[i].unsqueeze(0)) # 1x14x14
            elif pool == 'max':
                subjectness[i] = self.maxpool(subjectness[i].unsqueeze(0)) # 1x14x14
                objectness[i] = self.maxpool(objectness[i].unsqueeze(0)) # 1x14x14

        subjectness = torch.stack(subjectness, dim=0) # Nx1x14x14
        objectness = torch.stack(objectness, dim=0) # Nx1x14x14
        background = torch.ones(subjectness.size(0), 1, 14, 14) - subjectness - objectness # Nx1x14x14
        background[background < 0] = 0

        return subjectness, objectness, background # Nx1x14x14, Nx1x14x14, Nx1x14x14

    def _graph_mask_att(self, x, proposal_pairs):
        proposals_union = [proposal_pair.copy_with_union() for proposal_pair in proposal_pairs]
        x_union = self.pooler(x, proposals_union)

        subject_mask, object_mask, background_mask = self._graph_mask(proposal_pairs, proposals_union) # Nx1x14x14
        x_subject = x_union * subject_mask.to(x_union.device) # Nx1024x14x14
        x_object = x_union * object_mask.to(x_union.device) # Nx1024x14x14
        x_background = x_union * background_mask.to(x_union.device) # Nx1024x14x14

        subj_att, obj_att, bg_att = self.instance_att_1024(x_subject, x_object, x_background) # Nx1024x14x14
        x_att = subj_att + obj_att + bg_att # Nx1024x14x14

        x = self.head(x_att).squeeze() # Nx1024x14x14 -> Nx2048x7x7 -> Nx1024x1x1 -> Nx1024
        x = self.embed(x) # Nx1024 -> Nx1024
        return x # Nx1024

    def _graph_mask_interact(self, x, proposal_pairs):
        proposals_union = [proposal_pair.copy_with_union() for proposal_pair in proposal_pairs]
        x_union = self.pooler(x, proposals_union)

        subject_mask, object_mask, background_mask = self._graph_mask(proposal_pairs, proposals_union) # Nx1x14x14
        x_subject = x_union * subject_mask.to(x_union.device) # Nx1024x14x14
        x_object = x_union * object_mask.to(x_union.device) # Nx1024x14x14
        x_background = x_union * background_mask.to(x_union.device) # Nx1024x14x14

        x_subject, x_object, x_background = self.instance_att_1024(x_subject, x_object, x_background)
        subj_conv, obj_conv, bg_conv = self.instance_conv(x_subject, x_object, x_background) # Nx2048x7x7
        subj_att, obj_att, bg_att = self.instance_att_2048(subj_conv, obj_conv, bg_conv) # Nx2048x7x7
        subj_att = self.subj_conv(subj_att).squeeze() # Nx1024
        obj_att = self.obj_conv(obj_att).squeeze() # Nx1024
        bg_att = self.bg_conv(bg_att).squeeze() # Nx1024
        subj_emb, obj_emb, bg_emb = self.instance_embedding(subj_att, obj_att, bg_att) # Nx1024

        # subj_att, obj_att, bg_att = self.instance_att(x_subject, x_object, x_background) # Nx1024x14x14
        # subj_conv, obj_conv, bg_conv = self.instance_conv(subj_att, obj_att, bg_att)
        # subj_emb, obj_emb, bg_emb = self.instance_embedding(subj_conv, obj_conv, bg_conv) # Nx1024, Nx1024, Nx1024
 
        x = self.rel_embedding(subj_emb, obj_emb, bg_emb) # Nx1024
        return x # Nx1024

    def _multi_class_position(self, x, proposals, proposal_pairs):
        labels_list = [proposal.get_field('labels') for proposal in proposals] # list[1xTensor(N)] (N: num_proposals)
        idx_pairs_list = [proposal_pair.get_field('idx_pairs') for proposal_pair in proposal_pairs] # list[1xTensor(Mx2)] (M: num_proposal_pairs)

        onehot_labels_list = []
        for labels, idx_pairs in zip(labels_list, idx_pairs_list):
            onehot_labels = []
            for idx_pair in idx_pairs:
                subj_idx, obj_idx = idx_pair
                subj_label = labels[subj_idx]; obj_label = labels[obj_idx]

                onehot_label = torch.zeros(151)
                onehot_label[subj_label] = 1; onehot_label[obj_label] = 1

                onehot_labels.append(onehot_label) # list[Tensor(151)]
            onehot_labels = torch.stack(onehot_labels, dim=0) # Tensor(Mx151)
            onehot_labels_list.append(onehot_labels) # list[1xTensor(Mx151)]

        proposals_union = [proposal_pair.copy_with_union() for proposal_pair in proposal_pairs]
        x_union = self.pooler(x, proposals_union) # Nx1024x14x14

        subject_mask, object_mask, background_mask = self._graph_mask(proposal_pairs, proposals_union, pool='max') # Nx1x14x14
        x_subject = x_union * subject_mask.to(x_union.device) # Nx1024x14x14
        x_object = x_union * object_mask.to(x_union.device) # Nx1024x14x14
        x_background = x_union * background_mask.to(x_union.device) # Nx1024x14x14

        # target_pos_mask = subject_mask + object_mask # Nx1x14x14
        # target_pos_mask[target_pos_mask > 1] = 1 # Nx1x14x14
        # output_pos_mask = self.position_encoder(x_union) # Nx1024x14x14 -> Nx1x14x14
        # loss_pos = F.binary_cross_entropy(output_pos_mask, target_pos_mask.to(output_pos_mask))

        subj_conv, obj_conv, bg_conv = self.instance_conv(x_subject, x_object, x_background) # Nx1024x14x14 -> Nx1024x7x7
        subj_att, obj_att, bg_att = self.instance_att_1024(subj_conv, obj_conv, bg_conv) # Nx1024x7x7 -> Nx1024x7x7

        subj_att = self.subj_conv(subj_att).squeeze() # Nx1024x7x7 -> Nx1024x1x1 -> Nx1024
        obj_att = self.obj_conv(obj_att).squeeze() # Nx1024x7x7 -> Nx1024x1x1 -> Nx1024
        bg_att = self.bg_conv(bg_att).squeeze() # Nx1024x7x7 -> Nx1024x1x1 -> Nx1024
        subj_emb, obj_emb, bg_emb = self.instance_embedding(subj_att, obj_att, bg_att) # Nx1024, Nx1024, Nx1024
        ''''''
        output_subj_mask = self.position_encoder_v2(subj_emb).reshape_as(subject_mask) # Nx1024 -> Nx196 -> Nx1x14x14
        loss_subj_pos = F.binary_cross_entropy(output_subj_mask, subject_mask.to(output_subj_mask))

        output_obj_mask = self.position_encoder_v2(obj_emb).reshape_as(object_mask) # Nx1024 -> Nx196 -> Nx1x14x14
        loss_obj_pos = F.binary_cross_entropy(output_obj_mask, object_mask.to(output_obj_mask))

        output_bg_mask = self.position_encoder_v2(bg_emb).reshape_as(background_mask) # Nx1024 -> Nx196 -> Nx1x14x14
        loss_bg_pos = F.binary_cross_entropy(output_bg_mask, background_mask.to(output_bg_mask))

        loss_pos = loss_subj_pos + loss_obj_pos + loss_bg_pos
        
        target_class = onehot_labels_list[0] # Nx151
        x, loss_multi = self.rel_embedding(subj_emb, obj_emb, bg_emb, target_class) # Nx1024
        ''''''
        # x = self.rel_embedding(subj_emb, obj_emb, bg_emb) # Nx1024

        # target_class = onehot_labels_list[0] # Nx151
        # predict_class = self.union_box_encoder(x) # Nx1024 -> Nx151
        # loss_multi = F.binary_cross_entropy(predict_class, target_class.to(predict_class))

        return x, loss_multi, loss_pos # Nx1024

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
        # x = self._union_box_feats(x, proposal_pairs)
        # x = self._graph_mask_att(x, proposal_pairs)
        # x = self._graph_mask_interact(x, proposal_pairs)
        x, loss_multi, loss_pos = self._multi_class_position(x, proposals, proposal_pairs)
        return x, rel_inds, loss_multi, loss_pos

def make_roi_relation_feature_extractor(cfg, in_channels):
    func = registry.ROI_RELATION_FEATURE_EXTRACTORS[
        cfg.MODEL.ROI_RELATION_HEAD.FEATURE_EXTRACTOR
    ]
    return func(cfg, in_channels)
