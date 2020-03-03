import torch 
import torch.nn as nn
import torch.nn.functional as F
from lib.scene_parser.rcnn.modeling.box_coder import BoxCoder
from lib.scene_parser.rcnn.modeling.matcher import Matcher
from lib.scene_parser.rcnn.modeling.pair_matcher import PairMatcher
from lib.scene_parser.rcnn.structures.boxlist_ops import boxlist_iou
from lib.scene_parser.rcnn.structures.bounding_box_pair import BoxPairList
from lib.scene_parser.rcnn.modeling.balanced_positive_negative_pair_sampler import (
    BalancedPositiveNegativePairSampler
)
from lib.scene_parser.rcnn.modeling.utils import cat
from .relationshipness import Relationshipness

class RelPN(nn.Module):
    def __init__(
        self,
        cfg,
        proposal_pair_matcher,
        fg_bg_pair_sampler,
        box_coder,
        cls_agnostic_bbox_reg=False,
        use_matched_pairs_only=False,
        minimal_matched_pairs=0,
    ):
        super(RelPN, self).__init__()
        self.cfg = cfg
        self.proposal_pair_matcher = proposal_pair_matcher
        self.fg_bg_pair_sampler = fg_bg_pair_sampler
        self.box_coder = box_coder
        self.cls_agnostic_bbox_reg = cls_agnostic_bbox_reg
        self.use_matched_pairs_only = use_matched_pairs_only
        self.minimal_matched_pairs = minimal_matched_pairs
        self.relationshipness = Relationshipness(
                                    vis_dim=2048,
                                    num_classes=self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES,
                                    pos_encoding=True
                                )

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        temp = []
        target_box_pairs = []
        for i in range(match_quality_matrix.shape[0]):
            for j in range(match_quality_matrix.shape[0]):
                match_i = match_quality_matrix[i].view(-1, 1)
                match_j = match_quality_matrix[j].view(1, -1)
                match_ij = ((match_i + match_j) / 2)
                # remove duplicate index
                non_duplicate_idx = (torch.eye(match_ij.shape[0]).view(-1) == 0).nonzero().view(-1).to(match_ij.device)
                match_ij = match_ij.view(-1) # [::match_quality_matrix.shape[1]] = 0
                match_ij = match_ij[non_duplicate_idx]
                temp.append(match_ij)
                boxi = target.bbox[i]; boxj = target.bbox[j]
                box_pair = torch.cat((boxi, boxj), 0)
                target_box_pairs.append(box_pair)

        match_pair_quality_matrix = torch.stack(temp, 0).view(len(temp), -1)
        target_box_pairs = torch.stack(target_box_pairs, 0)
        target_pair = BoxPairList(target_box_pairs, target.size, target.mode)
        target_pair.add_field("labels", target.get_field("pred_labels").view(-1))

        box_subj = proposal.bbox
        box_obj = proposal.bbox
        box_subj = box_subj.unsqueeze(1).repeat(1, box_subj.shape[0], 1)
        box_obj = box_obj.unsqueeze(0).repeat(box_obj.shape[0], 1, 1)
        proposal_box_pairs = torch.cat((box_subj.view(-1, 4), box_obj.view(-1, 4)), 1)

        idx_subj = torch.arange(box_subj.shape[0]).view(-1, 1, 1).repeat(1, box_obj.shape[0], 1).to(proposal.bbox.device)
        idx_obj = torch.arange(box_obj.shape[0]).view(1, -1, 1).repeat(box_subj.shape[0], 1, 1).to(proposal.bbox.device)
        proposal_idx_pairs = torch.cat((idx_subj.view(-1, 1), idx_obj.view(-1, 1)), 1)

        non_duplicate_idx = (proposal_idx_pairs[:, 0] != proposal_idx_pairs[:, 1]).nonzero()
        proposal_box_pairs = proposal_box_pairs[non_duplicate_idx.view(-1)]
        proposal_idx_pairs = proposal_idx_pairs[non_duplicate_idx.view(-1)]
        proposal_pairs = BoxPairList(proposal_box_pairs, proposal.size, proposal.mode)
        proposal_pairs.add_field("idx_pairs", proposal_idx_pairs)

        # matched_idxs = self.proposal_pair_matcher(match_quality_matrix)
        matched_idxs = self.proposal_pair_matcher(match_pair_quality_matrix)

        # Fast RCNN only need "labels" field for selecting the targets
        # target = target.copy_with_fields("pred_labels")
        # get the targets corresponding GT for each proposal
        # NB: need to clamp the indices because we can have a single
        # GT in the image, and matched_idxs can be -2, which goes
        # out of bounds

        if self.use_matched_pairs_only and \
            (matched_idxs >= 0).sum() > self.minimal_matched_pairs:
            # filter all matched_idxs < 0
            proposal_pairs = proposal_pairs[matched_idxs >= 0]
            matched_idxs = matched_idxs[matched_idxs >= 0]

        matched_targets = target_pair[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets, proposal_pairs

    def prepare_targets(self, proposals, targets):
        '''
        proposals
        [BoxList(num_boxes=36, image_width=1024, image_height=681, mode=xyxy),
        BoxList(num_boxes=21, image_width=1024, image_height=768, mode=xyxy),
        BoxList(num_boxes=56, image_width=1024, image_height=679, mode=xyxy),
        BoxList(num_boxes=20, image_width=1024, image_height=681, mode=xyxy)]

        targets
        [BoxList(num_boxes=17, image_width=1024, image_height=681, mode=xyxy),
        BoxList(num_boxes=20, image_width=1024, image_height=768, mode=xyxy),
        BoxList(num_boxes=11, image_width=1024, image_height=679, mode=xyxy),
        BoxList(num_boxes=2, image_width=1024, image_height=681, mode=xyxy)]
        '''
        labels = []
        proposal_pairs = []
        for proposals_per_image, targets_per_image in zip(proposals, targets): # number of images: M
            matched_targets, proposal_pairs_per_image = \
                self.match_targets_to_proposals(proposals_per_image, targets_per_image)

            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            # Label background (below the low threshold)
            bg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_inds] = 0

            # Label ignore proposals (between low and high thresholds)
            ignore_inds = matched_idxs == Matcher.BETWEEN_THRESHOLDS
            labels_per_image[ignore_inds] = -1  # -1 is ignored by sampler

            labels.append(labels_per_image)
            proposal_pairs.append(proposal_pairs_per_image)

        return labels, proposal_pairs


    def _relpnsample_train(self, proposals, targets):
        """
        perform relpn based sampling during training
        """
        labels, proposal_pairs = self.prepare_targets(proposals, targets)
        '''
        proposal_pairs
        [BoxPairList(num_boxes=1260, image_width=1024, image_height=681, mode=xyxy), 36*35=1260
        BoxPairList(num_boxes=420, image_width=1024, image_height=768, mode=xyxy), 21*20=420
        BoxPairList(num_boxes=3080, image_width=1024, image_height=679, mode=xyxy), 56*55=3080
        BoxPairList(num_boxes=380, image_width=1024, image_height=681, mode=xyxy)] 20*19=380
        '''

        # add corresponding label and regression_targets information to the bounding boxes
        for labels_per_image, proposal_pairs_per_image in zip(labels, proposal_pairs):
            proposal_pairs_per_image.add_field("labels", labels_per_image)

        losses = 0
        for img_idx, proposals_per_image in enumerate(proposals):
            obj_features = proposals_per_image.get_field('features') # Nx2048x1x1
            obj_logits = proposals_per_image.get_field('logits') # Nx151
            obj_bboxes = proposals_per_image.bbox # Nx4

            relness = self.relationshipness(obj_features, obj_logits, obj_bboxes, proposals_per_image.size) # Tensor(256x256)
            nondiag = (1 - torch.eye(obj_logits.shape[0]).to(relness.device)).view(-1) # Tensor(65536)
            relness = relness.view(-1)[nondiag.nonzero()] # Tensor(65280, 1)
            relness_sorted, order = torch.sort(relness.view(-1), descending=True) # Tensor(65280), Tensor(65280)
            img_sampled_inds = order[:self.cfg.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE].view(-1) # # Tensor(256)
            proposal_pairs_per_image = proposal_pairs[img_idx][img_sampled_inds]
            proposal_pairs[img_idx] = proposal_pairs_per_image

            losses += F.binary_cross_entropy(relness, (labels[img_idx] > 0).view(-1, 1).float())

        self._proposal_pairs = proposal_pairs

        return proposal_pairs, losses

    def _fullsample_test(self, proposals):
        """
        This method get all subject-object pairs, and return the proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])

        Returns:
            proposal_pairs (list[BoxPairList])
        """
        proposal_pairs = []
        for i, proposals_per_image in enumerate(proposals): # num of images = M; num of proposals per image = N 
            box_subj = proposals_per_image.bbox # Nx4
            box_obj = proposals_per_image.bbox # Nx4

            box_subj = box_subj.unsqueeze(1).repeat(1, box_subj.shape[0], 1) # Nx1x4 -> NxNx4
            box_obj = box_obj.unsqueeze(0).repeat(box_obj.shape[0], 1, 1) # 1xNx4 -> NxNx4
            proposal_box_pairs = torch.cat((box_subj.view(-1, 4), box_obj.view(-1, 4)), 1) # N^2x4 + N^2x4 -> N^2x8

            idx_subj = torch.arange(box_subj.shape[0]).view(-1, 1, 1).repeat(1, box_obj.shape[0], 1).to(proposals_per_image.bbox.device) # Nx1x1 -> NxNx1
            idx_obj = torch.arange(box_obj.shape[0]).view(1, -1, 1).repeat(box_subj.shape[0], 1, 1).to(proposals_per_image.bbox.device) # 1xNx1 -> NxNx1
            proposal_idx_pairs = torch.cat((idx_subj.view(-1, 1), idx_obj.view(-1, 1)), 1) # N^2x1 + N^2x1 -> N^2x2

            keep_idx = (proposal_idx_pairs[:, 0] != proposal_idx_pairs[:, 1]).nonzero().view(-1) # Kx1 -> K (remove 00, 11, ...)

            # if we filter non overlap bounding boxes
            if self.cfg.MODEL.ROI_RELATION_HEAD.FILTER_NON_OVERLAP: # True
                ious = boxlist_iou(proposals_per_image, proposals_per_image).view(-1) # NxN -> N^2
                ious = ious[keep_idx] # K
                keep_idx = keep_idx[(ious > 0).nonzero().view(-1)] # K -> k(remove negative ious)
            proposal_idx_pairs = proposal_idx_pairs[keep_idx] # N^2x2 -> kx2 
            proposal_box_pairs = proposal_box_pairs[keep_idx] # N^2x8 -> kx8
            proposal_pairs_per_image = BoxPairList(proposal_box_pairs, proposals_per_image.size, proposals_per_image.mode) # kx8, (1024, 1024), xyxy
            proposal_pairs_per_image.add_field("idx_pairs", proposal_idx_pairs) # + kx2

            proposal_pairs.append(proposal_pairs_per_image) # list(BoxPairList[kx8, (1024x1024), 'xyxy', kx2])
        return proposal_pairs # list(M x BoxPairList)

    def _relpnsample_test(self, proposals):
        """
        perform relpn based sampling during testing
        """
        proposals[0] = proposals[0]
        proposal_pairs = self._fullsample_test(proposals)
        proposal_pairs = list(proposal_pairs)

        relnesses = []
        for img_idx, proposals_per_image in enumerate(proposals):
            obj_features = proposals_per_image.get_field('features') # Nx2048x1x1
            obj_logits = proposals_per_image.get_field('logits') # Nx151
            obj_bboxes = proposals_per_image.bbox # Nx4
            relness = self.relationshipness(obj_features, obj_logits, obj_bboxes, proposals_per_image.size)
            nondiag = (1 - torch.eye(obj_logits.shape[0]).to(relness.device)).view(-1)
            relness = relness.view(-1)[nondiag.nonzero()]
            relness_sorted, order = torch.sort(relness.view(-1), descending=True)
            img_sampled_inds = order[:self.cfg.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE].view(-1)
            relness = relness_sorted[:self.cfg.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE].view(-1)
            proposal_pairs_per_image = proposal_pairs[img_idx][img_sampled_inds]
            proposal_pairs[img_idx] = proposal_pairs_per_image
            relnesses.append(relness)

        self._proposal_pairs = proposal_pairs

        return proposal_pairs, relnesses

    def forward(self, proposals, targets=None):
        """
        This method performs the positive/negative sampling, and return
        the sampled proposals.
        Note: this function keeps a state.

        Arguments:
            proposals (list[BoxList])
            targets (list[BoxList])
        """
        if self.training:
            return self._relpnsample_train(proposals, targets)
        else:
            return self._relpnsample_test(proposals)

    def pred_classification_loss(self, class_logits):
        """
        Computes the loss for Faster R-CNN.
        This requires that the subsample method has been called beforehand.

        Arguments:
            class_logits (list[Tensor])

        Returns:
            classification_loss (Tensor)
        """
        class_logits = cat(class_logits, dim=0)
        device = class_logits.device

        if not hasattr(self, "_proposal_pairs"):
            raise RuntimeError("subsample needs to be called before")

        proposals = self._proposal_pairs

        labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)

        rel_fg_cnt = len(labels.nonzero())
        rel_bg_cnt = labels.shape[0] - rel_fg_cnt
        ce_weights = labels.new(class_logits.size(1)).fill_(1).float()
        ce_weights[0] = float(rel_fg_cnt) / (rel_bg_cnt + 1e-5)
        classification_loss = F.cross_entropy(class_logits, labels, weight=ce_weights)

        return classification_loss


def make_relation_proposal_network(cfg):
    proposal_pair_matcher = PairMatcher(
        cfg.MODEL.ROI_HEADS.FG_IOU_THRESHOLD, # class=?(foreground) if overlap >= 0.5
        cfg.MODEL.ROI_HEADS.BG_IOU_THRESHOLD, # class=0(background) if overlap in [0, 0.5)
        allow_low_quality_matches=False,
    )
    # weights on (dx, dy, dw, dh) for normalizing bbox regression targets
    bbox_reg_weights = cfg.MODEL.ROI_HEADS.BBOX_REG_WEIGHTS
    box_coder = BoxCoder(weights=bbox_reg_weights)

    fg_bg_pair_sampler = BalancedPositiveNegativePairSampler(
        cfg.MODEL.ROI_RELATION_HEAD.BATCH_SIZE_PER_IMAGE, # 512
        cfg.MODEL.ROI_RELATION_HEAD.POSITIVE_FRACTION # 0.25
    )

    cls_agnostic_bbox_reg = cfg.MODEL.CLS_AGNOSTIC_BBOX_REG # True / False

    relpn = RelPN(
        cfg,
        proposal_pair_matcher,
        fg_bg_pair_sampler,
        box_coder,
        cls_agnostic_bbox_reg
    )
    return relpn
