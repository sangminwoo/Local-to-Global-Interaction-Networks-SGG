import torch
import torch.nn as nn
import torch.nn.functional as F
from maskrcnn_benchmark.modeling.utils import cat
from .modules_attention import MultiHeadAttention

class CUT(nn.Module):
    def __init__(self, cfg, embed_dim):
        super(CUT, self).__init__()
        self.cfg = cfg
        self.obj_classes = cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.subj_embedding = nn.Sequential(
            nn.Linear(self.obj_classes+6, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.obj_embedding = nn.Sequential(
            nn.Linear(self.obj_classes+6, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim)
        )

        self.subj_att = MultiHeadAttention(8, embed_dim)
        self.obj_att = MultiHeadAttention(8, embed_dim)

        self.cut_loss = nn.BCEWithLogitsLoss()

    def get_target_binarys(self, targets):
        rel_binarys = []
        for target in targets:
            device = target.bbox.device
            num_prp = target.bbox.shape[0]

            tgt_rel_matrix = target.get_field("relation") # [tgt, tgt]
            tgt_pair_idxs = torch.nonzero(tgt_rel_matrix > 0)
            assert tgt_pair_idxs.shape[1] == 2
            tgt_head_idxs = tgt_pair_idxs[:, 0].contiguous().view(-1)
            tgt_tail_idxs = tgt_pair_idxs[:, 1].contiguous().view(-1)

            binary_rel = torch.zeros((num_prp, num_prp), device=device)
            binary_rel[tgt_head_idxs, tgt_tail_idxs] = 1
            rel_binarys.append(binary_rel)

        return rel_binarys

    def encode_bbox(self, bboxes, im_size):
        bboxes_enc = bboxes.clone()
        width, height = im_size

        dim0 = bboxes_enc[:, 0] / width
        dim1 = bboxes_enc[:, 1] / height
        dim2 = bboxes_enc[:, 2] / width
        dim3 = bboxes_enc[:, 3] / height
        dim4 = (bboxes_enc[:, 2] - bboxes_enc[:, 0]) * (bboxes_enc[:, 3] - bboxes_enc[:, 1]) / height / width
        dim5 = (bboxes_enc[:, 3] - bboxes_enc[:, 1]) / (bboxes_enc[:, 2] - bboxes_enc[:, 0] + 1)

        return torch.stack((dim0, dim1, dim2, dim3, dim4, dim5), dim=1)

    def calc_relevance(self, logit, bbox):
        features = torch.cat((logit, bbox), dim=1)

        subj_emb = self.subj_embedding(features)
        subj_emb = self.subj_att(subj_emb, subj_emb, subj_emb).squeeze(1)
        obj_emb = self.obj_embedding(features)
        obj_emb = self.obj_att(obj_emb, obj_emb, obj_emb).squeeze(1)
        relevance = torch.mm(subj_emb, obj_emb.t())
        return relevance

    def forward(self, proposals, targets, num_pair_proposals=64):
        if self.training:
            rel_binarys = self.get_target_binarys(targets)
            tgt_rel_matrices = [target.get_field("relation") for target in targets]

            # semantics
            if self.mode == 'predcls':
                obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                obj_logits = torch.eye(self.obj_classes, device=obj_labels.device)[obj_labels]
                obj_logits[obj_logits==0] = -1
            else:
                obj_logits = torch.cat([proposal.get_field("predict_logits") for proposal in proposals], 0) # 20x151

            # spatial
            bboxes = torch.cat([self.encode_bbox(proposal.bbox, proposal.size) for proposal in proposals], 0)

            num_objs = [len(proposal) for proposal in proposals]
            obj_logits = obj_logits.split(num_objs, dim=0)
            bboxes = bboxes.split(num_objs, dim=0)

            rel_labels = []
            rel_pair_idxs = []
            cut_losses = 0
            for obj_logit, bbox, rel_binary, tgt_rel_matrix in zip(obj_logits, bboxes, rel_binarys, tgt_rel_matrices):
                relevance = self.calc_relevance(obj_logit, bbox)
                cut_losses += self.cut_loss(relevance, rel_binary)
                sorted_idxs = torch.sort(relevance.contiguous().view(-1), descending=True)[1]
                rel_pair_idx = torch.cat([(sorted_idxs//obj_logit.shape[0]).contiguous().view(-1,1),
                                          (sorted_idxs%obj_logit.shape[0]).contiguous().view(-1,1)], dim=1)
                # leave top pairs
                rel_pair_idx = rel_pair_idx[:num_pair_proposals]
                rel_pair_idxs.append(rel_pair_idx)
                rel_labels.append(tgt_rel_matrix[rel_pair_idx[:,0], rel_pair_idx[:,1]])

            return rel_labels, rel_pair_idxs, cut_losses

        else: # test mode
            with torch.no_grad():
                if self.mode == 'predcls':
                    obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                    obj_logits = torch.eye(self.obj_classes, device=obj_labels.device)[obj_labels]
                    obj_logits[obj_logits==0] = -1
                else:
                    obj_logits = torch.cat([proposal.get_field("predict_logits") for proposal in proposals], 0) # 20x151

                # spatial
                bboxes = torch.cat([self.encode_bbox(proposal.bbox, proposal.size) for proposal in proposals], 0)

                num_objs = [len(proposal) for proposal in proposals]
                obj_logits = obj_logits.split(num_objs, dim=0)
                bboxes = bboxes.split(num_objs, dim=0)

                rel_pair_idxs = []
                for obj_logit, bbox in zip(obj_logits, bboxes):
                    relevance = self.calc_relevance(obj_logit, bbox)
                    sorted_idxs = torch.sort(relevance.contiguous().view(-1), descending=True)[1]
                    rel_pair_idx = torch.cat([(sorted_idxs//obj_logit.shape[0]).contiguous().view(-1,1),
                                              (sorted_idxs%obj_logit.shape[0]).contiguous().view(-1,1)], dim=1)
                    # leave top pairs
                    rel_pair_idx = rel_pair_idx[:num_pair_proposals]
                    rel_pair_idxs.append(rel_pair_idx)

                return rel_pair_idxs

def make_relation_pruner(cfg, embed_dim):
    cut = CUT(cfg, embed_dim)
    return cut
