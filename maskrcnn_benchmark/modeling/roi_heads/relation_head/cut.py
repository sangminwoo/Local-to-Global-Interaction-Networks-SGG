import torch
import torch.nn as nn
import torch.nn.functional as F
from maskrcnn_benchmark.modeling.utils import cat
from .utils_csinet import MultiHeadAttention

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
            nn.Linear(self.obj_classes, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.obj_embedding = nn.Sequential(
            nn.Linear(self.obj_classes, embed_dim),
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

    def calc_relevance(self, logits):
        subj_emb = self.subj_embedding(logits)
        subj_emb = self.subj_att(subj_emb, subj_emb, subj_emb).squeeze(1)
        obj_emb = self.obj_embedding(logits)
        obj_emb = self.obj_att(obj_emb, obj_emb, obj_emb).squeeze(1)

        relevance = torch.mm(subj_emb, obj_emb.t())
        return relevance

    def forward(self, proposals, targets, num_pair_proposals=64):
        if self.training:
            rel_binarys = self.get_target_binarys(targets)
            tgt_rel_matrices = [target.get_field("relation") for target in targets]

            if self.mode == 'predcls':
                obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                obj_logits = torch.eye(self.obj_classes, device=obj_labels.device)[obj_labels]
            else:
                obj_logits = torch.cat([proposal.get_field("predict_logits") for proposal in proposals], 0) # 20x151

            num_objs = [len(proposal) for proposal in proposals]
            obj_logits = obj_logits.split(num_objs, dim=0)

            rel_labels = []
            rel_pair_idxs = []
            cut_losses = 0
            for obj_logit, rel_binary, tgt_rel_matrix in zip(obj_logits, rel_binarys, tgt_rel_matrices):
                relevance = self.calc_relevance(obj_logit)
                top_idxs = torch.sort(relevance.contiguous().view(-1), descending=True)[1][:num_pair_proposals]
                rel_pair_idx = torch.cat([(top_idxs//obj_logit.shape[0]).contiguous().view(-1,1),
                                          (top_idxs%obj_logit.shape[0]).contiguous().view(-1,1)], dim=1)
                rel_pair_idxs.append(rel_pair_idx)
                rel_labels.append(tgt_rel_matrix[rel_pair_idx[:,0], rel_pair_idx[:,1]])
                cut_losses += self.cut_loss(relevance, rel_binary)

            return rel_labels, rel_pair_idxs, cut_losses

        else: # test mode
            with torch.no_grad():
                if self.mode == 'predcls':
                    obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0)
                    obj_logits = torch.eye(self.obj_classes, device=obj_labels.device)[obj_labels]
                else:
                    obj_logits = torch.cat([proposal.get_field("predict_logits") for proposal in proposals], 0) # 20x151

                num_objs = [len(proposal) for proposal in proposals]
                obj_logits = obj_logits.split(num_objs, dim=0)

                rel_pair_idxs = []
                for obj_logit in obj_logits:
                    relevance = self.calc_relevance(obj_logit)
                    top_idxs = torch.sort(relevance.contiguous().view(-1), descending=True)[1][:num_pair_proposals]
                    rel_pair_idx = torch.cat([(top_idxs//obj_logit.shape[0]).contiguous().view(-1,1),
                                              (top_idxs%obj_logit.shape[0]).contiguous().view(-1,1)], dim=1)
                    rel_pair_idxs.append(rel_pair_idx)

                return rel_pair_idxs

def make_relation_pruner(cfg, embed_dim):
    cut = CUT(cfg, embed_dim)
    return cut