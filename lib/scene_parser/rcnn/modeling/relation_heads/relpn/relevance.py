import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import box_pos_encoder
from .multi_head_att import MultiHeadAttention

class Relevance(nn.Module):
    """
    calculate relavance score between subjects and objects
    """
    def __init__(self, vis_dim, num_classes, embed_dim=64):
        super(Relevance, self).__init__()

        self.vis_subj = nn.Sequential(
            nn.Linear(vis_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.vis_obj = nn.Sequential(
            nn.Linear(vis_dim, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim)
        )

        # self.word_subj = nn.Sequential(
        #     nn.Linear(word_dim, embed_dim),
        #     nn.ReLU(True),
        #     nn.Linear(embed_dim, embed_dim)
        # )
        # self.word_obj = nn.Sequential(
        #     nn.Linear(word_dim, embed_dim),
        #     nn.ReLU(True),
        #     nn.Linear(embed_dim, embed_dim)
        # )

        self.logit_subj = nn.Sequential(
            nn.Linear(num_classes, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.logit_obj = nn.Sequential(
            nn.Linear(num_classes, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim)
        )

        self.pos_subj = nn.Sequential(
            nn.Linear(6, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim)
        )
        self.pos_obj = nn.Sequential(
            nn.Linear(6, embed_dim),
            nn.ReLU(True),
            nn.Linear(embed_dim, embed_dim)
        )

        self.vis_att_subj = MultiHeadAttention(8, embed_dim)
        self.vis_att_obj = MultiHeadAttention(8, embed_dim)

        # self.word_att_subj = MultiHeadAttention(8, embed_dim)
        # self.word_att_obj = MultiHeadAttention(8, embed_dim)

        self.logit_att_subj = MultiHeadAttention(8, embed_dim)
        self.logit_att_obj = MultiHeadAttention(8, embed_dim)

        self.pos_att_subj = MultiHeadAttention(8, embed_dim)
        self.pos_att_obj = MultiHeadAttention(8, embed_dim)

    def forward(self, vis_feats, logits, bbox=None, imsize=None):
        # vis_subj = self.vis_subj(vis_feats) # kx128
        # vis_subj = self.vis_att_subj(vis_subj, vis_subj, vis_subj).squeeze(1)
        # vis_obj = self.vis_obj(vis_feats) # kx128
        # vis_obj = self.vis_att_obj(vis_obj, vis_obj, vis_obj).squeeze(1)
        # norm_vs = torch.sqrt(torch.sum(torch.mul(vis_subj, vis_subj), dim=1, keepdim=True))
        # norm_vo = torch.sqrt(torch.sum(torch.mul(vis_obj, vis_obj), dim=1, keepdim=True))
        # vis_subj /= norm_vs; vis_obj /= norm_vo
        # vis_scores = torch.mm(vis_subj, vis_obj.t()) # k x k
        # # vis_scores = torch.sigmoid(vis_scores)

        # word_subj = self.word_subj(word_feats) # kx128
        # word_subj = self.word_att_subj(word_subj, word_subj, word_subj).squeeze(1)
        # word_obj = self.word_obj(word_feats) # kx128
        # word_obj = self.word_att_obj(word_obj, word_obj, word_obj).squeeze(1)
        # word_scores = torch.mm(word_subj, word_obj.t()) # kxk
        # # word_scores = torch.sigmoid(word_scores)

        logits_subj = self.logit_subj(logits) # k x 128
        logits_subj = self.logit_att_subj(logits_subj, logits_subj, logits_subj).squeeze(1)
        logits_obj = self.logit_obj(logits)   # k x 128
        logits_obj = self.logit_att_obj(logits_obj, logits_obj, logits_obj).squeeze(1)
        norm_ls = torch.sqrt(torch.sum(torch.mul(logits_subj, logits_subj), dim=1, keepdim=True))
        norm_lo = torch.sqrt(torch.sum(torch.mul(logits_obj, logits_obj), dim=1, keepdim=True))
        logits_subj = logits_subj / norm_ls; logits_obj = logits_obj / norm_lo
        logits_scores = torch.mm(logits_subj, logits_obj.t()) # k x k
        # logits_scores = torch.sigmoid(logits_scores)

        pos = box_pos_encoder(bbox, imsize[0], imsize[1])
        pos_subj = self.pos_subj(pos) # k x 128
        pos_subj = self.pos_att_subj(pos_subj, pos_subj, pos_subj).squeeze(1)
        pos_obj = self.pos_obj(pos) # k x 128
        pos_obj = self.pos_att_obj(pos_obj, pos_obj, pos_obj).squeeze(1)
        norm_ps = torch.sqrt(torch.sum(torch.mul(pos_subj, pos_subj), dim=1, keepdim=True))
        norm_po = torch.sqrt(torch.sum(torch.mul(pos_obj, pos_obj), dim=1, keepdim=True))
        pos_subj = pos_subj / norm_ps; pos_obj = pos_obj / norm_po
        pos_scores = torch.mm(pos_subj, pos_obj.t()) # k x k
        # pos_scores = torch.sigmoid(pos_scores)

        scores = logits_scores + pos_scores
        # scores = vis_scores + logits_scores + pos_scores
        # scores = vis_scores + word_scores + logits_scores + pos_scores
        relness = torch.sigmoid(scores)
        
        return relness # k x k