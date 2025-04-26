import torch
import os
import os.path as osp
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import random
from sklearn.cluster import KMeans
from utils.utils import load_pkl


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    def _gen_gumbels():
        gumbels = -torch.empty_like(logits).exponential_().log()
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
            gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret
class Linear_CLS(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_func=None,
                 **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        self.init_weights()

        self.loss_func = loss_func

    
    def forward(self, x, gt_label=None, mode="train"):
        out_cls = self.fc_cls(x)

        if mode == "train":
            loss = self.loss_func(out_cls, gt_label)
            return out_cls, loss
        else:
            return out_cls, None

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

class End2End_IP(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 concept_path,
                 num_concepts=200,
                 loss_func=None,
                 sparsity_ratio=0,
                 **kwargs):
        super().__init__()
        
        self.temperature = 0.1
        self.in_channels = in_channels
        self.num_classes = num_classes
        print(f'load concept vectors from {concept_path}')
        concept_matrix = load_pkl(concept_path)
        concept_matrix = F.normalize(concept_matrix, dim=-1).cpu()[:20000]

        self.selected_concept = nn.Parameter(F.normalize(torch.Tensor(concept_matrix), dim=-1), requires_grad=False)
        self.num_concepts = num_concepts
        self.num_elements = self.selected_concept.shape[0]
        self.fc_cls = nn.Linear(self.num_concepts, self.num_classes)

        self.q = nn.Parameter(torch.rand(self.num_elements, self.num_concepts), requires_grad=True)
        
        self.q_norm = nn.LayerNorm(self.num_concepts)
        self.feat_norm = nn.LayerNorm(self.num_concepts)

        
        self.feat_idx = int(self.num_concepts * sparsity_ratio)

        self.init_weights()
        trunc_normal_(self.q, std=0.02)

        self.loss_func = loss_func

    def l2_normalize(self, x):
        return F.normalize(x, p=2, dim=-1)

    def soft_threshold(self, feat, thres):
        abs_feat = torch.abs(feat)
        sub = abs_feat - thres
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        one_sub = ch.ones_like(n_sub)
        n_sub = ch.where(n_sub != 0, one_sub, n_sub)
        feat = feat * n_sub
        return feat

    def forward(self, x, gt_label=None, mode="train"):
        """Forward function for both train and test."""
        # get higher dimension if comment out, no dimension expanded
        temperature = 0.1
        q = self.q_norm(self.q)

        # get thres soft-thres
        q = gumbel_softmax(q / temperature, dim=-1, hard=True)
        prototypes = q.T.matmul(self.selected_concept)
        prototypes = self.l2_normalize(prototypes)

        # bs * self.num_subcentroids
        concept_feat = x.matmul(prototypes.T)
        concept_feat = self.feat_norm(concept_feat)
        value_feat, _ = torch.sort(torch.abs(concept_feat.detach()), dim=-1)
        # bs
        feat_thres = value_feat[:, self.feat_idx].unsqueeze(-1)
        concept_feat = self.soft_threshold(concept_feat, feat_thres)
        out_cls = self.fc_cls(concept_feat)

        if mode == "train":
            loss = self.loss_func(out_cls, gt_label)
            return out_cls, loss
        else:
            return out_cls, None

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

class Prototype_IP(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 concept_path,
                 num_concepts=200,
                 loss_func=None,
                 sparsity_ratio=0,
                 **kwargs):
        super().__init__()
        
        self.temperature = 0.1
        self.in_channels = in_channels
        self.num_classes = num_classes
        print(f'load concept vectors from {concept_path}')
        concept_matrix = load_pkl(concept_path)
        concept_matrix = F.normalize(concept_matrix, dim=-1).cpu()[:num_concepts]

        self.selected_concept = nn.Parameter(F.normalize(torch.Tensor(concept_matrix), dim=-1), requires_grad=False)
        self.num_concepts = num_concepts
        self.feat_norm = nn.LayerNorm(self.num_concepts)
        self.fc_cls = nn.Linear(self.num_concepts, self.num_classes)

        self.feat_idx = int(self.num_concepts * sparsity_ratio)

        self.init_weights()

        self.loss_func = loss_func

    def l2_normalize(self, x):
        return F.normalize(x, p=2, dim=-1)

    def soft_threshold(self, feat, thres):
        abs_feat = torch.abs(feat)
        sub = abs_feat - thres
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        one_sub = ch.ones_like(n_sub)
        n_sub = ch.where(n_sub != 0, one_sub, n_sub)
        feat = feat * n_sub
        return feat

    def forward(self, x, gt_label=None, mode="train"):
        """Forward function for both train and test."""
        # get higher dimension if comment out, no dimension expanded
        temperature = 0.1

        prototypes = self.selected_concept
        prototypes = self.l2_normalize(prototypes)

        # bs * self.num_subcentroids
        concept_feat = x.matmul(prototypes.T)
        concept_feat = self.feat_norm(concept_feat)
        value_feat, _ = torch.sort(torch.abs(concept_feat.detach()), dim=-1)
        # bs
        feat_thres = value_feat[:, self.feat_idx].unsqueeze(-1)
        concept_feat = self.soft_threshold(concept_feat, feat_thres)
        out_cls = self.fc_cls(concept_feat)

        if mode == "train":
            loss = self.loss_func(out_cls, gt_label)
            return out_cls, loss
        else:
            return out_cls, None

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)


class Cluster_IP(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 concept_path,
                 num_concepts=200,
                 loss_func=None,
                 sparsity_ratio=0,
                 **kwargs):
        super().__init__()
        
        self.temperature = 0.1
        self.in_channels = in_channels
        self.num_classes = num_classes
        print(f'load concept vectors from {concept_path}')
        concept_matrix = load_pkl(concept_path)
        concept_matrix = F.normalize(concept_matrix, dim=-1).cpu()[:20000]

        cluster_result = KMeans(n_clusters=num_concepts, random_state=0, n_init="auto").fit(concept_matrix)
        prototypes = cluster_result.cluster_centers_
        self.selected_concept = nn.Parameter(F.normalize(torch.Tensor(prototypes), dim=-1), requires_grad=False)
        self.num_concepts = num_concepts
        self.fc_cls = nn.Linear(self.num_concepts, self.num_classes)
        self.feat_norm = nn.LayerNorm(self.num_concepts)

        self.feat_idx = int(self.num_concepts * sparsity_ratio)

        self.init_weights()

        self.loss_func = loss_func

    def l2_normalize(self, x):
        return F.normalize(x, p=2, dim=-1)

    def soft_threshold(self, feat, thres):
        abs_feat = torch.abs(feat)
        sub = abs_feat - thres
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        one_sub = ch.ones_like(n_sub)
        n_sub = ch.where(n_sub != 0, one_sub, n_sub)
        feat = feat * n_sub
        return feat

    def forward(self, x, gt_label=None, mode="train"):
        """Forward function for both train and test."""
        # get higher dimension if comment out, no dimension expanded

        prototypes = self.selected_concept
        prototypes = self.l2_normalize(prototypes)

        # bs * self.num_subcentroids
        concept_feat = x.matmul(prototypes.T)
        concept_feat = self.feat_norm(concept_feat)
        value_feat, _ = torch.sort(torch.abs(concept_feat.detach()), dim=-1)
        # bs
        feat_thres = value_feat[:, self.feat_idx].unsqueeze(-1)
        concept_feat = self.soft_threshold(concept_feat, feat_thres)
        out_cls = self.fc_cls(concept_feat)

        if mode == "train":
            loss = self.loss_func(out_cls, gt_label)
            return out_cls, loss
        else:
            return out_cls, None

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

