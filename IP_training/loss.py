import torch
import torch.nn as nn
import torch.nn.functional as F
def sparse_loss(feat):
    feat_prob = F.softmax(feat, dim=-1)
    feat_log_prob = F.log_softmax(feat, dim=-1)
    entropy_ = torch.sum(-(feat_prob * feat_log_prob), dim=-1)
    return torch.sum(entropy_)
    #return torch.norm(feat, p=1)

def orth_loss(feat):
    inner = feat.matmul(feat.T)
    return ((inner - torch.eye(feat.shape[0]).to(DEVICE))**2).sum()

def ce_criterion(y_pre, label):
    ce = nn.CrossEntropyLoss()
    return ce(y_pre, label)

def elastic_criterion(pred, label, model, alpha=0.5, beta=0.0001):
    ce = nn.CrossEntropyLoss()
    ce_loss = ce(pred, label)
    l1_loss, l2_loss = 0, 0
    for param in model.parameters():
        l1_loss += torch.norm(param, p=1)
        l2_loss += torch.norm(param, p=2)
    e_loss = alpha*l1_loss + (1-alpha)*l2_loss
    return ce_loss + beta*e_loss
    
def ce_sparse_criterion(y_pre, label, thres, feat, alpha=1e-1, beta=1e-1, gamma=1e-1):
    ce = nn.CrossEntropyLoss()
    performance_loss = ce(y_pre, label)
    if beta < 1e-6:
        thres_sparse_loss = 0
    else:
        thres_sparse_loss = sparse_loss(thres)
    if gamma < 1e-6:
        feat_sparse_loss = 0
    else:
        feat_sparse_loss = sparse_loss(feat)

    total_loss = alpha * performance_loss + beta * thres_sparse_loss + gamma * feat_sparse_loss
    return total_loss