import argparse
import os
import os.path as osp
import numpy as np
import torch
from tqdm import tqdm
from dataset import emb_dataset
from model import Prototype_IP, Linear_CLS
from loss import ce_criterion
from utils import build_optimizer
from torch.utils.data import DataLoader
from data_utils import get_targets_only

    
def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Grid Train Pipeline", add_help=add_help)

    parser.add_argument("--model_name", default='resnet50', type=str)
    parser.add_argument("--in_channels", default=2048, type=int)
    parser.add_argument("--concept_path", default='', type=str)
    parser.add_argument("--num_concepts", default=200, type=int)
    parser.add_argument("--num_classes", default=1000, type=int)
    parser.add_argument("--n_epoch", default=30, type=int)
    parser.add_argument("--batch_size", default=1024, type=int)
    parser.add_argument("--train_pt", default='', type=str)
    parser.add_argument("--val_pt", default='', type=str)
    parser.add_argument("--save_dir", default='', type=str)
    return parser.parse_args()


args = get_args_parser()
model_name = args.model_name
num_classes = args.num_classes
num_concepts = args.num_concepts
in_channels = args.in_channels
save_dir = args.save_dir
concept_path = args.concept_path

n_epoch = args.n_epoch
batch_size = args.batch_size

lrs = (1e-3, 1e-2, 1e-1)
sparsity_ratios = (0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.98)

optimizer_type = 'adam'
scheduler_type = 'exp'
device = 'cuda'


def top_k_accuracy(scores, labels, topk=(1, )):
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)
    return res

def train_model(train_dataloader, val_dataloader, scheduler, optimizer, n_epoch, model, device):
    best_loss, best_top1_acc, best_top5_acc = 9999, 0, 0
    for i in range(n_epoch):
        mean_loss = []
        model.train()
        for emb, label in train_dataloader:
            emb = emb.to(device)
            label = label.to(device)
            label = label.reshape(-1, 1).squeeze()
            optimizer.zero_grad()
            output, loss = model(emb, label, mode='train')
            loss.backward()
            mean_loss.append(loss.item())
            optimizer.step()
        scheduler.step()
        best_loss = min(sum(mean_loss)/len(mean_loss), best_loss)
        
        model.eval()
        all_output, all_label = [], []
        for emb, label in val_dataloader:
            emb = emb.to(device)
            label = label.to(device)
            label = label.reshape(-1, 1)
            output, _ = model(emb, mode='test')
            all_output.append(output.softmax(-1).detach().cpu())
            all_label.append(label.cpu())
        all_output = torch.cat(all_output, dim=0)
        all_label = torch.cat(all_label, dim=0)
        all_output = np.array(all_output)
        all_label = np.array(all_label).squeeze()
        top1_acc, top5_acc = top_k_accuracy(all_output, all_label, topk=(1,5))
        print('epoch {}, loss {}, top1_acc {}, top5_acc {}'.format(i, sum(mean_loss)/len(mean_loss), top1_acc, top5_acc))
        if best_top1_acc < top1_acc:
            best_top1_acc = top1_acc
            best_top5_acc = top5_acc

    return best_loss, best_top1_acc, best_top5_acc


# load dataset
# print(get_targets_only('imagenet_train'))
train_dataset = emb_dataset(args.train_pt, get_targets_only('imagenet_train'))
val_dataset = emb_dataset(args.val_pt, get_targets_only('imagenet_val'))

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

save_path = osp.join(save_dir)
if not osp.exists(save_path):
    os.makedirs(save_path)
trained_list = os.listdir(save_path)


for lr in lrs:
    model = Linear_CLS(in_channels=in_channels, num_classes=num_classes, loss_func=ce_criterion).to(device)
    # optimizer
    scheduler, optimizer = build_optimizer(optimizer_type, scheduler_type, model.parameters(), lr=lr)
    best_loss, best_top1_acc, best_top5_acc = train_model(train_dataloader, val_dataloader, scheduler, optimizer, n_epoch, model, device)
    print('loss:{}, top1_acc:{}, top5_acc:{}'.format(best_loss, best_top1_acc, best_top5_acc))

    model_filename = f"standard##{lr}##{best_loss}##{best_top1_acc}##{best_top5_acc}.pth"
    torch.save(model.state_dict(), osp.join(save_path, model_filename))

    for sr in sparsity_ratios:
        model = Prototype_IP(in_channels=in_channels, num_classes=num_classes, concept_path=concept_path, model_name=model_name, num_concepts=num_concepts, loss_func=ce_criterion, sparsity_ratio=sr).to(device)
        # optimizer
        scheduler, optimizer = build_optimizer(optimizer_type, scheduler_type, model.parameters(), lr=lr)
        best_loss, best_top1_acc, best_top5_acc = train_model(train_dataloader, val_dataloader, scheduler, optimizer, n_epoch, model, device)
        print('loss:{}, top1_acc:{}, top5_acc:{}'.format(best_loss, best_top1_acc, best_top5_acc))

        model_filename = f"{num_concepts}##{sr}##{lr}##{best_loss}##{best_top1_acc}##{best_top5_acc}.pth"
        torch.save(model.state_dict(), osp.join(save_path, model_filename))
