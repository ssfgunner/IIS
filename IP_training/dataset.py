import os
import os.path as osp
import torch
from utils.utils import load_pkl
from torch.utils.data import Dataset

class emb_dataset(Dataset):
    def __init__(self, pt_file, label):
        self.data = []
        emb = torch.load(pt_file)
        if len(emb.shape) == 1:
            emb = emb.reshape(label.shape[0], -1)
        for i in range(len(emb)):
            self.data.append({'data': emb[i].cpu(), 'label': torch.Tensor([label[i]]).long().cpu()})
    def __getitem__(self, idx):
        label = self.data[idx]['label']
        data = self.data[idx]['data']
        #if self.mode == 'train':
        #    data += torch.randn_like(data)
        return data, label
    def emb_size(self):
        return self.data[0]['data'].size()
    def __len__(self):
        return len(self.data)