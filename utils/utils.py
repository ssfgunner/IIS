import os
import pickle
from collections import defaultdict, deque, OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import yaml
import torch.optim as optim


import math
import clip
from . import data_utils

from tqdm import tqdm
from torch.utils.data import DataLoader

import pickle
import yaml
import torch.optim as optim

def build_optimizer(optimizer, scheduler, params, weight_decay=0.0, lr=0.08, opt_decay_step=40, opt_decay_rate=0.99, opt_restart=1):
    if optimizer == 'adam':
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=lr, momentum=0.95, weight_decay=weight_decay)
    elif optimizer == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'adagrad':
        optimizer = optim.Adagrad(params, lr=lr, weight_decay=weight_decay)
    if scheduler == 'none':
        return None, optimizer
    elif scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt_decay_step, gamma=opt_decay_rate)
    elif scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt_restart)
    elif scheduler == 'exp':
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=opt_decay_rate, last_epoch=-1)
    return scheduler, optimizer


PM_SUFFIX = {"max":"_max", "avg":""}
def load_pkl(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl(save_path, feats):
    with open(save_path, 'wb') as f:
        pickle.dump(feats, f)

def load_yaml(file_path):
    with open(file_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.Loader)
    return config

def write_yaml(yaml_dict, file_path):
    new_dict = {}
    new_dict.update(yaml_dict)
    with open(file_path, 'w') as f:
        yaml.dump(new_dict, f, sort_keys=False)

def load_txt(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    vars_list = [item.strip('\n') for item in data]
    return vars_list

def save_txt(file_path, data_list):
    data_list = ['{}\n'.format(item) for item in data_list]
    with open(file_path, 'w') as f:
        f.writelines(data_list)



def save_target_activations(target_model, dataset, save_name, target_layers = ["layer4"], batch_size = 1000,
                            device = "cuda", pool_mode='avg'):
    """
    save_name: save_file path, should include {} which will be formatted by layer names
    """
    _make_save_dir(save_name)
    save_names = {}    
    for target_layer in target_layers:
        save_names[target_layer] = save_name.format(target_layer)
        
    if _all_saved(save_names):
        return
    
    all_features = {target_layer:[] for target_layer in target_layers}
    
    hooks = {}
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(target_layer)
        hooks[target_layer] = eval(command)
    
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = target_model(images.to(device))
    for target_layer in target_layers:
        torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
        hooks[target_layer].remove()
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_clip_image_features(model, dataset, save_name, batch_size=1000 , device = "cuda"):
    _make_save_dir(save_name)
    all_features = []
    
    if os.path.exists(save_name):
        return
    
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = model.encode_image(images.to(device))
            all_features.append(features.cpu())
    torch.save(torch.cat(all_features), save_name)
    #free memory
    del all_features
    torch.cuda.empty_cache()
    return

def save_clip_text_features(model, text, save_name, batch_size=1000):
    
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text)/batch_size))):
            text_features.append(model.encode_text(text[batch_size*i:batch_size*(i+1)]))
    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)
    del text_features
    torch.cuda.empty_cache()
    return

def save_activations(clip_name, target_name, target_layers, d_probe, 
                     concept_set, batch_size, device, pool_mode, save_dir):
    
    
    target_save_name, clip_save_name, text_save_name = get_save_names(clip_name, target_name, 
                                                                    "{}", d_probe, concept_set, 
                                                                      pool_mode, save_dir)
    save_names = {"clip": clip_save_name, "text": text_save_name}
    for target_layer in target_layers:
        save_names[target_layer] = target_save_name.format(target_layer)
        
    if _all_saved(save_names):
        return
    
    clip_model, clip_preprocess = clip.load(clip_name, device=device)
    
    if target_name.startswith("clip_"):
        target_model, target_preprocess = clip.load(target_name[5:], device=device)
    else:
        target_model, target_preprocess = data_utils.get_target_model(target_name, device)
    # for name, module in target_model.named_modules():
    #     print(name)
    # setup data
    data_c = data_utils.get_data(d_probe, clip_preprocess)
    data_t = data_utils.get_data(d_probe, target_preprocess)

    with open(concept_set, 'r') as f: 
        words = (f.read()).split('\n')
    text = clip.tokenize(["{}".format(word) for word in words]).to(device)
    
    save_clip_text_features(clip_model, text, text_save_name, batch_size)
    
    save_clip_image_features(clip_model, data_c, clip_save_name, batch_size, device)
    if target_name.startswith("clip_"):
        save_clip_image_features(target_model, data_t, target_save_name, batch_size, device)
    else:
        save_target_activations(target_model, data_t, target_save_name, target_layers,
                                batch_size, device, pool_mode)
    
    return
    
    
def get_similarity_from_activations(target_save_name, clip_save_name, text_save_name, similarity_fn, 
                                   return_target_feats=True):
    image_features = torch.load(clip_save_name)
    text_features = torch.load(text_save_name)
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True).float()
        text_features /= text_features.norm(dim=-1, keepdim=True).float()
        clip_feats = (image_features @ text_features.T)
    del image_features, text_features
    torch.cuda.empty_cache()
    
    target_feats = torch.load(target_save_name)
    similarity = similarity_fn(clip_feats, target_feats)
    
    del clip_feats
    torch.cuda.empty_cache()
    
    if return_target_feats:
        return similarity, target_feats
    else:
        del target_feats
        torch.cuda.empty_cache()
        return similarity
    
def get_activation(outputs, mode):
    '''
    mode: how to pool activations: one of avg, max
    for fc neurons does no pooling
    '''
    if mode=='avg':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output.mean(dim=[2,3]).detach().cpu())
            elif len(output.shape)==2:
                outputs.append(output.detach().cpu())
            elif len(output.shape)==3:
                outputs.append(output[:, 0].detach().cpu())
    elif mode=='max':
        def hook(model, input, output):
            if len(output.shape)==4:
                outputs.append(output.amax(dim=[2,3]).detach().cpu())
            elif len(output.shape)==2:
                outputs.append(output.detach().cpu())
    return hook

    
def get_save_names(clip_name, target_name, target_layer, d_probe, concept_set, pool_mode, save_dir):
    
    if target_name.startswith("clip_"):
        target_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, target_name.replace('/', ''))
    else:
        target_save_name = "{}/{}_{}_{}{}.pt".format(save_dir, d_probe, target_name, target_layer,
                                                 PM_SUFFIX[pool_mode])
    clip_save_name = "{}/{}_clip_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
    concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
    text_save_name = "{}/{}_{}.pt".format(save_dir, concept_set_name, clip_name.replace('/', ''))
    
    return target_save_name, clip_save_name, text_save_name

    
def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True

def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return

def get_accuracy_cbm(model, dataset, device, batch_size=250, num_workers=2):
    correct = 0
    total = 0
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers,
                                           pin_memory=True)):
        with torch.no_grad():
            #outs = target_model(images.to(device))
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            correct += torch.sum(pred.cpu()==labels)
            total += len(labels)
    return correct/total

def get_preds_cbm(model, dataset, device, batch_size=250, num_workers=2):
    preds = []
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers,
                                           pin_memory=True)):
        with torch.no_grad():
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    return preds

def get_concept_act_by_pred(model, dataset, device):
    preds = []
    concept_acts = []
    for images, labels in tqdm(DataLoader(dataset, 500, num_workers=8, pin_memory=True)):
        with torch.no_grad():
            outs, concept_act = model(images.to(device))
            concept_acts.append(concept_act.cpu())
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    concept_acts = torch.cat(concept_acts, dim=0)
    concept_acts_by_pred=[]
    for i in range(torch.max(pred)+1):
        concept_acts_by_pred.append(torch.mean(concept_acts[preds==i], dim=0))
    concept_acts_by_pred = torch.stack(concept_acts_by_pred, dim=0)
    return concept_acts_by_pred