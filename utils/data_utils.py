import os
import torch
from torchvision import datasets, transforms, models

import clip
from pytorchcv.model_provider import get_model as ptcv_get_model

DATASET_ROOTS = {
    "imagenet_train": "/home/shufanshen/PICO/imagenet/train",
    "imagenet_val": "/home/shufanshen/PICO/imagenet/val",
    "cub_train":"./datasets/CUB_200_2011/train",
    "cub_val":"./datasets/CUB_200_2011/val"
}

LABEL_FILES = {
               "imagenet":"../concept_extractor/data/imagenet_classes.txt",
               "cifar10":"data/cifar10_classes.txt",
               "cifar100":"data/cifar100_classes.txt",
               "cub":"data/cub_classes.txt"}

def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                   transforms.ToTensor(), transforms.Normalize(mean=target_mean, std=target_std)])
    return preprocess


def get_data(dataset_name, preprocess=None):
    if dataset_name == "cifar100_train":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)

    elif dataset_name == "cifar100_val":
        data = datasets.CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False, 
                                   transform=preprocess)
        
    elif dataset_name == "cifar10_train":
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=True,
                                   transform=preprocess)
        
    elif dataset_name == "cifar10_val":
        data = datasets.CIFAR10(root=os.path.expanduser("~/.cache"), download=True, train=False,
                                   transform=preprocess)        
    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)
               
    return data

def get_targets_only(dataset_name):
    pil_data = get_data(dataset_name)
    return pil_data.targets

def get_target_model(target_name, device):
    
    if target_name.startswith("clip_"):
        target_name = target_name[5:]
        model, preprocess = clip.load(target_name, device=device)
        target_model = lambda x: model.encode_image(x).float()
    
    elif target_name == 'resnet18_places': 
        target_model = models.resnet18(pretrained=False, num_classes=365).to(device)
        state_dict = torch.load('data/resnet18_places365.pth.tar')['state_dict']
        new_state_dict = {}
        for key in state_dict:
            if key.startswith('module.'):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
        
    elif target_name == 'resnet18_cub':
        target_model = ptcv_get_model("resnet18_cub", pretrained=True).to(device)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()
    elif target_name == 'resnet34_cub':
        target_model = ptcv_get_model("resnet34_cub", pretrained=True).to(device)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()

    elif target_name.endswith("_v2"):
        target_name = target_name[:-3]
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V2".format(target_name_cap))
        target_model = eval("models.{}(weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
    else:
        target_name_cap = target_name.replace("resnet", "ResNet")
        target_name_cap = target_name_cap.replace("vit_b", "ViT_B")
        target_name_cap = target_name_cap.replace("vit_l", "ViT_L")
        target_name_cap = target_name_cap.replace("swin_b", "Swin_B")
        target_name_cap = target_name_cap.replace("swin_s", "Swin_S")
        target_name_cap = target_name_cap.replace("swin_t", "Swin_T")
        target_name_cap = target_name_cap.replace("convnext_tiny", "ConvNeXt_Tiny")
        target_name_cap = target_name_cap.replace("convnext_small", "ConvNeXt_Small")
        target_name_cap = target_name_cap.replace("convnext_base", "ConvNeXt_Base")
        target_name_cap = target_name_cap.replace("convnext_large", "ConvNeXt_Large")
        
        if "ViT_L" in target_name_cap:
            weights = eval("models.{}_Weights.IMAGENET1K_SWAG_E2E_V1".format(target_name_cap))
        else:
            weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()
    
    return target_model, preprocess