import sys
sys.path.append("..")
import torch
import os
import random
from utils import utils, data_utils, similarity
import argparse
import datetime
import json

parser = argparse.ArgumentParser(description='Settings for creating CBM')


parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--concept_set", type=str, default=None, 
                    help="path to concept set name")
parser.add_argument("--backbone", type=str, default="clip_RN50", help="Which pretrained model to use as backbone")
parser.add_argument("--clip_name", type=str, default="ViT-B/16", help="Which CLIP model to use")

parser.add_argument("--device", type=str, default="cuda", help="Which device to use")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size used when saving model/CLIP activations")

parser.add_argument("--feature_layer", type=str, default='layer4', 
                    help="Which layer to collect activations from. Should be the name of second to last layer in the model")
parser.add_argument("--activation_dir", type=str, default='saved_activations', help="save location for backbone and CLIP activations")

args = parser.parse_args()

    
similarity_fn = similarity.cos_similarity_cubed_single

d_train = args.dataset + "_train"
d_val = args.dataset + "_val"

#get concept set
cls_file = data_utils.LABEL_FILES[args.dataset]
with open(cls_file, "r") as f:
    classes = f.read().split("\n")

with open(args.concept_set) as f:
    concepts = f.read().split("\n")

#save activations and get save_paths
for d_probe in [d_train, d_val]:
    utils.save_activations(clip_name = args.clip_name, target_name = args.backbone, 
                            target_layers = [args.feature_layer], d_probe = d_probe,
                            concept_set = args.concept_set, batch_size = args.batch_size, 
                            device = args.device, pool_mode = "avg", save_dir = args.activation_dir)