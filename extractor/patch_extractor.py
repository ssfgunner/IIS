import datetime
import os
import os.path as osp
import time
import warnings


import numpy as np
from PIL import Image
import scipy.stats as stats
import skimage.segmentation as segmentation
from skimage import io
from tqdm import tqdm


def get_concept_patchs(image, patch_num=4):
    width, height = image.size
    item_width = int(width / patch_num)
    item_height = int(height / patch_num)
    box_list = []
    # (left, upper, right, lower)
    for i in range(0 ,patch_num): 
        for j in range(0 ,patch_num):
            box = ( j *item_width , i *item_height ,( j +1 ) *item_width ,( i +1 ) *item_height)
            box_list.append(box)
    image_list = [image.crop(box).resize([224, 224]) for box in box_list]  #Image.crop(left, up, right, below)
    return image_list



def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Grid Train Pipeline", add_help=add_help)

    parser.add_argument("--data_root", default='./ImageNet/train', type=str)
    parser.add_argument("--save_root", default='../concept_library/visual/patches', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args_parser()
    data_root = args.data_root
    save_root = args.save_root
    if not osp.exists(save_root):
        os.mkdir(save_root)
    class_dirs = os.listdir(data_root)
    img_per_class = 10

    for class_dir in tqdm(class_dirs):
        class_dir_path = osp.join(data_root, class_dir)
        img_list = os.listdir(class_dir_path)
        save_dir = osp.join(save_root, class_dir)
        if not osp.exists(save_dir):
            os.mkdir(save_dir)
        img_num = 0
        for img in img_list:
            img_path = osp.join(class_dir_path, img)
            img_data = Image.open(img_path)
            img_name = img.split('.')[0]
            patches = get_concept_patchs(img_data, patch_num=4)
            for i, patch in enumerate(patches):
                save_path = osp.join(save_dir, f'{img_name}_{i}.jpg')
                patch.save(save_path)
            img_num += 1
            if img_num >= img_per_class:
                break
            
