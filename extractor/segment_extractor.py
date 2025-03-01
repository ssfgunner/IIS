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

def extract_patch(image, mask, crop_size):
    """Extracts a patch out of an image.

    Args:
      image: The original image
      mask: The binary mask of the patch area

    Returns:
      image_resized: The resized patch such that its boundaries touches the
        image boundaries
      patch: The original patch. Rest of the image is padded with average value
    """
    average_image_value = 117

    mask_expanded = np.expand_dims(mask, -1)  
    patch = (mask_expanded * image + (
        1 - mask_expanded) * float(average_image_value)) 
    ones = np.where(mask == 1)
    h1, h2, w1, w2 = ones[0].min(), ones[0].max(), ones[1].min(), ones[1].max()
    image = Image.fromarray((patch[h1:h2, w1:w2]).astype('uint8')) 
    image_resized = np.array(image.resize(crop_size,
                                          Image.BICUBIC)).astype(float)
    return image_resized, patch


def get_concept_patchs(image, patch_num=5):
    n_segments = [10, 20, 30, ]
    n_params = len(n_segments) 
    compactnesses = [10] * n_params
    sigmas = [1.] * n_params
    unique_masks = []
    for i in range(n_params):
        param_masks = []
        assert len(image.shape) == 3, image.shape
        segments = segmentation.slic(
            image, n_segments=n_segments[i], compactness=compactnesses[i],
            sigma=sigmas[i])
        for s in range(segments.max()):
            
            mask = (segments == s).astype(float)
            unique = False
            if np.mean(mask) > 0.001:
                unique = True
            for seen_mask in unique_masks:  
                jaccard = np.sum(seen_mask * mask) / np.sum((seen_mask + mask) > 0)
                if jaccard > 0.5:
                    unique = False
                    break
            if unique:
                param_masks.append(mask)
            if len(param_masks) >= patch_num:
                break
        unique_masks.extend(param_masks)
    superpixels, patches = [], []
    while unique_masks:  
        superpixel, patch = extract_patch(np.array(image), unique_masks.pop(), [image.shape[1], image.shape[0]])  
        superpixels.append(superpixel)
        patches.append(patch)
    return superpixels, patches



def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Grid Train Pipeline", add_help=add_help)

    parser.add_argument("--data_root", default='./ImageNet/train', type=str)
    parser.add_argument("--save_root", default='../concept_library/visual/segments', type=str)
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
            img_data = io.imread(img_path)
            if len(img_data.shape) != 3:
                continue
            img_name = img.split('.')[0]
            superpixels, patches = get_concept_patchs(img_data, patch_num=5)
            for i, superpixel in enumerate(superpixels):
                segment_img = np.array(superpixel).astype('uint8')
                segment_img = Image.fromarray(segment_img).convert('RGB')
                save_path = osp.join(save_dir, f'{img_name}_{i}.jpg')
                segment_img.save(save_path)
            img_num += 1
            if img_num >= img_per_class:
                break
            
