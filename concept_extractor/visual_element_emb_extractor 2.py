import os
import os.path as osp

import presets
import torch
import torch.utils.data
import torchvision
import torchvision.transforms
import utils
from torchvision.transforms.functional import InterpolationMode

from utils import load_yaml, save_pkl
from hooks import InputHook
from model_zoo import get_model

def evaluate(model, data_loader, device, target_layer='fc'):
    model.eval()
    emb_mat, target_mat = [], []
    with torch.inference_mode():
        for i, (image, target) in enumerate(data_loader):
            
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with InputHook(model, outputs=[target_layer], as_tensor=True) as h:
                output = model(image)
                returned_features = h.layer_outputs
                returned_features = returned_features[target_layer][0].squeeze().detach().cpu()
            emb_mat.append(returned_features)
            target_mat.append(target.detach().cpu())
    emb_mat = torch.cat(emb_mat, dim=0)
    target_mat = torch.cat(target_mat, dim=0)

    return emb_mat, target_mat

def load_data(valdir, args):
    # Data loading code
    print("Loading data")
    val_resize_size, val_crop_size = (
        args.val_resize_size,
        args.val_crop_size,
    )
    interpolation = InterpolationMode(args.interpolation)

    print("Loading validation data")
    preprocessing = presets.ClassificationPresetEval(
        crop_size=val_crop_size,
        resize_size=val_resize_size,
        interpolation=interpolation,
        backend=args.backend,
    )

    dataset_test = torchvision.datasets.ImageFolder(
        valdir,
        preprocessing,
    )

    print("Creating data loaders")
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset_test, test_sampler


def main(args):

    print(args)

    device = torch.device(args.device)

    
    torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(args.data_path)
    val_dir = os.path.join(args.data_path)
    dataset_test, test_sampler = load_data(val_dir, args)


    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True
    )

    print("Creating model")
    model = get_model(args.model, args.weights)
    model.to(device)    
    

    model_without_ddp = model

    if args.resume and osp.exists(args.resume):
        print('load model from {}'.format(args.resume))
        checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    val_emb, val_label = evaluate(model, data_loader_test, device=device, target_layer=args.target_layer_name)
    save_dir = f'./embs/{args.model}/'
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    save_pkl(osp.join(save_dir, 'visual_element_emb.pkl'), val_emb)
    return



def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="PyTorch Classification Training", add_help=add_help)

    parser.add_argument("--config_yaml", default=None, type=str, help="config_file")
    parser.add_argument("--data-path", default="", type=str)
    parser.add_argument("--model", default="resnet18", type=str, help="model name")
    parser.add_argument("--target_layer_name", default="fc", type=str)
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument(
        "-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size"
    )
    parser.add_argument(
        "-j", "--workers", default=16, type=int, metavar="N", help="number of data loading workers (default: 16)"
    )
    parser.add_argument("--resume", default="", type=str, help="path of checkpoint")

    # distributed training parameters
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")
    parser.add_argument(
        "--interpolation", default="bilinear", type=str, help="the interpolation method (default: bilinear)"
    )
    parser.add_argument(
        "--val-resize-size", default=256, type=int, help="the resize size used for validation (default: 256)"
    )
    parser.add_argument(
        "--val-crop-size", default=224, type=int, help="the central crop size used for validation (default: 224)"
    )
    parser.add_argument("--weights", default=None, type=str, help="the weights enum name to load")
    parser.add_argument("--backend", default="PIL", type=str.lower, help="PIL or tensor - case insensitive")
    return parser


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    if args.config_yaml is not None:
        cfg = load_yaml(args.config_yaml)
        parser.set_defaults(**cfg)
        args = parser.parse_args()
    
    main(args)