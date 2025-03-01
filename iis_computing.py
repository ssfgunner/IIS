import os
import os.path as osp
from collections import defaultdict
def compute_iis(data_dict, standard_acc):
    srs = sorted(data_dict.keys())
    print(srs)
    accs = [data_dict[sr] for sr in srs]
    iis = 0
    for i in range(len(accs)-1):
        iis += (accs[i]/standard_acc + accs[i+1]/standard_acc) * (srs[i+1]-srs[i]) / 2
    return iis


def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="Grid Train Pipeline", add_help=add_help)

    parser.add_argument("--model_root", default='./IP_training/IP_Prototype/resnet50', type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args_parser()
    model_root = args.model_root
    model_list = os.listdir(model_root)
    data_dict = defaultdict(float)
    standard_acc = 0
    for model_pth in model_list:
        if 'standard' in model_pth:
            standard_acc = float(model_pth.split('##')[3])
            print(standard_acc)
        else:
            num_concepts, sparsity_ratio, lr, loss, acc1, acc5 = model_pth.split('.pth')[0].split('##')
            data_dict[float(sparsity_ratio)] = max(data_dict[float(sparsity_ratio)], float(acc1))
    print(compute_iis(data_dict, standard_acc))
