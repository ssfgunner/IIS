import pickle
import yaml
import torch.optim as optim





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