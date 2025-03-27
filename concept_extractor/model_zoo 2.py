import torch
import torchvision

def get_model(model_name, weights):
    model = None
    if model_name == 'vit_l_16':
        model = torchvision.models.vit_l_16(weights=weights)
    elif model_name == 'vit_b_16':
        model = torchvision.models.vit_b_16(weights=weights)
    elif model_name == 'swin_t':
        model = torchvision.models.swin_t(weights=weights)
    elif model_name == 'swin_s':
        model = torchvision.models.swin_s(weights=weights)
    elif model_name == 'swin_b':
        model = torchvision.models.swin_b(weights=weights)
    elif model_name == 'convnext_tiny':
        model = torchvision.models.convnext_tiny(weights=weights)
    elif model_name == 'convnext_small':
        model = torchvision.models.convnext_small(weights=weights)
    elif model_name == 'convnext_base':
        model = torchvision.models.convnext_base(weights=weights)
    elif model_name == 'convnext_large':
        model = torchvision.models.convnext_large(weights=weights)
    elif model_name == 'resnet18':
        model = torchvision.models.resnet18(weights=weights)
    elif model_name == 'resnet34':
        model = torchvision.models.resnet34(weights=weights)
    elif model_name == 'resnet50':
        model = torchvision.models.resnet50(weights=weights)
    elif model_name == 'resnet101':
        model = torchvision.models.resnet101(weights=weights)
    elif model_name == 'resnet152':
        model = torchvision.models.resnet152(weights=weights)
    else:
        print('Model not registered!')
    return model