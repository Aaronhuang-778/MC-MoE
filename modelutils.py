import torch
import torch.nn as nn
import transformers

DEV = torch.device('cuda:0')


def find_layers(module, layers=[nn.Conv2d, nn.Linear], name=''):
    if not layers:
        layers = [transformers.pytorch_utils.Conv1D, nn.Conv2d, nn.Linear]
    for layer in layers:
        if isinstance(module, layer):
            return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(child, layers=layers, name=name + "." + name1 if name != "" else name1))
    return res


