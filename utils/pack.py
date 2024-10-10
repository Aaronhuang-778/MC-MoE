import torch
from os.path import join as pjoin

def is_leaf_module(module) -> bool:
    return len(module._modules) == 0

def get_ignore_layers( model) -> list:
        layers = {""}
        for name, module in model.named_modules():
            if not is_leaf_module(module):
                layers.add(name)
        return list(layers)

def serialize_weights(model, verbose: bool = False) -> dict:
        weights = {}
        ignore_keys = get_ignore_layers(model)
        for name, module in model.named_modules():
            if name in ignore_keys:
                continue
            try:
                # disable state_dict encoding for safetensors
                module.encoded_state_dict = False
                state_dict = module.state_dict()

                if len(state_dict) > 0:
                    weights[name] = dict(state_dict)
            except Exception:
                if verbose:
                    print("Skipping", name)

        return weights



def cache_model(model, save_dir):
        model.config.save_pretrained(save_dir)

def get_weight_file(save_dir: str) -> str:
        return pjoin(save_dir, "qmodel.pt")

def save_weights(weights: dict, save_dir: str) -> None:
        torch.save(weights, get_weight_file(save_dir))

def save_quantized(model, save_dir: str, verbose: bool = False):
        # Save config
        cache_model(model, save_dir)

        # Serialization
        weights = serialize_weights(model, verbose=verbose)

        # Save
        save_weights(weights, save_dir)