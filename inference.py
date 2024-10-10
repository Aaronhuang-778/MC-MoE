import transformers
from transformers import AutoTokenizer
from os.path import join as pjoin
from accelerate import init_empty_weights
from typing import Callable
import torch
import torch.nn as nn
from quant.QLinear import QLinear
from utils.pack import get_weight_file
from tqdm import tqdm
import gc

_IGNORE_LINEAR = ["lm_head"]
_QUANT_LAYERS = [nn.Linear, QLinear]

def get_linear_tags():
        return [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "block_sparse_moe.experts.w1",
            "block_sparse_moe.experts.w2",
            "block_sparse_moe.experts.w3",
        ]

def patch_nonlinearlayers(model, patch_fct, verbose=True):
        base_model = model.model
        model.lm_head = patch_fct(model.lm_head)  ###
        base_model.embed_tokens = patch_fct(base_model.embed_tokens)
        base_model.norm = patch_fct(base_model.norm)

        layers = base_model.layers
        print(layers)
        for i in tqdm(range(len(layers)), disable=not verbose):
            layers[i].self_attn.rotary_emb = patch_fct(layers[i].self_attn.rotary_emb)
            layers[i].input_layernorm = patch_fct(layers[i].input_layernorm)
            layers[i].post_attention_layernorm = patch_fct(
                layers[i].post_attention_layernorm
            )

            layers[i].block_sparse_moe.gate = patch_fct(
                layers[i].block_sparse_moe.gate
            )  # Keep MOE gate as fp16 because it's small

            n_experts = len(layers[i].block_sparse_moe.experts)
            for k in range(n_experts):
                layers[i].block_sparse_moe.experts[k].act_fn = patch_fct(
                    layers[i].block_sparse_moe.experts[k].act_fn
                )

def patch_linearlayers( model, patch_fct, patch_params, verbose=True):
    base_model = model.model
    layers = base_model.layers
    for i in tqdm(range(len(layers)), disable=not verbose):
        layers[i].self_attn.q_proj = patch_fct(
            layers[i].self_attn.q_proj, patch_params["self_attn.q_proj"]
        )
        layers[i].self_attn.k_proj = patch_fct(
            layers[i].self_attn.k_proj, patch_params["self_attn.k_proj"]
        )
        layers[i].self_attn.v_proj = patch_fct(
            layers[i].self_attn.v_proj, patch_params["self_attn.v_proj"]
        )
        layers[i].self_attn.o_proj = patch_fct(
            layers[i].self_attn.o_proj, patch_params["self_attn.o_proj"]
        )

        n_experts = len(layers[i].block_sparse_moe.experts)
        for k in range(n_experts):
            layers[i].block_sparse_moe.experts[k].w1 = patch_fct(
                layers[i].block_sparse_moe.experts[k].w1,
                patch_params["block_sparse_moe.experts.w1"],
            )
            layers[i].block_sparse_moe.experts[k].w2 = patch_fct(
                layers[i].block_sparse_moe.experts[k].w2,
                patch_params["block_sparse_moe.experts.w2"],
            )
            layers[i].block_sparse_moe.experts[k].w3 = patch_fct(
                layers[i].block_sparse_moe.experts[k].w3,
                patch_params["block_sparse_moe.experts.w3"],
            )

def autoname_modules( model) -> None:
    for name, module in model.named_modules():
        module.name = name
def name_to_linear_tag(name: str) -> str:
    return ".".join(
        [
            n
            for n in name.split(".")
            if ((n not in ["model", "layers"]) and (not n.isnumeric()))
        ]
    )
def get_linear_tags_from_model(model, ignore: list) -> list:
    linear_tags = set()
    for name, module in model.named_modules():
        if (type(module) in _QUANT_LAYERS) and (name.split(".")[-1] not in ignore):
            linear_tags.add(name_to_linear_tag(name))
    return list(linear_tags)

def set_auto_linear_tags(model, ignore: list = _IGNORE_LINEAR) -> None:
        if hasattr(model, "linear_tags") is False:
            linear_tags = get_linear_tags()
            model.linear_tags = (
                linear_tags
                if len(linear_tags) > 0
                else get_linear_tags_from_model(model, ignore=ignore)
            )

def get_config_file(save_dir: str) -> str:
    return pjoin(save_dir, "config.json")

def load_weights(save_dir: str, map_location=None):
    return torch.load(get_weight_file(save_dir), map_location=map_location)

def create_model(save_dir, kwargs):
        model_kwargs = {}
        for key in ["attn_implementation"]:
            if key in kwargs:
                model_kwargs[key] = kwargs[key]

        config = transformers.AutoConfig.from_pretrained(
            get_config_file(save_dir)
        )

        auto_class = transformers.AutoModel

        # Todo: add support for other auto models
        archs = config.architectures
        if len(archs) == 1 and ("CausalLM" in archs[0]):
            auto_class = transformers.AutoModelForCausalLM

        with init_empty_weights():
            model = auto_class.from_config(config, **model_kwargs)

        return model

def freeze_model(model) -> None:
    for param in model.parameters():
        param.requires_grad = False
    try:
        for param in model.model.parameters():
            param.requires_grad = False
    except Exception:
        pass

def patch_model(
        model,
        patch_nonlinear_fct: Callable,
        patch_linear_fct: Callable,
        patch_params: dict,
        verbose: bool = True,
    ) -> None:
    model.eval()
    freeze_model(model)
    autoname_modules(model)
    patch_nonlinearlayers(model, patch_nonlinear_fct, verbose=verbose)
    patch_linearlayers(model, patch_linear_fct, patch_params, verbose=verbose)
    torch.cuda.empty_cache()
    gc.collect()

def setup_model(model):
    autoname_modules(model)
    set_auto_linear_tags(model)

def load_quantized_model(save_dir, kwargs):
    model = create_model(save_dir, kwargs)
    model.save_dir = save_dir
    setup_model(model)

    @torch.no_grad()
    def _load_module(module, params=None):
        device="cuda"
        compute_dtype=torch.float16
        if module.name not in weights:
            return module.to(device=device, dtype=compute_dtype, non_blocking=True)
        state_dict = weights[module.name]
        if "W_q" in state_dict:
            module = QLinear(
                # linear_layer=None,
                quant_config=None,
                compute_dtype=compute_dtype,
                device=device,
            )
            module.load_state_dict(state_dict)
        else:
            for key in state_dict:
                setattr(
                    module,
                    key,
                    nn.Parameter(
                        state_dict[key].to(
                            device=device, dtype=compute_dtype, non_blocking=True
                        ),
                        requires_grad=False,
                    ),
                )
        return module
    try:
        weights = load_weights(save_dir)
    except Exception:
        print("Failed to load the weights")
        raise FileNotFoundError
    patch_model(model, _load_module, _load_module, {k: None for k in model.linear_tags})
    return model


# tokenizer = AutoTokenizer.from_pretrained(save_dir)
# prompt = "Do you know Wei Huang?"
# inputs = tokenizer(prompt, return_tensors="pt")
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# inputs.input_ids = inputs.input_ids.to(device)
# inputs.attention_mask = inputs.attention_mask.to(device)
# # Generate
# outputs = model.generate(inputs.input_ids, 
#                         do_sample=True,
#                         temperature=0.7,
#                         top_p=0.95,
#                         top_k=40,max_new_tokens=20)

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))