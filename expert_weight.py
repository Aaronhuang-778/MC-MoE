import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
import logging
from argparse import Namespace
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.wrapper import PrunableMixtralSparseMoeBlockWrapper, QuantbleMixtralSparseMoeBlockWrapper, DynamicRankMixtralSparseMoeBlockWrapper, DynamicRankMixtralDecoderLayer
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM, MixtralDecoderLayer
import numpy as np
logger = logging.getLogger(__name__)



def replace_with_dynamic_rank(model: MixtralForCausalLM, args: Namespace, block_range: int):
    assert isinstance(
        model, MixtralForCausalLM), 'Currently only `Mixtral` is supported'
    
    pruning_loss = [5045.205078, 403.264374, 6.949406, 6.804956, 6.935846, 6.583706, 6.535251, 6.573341, 6.408963, 6.365987, 6.392511, 6.327014, 6.400521, 6.286018, 6.266879, 6.37049, 6.458919, 6.517889, 6.565863, 6.774975, 6.509265, 6.424187, 6.296724, 6.345919, 6.279357, 6.302347, 6.286659, 6.303178, 6.326483, 6.598198, 6.890288, 9.767349]
    pruning_loss = np.array(pruning_loss)
    sorted_indices = np.argsort(pruning_loss)
    order = np.empty_like(sorted_indices)
    order[sorted_indices] = np.arange(len(pruning_loss))

    beta = [0.402,0.494,0.463,0.484,0.478,0.491,0.523, 0.521,0.544,0.570,0.574,0.489,0.503,0.618,0.568, 0.535,0.559,0.519,0.537,0.487,0.469,0.461,0.461, 0.469,0.458,0.418,0.433,0.418,0.406,0.433,0.447, 0.535]

    for idx, _ in enumerate(model.model.layers):
        layer = model.model.layers[idx]
        model.model.layers[idx] = DynamicRankMixtralDecoderLayer(layer)
    for l, layer in enumerate(model.model.layers):

        layer.model.block_sparse_moe = DynamicRankMixtralSparseMoeBlockWrapper(
            layer.model.block_sparse_moe)
        layer.model.block_sparse_moe.block_index = l
        layer.model.block_sparse_moe.block_range = block_range
        layer.model.block_sparse_moe.loss_index = order[l]
        layer.model.block_sparse_moe.beta = beta[l]
        layer.model.block_sparse_moe.tau_l = 0
        layer.model.block_sparse_moe.tau_h = 0.02
    return model


def layerwise_experts_weights_frequencies(model: MixtralForCausalLM, calib_loader: DataLoader, args: Namespace):
    assert isinstance(
        model, MixtralForCausalLM), 'Currently only `Mixtral` is supported'

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = PrunableMixtralSparseMoeBlockWrapper(
            layer.block_sparse_moe, r=args.r)
        layer.block_sparse_moe.cache_X = True
        layer.block_sparse_moe.cache_Z = True

    with torch.inference_mode():
        for i, batch in enumerate(tqdm(calib_loader, desc='Model forwarding on sample set...')):
            model_inputs = model.prepare_inputs_for_generation(**batch)
            outputs = model(**model_inputs)
            assert outputs is not None

    print('Moving whole model to cpu...')
    model.to('cpu')
    torch.cuda.empty_cache()

    weights_tensor = dict()
    frequencies_tensor = dict()
    for l, layer in tqdm(list(enumerate(model.model.layers)), desc='Enumerating loss on sample set...'):
        b = layer.block_sparse_moe
        if not hasattr(b, 'cache_space'):
            continue
        if l < 16:
            b.to('cuda:0')
        else:
            b.to('cuda:1')
        # b.to('cuda:0')
        weights_tensor[l] = b.weights_tensor.to('cpu')
        frequencies_tensor[l] = b.number_tensor.to('cpu')

        print(weights_tensor[l])
        # b.prune()
        b.to('cpu')

    import pickle
    with open("experts_act_weight.pkl", "wb") as f:
        pickle.dump(weights_tensor, f)
    with open("experts_act_frequency.pkl", "wb") as f:
        pickle.dump(frequencies_tensor, f)



    return model

def layerwise_quant(model: MixtralForCausalLM, calib_loader: DataLoader, args: Namespace):
    assert isinstance(
        model, MixtralForCausalLM), 'Currently only `Mixtral` is supported'

    for l, layer in enumerate(model.model.layers):
        layer.block_sparse_moe = QuantbleMixtralSparseMoeBlockWrapper(
            layer.block_sparse_moe, r=args.r)
        layer.block_sparse_moe.cache_X = True
        layer.block_sparse_moe.cache_Z = True

    with torch.inference_mode():
        for i, batch in enumerate(tqdm(calib_loader, desc='Model forwarding on sample set...')):
            model_inputs = model.prepare_inputs_for_generation(**batch)
            outputs = model(**model_inputs)
            assert outputs is not None

    # print('Moving whole model to cpu...')
    # model.to('cpu')
    # torch.cuda.empty_cache()

    quant_expert_loss = dict()
    for l, layer in tqdm(list(enumerate(model.model.layers)), desc='Enumerating loss on sample set...'):
        b = layer.block_sparse_moe
        if not hasattr(b, 'cache_space'):
            continue
        # if l < 16:
        #     b.to('cuda:0')
        # else:
        #     b.to('cuda:1')
        loss_history = b.order_quant()
        quant_expert_loss[l] = loss_history
        
        # b.prune()
        # b.to('cpu')
    
    import pickle
    torch.cuda.empty_cache()
    with open("experts_quant_loss.pkl", "wb") as f:
        pickle.dump(quant_expert_loss, f)
    return model
