import os
import pickle
import time
import torch
import logging
import torch.nn as nn
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from gptq import GPTQ
from modelutils import find_layers
from datautils import get_loaders
from quant.QLinear import *

atten_modules = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
]
expert_modules = [
    "block_sparse_moe.experts.0.w1",
    "block_sparse_moe.experts.1.w1",
    "block_sparse_moe.experts.2.w1",
    "block_sparse_moe.experts.3.w1",
    "block_sparse_moe.experts.4.w1",
    "block_sparse_moe.experts.5.w1",
    "block_sparse_moe.experts.6.w1",
    "block_sparse_moe.experts.7.w1",
    "block_sparse_moe.experts.0.w3",
    "block_sparse_moe.experts.1.w3",
    "block_sparse_moe.experts.2.w3",
    "block_sparse_moe.experts.3.w3",
    "block_sparse_moe.experts.4.w3",
    "block_sparse_moe.experts.5.w3",
    "block_sparse_moe.experts.6.w3",
    "block_sparse_moe.experts.7.w3",
    "block_sparse_moe.experts.0.w2",
    "block_sparse_moe.experts.1.w2",
    "block_sparse_moe.experts.2.w2",
    "block_sparse_moe.experts.3.w2",
    "block_sparse_moe.experts.4.w2",
    "block_sparse_moe.experts.5.w2",
    "block_sparse_moe.experts.6.w2",
    "block_sparse_moe.experts.7.w2",
]


logger = logging.getLogger(__name__)


def get_model():
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    config = AutoConfig.from_pretrained(
        args.model, attn_implementation=args.attn_implementation
    )
    model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',torch_dtype=torch.float16)

    assert isinstance(
        model, MixtralForCausalLM), 'Successfully loaded `Mixtral` model!'
    model.seqlen = 2048
    return model


@torch.no_grad()
def mixtral_sequential(model, dataloader, dev, bit_config=None):
    print('Starting ...')

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev)
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):

        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    print('Ready.')
    quantizers = {}
    for i in range(len(layers)):

        print(f'Quantizing layer {i+1}/{len(layers)}..')
        print('+--------------------------------+------------+------------+------------+---------+')
        print('|              name              |weight_error| fp_inp_SNR | q_inp_SNR  |  time   |')
        print('+================================+============+============+============+=========+')

        layer = layers[i].to(dev)
        full = find_layers(layer)

        sequential = [list(full.keys())]
        # random generation
        if args.mixed_type == "random":
            import random
            numbers = list(range(8))
            low_bit_config = random.sample(numbers, 2)
            for num in low_bit_config:
                numbers.remove(num)
            high_bit_config = random.sample(numbers, 2)
            low_bit_experts = ["block_sparse_moe.experts."+str(j) for j in low_bit_config]   
            high_bit_experts = ["block_sparse_moe.experts."+str(j) for j in high_bit_config]
        elif args.mixed_type == "manual":
            if bit_config is not None:
                _, indices_max = torch.topk(bit_config[i], args.h_experts)
                _, indices_min = torch.topk(bit_config[i], args.l_experts, largest=False)
                low_bit_experts = ["block_sparse_moe.experts."+str(j.item()) for j in indices_min]   
                high_bit_experts = ["block_sparse_moe.experts."+str(j.item()) for j in indices_max]
            else:
                print("Please generate the high_experts.pkl and low_experts.pkl first!")
                exit()
        elif args.mixed_type == "mixed":
             if bit_config is not None:
                low_bit_experts = []
                high_bit_experts = []
                for expert_index in bit_config[i].keys():
                    if bit_config[i][expert_index] == 1:
                        low_bit_experts.append("block_sparse_moe.experts."+str(expert_index))
                    elif bit_config[i][expert_index] == 3:
                        high_bit_experts.append("block_sparse_moe.experts."+str(expert_index))
             else:
                print("Please generate the high_experts.pkl and low_experts.pkl first!")
                exit()
        


        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:

                gptq[name] = GPTQ(subset[name], logger, name, args.wbits)

                if args.mixed_type == "uniform":
                    gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False, pack=args.pack) 
                    gptq[name].wbits = args.wbits
                else:
                    if name not in expert_modules:
                        gptq[name].quantizer.configure(args.attn_bits, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                        gptq[name].wbits = args.attn_bits
                    else:
                        if name[:-3] in high_bit_experts:
                            gptq[name].quantizer.configure(args.wbits+1, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                            gptq[name].wbits = args.wbits+1
                        elif name[:-3] in low_bit_experts:
                            gptq[name].quantizer.configure(args.wbits-1, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                            gptq[name].wbits = args.wbits-1
                        else:
                            gptq[name].quantizer.configure(args.wbits, perchannel=True, sym=args.sym, mse=False, pack=args.pack)
                            gptq[name].wbits = args.wbits
            # print(layer)
            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                scale, zero, g_idx, error = gptq[name].fasterquant(percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, name=name)
                # quantizers['model.layers.%d.%s' % (i, name)] = (gptq[name].quantizer.cpu(), scale.cpu(), zero.cpu(), g_idx.cpu(), args.wbits, args.groupsize)
                quantizers['model.layers.%d.%s' % (i, name)] = None
                if args.pack:
                    # real quant for compact memory
                    quant_config = BaseQuantizeConfig(nbits=gptq[name].wbits, group_size=args.groupsize)
                    name_parts = name.split('.')
                    if len(name_parts) == 2: # atten layer
                        _module = getattr(layer, name_parts[-2])
                        linear_layer = getattr(_module, name_parts[-1])
                    else: 
                        experts = getattr(layer.block_sparse_moe, "experts")
                        _module = experts[int(name_parts[-2])]
                        linear_layer = getattr(_module, name_parts[-1])
                    quant_layer = QLinear(quant_config=quant_config, device=linear_layer.weight.device)
                    quant_layer.replace_quantized_weight(linear_layer.weight, scale, zero)
                    setattr(_module, name_parts[-1], quant_layer)
                    print(getattr(_module, name_parts[-1]).W_q.dtype)
                gptq[name].free()
            
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        del gptq
        torch.cuda.empty_cache()
        inps, outs = outs, inps
        print('+--------------------------------+------------+------------+------------+---------+')
        print('\n')

    model.config.use_cache = use_cache

    return quantizers


if __name__ == "__main__":
    import argparse

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    
    def list_of_floats(arg):
        return list(map(float, arg.split(',')))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="model to load; for example `huggyllama/llama-7b`."
    )
    parser.add_argument(
        "--wbits",
        type=str,
        choices=["1bit", "2bit", "3bit", "4bit", "5bit", "6bit", "7bit", "8bit"],
        help="weight bit-width",
    )
    parser.add_argument(
        "--attn_bits",
        type=str,
        choices=["1bit", "2bit", "3bit", "4bit", "5bit", "6bit", "7bit", "8bit"],
        help="attention weight bit-width",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4", "mix"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument("--load_quantized", action="store_true")
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--groupsize",
        type=int,
        default=128,
        help="Group size",
    )
    parser.add_argument(
        "--num_fewshot", 
        type=int, 
        default=0
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=1, 
        help="batch size."
    )
    parser.add_argument(
        "--attn_implementation",
        type=str, required=False, default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="attention implementation that the model works with",
    )
    parser.add_argument(
        '--sym', 
        action='store_true', 
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--act-order', 
        action='store_true', 
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        "--multigpu",
        action="store_true",
    )
    parser.add_argument(
        "--eval_ppl", action="store_true", help="Evaluate perplexity."
    )
    parser.add_argument(
        "--tasks", 
        type=str,
        default="",
        help="Test datasets",
    )
    parser.add_argument(
        "--save",
        action="store_true",
    )
    parser.add_argument(
        "--pack", action="store_true", help="Whether to save the packed model."
    )
    parser.add_argument(
        "--use_flash_attention_2", action="store_true", help="Whether to use flash_attention2 for inference."
    )
    parser.add_argument(
        '--r', type=int, default=7, help='Number of experts to preserve'
    )
    parser.add_argument(
        "--mixed_type",
        type=str,
        choices=["uniform", "mixed", "random", "manual"],
        help='Whether to use mixed-precision',
    )
    parser.add_argument(
        "--h_experts", 
        type=int, 
        default=2, 
        help="batch size." 
    )
    parser.add_argument(
        "--l_experts", 
        type=int, 
        default=2, 
        help="batch size." 
    )
    parser.add_argument(
        "--precisions", type=str, help="the file path of experts precision"
    )
    parser.add_argument(
        "--saving_path", type=str, help="the saving path of quantized model"
    )

    args = parser.parse_args()
    print(f'Arguments: {args}')

    groupsize = args.groupsize
    args.wbits = int(args.wbits[0])
    args.attn_bits = int(args.attn_bits[0])

    model = get_model()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    bit_config = None
    if args.mixed_type == "manual" or args.mixed_type == "mixed":
        high_bit = args.precisions
        if os.path.exists(high_bit):
            with open(high_bit, 'rb') as file:
                bit_config = pickle.load(file)
        else:
            print("Please generate the high_experts.pkl and low_experts.pkl first!")
            exit()

    dataloader, testloader = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )
    device = "cuda:0"
    tick = time.time()
    quantizers = mixtral_sequential(model, dataloader, device, bit_config)
    print("quantization time:", time.time() - tick, "s")
    print(model)

    if args.eval_ppl:
        for dataset in ["wikitext2", "c4", "ptb"]:
            dataloader, testloader = get_loaders(
                dataset, seed=args.seed, seqlen=2048, model=args.model
            )
            print(dataset)
            from eval_ppl_utils import llama_eval
            t1 = time.time()
            llama_eval(model, testloader, device, dataset)
            print("Time: ", time.time() - t1)
    if args.save:
        average_bits = int(args.precisions[-9:-7])/8
        saving_path = args.saving_path + f"Mixtral-8x7B-v0.1-atten_{args.attn_bits}-e_{average_bits}"
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.save_pretrained(saving_path)
        from utils.pack import save_quantized
        save_quantized(model, saving_path)