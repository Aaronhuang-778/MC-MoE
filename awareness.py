
import logging
from transformers.models.mixtral.modeling_mixtral import MixtralForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from gptq import GPTQ
from expert_weight import layerwise_experts_weights_frequencies, layerwise_quant
from data.build import build_calib_loader
from quant.QLinear import *
logger = logging.getLogger(__name__)


def get_model():
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    model = AutoModelForCausalLM.from_pretrained(args.model, device_map='auto',torch_dtype=torch.float16, attn_implementation="flash_attention_2" if args.use_flash_attention_2 else None)

    assert isinstance(
        model, MixtralForCausalLM), 'Successfully loaded `Mixtral` model!'
    model.seqlen = 2048
    return model



if __name__ == "__main__":
    import argparse

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))
    
    def list_of_floats(arg):
        return list(map(float, arg.split(',')))

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "model", type=str, help="model to load; for example `Mixtral 8*7b`."
    )
    parser.add_argument(
        "--calibration",
        type=str,
        choices=["math", "c4"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
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
        "--multigpu",
        action="store_true",
    )
    parser.add_argument(
        "--use_flash_attention_2", action="store_true", help="Whether to use flash_attention2 for inference."
    )
    parser.add_argument(
        '--r', type=int, default=7, help='Number of experts to preserve'
    )
    


    args = parser.parse_args()
    print(f'Arguments: {args}')

    ######## get the moe activated factors ########

    model = get_model()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    batch_size = 8
    num_workers = 4
    calib_loader = build_calib_loader(args.calibration, tokenizer, model.seqlen,
                                    args.nsamples, batch_size, num_workers, args.seed)  
    model = layerwise_experts_weights_frequencies(model, calib_loader, args)
    del model
    ######## get the moe quant loss ########
    model = get_model()
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    batch_size = 8
    num_workers = 4
    calib_loader = build_calib_loader(args.calibration, tokenizer, model.seqlen,
                                    args.nsamples, batch_size, num_workers, args.seed)  
    model = layerwise_quant(model, calib_loader, args)
    del model