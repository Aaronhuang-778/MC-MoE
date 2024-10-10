import os
import torch
from transformers import AutoTokenizer
from inference import load_quantized_model
from expert_weight import  replace_with_dynamic_rank
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.cuda.is_available()


kwargs = {"device_map": 'auto',
          "torch_dtype": "torch.float16"}
######## Input your save_dir of quantized model########
save_dir = ""

model = load_quantized_model(save_dir, kwargs)

######### Choose if you want to use dynamic pruning or not ##########
args = None
model = replace_with_dynamic_rank(model, args, block_range=10)
######### Choose if you want to use dynamic pruning or not ##########

tokenizer = AutoTokenizer.from_pretrained(save_dir)
prompt = "You are a writer. Please write a short story about two llamas in a forest."
prompt_template=f'''{prompt}'''

inputs = tokenizer(prompt_template, return_tensors="pt")
device = "cuda:0" if torch.cuda.is_available() else "cpu"
inputs.input_ids = inputs.input_ids.to(device)

inputs.attention_mask = inputs.attention_mask.to(device)
# Generate
outputs = model.generate(inputs.input_ids, 
                        max_new_tokens=512,
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1,
                        )

print(tokenizer.decode(outputs[0]))

