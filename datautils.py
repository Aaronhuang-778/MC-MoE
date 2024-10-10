import random

import numpy as np
import torch
from datasets import load_dataset
from tokenizer_wrapper import TokenizerWrapper
from transformers import AutoTokenizer, LlamaTokenizer
import os


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

'''
Generate tokenizer and return it to preload datasets by converting them to embedded vectors instead of natural words
'''
def get_tokenizer(model):
    tokenizer = AutoTokenizer.from_pretrained(model)
    return tokenizer

def get_wikitext2(nsamples, seed, seqlen, model, tokenizer):
    
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')

    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_ptb(nsamples, seed, seqlen, model, tokenizer):
    traindata = load_dataset('ptb_text_only', 'penn_treebank', split='train')
    testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')

    trainenc = tokenizer(" ".join(traindata['sentence']), return_tensors='pt')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

def get_c4(nsamples, seed, seqlen, model, tokenizer):
    traindata = load_dataset('json', data_files={'train': 'data/c4-train.00000-of-01024.json'})
    valdata = load_dataset('json', data_files={'validation': 'data/c4-validation.00000-of-00008.json'})

    traindata = traindata['train']
    valdata = valdata['validation']
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]

    valenc = TokenizerWrapper(valenc)

    return trainloader, valenc

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, model=''):
    model_name = model.split('/')[-1]
    cache_file=f'/mnt/afs/yliao/Tasks/moe/Expert_Quant/moeq/cache/{name}_{nsamples}_{seed}_{seqlen}/Mixtral-8x7B-v0.1.pt'
    try:
        test_enc = torch.load(cache_file)
        return test_enc
    except:
        pass

    tokenizer = get_tokenizer(model)
    
    if 'wikitext2' in name:
        loaders= get_wikitext2(nsamples, seed, seqlen, model, tokenizer)
    if 'ptb' in name:
        loaders= get_ptb(nsamples, seed, seqlen, model, tokenizer)
    if 'c4' in name:
        loaders= get_c4(nsamples, seed, seqlen, model, tokenizer)
    if 'mix' in name:
        wiki_train,wiki_val=get_wikitext2(nsamples//3, seed, seqlen, model, tokenizer)
        ptb_train,ptb_val=get_ptb(nsamples//3, seed, seqlen, model, tokenizer)
        c4_train,c4_val=get_c4(nsamples//3, seed, seqlen, model, tokenizer)
        mixed_loader=wiki_train+ptb_train+c4_train
        val=None

        directory='/'.join(cache_file.split('/')[:-1])
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save((mixed_loader, val),cache_file)

        return mixed_loader, val

    directory='/'.join(cache_file.split('/')[:-1])
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(loaders,cache_file)
    return loaders


# get_loaders("c4", nsamples=128, seed=0, model='/mnt/afs/share/LLMCKPTs/mistralai/Mixtral-8x7B-v0.1', seqlen=2048)