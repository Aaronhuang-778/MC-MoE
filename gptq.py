import math
import time

import torch
import torch.nn as nn
import transformers
from utils import mixed_quantizer, quantizer, quantizer_moe
from texttable import Texttable
from utils.reconstruct import torch_snr_error

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class GPTQ:

    def __init__(self, layer, logger, name, wbits):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = quantizer_moe.Quantizer()
        self.logger = logger
        self.name = name
        self.wbits = wbits
    
    def set_bit(self, bit):
        self.wbits = bit

    def add_batch(self, inp, out):
        # Hessian H = 2 X XT + λ I
        self.inp1 = None
        self.out1 = None

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(self.layer.kernel_size, dilation=self.layer.dilation, padding=self.layer.padding, stride=self.layer.stride)
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def print_loss(self, name, q_weight, weight_error, timecost, modules=None, bit=3):
        table = Texttable()
        name = name+"-"+str(bit)
        name += ' ' * (31 - len(name))

        table.header(['name', 'weight_error', 'fp_inp_SNR', 'q_inp_SNR', 'time'])

        # assign weight
        if modules is None:
            self.layer.weight.data = q_weight.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        else:
            modules.weight.data = q_weight.reshape(modules.weight.shape).to(modules.weight.data.dtype)

        if self.inp1 is not None:
            # quantize input to int8
            quantizer = quantizer.Quantizer()
            quantizer.configure(8, perchannel=False, sym=True, mse=False)
            quantizer.find_params(self.inp1)
            q_in = quantizer.quantize(self.inp1).type(torch.float16)
            q_out = self.layer(q_in)

            # get kinds of SNR
            q_SNR = torch_snr_error(q_out, self.out1).item()
            fp_SNR = torch_snr_error(self.layer(self.inp1), self.out1).item()
        else:
            q_SNR = '-'
            fp_SNR = '-'
        table.set_cols_width([31, 10, 10, 10, 7])
        table.add_row([name, weight_error, fp_SNR, q_SNR, timecost])
        print(table.draw().split('\n')[-2])

    def static_fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, name=''):

        self.layer.to(self.dev)

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        g_idx = []
        scale = []
        zero = []
        now_idx = 1

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)

                    if ((i1 + i) // groupsize) - now_idx == -1:
                        scale.append(self.quantizer.scale)
                        zero.append(self.quantizer.zero)
                        now_idx += 1
                if self.quantizer.pack:
                    q, s, z = self.quantizer.quantize(w.unsqueeze(1))
                    q_r = s * (q - z)
                    q_r = q_r.flatten()
                    q = q.flatten()
                    Q1[:, i] = q
                else:
                    q_r = self.quantizer.quantize(w.unsqueeze(1))
                    q_r = q_r.flatten()
                    Q1[:, i] = q_r

                Losses1[:, i] = (w - q_r)**2 / d**2
                err1 = (w - q_r) / d

                if self.wbits > 3:
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        error = torch.sum(Losses).item()

        groupsize = groupsize if groupsize != -1 else self.columns
        g_idx = [i // groupsize for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        if actorder:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.print_loss(name=name, q_weight=Q, weight_error=error, timecost=(time.time() - tick), bit=self.wbits)
        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
    
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)

        # z1 = zero.repeat_interleave(128, dim=1)
        # s1 = scale.repeat_interleave(128, dim=1)
        # print(Q.shape, scale.shape, zero.shape)
        # print(Q, zero, scale, zero.dtype)
        # self.dequant = s1 * (Q - z1)
        # print("W from gptq dequant", self.dequant)
        return scale, zero, g_idx, error

    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, name=''):
        return self.static_fasterquant(blocksize, percdamp, groupsize, actorder, name)

    def free(self):
        self.inp1 = None
        self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()