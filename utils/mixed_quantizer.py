
import torch
import torch.nn as nn
import math


@torch.no_grad()
def binary_scale(x):
    mean_tensor = torch.mean(x, dim=1)
    scale_tensor = torch.mean(torch.abs(x), dim=1)
    return scale_tensor, mean_tensor  

@torch.no_grad()
def binary(x, scale, zero):
    binary = torch.zeros_like(x)
    binary += x
    binary -= zero
    binary_slice = torch.sign(binary) * scale
    binary_slice += zero
    return binary_slice

@torch.no_grad()
def normal_quantize(x, scale, zero, maxq):

    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)

    return scale * (q - zero)


@torch.no_grad()
def ssim(x, y, C1=0.01**2, C2=0.03**2):
    n, m = x.shape
    mu_x = torch.nanmean(x, dim=1, keepdim=True)
    mu_y = torch.nanmean(y, dim=1, keepdim=True)
    sigma_x = ((x - mu_x)**2).nanmean(dim=1, keepdim=True)
    sigma_y = ((y - mu_y)**2).nanmean(dim=1, keepdim=True)
    x_centered = x - mu_x
    y_centered = y - mu_y
    sigma_xy = torch.nanmean(x_centered * y_centered, dim=1, keepdim=True)
    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x.pow(2) + mu_y.pow(2) + C1) * (sigma_x + sigma_y + C2)
    ssim_vals = numerator / denominator
    
    return ssim_vals.squeeze()


class Quantizer(nn.Module):
    def __init__(self, weight, method="2bit", groupsize=-1, norm=2.4, sym=True, maxshrink=.8, metric='mse', lambda_salience=1):
        super().__init__()
        oc,ic=weight.shape
        if groupsize==-1:
            groupsize=ic
        self.groupsize=groupsize
        self.n_groups=math.ceil(ic/groupsize)
        self.method=method
        self.norm = norm
        self.sym = sym
        self.maxshrink = maxshrink
        self.scale = None
        self.zero = None
        self.maxq = None
        self.lambda_salience = lambda_salience
        self.metric = metric

    def fit(self, w, Hinv1, bit_width=0):
        if bit_width > 0 and bit_width < 16:
            self.method = str(bit_width)+'bit'
        else:
            return
        # print(self.method, bit_width)

        if self.method=="1bit":
            scale, zero = binary_scale(w)
            maxq = None
        # print("this block bit_width: ", self.method)
        else:
            bits = int(self.method[0])
            perchannel = True
            weight = True
            dev = w.device
            maxq = torch.tensor(2 ** bits - 1)
            scale = torch.zeros(1)
            zero = torch.zeros(1)

            if dev != scale.device:
                scale=scale.to(dev)
                zero=zero.to(dev)
                maxq=maxq.to(dev)

            x = w.clone()
            shape = x.shape

            if perchannel:
                if weight:
                    x = x.flatten(1)
                else:
                    if len(shape) == 4:
                        x = x.permute([1, 0, 2, 3])
                        x = x.flatten(1)
                    if len(shape) == 3:
                        x = x.reshape((-1, shape[-1])).t()
                    if len(shape) == 2:
                        x = x.t()
            else:
                x = x.flatten().unsqueeze(0)
            tmp = torch.zeros(x.shape[0], device=dev)
            xmin = torch.minimum(x.min(1)[0], tmp)
            xmax = torch.maximum(x.max(1)[0], tmp)

            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            scale = (xmax - xmin) / maxq
            zero = torch.round(-xmin / scale)
            
            if maxq < 0:
                scale = xmax
                zero = xmin
            else:
                scale = (xmax - xmin) / maxq
                if self.sym:
                    zero = torch.full_like(scale, (maxq + 1) / 2)
                else:
                    zero = torch.round(-xmin / scale)
            tau_range = 0.1
            tau_n = 50
            # best = torch.zeros_like(x[:, 0], device=dev)
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            _p = torch.ones([x.shape[0]])
            p_left = 1 - tau_range
            p_right = 1 + tau_range
            for p in torch.cat([torch.ones(1),torch.linspace(1.0,p_right,tau_n+1)[1:],torch.linspace(1.0,p_left,tau_n+1)[1:]]):
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / maxq
                zero1 = torch.round(-xmin1 / scale1) if not self.sym else zero

                w_q = normal_quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), maxq)
                w_q -= w
                w_q.abs_()
                w_q.pow_(self.norm)
                
                err = torch.sum(w_q, 1)
                tmp = err < best
                if torch.any(tmp):
                    _p[tmp] = p
                    best[tmp] = err[tmp]
                    scale[tmp] = scale1[tmp]
                    zero[tmp] = zero1[tmp]

            if not perchannel:
                if weight:
                    tmp = shape[0]
                else:
                    tmp = shape[1] if len(shape) != 3 else shape[2]
                scale = scale.repeat(tmp)
                zero = zero.repeat(tmp)

            if weight:
                shape = [-1] + [1] * (len(shape) - 1)
                scale = scale.reshape(shape)
                zero = zero.reshape(shape)
        self.scale = scale
        self.zero = zero
        self.maxq = maxq

    def quantize(self, w, bit_width=0):

        if self.method=="1bit":
            q = binary(w, self.scale, self.zero)
            return q
        q = normal_quantize(w, self.scale.squeeze(), self.zero.squeeze(), self.maxq)
        return q
    
    def clear_quantize_paremeter(self):
        self.scale = None
        self.zero = None
        self.maxq = None
