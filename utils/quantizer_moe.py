from numpy import sign
import torch
import torch.nn as nn

@torch.no_grad()
def normal_quantize(x, scale, zero, maxq):
    return torch.clamp(torch.round(x / scale + zero), 0, maxq)

@torch.no_grad()
def binary_scale(x):
    scale_tensor = torch.mean(torch.abs(x), dim=1) * 2
    zero = 0.5 * torch.ones_like(scale_tensor)
    return scale_tensor, zero

@torch.no_grad()
def binary(x):
    x_ = torch.zeros_like(x)
    x_ += x
    binary_slice = torch.where(x_ >= 0, 1, -1)
    return binary_slice/2 + 0.5 # change (-1, 1) weights to (0, 1) weights

@torch.no_grad()
def residual_binary(x, scale, r_scale, zero, order=2):
    sum_order = torch.zeros_like(x)
    for od in range(2):
        residual = x - sum_order
        binary_slice = torch.where(residual >= 0, 1, -1)
        if od == 0:
            sum_order += scale * binary_slice
        else:
            sum_order += r_scale * binary_slice
    del binary_slice
    return sum_order

@torch.no_grad()
def residual_scale(x, order=2):
    sum_order = torch.zeros_like(x).to(x.device)
    scale_tensors = [torch.mean(torch.abs(x), dim=1)] * order
    zero = 0.5 * torch.ones_like(x)
    for od in range(order):
        residual = x - sum_order
        scale_tensors[od] = torch.mean(torch.abs(residual), dim=1).to(x.device)
        binary_slice = torch.where(residual >= 0, 1, -1)
        sum_order += scale_tensors[od].unsqueeze(1) * binary_slice
    del sum_order
    del binary_slice
    return scale_tensors[0], scale_tensors[1], zero

@torch.no_grad()
def r_residual_scale(x, order=3):
    sum_order = torch.zeros_like(x).to(x.device)
    scale_tensors = [torch.mean(torch.abs(x), dim=1)] * order
    zero = 0.5 * torch.ones_like(x)
    for od in range(order):
        residual = x - sum_order
        scale_tensors[od] = torch.mean(torch.abs(residual), dim=1).to(x.device)
        binary_slice = torch.where(residual >= 0, 1, -1)
        sum_order += scale_tensors[od].unsqueeze(1) * binary_slice
    del sum_order
    del binary_slice
    return scale_tensors[0], scale_tensors[1], scale_tensors[2], zero

@torch.no_grad()
def r_residual_binary(x, scale, r_scale, rr_scale, order=3):
    sum_order = torch.zeros_like(x)
    for od in range(3):
        residual = x - sum_order
        binary_slice = torch.where(residual >= 0, 1, -1)
        if od == 0:
            sum_order += scale * binary_slice
        elif od == 1:
            sum_order += r_scale * binary_slice
        else:
            sum_order += rr_scale * binary_slice
    del binary_slice
    return sum_order



class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0, dtype=torch.float16))
        self.register_buffer('scale', torch.zeros(shape, dtype=torch.float16))
        self.register_buffer('zero', torch.zeros(shape, dtype=torch.float16))

    def configure(self, bits, perchannel=False, sym=True, mse=False, norm=2.4, grid=100, maxshrink=.8, trits=False, pack=False):

        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if trits:
            self.maxq = torch.tensor(-1)
        self.scale = torch.zeros_like(self.scale, dtype=torch.float16)
        self.pack = pack

    def _quantize(self, x, scale, zero, maxq):
        if maxq < 0:
            return (x > scale / 2).float() * scale + (x < zero / 2).float() * zero
        if self.maxq == 1:
            q = binary(x)
        # elif self.maxq == 3 and (not self.pack):
        #     # TODO: pack the residual quantization
        #     return residual_binary(x, scale=scale, r_scale=self.r_scale, zero=zero, order=2)
        # elif self.maxq == 7 and (not self.pack):
        #     return r_residual_binary(x, scale=scale, r_scale=self.r_scale, rr_scale=self.rr_scale, order=3)
        else:
            q = normal_quantize(x, scale, zero, maxq)
        return scale * (q - zero) if not self.pack else q

    def find_params(self, w, weight=False):
        perchannel = True
        weight = True
        dev = w.device
        maxq = self.maxq
        scale = torch.zeros(1, dtype=torch.float16)
        zero = torch.zeros(1, dtype=torch.float16)
        shape = w.shape

        x = w.clone().to(torch.float16)

        if self.maxq == 1:
            scale, zero = binary_scale(x)
            if dev != scale.device:
                scale=scale.to(dev)
                zero=zero.to(dev)
                maxq=maxq.to(dev)
        # elif (self.maxq == 3) and (not self.pack):
        #     scale, r_scale, zero = residual_scale(x, order=2)
        #     if dev != scale.device:
        #         scale=scale.to(dev)
        #         r_scale=r_scale.to(dev)
        #         zero=zero.to(dev)
        #         maxq=maxq.to(dev)
        else:
            if dev != scale.device:
                scale=scale.to(dev)
                zero=zero.to(dev)
                maxq=maxq.to(dev)
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
            tmp = torch.zeros(x.shape[0], device=dev, dtype=torch.float16)
            xmin = torch.minimum(x.min(1)[0], tmp)
            xmax = torch.maximum(x.max(1)[0], tmp)

            tmp = (xmin == 0) & (xmax == 0)
            xmin[tmp] = -1
            xmax[tmp] = +1
            scale = (xmax - xmin) / maxq
            zero = -xmin / scale
            
            if maxq < 0:
                scale = xmax
                zero = xmin
            else:
                scale = (xmax - xmin) / maxq
                if self.sym:
                    zero = torch.full_like(scale, (maxq + 1) / 2, dtype=torch.float16)
                else:
                    zero = -xmin / scale
            tau_range = 0.1
            tau_n = 50
            # best = torch.zeros_like(x[:, 0], device=dev)
            best = torch.full([x.shape[0]], float('inf'), device=dev, dtype=torch.float16)
            # _p = torch.ones([x.shape[0]], dtype=torch.float16)
            p_left = 1 - tau_range
            p_right = 1 + tau_range
            for p in torch.cat([torch.ones(1),torch.linspace(1.0,p_right,tau_n+1)[1:],torch.linspace(1.0,p_left,tau_n+1)[1:]]):
                xmin1 = p * xmin
                xmax1 = p * xmax
                scale1 = (xmax1 - xmin1) / maxq
                # zero1 = torch.round(-xmin1 / scale1) if not self.sym else zero
                zero1 = -xmin1 / scale1

                w_q = self._quantize(x, scale1.unsqueeze(1), zero1.unsqueeze(1), maxq)
                w_q = scale1.unsqueeze(1) * (w_q - zero1.unsqueeze(1))

                w_q -= w
                w_q.abs_()
                w_q.pow_(self.norm)
                
                err = torch.sum(w_q, 1)
                tmp = err < best
                if torch.any(tmp):
                    # _p[tmp] = p
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

    def quantize(self, x):
        if self.ready():
            if self.pack:
                return self._quantize(x, self.scale, self.zero, self.maxq), self.scale, self.zero
            return self._quantize(x, self.scale, self.zero, self.maxq)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)


# random_tensor = torch.randn(10, 10)
# mask = torch.randn(10, 10) < 0.5  # Generate a mask where about 50% are True
# random_tensor[mask] = 0 

# sign_tensor = torch.sign(random_tensor)

# print(random_tensor)
# print(sign_tensor)

# sign_tensor = torch.where(random_tensor >= 0, 1, -1)
# print(sign_tensor)
