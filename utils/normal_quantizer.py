import torch

@torch.no_grad()
def binary(x):
    scale_tensor = torch.mean(torch.abs(x), dim=1) * 2
    zero = 0.5 * torch.ones_like(scale_tensor)
    x_ = torch.zeros_like(x)
    x_ += x
    binary_slice = torch.where(x_ >= 0, 1, -1)
    binary_slice = binary_slice/2 + 0.5

    scale_tensor = scale_tensor.unsqueeze(1).expand(-1, 128)
    zero = zero.unsqueeze(1).expand(-1, 128)
    return scale_tensor * (binary_slice - zero)

@torch.no_grad()
def _quantize(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)

def quantize(w, wbit):
    perchannel = True
    weight = True
    dev = w.device
    maxq = torch.tensor(2 ** wbit - 1)
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
    w = _quantize(w, scale, zero, maxq)
    return w

def normal_quantize(w=None, blocksize=128, wbit=2):
    columns = w.shape[1]
    w_q = torch.zeros_like(w)
    w_q = w_q.to(w.device)
    for i1 in range(0, columns, blocksize):
        i2 = min(i1 + blocksize, columns)
        count = i2 - i1

        W1 = w[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        if wbit == 1:
            Q1 = binary(W1)
        else:
            Q1 = quantize(W1, wbit)
        
        w_q[:, i1:i2] = Q1
    return w_q