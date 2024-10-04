import torch
import numpy as np
from scipy.interpolate import interp1d 

# 对mask后的tensor进行插值。interp1d()函数的kind参数决定了插值方法：nearest(最近邻)、linear(线性)
def interpolate(tensor,device,kind):
    tensor = tensor.cpu()
    B, T, N = tensor.shape
    filled_tensor = torch.zeros_like(tensor)
    for b in range(B):
        for n in range(N):
            data = tensor[b, :, n].numpy()
            mask = data != 0
            if np.any(mask):
                indices = np.arange(T)
                interp_func = interp1d(indices[mask], data[mask], bounds_error=False, fill_value="extrapolate",kind=kind)
                filled_tensor[b, :, n] = torch.tensor(interp_func(indices))
            else:
                filled_tensor[b, :, n] = torch.tensor(data)  # No non-zero data to interpolate

    return filled_tensor.to(device)

# 该函数用于计算掩码后在时间维度上的均值，并将均值填补到掩码位置
def masked_mean(x,mask):
    _,T,_ = x.shape
    # 计算时间维度上未被掩码的值
    count = mask.sum(dim=1)
    # 计算均值 mean.shape=[B,N]
    mean = x.sum(dim=1)/count.clamp(min=1)
    # 调整维度 [B,N]->[B,1,N]->[B,T,N]
    mean = mean.unsqueeze(1)
    mean = mean.expand(-1,T,-1)
    return mean
