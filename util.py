# adopted from
# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py
# and
# https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
# and
# https://github.com/openai/guided-diffusion/blob/0ba878e517b276c45d1195eb29f6f5f72659a05b/guided_diffusion/nn.py
#
# thanks!


import os
import math
import torch
import torch.nn as nn
import numpy as np
from einops import repeat
from utils import exists
from functools import partial
import torchvision.transforms as transforms
import torchvision.models as models
from scipy.linalg import sqrtm
from PIL import Image



def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas


def make_ddim_timesteps(ddim_discr_method, num_ddim_timesteps, num_ddpm_timesteps, verbose=True):
    if ddim_discr_method == 'uniform':
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == 'quad':
        ddim_timesteps = ((np.linspace(0, np.sqrt(num_ddpm_timesteps * .8), num_ddim_timesteps)) ** 2).astype(int)
    else:
        raise NotImplementedError(f'There is no ddim discretization method called "{ddim_discr_method}"')

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f'Selected timesteps for ddim sampler: {steps_out}')
    return steps_out


def make_ddim_sampling_parameters(alphacums, ddim_timesteps, eta, verbose=True):
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt((1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev))
    if verbose:
        print(f'Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}')
        print(f'For the chosen value of eta, which is {eta}, '
              f'this results in the following sigma_t schedule for ddim sampler {sigmas}')
    return sigmas, alphas, alphas_prev


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


def extract_into_tensor(a, t, x_shape):
    # 获取时间步 t 的批量大小 b
    b, *_ = t.shape

    # 从张量 a 中提取与时间步 t 对应的最后一维元素
    out = a.gather(-1, t)

    # 将提取的结果 reshaped 为 (b, 1, 1, ..., 1) 的形状
    # 其中 1 的个数与 x_shape 的维度（去掉第一个维度）相同
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def checkpoint(func, inputs, params, flag):
    """
    Evaluate a function without caching intermediate activations, allowing for
    reduced memory at the expense of extra compute in the backward pass.
    :param func: the function to evaluate.
    :param inputs: the argument sequence to pass to `func`.
    :param params: a sequence of parameters `func` depends on but does not
                   explicitly take as arguments.
    :param flag: if False, disable gradient checkpointing.
    """
    if flag:
        args = tuple(inputs) + tuple(params)
        return CheckpointFunction.apply(func, len(inputs), *args)
    else:
        return func(*inputs)


class CheckpointFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad(), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs):
            # Fixes a bug where the first op in run_function modifies the
            # Tensor storage in place, which is not allowed for detach()'d
            # Tensors.
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def timestep_embedding(timesteps, dim, max_period=10000, repeat_only=False):
    """
    创建正弦时间步嵌入。

    :param timesteps: 一维张量，包含 N 个索引，每个批次元素一个。
                      这些值可以是分数。
    :param dim: 输出嵌入的维度。
    :param max_period: 控制嵌入的最小频率。
    :return: 一个形状为 [N x dim] 的位置嵌入张量。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 如果 repeat_only 参数为 False，则计算正弦和余弦的嵌入
    if not repeat_only:
        half = dim // 2  # 计算要使用的频率数量（嵌入维度的一半）

        # 创建频率张量，使用指数函数调整频率
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device)  # 移动到与 timesteps 相同的设备上

        # 将时间步与频率相乘，生成正弦和余弦函数的输入

        args = timesteps[:, None].float() * freqs[None]

            # 计算正弦和余弦嵌入，并在最后一个维度上连接
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        # 如果 embedding 维度是奇数，则在最后一维添加一个零向量
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    else:
        # 如果 repeat_only 为 True，则直接复制时间步，使每个时间步的值在嵌入的所有维度上相同
        embedding = repeat(timesteps, 'b -> b d', d=dim)

    # 返回生成的嵌入，形状为 [N, dim]
    return embedding


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """
    Scale the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    return GroupNorm32(32, channels)


# PyTorch 1.7 has SiLU, but we support PyTorch 1.5.
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """
    Create a linear module.
    """
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D average pooling module.
    """
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def to_torch(array):
    if isinstance(array, np.ndarray):
        # 如果是 numpy 数组，将其转换为 PyTorch tensor，并指定 dtype
        return torch.from_numpy(array).to(dtype=torch.float32).clone()
    elif isinstance(array, torch.Tensor):
        # 如果是 PyTorch tensor，使用 clone() 保持相同的 requires_grad 属性，同时转换数据类型
        return array.clone().to(dtype=torch.float32)
    else:
        raise TypeError("Input must be a numpy array or a PyTorch tensor.")



def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def q_sample(x_start, t, noise=None):  #images：[10,:,128,128]  timestep:[10(batch_size)]  noisy：[10,:,128,128]
    # 对噪声进行初始化，如果未提供噪声，则生成与 x_start 形状相同的随机噪声

    # 根据给定的时间步 t 和起始样本 x_start 计算并返回加噪声的样本
    return (
        # 提取平方根的累积 alpha 值并与 x_start 进行逐元素相乘
            extract_into_tensor(register_schedule_sqrt_alphas_cumprod(timesteps=1000, linear_start=0.00085, linear_end=0.0120), t, x_start.shape) * x_start +

            # 提取平方根的 (1 - alpha) 值并与噪声进行逐元素相乘
            extract_into_tensor(register_schedule_sqrt_one_minus_alphas_cumprod(timesteps=1000, linear_start=0.00085, linear_end=0.0120), t, x_start.shape) * noise
    )

def register_schedule_sqrt_alphas_cumprod(given_betas=None, beta_schedule="linear", timesteps=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if exists(given_betas):
        betas = given_betas
    else:
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                   cosine_s=cosine_s)

    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

    timesteps, = betas.shape
    num_timesteps = int(timesteps)
    linear_start = linear_start
    linear_end = linear_end
    assert alphas_cumprod.shape[0] == num_timesteps, 'alphas have to be defined for each timestep'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return to_torch(np.sqrt(alphas_cumprod)).to(device) #返回一个长度为[1000]的一维张量

def register_schedule_sqrt_one_minus_alphas_cumprod(given_betas=None, beta_schedule="linear", timesteps=1000, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if exists(given_betas):
        betas = given_betas
    else:
        betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                   cosine_s=cosine_s)

    alphas = 1. - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

    timesteps, = betas.shape
    num_timesteps = int(timesteps)
    linear_start = linear_start
    linear_end = linear_end
    assert alphas_cumprod.shape[0] == num_timesteps, 'alphas have to be defined for each timestep'

    to_torch = partial(torch.tensor, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return to_torch(np.sqrt(1. - alphas_cumprod)).to(device)  #返回一个长度为[1000]的一维张量
