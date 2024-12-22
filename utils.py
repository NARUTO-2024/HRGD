from __future__ import division

import torch
import torch.nn as nn
import logging
import numpy as np
import os

from inspect import isfunction

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_logger(file_dir):
    logger = logging.getLogger()
    fhandler = logging.FileHandler(filename=file_dir, mode='a')
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")
    fhandler.setFormatter(formatter)
    logger.addHandler(fhandler)
    logger.setLevel(logging.INFO)
    return logger

def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))

class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = torch.abs(outputs - label) / label
        mrae = torch.mean(error.view(-1))
        return mrae

class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape
        error = outputs-label
        sqrt_error = torch.pow(error,2)
        rmse = torch.sqrt(torch.mean(sqrt_error.view(-1)))
        return rmse

class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        N = im_true.size()[0]
        C = im_true.size()[1]
        H = im_true.size()[2]
        W = im_true.size()[3]
        im_true = im_true.detach()
        im_fake = im_fake.detach()
        Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
        mse = nn.MSELoss(reduce=False)
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)
        return torch.mean(psnr)

def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second
    return time_filename

def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss):
    """ Record many results."""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))
    loss_csv.flush()
    loss_csv.close

def calc_gradient_penalty(netD, timesteps, real_data, generated_data, penalty_weight=10):
    #netD是一个判别器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = real_data.size()[0]  # 获取真实数据的批次大小

    # alpha取值在0到1之间，用于插值
    alpha = torch.rand(batch_size, 1) if real_data.dim() == 2 else torch.rand(batch_size, 1, 1, 1)
    #如果 real_data.dim() == 2 返回 [batch_size, 1] ，否则返回 [batch_size, 1, 1, 1]
    alpha = alpha.expand_as(real_data)  # 与真实数据形状相同
    #用于返回一个形状与另一张量相同但共享同一数据的新的张量
    alpha = alpha.cuda()  # 将alpha转移到GPU上

    # 进行插值，生成混合样本
    interpolated = alpha * real_data + (1 - alpha) * generated_data
    interpolated.requires_grad_()  # 需要计算梯度
    interpolated = interpolated.to(device)  # 转移到GPU上

    # 计算插值样本的概率
    prob_interpolated_patch_map, prob_interpolated = netD(x= interpolated, timesteps= timesteps, control= None, only_mid_control= False)
    # 确保 timesteps 作为张量直接参与计算

    print(interpolated.grad_fn)
    # 计算插值样本的概率的梯度
    gradients = torch.autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                                    grad_outputs=torch.ones(prob_interpolated.size()).to(device),
                                    create_graph=True, retain_graph=True,
                                    allow_unused=True)[0]

    # 将梯度形状转换为(batch_size, num_features)
    gradients = gradients.view(batch_size, -1)

    # 计算梯度的范数，防止接近0的情况引发问题，加入微小值epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # 返回梯度惩罚
    return penalty_weight * ((gradients_norm - 1) ** 2).mean()

def reset_grad(*nets):
    for net in nets:
        net.zero_grad()

def exists(x):
        return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def extract_into_tensor(a, t, x_shape):
    # 获取时间步 t 的批量大小 b
    b, *_ = t.shape

    # 从张量 a 中提取与时间步 t 对应的最后一维元素
    out = a.gather(-1, t)

    # 将提取的结果 reshaped 为 (b, 1, 1, ..., 1) 的形状
    # 其中 1 的个数与 x_shape 的维度（去掉第一个维度）相同
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

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
