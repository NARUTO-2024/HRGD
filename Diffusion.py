import torch
import numpy as np
import pytorch_lightning as pl
import csv
import matplotlib.pyplot as plt
import os
import hdf5storage
import random

from util import to_torch, q_sample
from AutoencoderKL import AutoencoderKL
from distribution import DiagonalGaussianDistribution
from einops import rearrange, repeat
from utils import exists, default, extract_into_tensor, make_beta_schedule, \
    AverageMeter, Loss_MRAE, Loss_RMSE, Loss_PSNR
from torch.autograd import Variable
from torchvision.utils import make_grid


class CombinedModel(pl.LightningModule):
    """
    ControlLDM是通过控制模型增强的Latent Diffusion Model，用于条件图像生成。
    """

    def __init__(self,
                 generator_diffusion,
                 generator_net,
                 generator_diffusion_true,
                 val_loader,
                 given_betas=None,
                 beta_schedule="linear",
                 loss_type="l2",
                 timesteps=1000,
                 linear_start=0.00085,
                 linear_end=0.0120,
                 cosine_s=8e-3,
                 control_key="hint",
                 only_mid_control=False,
                 num_timesteps_cond=1,
                 cond_stage_trainable=False,
                 conditioning_key="crossattn",
                 concat_mode=True,
                 logvar_init=0.,
                 l_simple_weight=1.,
                 original_elbo_weight=0.,
                 v_posterior=0.,
                 parameterization="eps",
                 ucg_training=None,
                 first_stage_key="jpg",
                 scale_factor=1.0,
                 scale_by_std=False,
                 ):

        super(CombinedModel, self).__init__()

        self.num_timesteps_cond = default(num_timesteps_cond, 1)

        # 实例化控制模型，使用从配置文件中提取的控制阶段配置
        self.diffusion_model = generator_diffusion

        self.control_model = generator_net

        self.diffusion_image_model = generator_diffusion_true

        self.loss_type = loss_type

        # 存储控制标签键
        self.control_key = control_key  # control_key: "hint"

        # 是否仅在中间阶段进行控制
        self.only_mid_control = only_mid_control  # only_mid_control: False

        self.cond_stage_trainable = cond_stage_trainable

        self.v_posterior = v_posterior
        self.parameterization = parameterization
        self.ucg_training = ucg_training or dict()
        if self.ucg_training:
            self.ucg_prng = np.random.RandomState()

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        self.register_buffer('logvar', logvar)
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'

        # 初始化控制尺度
        self.control_scales = [1.0] * 13

        self.l_simple_weight = l_simple_weight

        self.original_elbo_weight = original_elbo_weight

        self.first_stage_key = first_stage_key
        self.encode_first_stage = AutoencoderKL(embed_dim=4,
                                                double_z=True,
                                                ch_mult=[1, 2, 4, 4],
                                                dropout=0.0,
                                                target=torch.nn.Identity,
                                                )
        self.val_loader = val_loader
        self.record_mrae_loss = 1000000
        self.criterion_mrae = Loss_MRAE()
        self.criterion_rmse = Loss_RMSE()
        self.criterion_psnr = Loss_PSNR()

        if torch.cuda.is_available():
            self.criterion_mrae.cuda()
            self.criterion_rmse.cuda()
            self.criterion_psnr.cuda()
        self.save_path = "./restruction"

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            to_torch((1. - alphas_cumprod_prev)) * to_torch(np.sqrt(alphas)) / to_torch((1. - alphas_cumprod))))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

        assert not torch.isnan(self.lvlb_weights).all()

    def forward(self, x, label, pre=1 , *args, **kwargs):
        # 生成一组随机的时间步t
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        # 调用 p_losses 方法计算损失并返回
        return self.p_losses(x_start=x, t=t, cond=x, label=label, pre=pre)

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    def q_sample(self, x_start, t, noise=None):
        # 对噪声进行初始化，如果未提供噪声，则生成与 x_start 形状相同的随机噪声
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 根据给定的时间步 t 和起始样本 x_start 计算并返回加噪声的样本
        return (
            # 提取平方根的累积 alpha 值并与 x_start 进行逐元素相乘
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +

                # 提取平方根的 (1 - alpha) 值并与噪声进行逐元素相乘
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def get_v(self, x, noise, t):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def apply_model(self, x_noisy, t, cond=None, *args, **kwargs):
        # 根据条件拼接和控制信息应用扩散模型
        # 使用控制模型进行预测
        control = self.control_model(x=cond, timesteps=t)
        control = [c * scale for c, scale in zip(control, self.control_scales)]  # 应用控制尺度
        noise = x_noisy
        generated_noise = lambda: torch.randn_like(x_noisy)

        for i in range(10):
            noise_true = self.diffusion_model(x=noise, timesteps=t, control=None,
                                              only_mid_control=False)  # noisy：[10,:,128,128]  timestep:[10(batch_size)]
            noise = noise - noise_true
            noise_y = generated_noise()
            noise = q_sample(x_start=noise, t=t, noise=noise_y)

        generated_images = self.diffusion_image_model(x=noise, timesteps=t, control=control,
                                                      only_mid_control=self.only_mid_control)

        return generated_images  # 返回预测的图像

    def p_losses(self, x_start, t, cond, label, noise=None, pre=1):
        # 如果未提供噪声，则生成与 x_start 相同形状的随机噪声
        noise = default(noise, lambda: torch.randn_like(x_start))

        # 通过 q_sample 方法生成加噪声样本 x_noisy
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # 使用模型对加噪声样本进行处理，获取模型输出
        model_output = self.apply_model(x_noisy=x_noisy, t=t, cond=cond)
        if pre ==0:
            return model_output
        # 初始化损失字典用于存储不同损失值
        loss_dict = {}
        # 根据当前是否在训练模式，选择前缀
        prefix = 'train' if self.training else 'val'

        target = label

        # 计算损失，mean=False 表示不对所有维度取均值
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        # 将简单损失值存入损失字典
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        # 获取对应时间步 t 的 logvar
        logvar_t = self.logvar[t].to(self.device)
        # 计算最终损失，包含 logvar 的影响
        loss = loss_simple / torch.exp(logvar_t) + logvar_t

        # 乘以简单损失的权重，得出最终损失
        loss = self.l_simple_weight * loss.mean()

        # 计算与变分下界 (VLB) 相关的损失，并更新损失字典
        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})

        # 将 VLB 损失加入总损失中
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        if self.global_step % 100 == 0:
            losses_rmse = AverageMeter()
            losses_psnr = AverageMeter()
            loss_rmse = self.criterion_rmse(model_output, target)
            loss_psnr = self.criterion_psnr(model_output, target)
            losses_rmse.update(loss_rmse.data)
            losses_psnr.update(loss_psnr.data)
            print(
                f'l2_loss:{loss_simple}, RMSE: {losses_rmse.avg}, PNSR:{losses_psnr.avg}')
            self.save_losses_to_csv(loss_simple, losses_rmse.avg, losses_psnr.avg, self.global_step)
        # 返回最终损失值和损失字典
        return loss, loss_dict

    def get_input(self, batch, k, bs=None, *args, **kwargs):
        # 获取输入数据，包括潜在变量和条件
        x, y = batch
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()

        x = x.permute(0, 2, 1, 3)

        if bs is not None:
            x = x[:bs]

        x = x.to(self.device)

        encoder_posterior, encoder_posterior_z = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        out = z
        return out, y

    def shared_step(self, batch, batch_idx):
        x, y = self.get_input(batch=batch, k=batch_idx)
        loss, loss_dict = self(x=x, label=y)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        print(self.global_step)
        for k in self.ucg_training:
            p = self.ucg_training[k]["p"]
            val = self.ucg_training[k]["val"]
            if val is None:
                val = ""
            for i in range(len(batch[k])):
                if self.ucg_prng.choice(2, p=[1 - p, p]):
                    batch[k][i] = val
        #if self.global_step % 1000 == 0:
            #self.validation_ste()
        loss, loss_dict = self.shared_step(batch, batch_idx)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    def validation_ste(self):

        mrae_loss, rmse_loss, psnr_loss = self.validate(self.val_loader)
        current_epoch = self.current_epoch

        #if mrae_loss < self.record_mrae_loss:
            #self.record_mrae_loss = mrae_loss

        print(
            f'epoch:{current_epoch}, MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}, record_mrae_loss:{self.record_mrae_loss}')
        self.save_losses_to_csv(mrae_loss, rmse_loss, psnr_loss, current_epoch)


    def validate(self, val_loader):
        self.eval()
        losses_mrae = AverageMeter()
        losses_rmse = AverageMeter()
        losses_psnr = AverageMeter()
        for i, (input, target) in enumerate(val_loader):
            print(f"当前的i是: {i}")
            input = input.cuda()
            target = target.cuda()
            with torch.no_grad():
                # compute output
                output = self(input, pre=0 ,label= target)
                print(f"output shape: {output.shape}")
                loss_mrae = self.criterion_mrae(output, target)
                loss_rmse = self.criterion_rmse(output, target)
                loss_psnr = self.criterion_psnr(output, target)
            # record loss
            losses_mrae.update(loss_mrae.data)
            losses_rmse.update(loss_rmse.data)
            losses_psnr.update(loss_psnr.data)
        return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg

    def save_losses_to_csv(self, l2_loss, rmse_loss, psnr_loss, step, output_dir='loss_logs'):
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 定义CSV文件名
        csv_file = os.path.join(output_dir, 'losses.csv')

        # 如果文件不存在，则写入头部
        file_exists = os.path.isfile(csv_file)

        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                # 写入头部
                writer.writerow(['step', 'l2_loss', 'RMSE', 'PSNR'])
            # 写入当前的损失值
            writer.writerow([step, l2_loss, rmse_loss, psnr_loss])

        print(f"损失值已保存到 {csv_file}")

    def configure_optimizers(self):
        # 配置优化器
        lr = self.learning_rate  # 获取学习率
        params = list(self.control_model.parameters())  # 获取控制模型参数
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())  # 添加扩散模型参数
            params += list(self.model.diffusion_model.out.parameters())  # 添加扩散模型输出参数

        # 创建AdamW优化器
        opt = torch.optim.AdamW(params, lr=lr)
        return opt  # 返回优化器
请你修改优化器，要求锁住25%的随机参数即可

    def save_matv73(self, mat_name, var_name, var):
        hdf5storage.savemat(mat_name, {var_name: var}, format='7.3', store_python_metadata=True)

    @torch.no_grad()
    def log_images(self, batch, batch_idx, N=4, n_row=5, plot_diffusion_rows=False, **kwargs):
        # 日志记录图像生成过程的图像
        var_name = 'cube'

        log = dict()  # 初始化日志字典
        out, labels = self.get_input(batch, batch_idx, bs=N)  # 获取输入
        z = out
        print(f'z shape: {z.shape}')
        # 解码潜在变量生成重建图像
        t = torch.randint(0, self.num_timesteps, (z.shape[0],), device=self.device).long()
        generated_noise = lambda: torch.randn_like(z)
        noise = generated_noise()
        x_noisy = self.q_sample(x_start=z, t=t, noise=noise)

        control = self.control_model(x=z, timesteps=t)
        control = [c * scale for c, scale in zip(control, self.control_scales)]  # 应用控制尺度

        for i in range(10):
            noise_true = self.diffusion_model(x=x_noisy, timesteps=t, control=None,
                                              only_mid_control=False)  # noisy：[10,:,128,128]  timestep:[10(batch_size)]
            x_noisy = x_noisy - noise_true
            noise_y = generated_noise()
            x_noisy = q_sample(x_start=x_noisy, t=t, noise=noise_y)

        model_output = self.diffusion_image_model(x=x_noisy, timesteps=t, control=control,
                                                  only_mid_control=self.only_mid_control)

        log["reconstruction"] = model_output
        log["conditioning"] = labels  # 编码孔径条件日志

        model_output = model_output.cpu().numpy() * 1.0  # 将结果移动到 CPU 并转换为 NumPy 数组
        result = np.minimum(model_output, 1.0)  # 确保结果不超过 1.0
        result = np.maximum(result, 0)  # 确保结果不小于 0

        mat_name = f"{self.global_step}.mat"  # 去掉原文件扩展名并加上 .mat 后缀
        mat_dir = os.path.join(self.save_path, mat_name)  # 构建完整的保存路径

        self.save_matv73(mat_dir, var_name, var=result)

        return log  # 返回日志字典