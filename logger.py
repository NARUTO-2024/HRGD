import os
import numpy as np
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only


class ImageLogger(Callback):
    def __init__(self, batch_frequency=200, max_images=4, clamp=True, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False,
                 log_images_kwargs=None):
        super().__init__()

        # 初始化参数
        self.rescale = rescale  # 是否对图像进行缩放
        self.batch_freq = batch_frequency  # 记录图像的频率（每多少个批次记录一次）
        self.max_images = max_images  # 每次记录的最大图像数量
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]  # 日志记录步数
        self.clamp = clamp  # 是否限制图像值范围
        self.disabled = disabled  # 是否禁用记录
        self.log_on_batch_idx = log_on_batch_idx  # 是否根据批次索引进行记录
        self.log_images_kwargs = log_images_kwargs if log_images_kwargs else {}  # 日志记录图像的其他参数
        self.log_first_step = log_first_step  # 是否记录第一步

    @rank_zero_only  # 仅在主进程中执行
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
    # 记录图像到本地文件夹
        root = os.path.join(save_dir, "image_log", split)
        print(f'save_dir: {save_dir}')
        for k in images:
            # 将图像拼接为一个网格
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # 数据范围缩放到 [0, 1]

            # 转换形状
            grid = grid.permute(1, 2, 0)  # 转换为 [height, width, channels]
            grid = grid.squeeze()  # Squeeze去掉单一维度
            grid = grid.numpy()  # 转换为 NumPy 数组
            grid = (grid * 255).astype(np.uint8)  # 转换为 uint8 类型

            # 构建文件名
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(k, global_step, current_epoch, batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)  # 创建目录

            # 保存图像
            Image.fromarray(grid).save(path)
            print(f'path: {path}')

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        # 记录图像
        check_idx = batch_idx  # 当前批次索引
        # 检查是否需要记录
        if (self.check_frequency(check_idx) and
                hasattr(pl_module, "log_images") and
                callable(pl_module.log_images) and
                batch_idx ==0 and
                self.max_images > 0):
            logger = type(pl_module.logger)
            is_train = pl_module.training  # 检查当前是否在训练模式
            if is_train:
                pl_module.eval()  # 切换为评估模式

            with torch.no_grad():  # 不计算梯度
                images = pl_module.log_images(batch= batch, batch_idx= batch_idx, split=split, **self.log_images_kwargs)  # 获取需记录的图像

            for k in images:
                N = min(images[k].shape[0], self.max_images)  # 获取最大图像数
                images[k] = images[k][:N]  # 仅保留 N 张图像
                print(f'images[k] shape: {images[k].shape}')
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()  # 将张量移至 CPU
                    if self.clamp:
                        images[k] = torch.clamp(images[k], -1., 1.)  # 限制值范围
                rgb_image = images[k][:, [4, 16, 22], :, :]
                images[k] = rgb_image

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)  # 调用本地记录函数

            if is_train:
                pl_module.train()  # 切换回训练模式

    def check_frequency(self, check_idx):
        # 检查是否到达记录频率
        return check_idx % self.batch_freq == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        # 在每个训练批次结束时调用
        if not self.disabled:  # 如果没有禁用
            self.log_img(pl_module, batch, batch_idx, split="train")  # 记录图像