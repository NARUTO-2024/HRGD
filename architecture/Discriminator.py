import torch
import torch.nn as nn
from openai_ldm.openaimodel import UNetModel
from util import normalization


from util import timestep_embedding


class Discriminator(UNetModel):
    def __init__(
        self,
        image_size,           # 输入图像尺寸，例如 32
        in_channels,          # 输入通道数，例如 3（RGB 图像）
        model_channels,       # 基础通道数，例如 64
        num_res_blocks: 2,       # 每层残差块数量，例如 2
        attention_resolutions, # 注意力机制的分辨率，例如 [4, 2, 1]
        channel_mult: [ 1, 2, 4, 4 ], # 通道倍增，例如 (1, 2, 4, 8)
        num_heads: 8,          # 注意力头数量
        dropout=0.0,          # Dropout 概率
        patch_size= 2,         # PatchGAN 中的 Patch 大小
        **kwargs              # 其余参数传递给父类
    ):
        super().__init__(
            image_size=image_size,
            in_channels=in_channels,
            out_channels=1,  # 输出通道为 1，用于判断每个 Patch 的真假
            model_channels=model_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            channel_mult=channel_mult,
            num_heads=num_heads,
            dropout=dropout,
            **kwargs,
        )
        # 修改最后输出层：输出一个概率图而不是标量
        self.final_out = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            nn.Conv2d(model_channels, 1, kernel_size=patch_size, stride=1, padding=patch_size // 2),
        )

    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False,
                **kwargs):  # (x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=False)
        # 定义前向传播方法

        hs = []  # 用于存储每一层的输出特征图
        with torch.no_grad():  # 禁用梯度计算，节省显存，常用于评估或推理阶段
            # 获取时间步的嵌入  t_emb: [10, 320]
            t_emb = timestep_embedding(timesteps, self.model_channels,
                                       repeat_only=False)  # timestep:[10(batch_size)]   model_channels = 320,
            # 获取时间步的嵌入并通过嵌入层进行处理  emb:[10, 1280]
            emb = self.time_embed(t_emb)
            # 将输入张量转换为指定的数据类型  h:[10,:,128,128]
            x = x.type(self.dtype)

            # 输入块的前向传播
            for module in self.input_blocks:
                x = module(x, emb, context)  # 逐层处理输入，通过输入模块
                hs.append(x)  # 保存每一层的特征图

            # 中间块的前向传播
            x = self.middle_block(x, emb, context)

        # 如果有控制信号，则在当前特征图上添加控制信号
        if control is not None:
            x += control.pop()  # 从控制列表中取出并应用控制信号

        # 输出块的前向传播
        for i, module in enumerate(self.output_blocks):
            # 判断是否只使用中间控制或控制信号是否为 None
            if only_mid_control or control is None:
                x = torch.cat([x, hs.pop()], dim=1)  # 如果没有控制，拼接当前特征图和输入块的特征图
            else:
                x = torch.cat([x, hs.pop() + control.pop()], dim=1)  # 否则拼接当前特征图和控制信号

            x = module(x, emb, context)  # 通过输出模块处理拼接后的特征图

        features = x.type(x.dtype)

        patch_map = self.final_out(features)
        # 对第三、第四个维度（H 和 W）进行归一化（求平均值），并去掉多余维度
        scale = patch_map.mean(dim=[2, 3])  # 去掉 H 和 W 两个维度
        return patch_map, scale