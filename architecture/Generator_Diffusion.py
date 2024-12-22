import torch
from openai_ldm.openaimodel import UNetModel

from util import timestep_embedding



class generator_diffusion(UNetModel):  # ControlledUnetModel 继承自 UNetModel  noisy：[10,:,128,128]  timestep:[10(batch_size)]  control= None, only_mid_control= False
    def forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, **kwargs):  #(x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=False)
        # 定义前向传播方法

        hs = []  # 用于存储每一层的输出特征图
        with torch.no_grad():  # 禁用梯度计算，节省显存，常用于评估或推理阶段
            # 获取时间步的嵌入  t_emb: [10, 320]
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)  #timestep:[10(batch_size)]   model_channels = 320,
            # 获取时间步的嵌入并通过嵌入层进行处理  emb:[10, 1280]
            emb = self.time_embed(t_emb)
            # 将输入张量转换为指定的数据类型  h:[10,:,128,128]
            h = x.type(self.dtype)

            # 输入块的前向传播
            for module in self.input_blocks:
                h = module(h, emb, context)  # 逐层处理输入，通过输入模块
                hs.append(h)  # 保存每一层的特征图

            # 中间块的前向传播
            h = self.middle_block(h, emb, context)

        # 如果有控制信号，则在当前特征图上添加控制信号
        if control is not None:
            h += control.pop()  # 从控制列表中取出并应用控制信号

        # 输出块的前向传播
        for i, module in enumerate(self.output_blocks):
            # 判断是否只使用中间控制或控制信号是否为 None
            if only_mid_control or control is None:
                h = torch.cat([h, hs.pop()], dim=1)  # 如果没有控制，拼接当前特征图和输入块的特征图
            else:
                h = torch.cat([h, hs.pop() + control.pop()], dim=1)  # 否则拼接当前特征图和控制信号

            h = module(h, emb, context)  # 通过输出模块处理拼接后的特征图

        h = h.type(x.dtype)  # 将输出特征图转换为与输入相同的数据类型
        return self.out(h)  # 返回最终的输出