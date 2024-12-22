import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torch.utils.data import DataLoader
from torch.autograd import Variable

from logger import ImageLogger
from Diffusion import CombinedModel

import os
from scipy.io import loadmat
from hsi_dataset import TrainDataset, ValidDataset

from architecture.Generator_Diffusion import generator_diffusion
from architecture.Generator_Net import generator_net
from architecture.Generator_Diffusion_True import generator_diffusion_true
from architecture.Discriminator import Discriminator

from util import q_sample
from utils import time2file_name, reset_grad
import datetime

# 解析命令行参数  在python中使用input读入数据，使用命令行参数来解析输入
parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument("--batch_size", type=int, default=2, help="batch size")  # 批大小 10
parser.add_argument("--end_epoch", type=int, default=300, help="number of epochs")  # 训练轮数 100
parser.add_argument("--end_epoch_net", type=int, default=300, help="number of epochs")  # 训练轮数
parser.add_argument("--timestep", type=int, default=1000, help='number of timestep')
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")  # 初始学习率  4e-4
parser.add_argument("--outf", type=str, default='./exp/mst_plus_plus/',
                    help='path log files')  # 日志文件路径  ./exp/mst_plus_plus/
parser.add_argument("--data_root", type=str, default='../autodl-tmp/dataset/')  # 数据集路径  ../dataset/
parser.add_argument("--patch_size", type=int, default=32, help="patch size")  # 图像块大小  128
parser.add_argument("--stride", type=int, default=8, help="stride")  # 步长  8
parser.add_argument("--gpu_id", type=str, default='0', help='path log files')  # GPU ID  0
opt = parser.parse_args()  # 将用户通过命令行提供的参数解析为一个对象，然后将这个对象存储在opt变量中

# 设置CUDA设备
os.environ[
    "CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'  # os.environ是Python中的环境变量字典  "CUDA_DEVICE_ORDER"是一个CUDA环境变量的名称  'PCI_BUS_ID'是这个环境变量的值。  这个环境变量告诉PyTorch如何按照PCI总线ID来排序和选择GPU。
os.environ[
    "CUDA_VISIBLE_DEVICES"] = opt.gpu_id  # opt.gpu_id是从之前解析的命令行参数中获取的GPU ID，告诉PyTorch哪些GPU是可见的，也就是哪些GPU可以被PyTorch使用。

# 加载数据集
print("\nloading dataset ...")
train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, bgr2rgb=True, arg=True,
                          stride=opt.stride)  # crop_size表示图像块大小，arg 表示是否进行数据增强 ，bgr2rgb表示是否从BGR图像转换为RGB
print(f"Iteration per epoch: {len(train_data)}")
val_data = ValidDataset(data_root=opt.data_root, crop_size=opt.patch_size, bgr2rgb=True,
                          stride=opt.stride)
print("Validation set samples: ", len(val_data))
# val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
# print("Validation set samples: ", len(val_data))

# 设置迭代次数
per_epoch_iteration = 50
total_iteration = per_epoch_iteration * opt.end_epoch  # 迭代次数*训练轮数=总迭代次数
total_iteration_net = per_epoch_iteration * opt.end_epoch_net

# 创建输出路径
date_time = str(datetime.datetime.now())  # 只有一个datatime是不行的，datetime.now()是错误的用法，因为它返回的是time对象，只包含时间信息，没有日期信息。
date_time = time2file_name(date_time)  # time2file_name是一个自定义函数，它的作用是将日期和时间字符串转换为适合文件名的格式。
opt.outf = opt.outf + date_time  # opt.outf 日志文件路径  将转换后的日期和时间字符串添加到opt.outf的末尾，形成一个新的输出路径。
if not os.path.exists(opt.outf):  # 使用Python的os模块检查opt.outf指定的路径是否存在 如果路径不存在，则执行下面的代码。
    os.makedirs(opt.outf)  # os为python内置与操作系统交互的模块，makedirs()是os模块中的一个函数，用于创建目录。

generator_diffusion = generator_diffusion(image_size=32,
                                          in_channels=3,
                                          out_channels=3,
                                          model_channels=320,
                                          attention_resolutions=[4, 2, 1],
                                          num_res_blocks=2,
                                          channel_mult=[1, 2, 4, 4],
                                          num_heads=8,
                                          use_spatial_transformer=False,
                                          transformer_depth=1,
                                          context_dim=768,
                                          use_checkpoint=False,
                                          legacy=False)  # 扩散模型
generator_diffusion_true = generator_diffusion_true(image_size=32,
                                                    in_channels=3,
                                                    out_channels=31,
                                                    model_channels=320,
                                                    attention_resolutions=[4, 2, 1],
                                                    num_res_blocks=2,
                                                    channel_mult=[1, 2, 4, 4],
                                                    num_heads=8,
                                                    use_spatial_transformer=False,
                                                    transformer_depth=1,
                                                    context_dim=768,
                                                    use_checkpoint=False,
                                                    legacy=False)
generator_net = generator_net(image_size=32,
                              in_channels=3,
                              hint_channels=31,
                              model_channels=320,
                              attention_resolutions=[4, 2, 1],
                              num_res_blocks=2,
                              channel_mult=[1, 2, 4, 4],
                              num_heads=8,
                              use_spatial_transformer=False,
                              transformer_depth=1,
                              context_dim=768,
                              use_checkpoint=False,
                              legacy=False)  # control net
discriminator = Discriminator(image_size=32,
                              in_channels=31,
                              model_channels=320,
                              attention_resolutions=[4, 2, 1],
                              num_res_blocks=2,
                              channel_mult=[1, 2, 4, 4],
                              num_heads=8,
                              use_spatial_transformer=False,
                              transformer_depth=1,
                              context_dim=768,
                              use_checkpoint=False,
                              legacy=False)  # 判别器模型

# 移动模型和损失函数到GPU
if torch.cuda.is_available():  # 用于检查当前的PyTorch环境是否支持CUDA，即是否有可用的CUDA支持
    generator_diffusion.cuda()  # 如果CUDA支持可用，则将模型（model）转移到GPU上
    generator_net.cuda()
    discriminator.cuda()
    generator_diffusion_true.cuda()

# 如果有多个GPU, 使用DataParallel
if torch.cuda.device_count() > 1:
    generator_diffusion = nn.DataParallel(generator_diffusion)
    generator_net = nn.DataParallel(generator_net)
    discriminator = nn.DataParallel(discriminator)

# 创建优化器和学习率调度器
optimizer_diffusion = optim.Adam(generator_diffusion.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
optimizer_net = optim.Adam(generator_net.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
optimizer_disc = optim.Adam(discriminator.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))
optimizer_true = optim.Adam(generator_diffusion_true.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))

scheduler_diffusion = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_diffusion, total_iteration, eta_min=1e-6)
scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_net, total_iteration, eta_min=1e-6)
scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc, total_iteration, eta_min=1e-6)
scheduler_true = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_true, total_iteration, eta_min=1e-6)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    global timestep
    cudnn.benchmark = True  # 可以加速卷积层的前向和反向传播，但可能会牺牲一些准确性。cudnn是PyTorch中的一个模块，它提供了CUDA加速的深度神经网络库 benchmark = True是cudnn模块的一个属性，用于优化卷积神经网络的执行。
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=0,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    for i, (images, labels) in enumerate(train_loader):
        timestep = torch.randint(0, opt.timestep, (images.shape[0],)).long().to(device)
        break

    d_step = 5
    iteration = 0  # 用于记录当前的迭代次数
    record_mrae_loss = 1000  # 记录最小MRAE损失值  初始化值大，训练时逐渐缩小

    sd_locked = True
    only_mid_control = False
    logger_freq = 300
    
    num_params_1 = count_parameters(generator_diffusion)
    num_params_2 = count_parameters(generator_diffusion_true)
    num_params_3 = count_parameters(generator_net)
    num_params_4 = count_parameters(discriminator)
    num_params=num_params_1+num_params_2+num_params_3+num_params_4
    print(f"模型的参数量: {num_params}")
    
    loss_records = {"i": [], "D_loss": [], "G_loss": []}
    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)  # 创建保存文件的目录
    print(f"当前的total_iteration是: {total_iteration}")
    while iteration < 100:
        generator_diffusion.train()
        discriminator.train()
        print(f"当前的iteration是: {iteration}")
        iteration = iteration + 1
        for i, (images, labels) in enumerate(train_loader):  # 返回的都是[:,128,128]大小的块
            print(f"当前的i是: {i}")
            if i >= 500:
                break
            batch_size = labels.size(0)  # 获取当前batch的大小

            images, labels = images.float().to(device), labels.float().to(device)  # 转为float并移动到GPU
            images.requires_grad_()
            labels.requires_grad_()

            generated_noise = lambda: torch.randn_like(images)  # 噪声生成函数
            noise_y = generated_noise()
            noise_y.requires_grad_()
            noise = q_sample(x_start=images, t=timestep, noise=noise_y)  # 添加噪声
            noise.requires_grad_()

            optimizer_disc.zero_grad()

            D_real_patch_map, D_real = discriminator(x=labels, timesteps=timestep, control=None, only_mid_control=False)
            D_real_mean = D_real.mean()

            for j in range(10):
                noise_true = generator_diffusion(x=noise, timesteps=timestep, control=None, only_mid_control=False)
                noise = noise - noise_true
                noise_y = generated_noise()
                noise = q_sample(x_start=noise, t=timestep, noise=noise_y)

            generated_images = generator_diffusion_true(x=noise, timesteps=timestep, control=None,
                                                        only_mid_control=False)

            D_fake_patch_map, D_fake = discriminator(x=generated_images, timesteps=timestep, control=None,
                                                     only_mid_control=False)
            D_fake_mean = D_fake.mean()

            D_loss = D_fake_mean - D_real_mean

            # 反向传播
            D_loss.backward()
            optimizer_disc.step()
            scheduler_disc.step()

            # 仅在每5个判别器更新后更新一次生成器
            if i % d_step == 0:
                reset_grad(optimizer_diffusion, discriminator)

                noise_y = generated_noise()
                noise = q_sample(x_start=images, t=timestep, noise=noise_y)
                for k in range(10):
                    noise_true = generator_diffusion(x=noise, timesteps=timestep, control=None, only_mid_control=False)
                    noise = noise - noise_true
                    noise_y = generated_noise()
                    noise = q_sample(x_start=noise, t=timestep, noise=noise_y)

                generated_images = generator_diffusion_true(x=noise, timesteps=timestep, control=None,
                                                            only_mid_control=False)

                D_fake_patch_map, D_fake = discriminator(x=generated_images, timesteps=timestep, control=None,
                                                         only_mid_control=False)
                G_loss = -torch.mean(D_fake)

                # 反向传播
                G_loss.backward()
                optimizer_diffusion.step()
                optimizer_true.step()
                scheduler_diffusion.step()
                scheduler_true.step()

                # 保存D_loss和G_loss到CSV文件
                loss_records["i"].append(i)
                loss_records["D_loss"].append(D_loss.item())
                loss_records["G_loss"].append(G_loss.item())

                if len(loss_records["i"]) % 5 == 0:
                    pd.DataFrame(loss_records).to_csv(os.path.join(output_dir, "loss_records.csv"), index=False)

                # 每500次打印D_loss和G_loss时，生成折线图
                if i % 10 == 0 and i > 1 and iteration == 97:
                    plt.figure()
                    plt.plot(loss_records["i"], loss_records["D_loss"], label="D_loss", color="blue")
                    plt.xlabel("Iterations")
                    plt.ylabel("D_loss")
                    plt.title("Discriminator Loss")
                    plt.legend()
                    plt.savefig(os.path.join(output_dir, "D_loss_plot.png"))
                    plt.close()

                    plt.figure()
                    plt.plot(loss_records["i"], loss_records["G_loss"], label="G_loss", color="red")
                    plt.xlabel("Iterations")
                    plt.ylabel("G_loss")
                    plt.title("Generator Loss")
                    plt.legend()
                    plt.savefig(os.path.join(output_dir, "G_loss_plot.png"))
                    plt.close()

    # 第二功能训练
    model = CombinedModel(generator_diffusion, generator_net, generator_diffusion_true, val_loader)
    model.learning_rate = optimizer_net.param_groups[0]['lr']
    model.sd_locked = sd_locked
    model.only_mid_control = only_mid_control

    logger = ImageLogger(batch_frequency=logger_freq)
    trainer = pl.Trainer(devices=1, accelerator='gpu', precision=32, callbacks=[logger], val_check_interval=0.0001,
                         max_epochs=10)

    trainer.fit(model, train_loader)

    return 0


def project_validation_images(generator, generator_true, val_loader, device, output_dir, iteration):
    """
    从验证集加载前50张图片，并用生成器生成50张新图片。
    使用t-SNE将这100张图片投影到二维空间并保存结果。

    参数:
        generator (torch.nn.Module): 生成器模型
        val_loader (torch.utils.data.DataLoader): 验证集数据加载器
        device (torch.device): 运行设备（CPU 或 GPU）
        output_dir (str): 保存结果的目录路径
        iteration (int): 当前迭代次数，用于保存文件名
    """
    os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在

    real_images = []
    real_labels = []
    generated_images_list = []

    # 从验证集中逐张提取前50张图片
    for idx, (image, label) in enumerate(val_loader):
        image, label = image.float().to(device), label.float().to(device)
        if idx >= 10:  # 仅提取前50张图片
            break

        generated_noise = lambda: torch.randn_like(image)
        noise_y = generated_noise()
        noise = q_sample(x_start=image, t=timestep, noise=noise_y)
        for m in range(10):
            noise_true = generator(x=noise, timesteps=timestep, control=None, only_mid_control=False)
            noise = noise - noise_true
            noise_y = generated_noise()
            noise = q_sample(x_start=noise, t=timestep, noise=noise_y)

        generated_images = generator_true(x=noise, timesteps=timestep, control=None,
                                                    only_mid_control=False)
        generated_images_list.append(generated_images.to(device))
        real_images.append(image.to(device))
        real_labels.append(label.to(device))

    # 将逐张提取的图片和标签拼接成一个张量
    real_labels = torch.cat(real_labels, dim=0)
    generated_images_list = torch.cat(generated_images_list, dim=0)
    with torch.no_grad():

        # 将100张图片（50张真实，50张生成）进行拼接
        data_to_project = torch.cat([real_labels.flatten(1), generated_images_list.flatten(1)])

        # 使用t-SNE投影到二维空间
        tsne = TSNE(n_components=2, random_state=42)
        projections = tsne.fit_transform(data_to_project.cpu().numpy())

        # 绘制投影结果
        plt.figure(figsize=(8, 8))
        plt.scatter(projections[:50, 0], projections[:50, 1], color="blue", label="Real Labels")
        plt.scatter(projections[50:, 0], projections[50:, 1], color="red", label="Generated Images")
        plt.legend()
        plt.title("t-SNE Projection of Validation Images")
        plt.savefig(os.path.join(output_dir, f"tsne_projection_iter_{iteration}.png"))
        plt.close()

        print(f"t-SNE投影已保存: {os.path.join(output_dir, f'tsne_projection_iter_{iteration}.png')}")
if __name__ == '__main__':  # 用于执行一个名为main的函数
    main()