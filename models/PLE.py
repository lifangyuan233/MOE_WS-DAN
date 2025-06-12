import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from .wsdan import WSDAN
from models.wsdan import WSDAN
import config
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment

class Expert_net(nn.Module):
    """卷积专家基元，包含残差连接"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(Expert_net, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels))\
            
        # 通道注意力
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, max(out_channels//16, 4), 1),
            nn.ReLU(),
            nn.Conv2d(max(out_channels//16, 4), out_channels, 1),
            nn.Sigmoid())
        
        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels))
            
    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv(x)
        x = x * self.se(x)  # 通道注意力加权
        return F.relu(x + residual)
    

class CGCBlock(nn.Module):
    """层级化专家模块 (包含任务专属和共享专家)"""
    # 64/128   128  2  1  3
    def __init__(self, in_channels, expert_channels, num_shared_experts=2, GateNum=1):
        super(CGCBlock, self).__init__()

        self.GateNum = GateNum #输出几个Gate的结果，2表示最后一层只输出两个任务的Gate，3表示还要输出中间共享层的Gate
        
        '''两个任务模块，一个共享模块'''
        self.n_share = 1

        self.shared_experts = nn.ModuleList([
            Expert_net(in_channels, expert_channels) 
            for _ in range(num_shared_experts)])

        self.gateS = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_shared_experts),
            nn.Softmax(dim=1))

        

    def forward(self, x_S):
        bs, _, h, w = x_S.shape
        
        # 生成专家特征
        # 各专家输出形状: (bs, ec, h, w)
        experts_S_out = [e(x_S) for e in self.shared_experts]

        gate_weights_S = self.gateS(x_S).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (bs, num_experts, 1, 1, 1)

        combined_S = torch.stack(experts_S_out, dim=1)  # (bs, num_total, ec, h, w)
        Gate_S_out = (combined_S * gate_weights_S).sum(dim=1)

        
        return Gate_S_out



class DeepExpertNetwork(nn.Module):
    """深度多专家网络"""
    def __init__(self, in_channels=3, base_channels=64, expert_channels=128, num_blocks=2, num_shared_experts=2, mode='train'):
        super(DeepExpertNetwork, self).__init__()
        self.mode = mode
        # 基础特征提取
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, 1, 1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2))
        
        # 构建多层专家模块
        self.blocks = nn.ModuleList()
        current_channels = base_channels
        for i in range(num_blocks):
            # 在特定层进行下采样
            stride = 2 if i in [1, num_blocks//2 + 1] else 1
            
            block = CGCBlock(in_channels=current_channels, expert_channels=expert_channels, num_shared_experts=num_shared_experts, GateNum=1) # 最后一层只用任务门控
            self.blocks.append(block)
            current_channels = expert_channels
        
        # 多任务输出头
        self.taskA_head = WSDAN(num_classes=2, M=config.num_attentions, net=config.net, pretrained=True, mode=self.mode)
        self.taskB_head = smp.Unet(encoder_name="resnext50_32x4d", encoder_weights="imagenet", in_channels=128, classes=3, activation='sigmoid')



    def forward(self, x):
        x0 = self.stem(x)  # 初始下采样

        s = x0

        for index, block in enumerate(self.blocks):
            s = block(s)


        # print(a.shape, b.shape)   # a: [B, 128, 224, 224])   # b: [B, 128, 224, 224])
        # 任务预测
        

        if self.training:
            # print("PLE处于training")
            y_pred_raw, feature_matrix, attention_map = self.taskA_head(s)
            outB = self.taskB_head(s)
            # feature_matrix torch.Size([B, 65536])
            # y_pred_raw torch.Size([B, 2])
            # y_pred_crop torch.Size([B, 2])
            # y_pred_drop torch.Size([B, 2])
            # outB torch.Size([B, 2, 224, 224])

            

            with torch.no_grad():
                crop_images = batch_augment(x, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)
            
            crop_images = self.stem(crop_images)
            # crop images forward
            s0 = crop_images

            for index, block in enumerate(self.blocks):
                s0 = block(s0)


            y_pred_crop, _, _ = self.taskA_head(s0)

            with torch.no_grad():
                drop_images = batch_augment(x, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))

            drop_images = self.stem(drop_images)
            s1 = drop_images

            for index, block in enumerate(self.blocks):
                s1 = block(s1)
            # drop images forward
            y_pred_drop, _, _ = self.taskA_head(s1)
            
            return s1, s1, feature_matrix, y_pred_raw,  y_pred_crop, y_pred_drop, outB
        
        else:
            # print("PLE处于testing")
            with torch.no_grad():
                y_pred_raw, feature_matrix, attention_map = self.taskA_head(s)
                outB = self.taskB_head(s)

                crop_images = batch_augment(x, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)

                crop_images = self.stem(crop_images)
                s0 = crop_images

                for index, block in enumerate(self.blocks):
                    s0 = block(s0)
                        
                y_pred_crop, _, _ = self.taskA_head(s0)

                
                return s0, s0, feature_matrix, y_pred_raw,  y_pred_crop, None, outB

# # 示例使用
# model = DeepExpertNetwork(mode='train')
# x = torch.randn(1, 3, 448, 448)
# a, b, feature_matrix, y_pred_raw,  y_pred_crop, y_pred_drop, outB = model(x)
# print("a", a.shape)  # ([1, 128, 224, 224])
# print("b", b.shape)  # ([1, 128, 224, 224])
# print("feature_matrix", feature_matrix.shape)  # ([1, 65536])
# print("y_pred_raw", y_pred_raw.shape)  # ([1, 2])
# print("y_pred_crop", y_pred_crop.shape)  # ([1, 2])
# print("y_pred_drop", y_pred_drop.shape)  # ([1, 2])
# print("outB", outB.shape)  # ([1, 3, 224, 224])