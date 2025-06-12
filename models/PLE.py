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
    def __init__(self, in_channels, expert_channels, num_task_experts=1, num_shared_experts=1, GateNum=3):
        super(CGCBlock, self).__init__()

        self.GateNum = GateNum #输出几个Gate的结果，2表示最后一层只输出两个任务的Gate，3表示还要输出中间共享层的Gate
        
        '''两个任务模块，一个共享模块'''
        self.n_task = 2
        self.n_share = 1

        self.taskA_experts = nn.ModuleList([
            Expert_net(in_channels, expert_channels) 
            for _ in range(num_task_experts)])
        
        self.shared_experts = nn.ModuleList([
            Expert_net(in_channels, expert_channels) 
            for _ in range(num_shared_experts)])
        
        self.taskB_experts = nn.ModuleList([
            Expert_net(in_channels, expert_channels) 
            for _ in range(num_task_experts)])

        # 门控网络
        self.gateA = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_task_experts + num_shared_experts),
            nn.Softmax(dim=1))

        if GateNum == 3:
            self.gateS = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_channels, 128),
                nn.ReLU(),
                nn.Linear(128, 2*num_task_experts + num_shared_experts),
                nn.Softmax(dim=1))
        
        self.gateB = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, num_task_experts + num_shared_experts),
            nn.Softmax(dim=1))
        

    def forward(self, x_A, x_S, x_B):
        bs, _, h, w = x_A.shape
        
        # 生成专家特征
        experts_A_out = [e(x_A) for e in self.taskA_experts]  # 各专家输出形状: (bs, ec, h, w)
        experts_S_out = [e(x_S) for e in self.shared_experts]
        experts_B_out = [e(x_B) for e in self.taskB_experts]
        
        # 门控权重计算
        gate_weights_A = self.gateA(x_A).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (bs, num_experts, 1, 1, 1)
        if self.GateNum == 3:
            gate_weights_S = self.gateS(x_S).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (bs, num_experts, 1, 1, 1)
        gate_weights_B = self.gateB(x_B).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (bs, num_experts, 1, 1, 1)
        
        # 组合专家输出
        combined_A = torch.stack(experts_A_out + experts_S_out, dim=1)  # (bs, num_total, ec, h, w)
        Gate_A_out = (combined_A * gate_weights_A).sum(dim=1)

        if self.GateNum == 3:
            combined_S = torch.stack(experts_A_out + experts_S_out + experts_B_out, dim=1)  # (bs, num_total, ec, h, w)
            Gate_S_out = (combined_S * gate_weights_S).sum(dim=1)
        
        combined_B = torch.stack(experts_B_out + experts_S_out, dim=1)
        Gate_B_out = (combined_B * gate_weights_B).sum(dim=1)
        
        if self.GateNum == 3:
            return Gate_A_out, Gate_S_out, Gate_B_out
        else:
            return Gate_A_out, Gate_B_out



class DeepExpertNetwork(nn.Module):
    """深度多专家网络"""
    def __init__(self, in_channels=3, base_channels=64, expert_channels=128, num_blocks=2, num_task_experts=2, num_shared_experts=1, mode='train'):
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
            
            block = CGCBlock(in_channels=current_channels, expert_channels=expert_channels, num_task_experts=num_task_experts, num_shared_experts=num_shared_experts, GateNum=3 if i < num_blocks-1 else 2) # 最后一层只用任务门控
            self.blocks.append(block)
            current_channels = expert_channels
        
        # 多任务输出头
        self.taskA_head = WSDAN(num_classes=2, M=config.num_attentions, net=config.net, pretrained=True, mode=self.mode)
        self.taskB_head = smp.Unet(encoder_name="resnext50_32x4d", encoder_weights="imagenet", in_channels=128, classes=3, activation='sigmoid')



    def forward(self, x):
        x0 = self.stem(x)  # 初始下采样
        
        a = x0
        s = x0
        b = x0

        for index, block in enumerate(self.blocks):
            if index != len(self.blocks) - 1:
                a, s, b = block(a, s, b)
            else:
                a, b = block(a, s, b)

        # print(a.shape, b.shape)   # a: [B, 128, 224, 224])   # b: [B, 128, 224, 224])
        # 任务预测
        

        if self.training:
            # print("PLE处于training")
            y_pred_raw, feature_matrix, attention_map = self.taskA_head(a) 
            outB = self.taskB_head(b)
            # feature_matrix torch.Size([B, 65536])
            # y_pred_raw torch.Size([B, 2])
            # y_pred_crop torch.Size([B, 2])
            # y_pred_drop torch.Size([B, 2])
            # outB torch.Size([B, 2, 224, 224])
            
            return a, b, feature_matrix, y_pred_raw, outB
        
        else:
            # print("PLE处于testing")
            with torch.no_grad():
                y_pred_raw, feature_matrix, attention_map = self.taskA_head(a)
                outB = self.taskB_head(b)

                return a, b, feature_matrix, y_pred_raw, outB

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