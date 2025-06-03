"""Utils
Created: Nov 11,2019 - Yuchong Gu
Revised: Dec 03,2019 - Yuchong Gu
"""
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from TNTPFNFP import resulting0, resulting1, resulting2




# ------------------------- V3（裁剪前，无mask）-------------------------
##############################################
# Center Loss for Attention Regularization
##############################################
class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.l2_loss = nn.MSELoss(reduction='sum')

    def forward(self, outputs, targets):
        return self.l2_loss(outputs, targets) / outputs.size(0)


##################################
# Metric
##################################
class Metric(object):
    pass


class AverageMeter(Metric):
    def __init__(self, name='loss'):
        self.name = name
        self.reset()

    def reset(self):
        self.scores = 0.
        self.total_num = 0.

    def __call__(self, batch_score, sample_num=1):
        self.scores += batch_score
        self.total_num += sample_num
        return self.scores / self.total_num


class TopKAccuracyMetric(Metric):
    def __init__(self, topk=(1,)):
        self.name = 'topk_accuracy'
        self.topk = topk
        self.maxk = max(topk)
        self.reset()

    def reset(self):
        self.corrects = np.zeros(len(self.topk))
        self.num_samples = 0.

    def __call__(self, output, target):
        """Computes the precision@k for the specified values of k"""
        self.num_samples += target.size(0)
        _, pred = output.topk(self.maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        for i, k in enumerate(self.topk):
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            self.corrects[i] += correct_k.item()


        return self.corrects * 100. / self.num_samples


import torch
import numpy as np

class MixedMetric:
    def __init__(self):
        self.name = 'acc_sen_spe'
        self.reset()

    def reset(self):
        self.tp = 0  # True Positives
        self.fp = 0  # False Positives
        self.tn = 0  # True Negatives
        self.fn = 0  # False Negatives
        self.num_samples = 0

    def __call__(self, output, target):
        """
        计算 Sensitivity 和 Specificity
        output: 模型的输出张量 (n_samples, n_classes)，通常为 logits 或 softmax 概率
        target: 真实标签 (n_samples,)
        """
        # 获取模型的预测类别 (选取概率最大的类)
        _, predicted = torch.max(output, 1)
        
        # 更新样本数量
        self.num_samples += target.size(0)

        
        # 计算混淆矩阵中的 TP, FP, TN, FN
        # print('aaaa',target)
        # print('bbbb', predicted)
        self.tp += ((predicted == 1) & (target == 1)).sum().item()
        self.fp += ((predicted == 1) & (target == 0)).sum().item()
        self.tn += ((predicted == 0) & (target == 0)).sum().item()
        self.fn += ((predicted == 0) & (target == 1)).sum().item()

        # 计算 Sensitivity 和 Specificity
        acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn) if (self.tp + self.tn + self.fp + self.fn) > 0 else 0
        sensitivity = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
        specificity = self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0

        # resulting0(ground_truth=target, y_train_pred=predicted)


        return self.tp, self.tn, self.fp, self.fn, acc * 100, sensitivity * 100, specificity * 100


##################################
# Callback
##################################
class Callback(object):
    def __init__(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, *args):
        pass


class ModelCheckpoint(Callback):
    def __init__(self, savepath, monitor='val_topk_accuracy', mode='max'):
        self.savepath = savepath
        self.monitor = monitor
        self.mode = mode
        self.reset()
        super(ModelCheckpoint, self).__init__()

    def reset(self):
        if self.mode == 'max':
            self.best_score = float('-inf')
        else:
            self.best_score = float('inf')

    def set_best_score(self, score):
        if isinstance(score, np.ndarray):
            self.best_score = score[0]
        else:
            self.best_score = score

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self, logs, net, **kwargs):
        current_score = logs[self.monitor]
        if isinstance(current_score, np.ndarray):
            current_score = current_score[0]

        if (self.mode == 'max' and current_score > self.best_score) or \
            (self.mode == 'min' and current_score < self.best_score):
            self.best_score = current_score

            if isinstance(net, torch.nn.DataParallel):
                state_dict = net.module.state_dict()
            else:
                state_dict = net.state_dict()

            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            if 'feature_center' in kwargs:
                feature_center = kwargs['feature_center']
                feature_center = feature_center.cpu()

                torch.save({
                    'logs': logs,
                    'state_dict': state_dict,
                    'feature_center': feature_center}, self.savepath)
            else:
                torch.save({
                    'logs': logs,
                    'state_dict': state_dict}, self.savepath)


##################################
# augment function
##################################
def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c  # 把这张attention map上采样到图像的大小，大于theta_c的部分标记为true，表示这些区域将被保留
            # print(crop_mask.shape)
            # print(mask_tensor.shape)

            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])  # 根据掩码中的非零值（nonzero_indices），找到要保留的小块。这个区域会依据 padding_ratio 稍微扩展一些，以避免裁剪过于紧凑。

            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(  # 放缩
                F.upsample_bilinear(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)   # 把这张attention map上采样到图像的大小，小于theta_d的部分标记为true，表示这些区域将被丢弃
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)


##################################
# transform in dataset
##################################
def get_transform(resize, phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.RandomCrop(resize),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])



def resize_mask_if_needed(mask, target_size):
    """
    动态调整 Mask 尺寸（仅在尺寸不匹配时执行）
    :param mask: 输入 Mask，形状为 (B, H, W) 或 (B, 1, H, W)
    :param target_size: 目标尺寸 (height, width)
    :return: 调整后的 Mask，形状为 (B, H, W)
    """
    current_size = mask.shape[-2:]  # 获取 H, W
    
    # 尺寸一致时直接返回
    if current_size == target_size:
        return mask
    
    # 确保输入为四维 (B, C, H, W)，若为三维则添加通道维度
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # (B, 1, H, W)
    
    # 调整尺寸
    resized_mask = F.interpolate(
        mask.float(),
        size=target_size,
        mode='nearest'
    )
    
    # 移除通道维度并恢复原始数据类型
    return resized_mask.squeeze(1).long()  # (B, H, W)


# 合并三个二值 Mask 为单通道类别索引（0=肿瘤，1=血管，2=背景）
def combine_masks(tumor_mask, vessel_mask, background_mask):
    combined = torch.full_like(background_mask, 2)  # 初始化为背景
    combined[vessel_mask == 1] = 1
    combined[tumor_mask == 1] = 0  # 肿瘤优先级最高
    return combined  # (B, H, W)
















# # # ------------------------- V4（裁剪前，有mask）-------------------------
# ##############################################
# # Center Loss for Attention Regularization
# ##############################################
# class CenterLoss(nn.Module):
#     def __init__(self):
#         super(CenterLoss, self).__init__()
#         self.l2_loss = nn.MSELoss(reduction='sum')

#     def forward(self, outputs, targets):
#         return self.l2_loss(outputs, targets) / outputs.size(0)


# ##################################
# # Metric
# ##################################
# class Metric(object):
#     pass


# class AverageMeter(Metric):
#     def __init__(self, name='loss'):
#         self.name = name
#         self.reset()

#     def reset(self):
#         self.scores = 0.
#         self.total_num = 0.

#     def __call__(self, batch_score, sample_num=1):
#         self.scores += batch_score
#         self.total_num += sample_num
#         return self.scores / self.total_num


# class TopKAccuracyMetric(Metric):
#     def __init__(self, topk=(1,)):
#         self.name = 'topk_accuracy'
#         self.topk = topk
#         self.maxk = max(topk)
#         self.reset()

#     def reset(self):
#         self.corrects = np.zeros(len(self.topk))
#         self.num_samples = 0.

#     def __call__(self, output, target):
#         """Computes the precision@k for the specified values of k"""
#         self.num_samples += target.size(0)
#         _, pred = output.topk(self.maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))

#         for i, k in enumerate(self.topk):
#             correct_k = correct[:k].contiguous().view(-1).float().sum(0)
#             self.corrects[i] += correct_k.item()


#         return self.corrects * 100. / self.num_samples


# import torch
# import numpy as np

# class MixedMetric:
#     def __init__(self):
#         self.name = 'acc_sen_spe'
#         self.reset()

#     def reset(self):
#         self.tp = 0  # True Positives
#         self.fp = 0  # False Positives
#         self.tn = 0  # True Negatives
#         self.fn = 0  # False Negatives
#         self.num_samples = 0

#     def __call__(self, output, target):
#         """
#         计算 Sensitivity 和 Specificity
#         output: 模型的输出张量 (n_samples, n_classes)，通常为 logits 或 softmax 概率
#         target: 真实标签 (n_samples,)
#         """
#         # 获取模型的预测类别 (选取概率最大的类)
#         _, predicted = torch.max(output, 1)
        
#         # 更新样本数量
#         self.num_samples += target.size(0)

        
#         # 计算混淆矩阵中的 TP, FP, TN, FN
#         # print('aaaa',target)
#         # print('bbbb', predicted)
#         self.tp += ((predicted == 1) & (target == 1)).sum().item()
#         self.fp += ((predicted == 1) & (target == 0)).sum().item()
#         self.tn += ((predicted == 0) & (target == 0)).sum().item()
#         self.fn += ((predicted == 0) & (target == 1)).sum().item()

#         # 计算 Sensitivity 和 Specificity
#         acc = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn) if (self.tp + self.tn + self.fp + self.fn) > 0 else 0
#         sensitivity = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
#         specificity = self.tn / (self.tn + self.fp) if (self.tn + self.fp) > 0 else 0

#         # resulting0(ground_truth=target, y_train_pred=predicted)


#         return self.tp, self.tn, self.fp, self.fn, acc * 100, sensitivity * 100, specificity * 100


# ##################################
# # Callback
# ##################################
# class Callback(object):
#     def __init__(self):
#         pass

#     def on_epoch_begin(self):
#         pass

#     def on_epoch_end(self, *args):
#         pass


# class ModelCheckpoint(Callback):
#     def __init__(self, savepath, monitor='val_topk_accuracy', mode='max'):
#         self.savepath = savepath
#         self.monitor = monitor
#         self.mode = mode
#         self.reset()
#         super(ModelCheckpoint, self).__init__()

#     def reset(self):
#         if self.mode == 'max':
#             self.best_score = float('-inf')
#         else:
#             self.best_score = float('inf')

#     def set_best_score(self, score):
#         if isinstance(score, np.ndarray):
#             self.best_score = score[0]
#         else:
#             self.best_score = score

#     def on_epoch_begin(self):
#         pass

#     def on_epoch_end(self, logs, net, **kwargs):
#         current_score = logs[self.monitor]
#         if isinstance(current_score, np.ndarray):
#             current_score = current_score[0]

#         if (self.mode == 'max' and current_score > self.best_score) or \
#             (self.mode == 'min' and current_score < self.best_score):
#             self.best_score = current_score

#             if isinstance(net, torch.nn.DataParallel):
#                 state_dict = net.module.state_dict()
#             else:
#                 state_dict = net.state_dict()

#             for key in state_dict.keys():
#                 state_dict[key] = state_dict[key].cpu()

#             if 'feature_center' in kwargs:
#                 feature_center = kwargs['feature_center']
#                 feature_center = feature_center.cpu()

#                 torch.save({
#                     'logs': logs,
#                     'state_dict': state_dict,
#                     'feature_center': feature_center}, self.savepath)
#             else:
#                 torch.save({
#                     'logs': logs,
#                     'state_dict': state_dict}, self.savepath)


# ##################################
# # augment function
# ##################################
# def batch_augment(masks, images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
#     batches, _, imgH, imgW = images.size()

#     if mode == 'crop':
#         crop_images = []
#         for batch_index in range(batches):
#             mask_tensor = masks[batch_index:batch_index + 1]
#             atten_map = attention_map[batch_index:batch_index + 1]
#             if isinstance(theta, tuple):
#                 theta_c = random.uniform(*theta) * atten_map.max()
#             else:
#                 theta_c = theta * atten_map.max()

#             crop_mask = F.upsample_bilinear(atten_map, size=(imgH, imgW)) >= theta_c  # 把这张attention map上采样到图像的大小，大于theta_c的部分标记为true，表示这些区域将被保留
#             # print(crop_mask.shape)
#             # print(mask_tensor.shape)

#             crop_mask = crop_mask & mask_tensor  # 只保留 mask 为True的部分
#             nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])  # 根据掩码中的非零值（nonzero_indices），找到要保留的小块。这个区域会依据 padding_ratio 稍微扩展一些，以避免裁剪过于紧凑。

#             if nonzero_indices.numel() > 0:
#                 height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
#                 height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
#                 width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
#                 width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)
#             else:
#                 # 处理 empty case，设置默认值
#                 height_min, height_max = 0, imgH
#                 width_min, width_max = 0, imgW


#             crop_images.append(  # 放缩
#                 F.upsample_bilinear(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
#                                     size=(imgH, imgW)))
#         crop_images = torch.cat(crop_images, dim=0)
#         return crop_images

#     elif mode == 'drop':
#         drop_masks = []
#         for batch_index in range(batches):
#             atten_map = attention_map[batch_index:batch_index + 1]
#             if isinstance(theta, tuple):
#                 theta_d = random.uniform(*theta) * atten_map.max()
#             else:
#                 theta_d = theta * atten_map.max()

#             drop_masks.append(F.upsample_bilinear(atten_map, size=(imgH, imgW)) < theta_d)   # 把这张attention map上采样到图像的大小，小于theta_d的部分标记为true，表示这些区域将被丢弃
#         drop_masks = torch.cat(drop_masks, dim=0)
#         drop_images = images * drop_masks.float()
#         return drop_images

#     else:
#         raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)


# ##################################
# # transform in dataset
# ##################################
# def get_transform(resize, phase='train'):
#     if phase == 'train':
#         return transforms.Compose([
#             transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
#             transforms.RandomCrop(resize),
#             transforms.RandomHorizontalFlip(0.5),
#             transforms.ColorJitter(brightness=0.126, saturation=0.5),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])
#     else:
#         return transforms.Compose([
#             transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
#             transforms.CenterCrop(resize),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         ])