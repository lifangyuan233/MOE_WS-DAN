"""TRAINING
Created: May 04,2019 - Yuchong Gu
Revised: Dec 03,2019 - Yuchong Gu
"""
import os
import time
import logging
import warnings
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from models import WSDAN, CombinedLoss
from models.PLE import DeepExpertNetwork
from datasets import get_trainval_datasets
from utils import CenterLoss, AverageMeter, TopKAccuracyMetric, ModelCheckpoint, batch_augment,  MixedMetric, resize_mask_if_needed, combine_masks

import numpy as np
import pandas as pd
from TNTPFNFP import resulting0, resulting1, resulting2
# GPU settings
assert torch.cuda.is_available()



# 
# 
# 
# #
# #
# # ------------------------- V3（裁剪前，无mask）-------------------------
# version = config.version
# if config.version == '1':
#     version = '1'
# elif config.version == '2':
#     version = '1'
# elif config.version == '3':
#     version = '3'
# elif config.version == '4':
#     version = '3'
# os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
# device = torch.device("cuda")
# torch.backends.cudnn.benchmark = True

# # General loss functions
# cross_entropy_loss = nn.CrossEntropyLoss()
# center_loss = CenterLoss()

# # loss and metric
# loss_container = AverageMeter(name='loss')
# raw_metric = TopKAccuracyMetric(topk=(1, 1))
# crop_metric = TopKAccuracyMetric(topk=(1, 1))
# drop_metric = TopKAccuracyMetric(topk=(1, 1))
# # raw_metric1 = MixedMetric()
# # crop_metric1 = MixedMetric()
# # drop_metric1 = MixedMetric()


# # 过滤测试集
# def filter_number1_by_number2(filename):
#     result = []
#     # 打开文件并逐行读取
#     with open(filename, 'r', encoding='utf-8') as file:
#         for line in file:
#             # 将每行的两个数拆分
#             number1, number2 = map(int, line.split())
#             # 如果 number2 等于 0，保存 number1
#             if number2 == 0:
#                 result.append(number1)
#     return result

# # 找到测试集的名称.dcm
# def filter_and_strip_prefix(filename, numbers):
#     result = []
    
#     # 定义要去掉的前缀
#     prefixes = ['000.no_tumor/', '001.tumor/']
    
#     # 打开文件并逐行读取
#     with open(filename, 'r', encoding='utf-8') as file:
#         for line in file:
#             # 将每行的 number1 和 string2 分开
#             number1, string2 = line.split(maxsplit=1)
#             number1 = int(number1)  # 转换 number1 为整数
            
#             # 检查 number1 是否在给定的 numbers 列表中
#             if number1 in numbers:
#                 # 去掉 string2 的前缀
#                 for prefix in prefixes:
#                     if string2.startswith(prefix):
#                         string2 = string2[len(prefix):]  # 去掉前缀
#                         break  # 如果匹配到前缀就不再继续检查
                
#                 # 去除可能的换行符并保存结果
#                 result.append(string2.strip()[:-4] + '.dcm')
    
#     return result


# filtered_numbers = filter_number1_by_number2('D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V' + version + ')\\train_test_split.txt')  # “测试集”索引
# filtered_data = filter_and_strip_prefix('D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V' + version + ')\\images.txt', filtered_numbers)





# def main():
#     ##################################
#     # Initialize saving directory
#     ##################################
#     if not os.path.exists(config.save_dir):
#         os.makedirs(config.save_dir)

#     ##################################
#     # Logging setting
#     ##################################
#     logging.basicConfig(
#         filename=os.path.join(config.save_dir, config.log_name),
#         filemode='w',
#         format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
#         level=logging.INFO)
#     warnings.filterwarnings("ignore")

#     ##################################
#     # Load dataset
#     ##################################
#     train_dataset, validate_dataset = get_trainval_datasets(config.tag, config.image_size)

#     train_loader, validate_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
#                                                num_workers=config.workers, pin_memory=True), \
#                                     DataLoader(validate_dataset, batch_size=config.batch_size * 4, shuffle=False,
#                                                num_workers=config.workers, pin_memory=True)
#     num_classes = train_dataset.num_classes

#     ##################################
#     # Initialize model
#     ##################################
#     logs = {}
#     start_epoch = 0
#     net = WSDAN(num_classes=num_classes, M=config.num_attentions, net=config.net, pretrained=True, mode='train')

#     # feature_center: size of (#classes, #attention_maps * #channel_features)
#     feature_center = torch.zeros(num_classes, config.num_attentions * net.num_features).to(device)

#     if config.ckpt:
#         # Load ckpt and get state_dict
#         checkpoint = torch.load(config.ckpt)

#         # Get epoch and some logs
#         logs = checkpoint['logs']
#         start_epoch = int(logs['epoch'])

#         # Load weights
#         state_dict = checkpoint['state_dict']
#         net.load_state_dict(state_dict)
#         logging.info('Network loaded from {}'.format(config.ckpt))

#         # load feature center
#         if 'feature_center' in checkpoint:
#             feature_center = checkpoint['feature_center'].to(device)
#             logging.info('feature_center loaded from {}'.format(config.ckpt))

#     logging.info('Network weights save to {}'.format(config.save_dir))

#     ##################################
#     # Use cuda
#     ##################################
#     net.to(device)
#     if torch.cuda.device_count() > 1:
#         net = nn.DataParallel(net)

#     ##################################
#     # Optimizer, LR Scheduler
#     ##################################
#     learning_rate = logs['lr'] if 'lr' in logs else config.learning_rate
#     optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

#     # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=2)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

#     ##################################
#     # ModelCheckpoint
#     ##################################
#     callback_monitor = 'val_{}'.format(raw_metric.name)
#     callback = ModelCheckpoint(savepath=os.path.join(config.save_dir, config.model_name),
#                                monitor=callback_monitor,
#                                mode='max')
#     if callback_monitor in logs:
#         callback.set_best_score(logs[callback_monitor])
#     else:
#         callback.reset()

#     ##################################
#     # TRAINING
#     ##################################
#     logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
#                  format(config.epochs, config.batch_size, len(train_dataset), len(validate_dataset)))
#     logging.info('')

#     for epoch in range(start_epoch, config.epochs):
#         callback.on_epoch_begin()

#         logs['epoch'] = epoch + 1
#         logs['lr'] = optimizer.param_groups[0]['lr']

#         logging.info('Epoch {:03d}, Learning Rate {:g}'.format(epoch + 1, optimizer.param_groups[0]['lr']))

#         pbar = tqdm(total=len(train_loader), unit=' batches')
#         pbar.set_description('Epoch {}/{}'.format(epoch + 1, config.epochs))

#         train(logs=logs,
#               data_loader=train_loader,
#               net=net,
#               feature_center=feature_center,
#               optimizer=optimizer,
#               pbar=pbar)

#         validate(logs=logs,
#                  data_loader=validate_loader,
#                  net=net,
#                  pbar=pbar)

#         if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
#             scheduler.step(logs['val_loss'])
#         else:
#             scheduler.step()

#         callback.on_epoch_end(logs, net, feature_center=feature_center)
#         torch.save({
#             'epoch': epoch + 1,
#             'state_dict': net.state_dict(),
#             'optimizer': optimizer.state_dict(),
#             'feature_center': feature_center,
#             'logs': logs
#         }, os.path.join(config.save_dir, f'model_epoch_{epoch + 1}.ckpt'))

#         pbar.close()


# def train(**kwargs):
#     # Retrieve training configuration
#     logs = kwargs['logs']
#     data_loader = kwargs['data_loader']
#     net = kwargs['net']
#     feature_center = kwargs['feature_center']
#     optimizer = kwargs['optimizer']
#     pbar = kwargs['pbar']

#     # metrics initialization
#     loss_container.reset()
#     raw_metric.reset()
#     crop_metric.reset()
#     drop_metric.reset()
#     # raw_metric1.reset()
#     # crop_metric1.reset()
#     # drop_metric1.reset()

#     # begin training
#     start_time = time.time()
#     net.train()

#     batch_info1 = ""
#     all_targets = []
#  #   all_predictions_raw = []
#  #   all_predictions_crop = []
#  #   all_predictions_drop = []

#     for i, (X, y) in enumerate(data_loader):
#         optimizer.zero_grad()

#         # obtain data for training
#         X = X.to(device)
#         y = y.to(device)

#         ##################################
#         # Raw Image
#         ##################################
#         # raw images forward
#         y_pred_raw, feature_matrix, attention_map = net(X)
#         # print(X.shape)   # torch.Size([16, 3, 448, 448])

#         # Update Feature Center
#         feature_center_batch = F.normalize(feature_center[y], dim=-1)
#         feature_center[y] += config.beta * (feature_matrix.detach() - feature_center_batch)

#         ##################################
#         # Attention Cropping
#         ##################################
#         with torch.no_grad():
#             # print("crop")
#             # print(X.shape)            # torch.Size([12, 3, 448, 448])
#             # print( attention_map[:, :1, :, :].shape)       # torch.Size([12, 1, 14, 14])
#             crop_images = batch_augment(X, attention_map[:, :1, :, :], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)

#         # crop images forward
#         y_pred_crop, _, _ = net(crop_images)

#         ##################################
#         # Attention Dropping
#         ##################################
#         with torch.no_grad():
#             # print("drop")
#             # print(X.shape)      # torch.Size([12, 3, 448, 448])
#             # print( attention_map[:, 1:, :, :].shape)      # torch.Size([12, 1, 14, 14])
#             drop_images = batch_augment(X, attention_map[:, 1:, :, :], mode='drop', theta=(0.2, 0.5))

#         # drop images forward
#         y_pred_drop, _, _ = net(drop_images)

#         # loss
#         batch_loss = cross_entropy_loss(y_pred_raw, y) / 3. + \
#                      cross_entropy_loss(y_pred_crop, y) / 3. + \
#                      cross_entropy_loss(y_pred_drop, y) / 3. + \
#                      center_loss(feature_matrix, feature_center_batch)
        

#         # backward
#         batch_loss.backward()
#         optimizer.step()

#         # metrics: loss and top-1,5 error
#         with torch.no_grad():
#             epoch_loss = loss_container(batch_loss.item())
#             epoch_raw_acc = raw_metric(y_pred_raw, y)
#             epoch_crop_acc = crop_metric(y_pred_crop, y)
#             epoch_drop_acc = drop_metric(y_pred_drop, y)
#             # epoch_raw_tp1, epoch_raw_tn1, epoch_raw_fp1, epoch_raw_fn1, epoch_raw_acc1, epoch_raw_sen1, epoch_raw_spe1 = raw_metric1(y_pred_raw, y)
#             # epoch_crop_tp1, epoch_crop_tn1, epoch_crop_fp1, epoch_crop_fn1, epoch_crop_acc1,  epoch_crop_sen1, epoch_crop_spe1 = crop_metric1(y_pred_crop, y)
#             # epoch_drop_tp1, epoch_drop_tn1, epoch_drop_fp1, epoch_drop_fn1, epoch_drop_acc1,  epoch_drop_sen1, epoch_drop_spe1 = drop_metric1(y_pred_drop, y)
            
#    #         _, predicted_raw = torch.max(y_pred_raw, 1)
#    #         _, predicted_crop = torch.max(y_pred_crop, 1)
#    #         _, predicted_drop = torch.max(y_pred_drop, 1)

#             all_targets.append(y.cpu().numpy())
#     #        all_predictions_raw.append(predicted_raw.cpu().numpy())
#     #        all_predictions_crop.append(predicted_crop.cpu().numpy())
#     #        all_predictions_drop.append(predicted_drop.cpu().numpy())
#         # end of this batch
#         batch_info = 'Loss {:.4f}, Raw Acc ({:.2f}, {:.2f}), Crop Acc ({:.2f}, {:.2f}), Drop Acc ({:.2f}, {:.2f})'.format(
#             epoch_loss, epoch_raw_acc[0], epoch_raw_acc[1],
#             epoch_crop_acc[0], epoch_crop_acc[1], epoch_drop_acc[0], epoch_drop_acc[1])
#         # batch_info1 = 'Raw Acc1 Sen1 Spe1 ({:.2f}, {:.2f}, {:.2f}), Crop Acc1 Sen1 Spe1 {:.2f}, {:.2f}, {:.2f}), Drop Acc1 Sen1 Spe1 ({:.2f}, {:.2f}, {:.2f})'.format(
#         #     epoch_raw_acc1, epoch_raw_sen1, epoch_raw_spe1,
#         #     epoch_crop_acc1, epoch_crop_sen1, epoch_crop_spe1, 
#         #     epoch_drop_acc1, epoch_drop_sen1, epoch_drop_spe1)
        
#         combined_string = f"{batch_info} | {batch_info1}"
#         pbar.update()
#         pbar.set_postfix_str(batch_info)
    
#     all_targets = np.concatenate(all_targets)
#   #  all_predictions_raw = np.concatenate(all_predictions_raw)
#   #  all_predictions_crop = np.concatenate(all_predictions_crop)
#   #  all_predictions_drop = np.concatenate(all_predictions_drop)    
#   #  resulting0(all_targets, all_predictions_raw)
#   #  resulting0(all_targets, all_predictions_crop)
#   #  resulting0(all_targets, all_predictions_drop)


#     # end of this epoch
#     logs['train_{}'.format(loss_container.name)] = epoch_loss
#     logs['train_raw_{}'.format(raw_metric.name)] = epoch_raw_acc
#     logs['train_crop_{}'.format(crop_metric.name)] = epoch_crop_acc
#     logs['train_drop_{}'.format(drop_metric.name)] = epoch_drop_acc
#     logs['train_info'] = batch_info1
#     end_time = time.time()

#     # write log for this epoch
#     logging.info('Train: {}, Time {:3.2f}'.format(batch_info1, end_time - start_time))
#     logging.info("Train completed. Final metrics: %s", batch_info1)



# def validate(**kwargs):
#     # Retrieve training configuration
#     logs = kwargs['logs']
#     data_loader = kwargs['data_loader']
#     net = kwargs['net']
#     pbar = kwargs['pbar']

#     # metrics initialization
#     loss_container.reset()
#     raw_metric.reset()

#     # begin validation
#     start_time = time.time()
#     all_targets = []
#     all_predictions = []


#     net.eval()
#     # batch_info1 = ""
#     with torch.no_grad():
#         for i, (X, y) in enumerate(data_loader):
#             # obtain data
#             X = X.to(device)
#             y = y.to(device)

#             ##################################
#             # Raw Image
#             ##################################
#             y_pred_raw, _, attention_map = net(X)

#             ##################################
#             # Object Localization and Refinement
#             ##################################
#             crop_images = batch_augment(X, attention_map, mode='crop', theta=0.1, padding_ratio=0.05)
#             y_pred_crop, _, _ = net(crop_images)

#             ##################################
#             # Final prediction
#             ##################################
#             y_pred = (y_pred_raw + y_pred_crop) / 2.

#             _, predicted = torch.max(y_pred, 1)

#             all_targets.append(y.cpu().numpy())
#             all_predictions.append(predicted.cpu().numpy())

#             # loss
#             batch_loss = cross_entropy_loss(y_pred, y)
#             epoch_loss = loss_container(batch_loss.item())

#             # metrics: top-1,5 error
#             epoch_acc = raw_metric(y_pred, y)
#             # epoch_tp1, epoch_tn1, epoch_fp1, epoch_fn1, epoch_acc1, epoch_sen1, epoch_spe1 = raw_metric1(y_pred, y)

#         # 合并所有批次的结果
#         all_targets = np.concatenate(all_targets)
#         all_predictions = np.concatenate(all_predictions)

#         resulting1(filtered_data, all_targets, all_predictions)



#     # end of validation
#     logs['val_{}'.format(loss_container.name)] = epoch_loss
#     logs['val_{}'.format(raw_metric.name)] = epoch_acc
#     end_time = time.time()

#     batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f})'.format(epoch_loss, epoch_acc[0], epoch_acc[1])
#     # batch_info1 = 'Val Acc1 ({:.2f}), Val Sen1 ({:.2f}), Val Spe1 ({:.2f})'.format(epoch_acc1, epoch_sen1, epoch_spe1)
#     # combined_string = f"{batch_info} | {batch_info1}"

#     pbar.set_postfix_str('{}, {}'.format(logs['train_info'], batch_info))

#     # write log for this epoch
#     logging.info('Valid: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))
#     # logging.info("Validation completed. Final metrics: %s", batch_info1)
#     logging.info('')


# if __name__ == '__main__':
#     main()



























# # ------------------------- V4（裁剪前，有mask）-------------------------
torch.cuda.empty_cache()  # 清理缓存
version = config.version
if config.version == '1':
    version = '1'
elif config.version == '2':
    version = '1'
elif config.version == '3':
    version = '3'
elif config.version == '4':
    version = '3'
else:
    version = 'VV3'

os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# General loss functions
cross_entropy_loss = nn.CrossEntropyLoss()
center_loss = CenterLoss()

# loss and metric
loss_container = AverageMeter(name='loss')
raw_metric = TopKAccuracyMetric(topk=(1, 1))
crop_metric = TopKAccuracyMetric(topk=(1, 1))
drop_metric = TopKAccuracyMetric(topk=(1, 1))
# raw_metric1 = MixedMetric()
# crop_metric1 = MixedMetric()
# drop_metric1 = MixedMetric()


# 过滤测试集
def filter_number1_by_number2(filename):
    result = []
    # 打开文件并逐行读取
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # 将每行的两个数拆分
            number1, number2 = map(int, line.split())
            # 如果 number2 等于 0，保存 number1
            if number2 == 0:
                result.append(number1)
    return result

# 找到测试集的名称.dcm  return[]
def filter_and_strip_prefix(filename, numbers):
    result = []
    
    # 定义要去掉的前缀
    prefixes = ['000.no_tumor/', '001.tumor/']
    
    # 打开文件并逐行读取
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # 将每行的 number1 和 string2 分开
            number1, string2 = line.split(maxsplit=1)
            number1 = int(number1)  # 转换 number1 为整数
            
            # 检查 number1 是否在给定的 numbers 列表中
            if number1 in numbers:
                # 去掉 string2 的前缀
                for prefix in prefixes:
                    if string2.startswith(prefix):
                        string2 = string2[len(prefix):]  # 去掉前缀
                        break  # 如果匹配到前缀就不再继续检查
                
                # 去除可能的换行符并保存结果
                result.append(string2.strip()[:-4] + '.dcm')
    
    return result

# 所有测试集数据的indexs
filtered_numbers = filter_number1_by_number2('D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V' + version + ')\\train_test_split.txt')  # “测试集”索引
# 所有测试集数据的names
filtered_data = filter_and_strip_prefix('D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V' + version + ')\\images.txt', filtered_numbers)


def main():
    ##################################
    # Initialize saving directory
    ##################################
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)

    ##################################
    # Logging setting
    ##################################
    logging.basicConfig(
        filename=os.path.join(config.save_dir, config.log_name),
        filemode='w',
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")

    ##################################
    # Load dataset
    ##################################
    train_dataset, validate_dataset = get_trainval_datasets(config.tag, config.image_size)
    # print(len(train_dataset), len(validate_dataset))

    train_loader, validate_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                               num_workers=config.workers, pin_memory=True), \
                                    DataLoader(validate_dataset, batch_size=config.batch_size * 4, shuffle=False,
                                               num_workers=config.workers, pin_memory=True)
    num_classes = train_dataset.num_classes

    ##################################
    # Initialize model
    ##################################
    logs = {}
    start_epoch = 0

    net = DeepExpertNetwork(in_channels=3, base_channels=64, expert_channels=128, num_blocks=2, num_shared_experts=2, mode='train')

    # feature_center: size of (#classes, #attention_maps * #channel_features)
    # feature_center = torch.zeros(num_classes, config.num_attentions * net.num_features).to(device)
    feature_center = torch.zeros(num_classes, config.num_attentions * net.taskA_head.num_features).to(device)

    if config.ckpt:
        # Load ckpt and get state_dict
        checkpoint = torch.load(config.ckpt)

        # Get epoch and some logs
        logs = checkpoint['logs']
        start_epoch = int(logs['epoch'])

        # Load weights
        state_dict = checkpoint['state_dict']
        net.load_state_dict(state_dict)
        logging.info('Network loaded from {}'.format(config.ckpt))

        # load feature center
        if 'feature_center' in checkpoint:
            feature_center = checkpoint['feature_center'].to(device)
            logging.info('feature_center loaded from {}'.format(config.ckpt))

    logging.info('Network weights save to {}'.format(config.save_dir))

    ##################################
    # Use cuda
    ##################################
    net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    ##################################
    # Optimizer, LR Scheduler
    ##################################
    learning_rate = logs['lr'] if 'lr' in logs else config.learning_rate
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

    # Unet权重
    ce_weight = torch.tensor([1.0,  # 肿瘤权重高（小目标需重点优化））
                            0.5,    # 血管权重中等
                            0.1])   # 背景权重较低（若背景像素多
    dice_class_weights = [1.0, 0.5, 0.1]  # 与 CE 权重对齐

    # --- 创建损失函数实例 ---
    criterion = CombinedLoss(
        ce_weight=ce_weight,          # 控制交叉熵的类别权重
        dice_smooth=1e-5,             # 防止分母为0
        dice_class_weights=dice_class_weights  # 控制Dice的类别权重
    )
    criterion = criterion.to(device)  # 确保与模型在同一设备（GPU/CPU）

    ##################################
    # ModelCheckpoint
    ##################################
    callback_monitor = 'val_{}'.format(raw_metric.name)
    callback = ModelCheckpoint(savepath=os.path.join(config.save_dir, config.model_name),
                               monitor=callback_monitor,
                               mode='max')
    if callback_monitor in logs:
        callback.set_best_score(logs[callback_monitor])
    else:
        callback.reset()

    ##################################
    # TRAINING
    ##################################
    logging.info('Start training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}'.
                 format(config.epochs, config.batch_size, len(train_dataset), len(validate_dataset)))
    logging.info('')

    for epoch in range(start_epoch, config.epochs):
        callback.on_epoch_begin()

        logs['epoch'] = epoch + 1
        logs['lr'] = optimizer.param_groups[0]['lr']

        logging.info('Epoch {:03d}, Learning Rate {:g}'.format(epoch + 1, optimizer.param_groups[0]['lr']))

        pbar = tqdm(total=len(train_loader), unit=' batches')
        pbar.set_description('Epoch {}/{}'.format(epoch + 1, config.epochs))

        train(logs=logs,
              data_loader=train_loader,
              net=net,
              feature_center=feature_center,
              optimizer=optimizer,
              criterion=criterion,
              pbar=pbar)

        validate(logs=logs,
                 data_loader=validate_loader,
                 net=net,
                 pbar=pbar)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(logs['val_loss'])
        else:
            scheduler.step()

        callback.on_epoch_end(logs, net, feature_center=feature_center)
        if epoch % 1 == 0:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'feature_center': feature_center,
                'logs': logs
            }, os.path.join(config.save_dir, f'model_epoch_{epoch + 1}.ckpt'))

        pbar.close()


def train(**kwargs):
    # Retrieve training configuration
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    feature_center = kwargs['feature_center']
    optimizer = kwargs['optimizer']
    criterion = kwargs['criterion']
    pbar = kwargs['pbar']
    

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()
    crop_metric.reset()
    drop_metric.reset()
    # raw_metric1.reset()
    # crop_metric1.reset()
    # drop_metric1.reset()

    # begin training
    start_time = time.time()
    net.train()

    batch_info1 = ""
    all_targets = []
 #   all_predictions_raw = []
 #   all_predictions_crop = []
 #   all_predictions_drop = []

    # mask_origin, mask_tumor, mask_vessel, image, label
    for i, (mask_origin, mask_tumor, mask_vessel, mask_background, X, y) in enumerate(data_loader):
        optimizer.zero_grad()

        # obtain data for training
        mask_origin = mask_origin.to(device)
        mask_tumor = mask_tumor.to(device)
        mask_vessel = mask_vessel.to(device)
        mask_background = mask_background.to(device)
        X = X.to(device)
        y = y.to(device)
        print(X.shape)

        with torch.no_grad():
            mask_origin0 = resize_mask_if_needed(mask_origin, (config.image_size[0] // 2, config.image_size[0] // 2))
            mask_background0 = resize_mask_if_needed(mask_background, (config.image_size[0] // 2, config.image_size[0] // 2))
            mask_tumor0 = resize_mask_if_needed(mask_tumor, (config.image_size[0] // 2, config.image_size[0] // 2))
            mask_vessel0 = resize_mask_if_needed(mask_vessel, (config.image_size[0] // 2, config.image_size[0] // 2)) # 224
            masks0 = combine_masks(mask_tumor0, mask_vessel0, mask_background0)  # (B,H_in,W_in)  （0=肿瘤，1=血管，2=背景）
            # print(masks0.shape)   # B 224 224 
        
        
        ##################################
        # Raw Image
        ##################################
        # raw images forward
        a, b, feature_matrix, y_pred_raw,  y_pred_crop, y_pred_drop, outB = net(X)
        # y_pred_raw, feature_matrix, attention_map = net(X)

        # Update Feature Center
        feature_center_batch = F.normalize(feature_center[y], dim=-1)
        feature_center[y] += config.beta * (feature_matrix.detach() - feature_center_batch)


        # loss
        batch_loss = cross_entropy_loss(y_pred_raw, y) / 3. + \
                     cross_entropy_loss(y_pred_crop, y) / 3. + \
                     cross_entropy_loss(y_pred_drop, y) / 3. + \
                     center_loss(feature_matrix, feature_center_batch) + \
                     criterion(outB, masks0)
        

        # backward
        batch_loss.backward()
        optimizer.step()

        # metrics: loss and top-1,5 error
        with torch.no_grad():
            epoch_loss = loss_container(batch_loss.item())
            epoch_raw_acc = raw_metric(y_pred_raw, y)
            epoch_crop_acc = crop_metric(y_pred_crop, y)
            epoch_drop_acc = drop_metric(y_pred_drop, y)
            # epoch_raw_tp1, epoch_raw_tn1, epoch_raw_fp1, epoch_raw_fn1, epoch_raw_acc1, epoch_raw_sen1, epoch_raw_spe1 = raw_metric1(y_pred_raw, y)
            # epoch_crop_tp1, epoch_crop_tn1, epoch_crop_fp1, epoch_crop_fn1, epoch_crop_acc1,  epoch_crop_sen1, epoch_crop_spe1 = crop_metric1(y_pred_crop, y)
            # epoch_drop_tp1, epoch_drop_tn1, epoch_drop_fp1, epoch_drop_fn1, epoch_drop_acc1,  epoch_drop_sen1, epoch_drop_spe1 = drop_metric1(y_pred_drop, y)
            
  #          _, predicted_raw = torch.max(y_pred_raw, 1)
  #          _, predicted_crop = torch.max(y_pred_crop, 1)
  #          _, predicted_drop = torch.max(y_pred_drop, 1)

            all_targets.append(y.cpu().numpy())
   #         all_predictions_raw.append(predicted_raw.cpu().numpy())
   #         all_predictions_crop.append(predicted_crop.cpu().numpy())
   #         all_predictions_drop.append(predicted_drop.cpu().numpy())
        # end of this batch
        batch_info = 'Loss {:.4f}, Raw Acc ({:.2f}, {:.2f}), Crop Acc ({:.2f}, {:.2f}), Drop Acc ({:.2f}, {:.2f})'.format(
            epoch_loss, epoch_raw_acc[0], epoch_raw_acc[1],
            epoch_crop_acc[0], epoch_crop_acc[1], epoch_drop_acc[0], epoch_drop_acc[1])
        # batch_info1 = 'Raw Acc1 Sen1 Spe1 ({:.2f}, {:.2f}, {:.2f}), Crop Acc1 Sen1 Spe1 {:.2f}, {:.2f}, {:.2f}), Drop Acc1 Sen1 Spe1 ({:.2f}, {:.2f}, {:.2f})'.format(
        #     epoch_raw_acc1, epoch_raw_sen1, epoch_raw_spe1,
        #     epoch_crop_acc1, epoch_crop_sen1, epoch_crop_spe1, 
        #     epoch_drop_acc1, epoch_drop_sen1, epoch_drop_spe1)
        
        combined_string = f"{batch_info} | {batch_info1}"
        pbar.update()
        pbar.set_postfix_str(batch_info)
    
    all_targets = np.concatenate(all_targets)
 #   all_predictions_raw = np.concatenate(all_predictions_raw)
 #   all_predictions_crop = np.concatenate(all_predictions_crop)
 #   all_predictions_drop = np.concatenate(all_predictions_drop)    
 #   resulting0(all_targets, all_predictions_raw)
 #   resulting0(all_targets, all_predictions_crop)
 #   resulting0(all_targets, all_predictions_drop)


    # end of this epoch
    logs['train_{}'.format(loss_container.name)] = epoch_loss
    logs['train_raw_{}'.format(raw_metric.name)] = epoch_raw_acc
    logs['train_crop_{}'.format(crop_metric.name)] = epoch_crop_acc
    logs['train_drop_{}'.format(drop_metric.name)] = epoch_drop_acc
    logs['train_info'] = batch_info1
    end_time = time.time()

    # write log for this epoch
    logging.info('Train: {}, Time {:3.2f}'.format(batch_info1, end_time - start_time))
    logging.info("Train completed. Final metrics: %s", batch_info1)



def validate(**kwargs):
    # Retrieve training configuration
    logs = kwargs['logs']
    data_loader = kwargs['data_loader']
    net = kwargs['net']
    pbar = kwargs['pbar']

    # metrics initialization
    loss_container.reset()
    raw_metric.reset()

    # begin validation
    start_time = time.time()
    all_targets = []
    all_predictions = []


    net.eval()
    # batch_info1 = ""
    with torch.no_grad():
        # mask_origin, mask_tumor, mask_vessel, image, label

        for i, (mask_origin, mask_tumor, mask_vessel, mask_background, X, y) in enumerate(data_loader):
            # obtain data
            mask_origin = mask_origin.to(device)
            mask_tumor = mask_tumor.to(device)
            mask_vessel = mask_vessel.to(device)
            mask_background = mask_background.to(device)
            X = X.to(device)
            y = y.to(device)

            ##################################
            # Raw Image
            ##################################
            # y_pred_raw, _, attention_map = net(X)
            a, b, feature_matrix, y_pred_raw,  y_pred_crop, y_pred_drop, outB = net(X)

            y_pred = (y_pred_raw + y_pred_crop) / 2.

            _, predicted = torch.max(y_pred, 1)

            all_targets.append(y.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())

            # loss
            batch_loss = cross_entropy_loss(y_pred, y)
            epoch_loss = loss_container(batch_loss.item())

            # metrics: top-1,5 error
            epoch_acc = raw_metric(y_pred, y)
            # epoch_tp1, epoch_tn1, epoch_fp1, epoch_fn1, epoch_acc1, epoch_sen1, epoch_spe1 = raw_metric1(y_pred, y)

        # 合并所有批次的结果
        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)

        resulting1(filtered_data, all_targets, all_predictions)



    # end of validation
    logs['val_{}'.format(loss_container.name)] = epoch_loss
    logs['val_{}'.format(raw_metric.name)] = epoch_acc
    end_time = time.time()

    batch_info = 'Val Loss {:.4f}, Val Acc ({:.2f}, {:.2f})'.format(epoch_loss, epoch_acc[0], epoch_acc[1])
    # batch_info1 = 'Val Acc1 ({:.2f}), Val Sen1 ({:.2f}), Val Spe1 ({:.2f})'.format(epoch_acc1, epoch_sen1, epoch_spe1)
    # combined_string = f"{batch_info} | {batch_info1}"

    pbar.set_postfix_str('{}, {}'.format(logs['train_info'], batch_info))

    # write log for this epoch
    logging.info('Valid: {}, Time {:3.2f}'.format(batch_info, end_time - start_time))
    # logging.info("Validation completed. Final metrics: %s", batch_info1)
    logging.info('')


if __name__ == '__main__':
    main()