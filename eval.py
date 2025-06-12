"""EVALUATION
Created: Nov 22,2019 - Yuchong Gu
Revised: Dec 03,2019 - Yuchong Gu
"""
import os
import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

import config
from models import WSDAN
from models.PLE import DeepExpertNetwork
from datasets import get_trainval_datasets
from utils import TopKAccuracyMetric, batch_augment, MixedMetric
from TNTPFNFP import resulting0, resulting1, resulting2
import pandas as pd
import os


# GPU settings
assert torch.cuda.is_available()


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
# elif config.version == 'V':
#     version = 'V'
# os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
# device = torch.device("cuda")
# torch.backends.cudnn.benchmark = True

# # visualize
# visualize = config.visualize
# savepath = config.eval_savepath
# if visualize:
#     os.makedirs(savepath, exist_ok=True)

# ToPILImage = transforms.ToPILImage()
# MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
# STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# def generate_heatmap(attention_maps):
#     heat_attention_maps = []
#     heat_attention_maps.append(attention_maps[:, 0, ...])  # R
#     heat_attention_maps.append(attention_maps[:, 0, ...] * (attention_maps[:, 0, ...] < 0.5).float() + \
#                                (1. - attention_maps[:, 0, ...]) * (attention_maps[:, 0, ...] >= 0.5).float())  # G
#     heat_attention_maps.append(1. - attention_maps[:, 0, ...])  # B
#     return torch.stack(heat_attention_maps, dim=1)

# # 过滤测试集
# def filter_number1_by_number2(filename):
#     result = []
#     # 打开文件并逐行读取
#     with open(filename,  'r', encoding='utf-8') as file:
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
#     with open(filename,  'r', encoding='utf-8') as file:
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




# def main():
#     logging.basicConfig(
#         format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
#         level=logging.INFO)
#     warnings.filterwarnings("ignore")

#     try:
#         ckpt = config.eval_ckpt
#     except:
#         logging.info('Set ckpt for evaluation in config.py')
#         return

#     ##################################
#     # Dataset for testing
#     ##################################
#     _, test_dataset = get_trainval_datasets(config.tag, resize=config.image_size)
#     test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
#                              num_workers=2, pin_memory=True)
    

#     filtered_numbers = filter_number1_by_number2('D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V' + version + ')\\train_test_split' + config.choose_name + '.txt')  # “测试集”索引
#     filtered_data = filter_and_strip_prefix('D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V' + version + ')\\images.txt', filtered_numbers)

#     # print(len(filtered_numbers))
#     # print(len(filtered_data))

#     # print(filtered_numbers)
#     # print(filtered_data)



#     ##################################
#     # Initialize model
#     ##################################
#     net = WSDAN(num_classes=test_dataset.num_classes, M=config.num_attentions, net=config.net, mode='test')

#     # Load ckpt and get state_dict
#     checkpoint = torch.load(ckpt)
#     state_dict = checkpoint['state_dict']

#     # Load weights
#     net.load_state_dict(state_dict)
#     logging.info('Network loaded from {}'.format(ckpt))

#     ##################################
#     # use cuda
#     ##################################
#     net.to(device)
#     if torch.cuda.device_count() > 1:
#         net = nn.DataParallel(net)

#     ##################################
#     # Prediction
#     ##################################
#     raw_accuracy = TopKAccuracyMetric(topk=(1, 1))
#     ref_accuracy = TopKAccuracyMetric(topk=(1, 1))
#     # raw_metric1 = MixedMetric()
#     raw_accuracy.reset()
#     ref_accuracy.reset()
#     # raw_metric1.reset()

#     all_targets = []
#     all_predictions = []
#     # all_predictions_raw = []
#     # all_predictions_crop = []
#     net.eval()
#     with torch.no_grad():
#         pbar = tqdm(total=len(test_loader), unit=' batches')
#         pbar.set_description('Validation')
#         # combined_string = ""
#         # batch_info1 = ""
#         for i, (X, y) in enumerate(test_loader):
#             X = X.to(device)
#             y = y.to(device)

#             # WS-DAN
#             y_pred_raw, _, attention_maps = net(X)

#             # Augmentation with crop_mask
#             crop_image = batch_augment(X, attention_maps, mode='crop', theta=0.1, padding_ratio=0.05)

#             y_pred_crop, _, _ = net(crop_image)
#             y_pred = (y_pred_raw + y_pred_crop) / 2.

#             # _, predicted_raw = torch.max(y_pred_raw, 1)
#             # _, predicted_crop = torch.max(y_pred_crop, 1)
#             _, predicted = torch.max(y_pred, 1)

#             all_targets.append(y.cpu().numpy())
#             # all_predictions_raw.append(predicted_raw.cpu().numpy())
#             # all_predictions_crop.append(predicted_crop.cpu().numpy())
#             all_predictions.append(predicted.cpu().numpy())


#             if visualize:
#                 # reshape attention maps
#                 attention_maps = F.upsample_bilinear(attention_maps, size=(X.size(2), X.size(3)))
#                 attention_maps = torch.sqrt(attention_maps.cpu() / attention_maps.max().item())

#                 # get heat attention maps
#                 heat_attention_maps = generate_heatmap(attention_maps)

#                 # raw_image, heat_attention, raw_attention
#                 raw_image = X.cpu() * STD + MEAN
#                 heat_attention_image = raw_image * 0.5 + heat_attention_maps * 0.5
#                 raw_attention_image = raw_image * attention_maps

#                 for batch_idx in range(X.size(0)):
#                     rimg = ToPILImage(raw_image[batch_idx])
#                     raimg = ToPILImage(raw_attention_image[batch_idx])
#                     haimg = ToPILImage(heat_attention_image[batch_idx])
#                     rimg.save(os.path.join(savepath, '%03d_raw.jpg' % (i * config.batch_size + batch_idx)))
#                     raimg.save(os.path.join(savepath, '%03d_raw_atten.jpg' % (i * config.batch_size + batch_idx)))
#                     haimg.save(os.path.join(savepath, '%03d_heat_atten.jpg' % (i * config.batch_size + batch_idx)))

#             # Top K
#             epoch_raw_acc = raw_accuracy(y_pred_raw, y)
#             epoch_ref_acc = ref_accuracy(y_pred, y)
#             # epoch_tp1, epoch_tn1, epoch_fp1, epoch_fn1, epoch_acc1, epoch_sen1, epoch_spe1 = raw_metric1(y_pred, y)

#             # end of this batch
#             batch_info = 'Val Acc: Raw ({:.2f}, {:.2f}), Refine ({:.2f}, {:.2f})'.format(
#                 epoch_raw_acc[0], epoch_raw_acc[1], epoch_ref_acc[0], epoch_ref_acc[1])
#             # batch_info1 = 'tp ({:d}), tn ({:d}), fp ({:d}), fn ({:d}), Val Acc1 ({:.2f}), Val Sen1 ({:.2f}), Val Spe1 ({:.2f})'.format(epoch_tp1, epoch_tn1, epoch_fp1, epoch_fn1, epoch_acc1, epoch_sen1, epoch_spe1)
#             # combined_string = f"{batch_info} | {batch_info1}"
#             pbar.update()
#             pbar.set_postfix_str(batch_info)
#             # logging.info(combined_string)  # 完整信息记录在日志里

#         # 合并所有批次的结果
#         all_targets = np.concatenate(all_targets)
#         all_predictions = np.concatenate(all_predictions)
#         # all_predictions_raw = np.concatenate(all_predictions_raw)
#         # all_predictions_crop = np.concatenate(all_predictions_crop)


#         # 将目标和预测保存到 Excel
#         df = pd.DataFrame({
#             'Target': all_targets,
#             'Prediction': all_predictions,
#             # 'Prediction_raw': all_predictions_raw,
#             # 'Prediction_crop': all_predictions_crop,


#         })
#         df.to_excel('predictions.xlsx', index=False)

#         # # 这里可以打印或者记录所有的目标和预测
#         # print("所有目标:", all_targets)
#         # print("所有预测:", all_predictions)
#         resulting2(config.choose_name + 'pred', filtered_data, all_targets, all_predictions)
#         # resulting2(name + 'raw', filtered_data, all_targets, all_predictions_raw)
#         # resulting2(name + 'crop', filtered_data, all_targets, all_predictions_crop)
#         pbar.close()
#         # logging.info("看我看我看我: %s", batch_info1)


# if __name__ == '__main__':
#     main()























#





#  # ------------------------- V4（裁剪前，有mask）-------------------------
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
    version = config.version[1:]

os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU
device = torch.device("cuda")
torch.backends.cudnn.benchmark = True

# visualize
visualize = config.visualize
savepath = config.eval_savepath
if visualize:
    os.makedirs(savepath, exist_ok=True)

ToPILImage = transforms.ToPILImage()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def generate_heatmap(attention_maps):
    heat_attention_maps = []
    heat_attention_maps.append(attention_maps[:, 0, ...])  # R
    heat_attention_maps.append(attention_maps[:, 0, ...] * (attention_maps[:, 0, ...] < 0.5).float() + \
                               (1. - attention_maps[:, 0, ...]) * (attention_maps[:, 0, ...] >= 0.5).float())  # G
    heat_attention_maps.append(1. - attention_maps[:, 0, ...])  # B
    return torch.stack(heat_attention_maps, dim=1)

# 过滤测试集
def filter_number1_by_number2(filename):
    result = []
    # 打开文件并逐行读取
    with open(filename,  'r', encoding='utf-8') as file:
        for line in file:
            # 将每行的两个数拆分
            number1, number2 = map(int, line.split())
            # 如果 number2 等于 0，保存 number1
            if number2 == 0:
                result.append(number1)
    return result

# 找到测试集的名称.dcm
def filter_and_strip_prefix(filename, numbers):
    result = []
    
    # 定义要去掉的前缀
    prefixes = ['000.no_tumor/', '001.tumor/']
    
    # 打开文件并逐行读取
    with open(filename,  'r', encoding='utf-8') as file:
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




def main():
    logging.basicConfig(
        format='%(asctime)s: %(levelname)s: [%(filename)s:%(lineno)d]: %(message)s',
        level=logging.INFO)
    warnings.filterwarnings("ignore")

    try:
        ckpt = config.eval_ckpt
    except:
        logging.info('Set ckpt for evaluation in config.py')
        return

    ##################################
    # Dataset for testing
    ##################################
    _, test_dataset = get_trainval_datasets(config.tag, resize=config.image_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)
    

    filtered_numbers = filter_number1_by_number2('D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V' + version + ')\\train_test_split' + config.choose_name + '.txt')  # “测试集”索引
    filtered_data = filter_and_strip_prefix('D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V' + version + ')\\images.txt', filtered_numbers)

    # print(len(filtered_numbers))
    # print(len(filtered_data))

    # print(filtered_numbers)
    # print(filtered_data)



    ##################################
    # Initialize model
    ##################################
    # net = WSDAN(num_classes=test_dataset.num_classes, M=config.num_attentions, net=config.net)
    net = DeepExpertNetwork(in_channels=3, base_channels=64, expert_channels=128, num_blocks=2, num_task_experts=2, mode='test')


    # Load ckpt and get state_dict
    checkpoint = torch.load(ckpt)
    state_dict = checkpoint['state_dict']

    # Load weights
    net.load_state_dict(state_dict)
    logging.info('Network loaded from {}'.format(ckpt))

    ##################################
    # use cuda
    ##################################
    net.to(device)
    if torch.cuda.device_count() > 1:
        net = nn.DataParallel(net)

    ##################################
    # Prediction
    ##################################
    raw_accuracy = TopKAccuracyMetric(topk=(1, 1))
    ref_accuracy = TopKAccuracyMetric(topk=(1, 1))
    # raw_metric1 = MixedMetric()
    raw_accuracy.reset()
    ref_accuracy.reset()
    # raw_metric1.reset()

    all_targets = []
    all_predictions = []
 #   all_predictions_raw = []
 #   all_predictions_crop = []
    net.eval()
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader), unit=' batches')
        pbar.set_description('Validation')
        # combined_string = ""
        # batch_info1 = ""

        # mask_origin, mask_tumor, mask_vessel, image, label

        for i, (mask_origin, mask_tumor, mask_vessel, mask_background, X, y) in enumerate(test_loader):
            mask_origin = mask_origin.to(device)
            mask_tumor = mask_tumor.to(device)
            mask_vessel = mask_vessel.to(device)
            mask_background =  mask_background.to(device)
            X = X.to(device)
            y = y.to(device)

            # WS-DAN
            a, b, feature_matrix, y_pred_raw,  y_pred_crop, y_pred_drop, outB = net(X)

            y_pred = (y_pred_raw + y_pred_crop) / 2.

      #      _, predicted_raw = torch.max(y_pred_raw, 1)
      #      _, predicted_crop = torch.max(y_pred_crop, 1)
            _, predicted = torch.max(y_pred, 1)

            all_targets.append(y.cpu().numpy())
     #       all_predictions_raw.append(predicted_raw.cpu().numpy())
     #       all_predictions_crop.append(predicted_crop.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())


            if visualize:
                # reshape attention maps
                attention_maps = F.upsample_bilinear(attention_maps, size=(X.size(2), X.size(3)))
                attention_maps = torch.sqrt(attention_maps.cpu() / attention_maps.max().item())

                # get heat attention maps
                heat_attention_maps = generate_heatmap(attention_maps)

                # raw_image, heat_attention, raw_attention
                raw_image = X.cpu() * STD + MEAN
                heat_attention_image = raw_image * 0.5 + heat_attention_maps * 0.5
                raw_attention_image = raw_image * attention_maps

                for batch_idx in range(X.size(0)):
                    rimg = ToPILImage(raw_image[batch_idx])
                    raimg = ToPILImage(raw_attention_image[batch_idx])
                    haimg = ToPILImage(heat_attention_image[batch_idx])
                    rimg.save(os.path.join(savepath, '%03d_raw.jpg' % (i * config.batch_size + batch_idx)))
                    raimg.save(os.path.join(savepath, '%03d_raw_atten.jpg' % (i * config.batch_size + batch_idx)))
                    haimg.save(os.path.join(savepath, '%03d_heat_atten.jpg' % (i * config.batch_size + batch_idx)))

            # Top K
            epoch_raw_acc = raw_accuracy(y_pred_raw, y)
            epoch_ref_acc = ref_accuracy(y_pred, y)
            # epoch_tp1, epoch_tn1, epoch_fp1, epoch_fn1, epoch_acc1, epoch_sen1, epoch_spe1 = raw_metric1(y_pred, y)

            # end of this batch
            batch_info = 'Val Acc: Raw ({:.2f}, {:.2f}), Refine ({:.2f}, {:.2f})'.format(
                epoch_raw_acc[0], epoch_raw_acc[1], epoch_ref_acc[0], epoch_ref_acc[1])
            # batch_info1 = 'tp ({:d}), tn ({:d}), fp ({:d}), fn ({:d}), Val Acc1 ({:.2f}), Val Sen1 ({:.2f}), Val Spe1 ({:.2f})'.format(epoch_tp1, epoch_tn1, epoch_fp1, epoch_fn1, epoch_acc1, epoch_sen1, epoch_spe1)
            # combined_string = f"{batch_info} | {batch_info1}"
            pbar.update()
            pbar.set_postfix_str(batch_info)
            # logging.info(combined_string)  # 完整信息记录在日志里

        # 合并所有批次的结果
        all_targets = np.concatenate(all_targets)
        all_predictions = np.concatenate(all_predictions)
 #       all_predictions_raw = np.concatenate(all_predictions_raw)
  #      all_predictions_crop = np.concatenate(all_predictions_crop)


        # 将目标和预测保存到 Excel
        df = pd.DataFrame({
            'tumor_nature': all_targets,
            'Prediction': all_predictions,
     #       'Prediction_raw': all_predictions_raw,
     #       'Prediction_crop': all_predictions_crop,


        })
        df.to_excel('predictions.xlsx', index=False)

        # # 这里可以打印或者记录所有的目标和预测
        # print("所有目标:", all_targets)
        # print("所有预测:", all_predictions)
        resulting2(config.choose_name + 'pred', filtered_data, all_targets, all_predictions)
  #      resulting2(name + 'raw', filtered_data, all_targets, all_predictions_raw)
  #      resulting2(name + 'crop', filtered_data, all_targets, all_predictions_crop)
        pbar.close()
        # logging.info("看我看我看我: %s", batch_info1)


if __name__ == '__main__':
    main()