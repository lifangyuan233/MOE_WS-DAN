import json
import numpy as np
import cv2
from PIL import Image
import os

def json_to_mask(json_path, output_path=None):
    # 加载JSON文件
    with open(json_path) as f:
        data = json.load(f)
    
    # 获取图像尺寸
    height = data['imageHeight']
    width = data['imageWidth']
    
    # 创建空白mask
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # 遍历所有形状
    for shape in data['shapes']:
        if shape['shape_type'] == 'polygon':
            # 将点坐标转换为numpy数组
            points = np.array(shape['points'], dtype=np.float32)
            
            # 将归一化坐标转换为像素坐标（如果需要）
            # 如果坐标已经是像素坐标，可以跳过这步
            # points[:, 0] *= width
            # points[:, 1] *= height
            
            # 转换为整数坐标
            points = points.astype(np.int32)
            
            # 填充多边形
            cv2.fillPoly(mask, [points], color=255)
    
    # 如果需要保存mask
    if output_path:
        cv2.imwrite(output_path, mask)
        print(f"Mask saved to {output_path}")
    
    return mask


files = os.listdir('D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V3)\\images\\ttt0')
for file in files:
    json_path = "D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V3)\\images\\ttt0\\" + file

    mask = json_to_mask(json_path, 'D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V3)\\images\\1\\' + file[:-4] + 'png')

# # 可视化（可选）
# Image.fromarray(mask).show()