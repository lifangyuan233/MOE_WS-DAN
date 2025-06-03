import cv2
import numpy as np
import os

def convert0(mask1_path, mask2_path, save_path):
    # 读取mask1和mask2，确保为灰度模式
    mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)

    # # 检查尺寸是否相同
    # if mask1.shape != mask2.shape:
    #     raise ValueError("Masks must have the same dimensions")

    # 生成mask3：mask1为255且mask2为0的位置设为255，其他为0
    mask3 = np.where((mask1 == 255) & (mask2 == 0), 255, 0).astype(np.uint8)

    # 保存结果

    cv2.imwrite(save_path, mask3)

files = os.listdir("D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V3)\\images\\病灶mask")
for file in files:
    mask1_path = "D:\\data\\masks-all-0(V3)\\" + file
    mask2_path = "D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V3)\\images\\病灶mask\\" + file
    save_path = "D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V3)\\images\\血管mask\\" + file
    convert0(mask1_path, mask2_path, save_path)

