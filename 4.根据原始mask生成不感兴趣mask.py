import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import scipy.io
import numpy as np
import cv2

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image



def generate_mask1(mask):
    # Step 1: 创建用于膨胀和腐蚀的卷积核
    kernel_size = 7  # 控制轮廓的宽度，可以根据需求调整
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Step 2: 进行膨胀操作，使得255的区域向外扩展
    dilated = cv2.dilate(mask, kernel, iterations=1)

    # Step 3: 进行腐蚀操作，使得255的区域向内收缩
    eroded = cv2.erode(mask, kernel, iterations=1)

    # Step 4: 两者相减，得到膨胀后的图像减去腐蚀后的图像，结果就是轮廓
    contour_mask = cv2.subtract(dilated, eroded)

    # Step 5: 将结果二值化，确保轮廓部分为255，其他部分为0
    _,mask1 = cv2.threshold(contour_mask, 127, 255, cv2.THRESH_BINARY)

    return mask1

def generate_mask2(point):
    # 创建一个与原图像大小相同的空白mask
    mask = np.zeros((102, 128), dtype=np.uint8)

    # 在指定的像素位置设置255
    x, y = point[0]
    mask[y, x] = 255  # 设定有效区域

    # 设置圆形的半径
    radius = 10  # 根据需要调整圆的半径

    # 在指定位置绘制圆形，颜色为255
    cv2.circle(mask, (point[0][0], point[0][1]), radius, (255), thickness=-1)

    return mask

# # 单个举例
# mat_file = 'P23001A_0000118-高美丽-202408191118-D-R.mat'
# mat = scipy.io.loadmat('D:\\data\\mats-all\\' + mat_file)

# mask = np.round(np.clip(mat['Mms'] * 255, 0, 255))
# mask1 = generate_mask1(mask)
# if len(mat['BreastPoint']) != 0:
#     mask2 = generate_mask2(mat['BreastPoint'])

#     # 叠加两个mask
#     combined_mask = cv2.bitwise_or(mask1, mask2)
# else:
#     combined_mask = mask1
# # # cv2.imshow("m1", mask)
# # # cv2.imshow("m2",mask1)
# # # cv2.imshow("m3",mask2)
# cv2.imshow("m4",combined_mask)
# cv2.waitKey(0)


# 批量处理
import os
filePath = 'D:\\data\\mats-all'
files = os.listdir(filePath)

files = ['445-DGD-202405201133-D-L.mat', '5456-hlhjljhlv-202408171713-D-L.mat']
for mat_file in files:
    mat = scipy.io.loadmat('D:\\data\\mats-all\\' + mat_file)
    mask = np.round(np.clip(mat['Mms'] * 255, 0, 255))

    mask1 = generate_mask1(mask)
    if len(mat['BreastPoint']) != 0:
        mask2 = generate_mask2(mat['BreastPoint'])
        combined_mask = cv2.bitwise_or(mask1, mask2)
    else:
        combined_mask = mask1

    combined_mask_image = Image.fromarray(combined_mask)
    combined_mask_image.save(os.path.join('D:\\data\\masks-all(V2)', mat_file[:-4] + ".png"), format='PNG')
    print(mat_file +  " finish!")
    # cv2.waitKey(0)




# from PIL import Image
# i = Image.open('D:\\project\\BBN\\mask-all\\010-BJBA-00010-WHL-201709281420-D.png')
# print(i)
# print(type(i))