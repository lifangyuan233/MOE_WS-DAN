import pandas as pd
from PIL import Image
import os
from PIL import ImageEnhance

# 0、原图
def origin_image(input_path, output_path):
    img = Image.open(input_path)

    img.save(output_path)

# 1、水平翻转
def flip_image(input_path, output_path):
    img = Image.open(input_path)

    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_img.save(output_path)

# 2、亮度1.2
def adjust_brightness(input_path, output_path, factor=1.2):
    img = Image.open(input_path)
    enhancer = ImageEnhance.Brightness(img)
    brightened_img = enhancer.enhance(factor)
    brightened_img.save(output_path)

# 3、亮度1.5
def adjust_brightness1(input_path, output_path, factor=1.5):
    img = Image.open(input_path)
    enhancer = ImageEnhance.Brightness(img)
    brightened_img = enhancer.enhance(factor)
    brightened_img.save(output_path)

# 4、对比度1.5
def adjust_contrast(input_path, output_path, factor=1.5):
    img = Image.open(input_path)
    enhancer = ImageEnhance.Contrast(img)
    contrast_img = enhancer.enhance(factor)
    contrast_img.save(output_path)

# 5、颜色1.5
def adjust_color(input_path, output_path, factor=1.5):
    img = Image.open(input_path)
    enhancer = ImageEnhance.Color(img)
    color_adjusted_img = enhancer.enhance(factor)
    color_adjusted_img.save(output_path)



# 6、模糊1.2
from PIL import ImageFilter

def blur_image(input_path, output_path, radius=1.2):
    img = Image.open(input_path)
    blurred_img = img.filter(ImageFilter.GaussianBlur(radius))
    blurred_img.save(output_path)

# 7、mixup 0.4
import numpy as np
from PIL import Image

def mixup(image1_path, image2_path, output_path, alpha=0.35): # alpha是img1的比例，主：img2
    # 读取两张图片
    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)
    
    # 调整两张图片到相同大小
    img1 = img1.resize(img2.size)
    
    # 将图片转换为 NumPy 数组
    np_img1 = np.array(img1).astype(np.float32)
    np_img2 = np.array(img2).astype(np.float32)
    
    # 生成lambda参数，使用beta分布
    lambda_value = np.random.beta(alpha, alpha)
    lambda_value = min(lambda_value, 1 - lambda_value)
    
    # 进行Mixup操作
    mixed_img = lambda_value * np_img1 + (1 - lambda_value) * np_img2
    mixed_img = np.clip(mixed_img, 0, 255).astype(np.uint8)  # 保持像素值在0到255之间
    
    # 将结果转换回Pillow图像并保存
    mixed_img = Image.fromarray(mixed_img)
    mixed_img.save(output_path)

    return lambda_value  # 返回lambda值，以便用于标签的混合


# 8、饱和度 1.2
from PIL import Image, ImageEnhance
import numpy as np

def adjust_random_saturation(input_path, output_path, factor=1.2):
    """
    随机调整图像的饱和度。

    :param input_path: 输入图像路径
    :param output_path: 输出图像路径
    :param min_factor: 最小饱和度因子，默认为0.5
    :param max_factor: 最大饱和度因子，默认为1.5
    """
    img = Image.open(input_path)
    enhancer = ImageEnhance.Color(img)
    
    # 调整饱和度
    saturated_img = enhancer.enhance(factor)
    
    # 保存结果
    saturated_img.save(output_path)


# 9、噪声
import numpy as np
from PIL import Image

def add_small_noise(input_path, output_path, mean=0, std=5):
    """
    向图像中添加小随机高斯噪声。

    :param input_path: 输入图像路径
    :param output_path: 输出图像路径
    :param mean: 噪声的均值，默认为0
    :param std: 噪声的标准差，默认为5
    """
    img = Image.open(input_path)
    np_img = np.array(img).astype(np.float32)
    
    # 生成与图像大小相同的小高斯噪声
    noise = np.random.normal(mean, std, np_img.shape)
    
    # 将噪声添加到图像上
    noisy_img = np_img + noise
    
    # 确保像素值在0到255之间
    noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
    
    # 将结果转换回Pillow图像并保存
    noisy_img = Image.fromarray(noisy_img)
    noisy_img.save(output_path)



'''
df = pd.read_excel("D:\\project\\MOE_WS-DAN\\train_adjust.xlsx")
all_names = df['dcm_name'].values.tolist()
dcm_names = df[df['tumor_nature'] == 1]['dcm_name'].tolist()
print(len(dcm_names))  # 89


# dcm_names = ['C19004A_0003627-Anonymized-202205231529-D-L.dcm', 'C20001D_0007253-Anonymized-202304271547-D-R.dcm', 'C20005B_0000194-Anonymized-202207181606-D-R.dcm', 'C19007A_0000109-Anonymized-202009030920-D-L.dcm', 'C19004A_0001700-Anonymized-202105211426-D-R.dcm', 'C20001D_0005899-Anonymized-202302280958-D-L.dcm', 'C20001D_0002870-Anonymized-202108061002-D-L.dcm', 'C20001D_0002593-Anonymized-202107271523-D-L.dcm', 'C20001D_0000121-Anonymized-202104221406-D-L.dcm', 'C19004A_0003144-Anonymized-202108241619-D-L.dcm', 'C20001D_0000068-Anonymized-202104211358-D-L.dcm', 'C19005B_0000346-Anonymized-202105281616-D-R.dcm', 'C19004A_0004252-Anonymized-202209211537-D-R.dcm', 'C19005C_0000280-Anonymized-202107211519-D-L.dcm', 'C20001D_0003443-Anonymized-202109061715-D-L.dcm', 'C20001D_0002651-Anonymized-202107290844-D-L.dcm', 'C20001D_0001174-Anonymized-202106041457-D-R.dcm', 'C16004_0000157-Anonymized-202106031202-D-L.dcm', 'C19004A_0004570-Anonymized-202211041453-D-R.dcm', 'C18005D_0000540-Anonymized-202208101459-D-L.dcm', 'C19004A_0004335-Anonymized-202209281624-D-L.dcm', 'C20001D_0003937-Anonymized-202110191404-D-L.dcm', 'C18005B_0000130-Anonymized-202206060825-D-L.dcm', 'C19005B_0000492-Anonymized-202106081343-D-R.dcm', 'C20001D_0002963-Anonymized-202108110941-D-R.dcm', 'C19004A_0002584-Anonymized-202107081641-D-L.dcm', 'C19004A_0004299-Anonymized-202209261637-D-R.dcm', 'C19004A_0002990-Anonymized-202108091614-D-R.dcm', 'C19007_0002338-Anonymized-202008201456-D-L.dcm', 'C16002A_0000462-Anonymized-202107190924-D-L.dcm', 'C20004C_0000080-Anonymized-202007081530-D-R.dcm', 'C16004B_0000284-Anonymized-202108201052-D-L.dcm', 'C16004B_0000201-Anonymized-202108170910-D-L.dcm', 'C16004B_0000099-Anonymized-202108111320-D-L.dcm', 'C18002A_0000108-Anonymized-202007091723-D-L.dcm', 'C20004_0000913-Anonymized-202005301058-D-L.dcm', 'C20003_0001064-Anonymized-202007201406-D-R.dcm', 'C16004_0000031-Anonymized-202105311122-D-R.dcm', 'C20004C_0000048-Anonymized-202007080919-D-L.dcm', 'C16002A_0000462-Anonymized-202107190927-D-R.dcm', 'C20004D_0000137-Anonymized-202007241640-D-L.dcm', 'C20004D_0000345-Anonymized-202007301753-D-L.dcm', 'C20004C_0000312-Anonymized-202007161121-D-L.dcm', 'C20004C_0000223-Anonymized-202007140851-D-L.dcm', 'C20004_0000208-Anonymized-202005201003-D-L.dcm', 'C20004B_0000467-Anonymized-202007041040-D-L.dcm', 'C20004_0000377-Anonymized-202005221602-D-R.dcm', 'C20004B_0000412-Anonymized-202007031050-D-R.dcm', 'C16002A_0000478-Anonymized-202107200840-D-R.dcm', 'C20003_0000814-Anonymized-202007071435-D-L.dcm', 'C20004D_0000119-Anonymized-202007241434-D-R.dcm', 'C20003_0000779-Anonymized-202007061434-D-R.dcm', 'C20003_0001064-Anonymized-202007201403-D-L.dcm', 'C20004D_0000321-Anonymized-202007301530-D-L.dcm', 'C20004_0000423-Anonymized-202005231141-D-L.dcm', 'C16004_0000548-Anonymized-202106141001-D-R.dcm', 'C19002B_0000354-Anonymized-202106211136-D-R.dcm', 'C20001_0000245-Anonymized-202006091231-D-L.dcm', 'C20004B_0000123-Anonymized-202006251004-D-R.dcm', 'C18004B_0000119-Anonymized-202108281153-D-R.dcm', 'C18006B_0000056-Anonymized-202007290824-D-R.dcm', 'C20001_0000277-Anonymized-202006101332-D-R.dcm']


for row in df.itertuples():
    if row.tumor_nature != 1:
        continue
    index = row.Index
    dcm_name = row.dcm_name
    input_path = "D:\\data\\pngs(V3)\\" + dcm_name[:-4] + ".png" 
    print(index)
'''

    # # # # 原图
    # output_path = "D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V3)\\images\\001.tumor\\" + dcm_name[:-4] + "-00-000000.png" 
    # origin_image(input_path, output_path)
    # print(f"图片已保存为: {output_path}")

    # # 水平旋转
    # output_path = "D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V3)\\images\\001.tumor\\" + dcm_name[:-4] + "-01-000000.png" 
    # flip_image(input_path, output_path)
    # print(f"图片已保存为: {output_path}")

    # # 亮度1.2
    # output_path = "D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V3)\\images\\001.tumor\\" + dcm_name[:-4] + "-02-000000.png" 
    # adjust_brightness(input_path, output_path)
    # print(f"图片已保存为: {output_path}")

    # # 亮度1.5
    # output_path = "D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V3)\\images\\001.tumor\\" + dcm_name[:-4] + "-03-000000.png" 
    # adjust_brightness1(input_path, output_path)
    # print(f"图片已保存为: {output_path}")

    # # 对比度1.5
    # output_path = "D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V3)\\images\\001.tumor\\" + dcm_name[:-4] + "-04-000000.png" 
    # adjust_contrast(input_path, output_path)
    # print(f"图片已保存为: {output_path}")


    # # 颜色1.5
    # output_path = "D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V3)\\images\\001.tumor\\" + dcm_name[:-4] + "-05-000000.png" 
    # adjust_color(input_path, output_path)
    # print(f"图片已保存为: {output_path}")

    # # 模糊1.2
    # output_path = "D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V3)\\images\\001.tumor\\" + dcm_name[:-4] + "-06-000000.png" 
    # blur_image(input_path, output_path)
    # print(f"图片已保存为: {output_path}")



    # # mixup 0.4（在2000中找61个，61 + 61中找20个）
    # # numbers1 = np.random.choice(2000, 61, replace=False)
    # # numbers2 = np.delete(np.arange(2000,2062), i)
    # numbers1 = np.random.choice(2000, 89, replace=False)
    # numbers2 = np.delete(np.arange(2000,2089), index-2000)
    # random_numbers = np.concatenate((numbers1, numbers2))
    # random_selections = np.random.choice(random_numbers, 13, replace=False)  # replace=False 表示不重复选择(20)

    # for random_selection in random_selections:
    #     str_num = str(random_selection).zfill(6)
    #     output_path = "D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V3)\\images\\ttt\\" + dcm_name[:-4] + "-07-" + str_num + ".png" 
    #     image1_path = "D:\\data\\pngs(V3)\\" + all_names[random_selection][:-4] + ".png" 

    #     mixup(image1_path, input_path, output_path, alpha=0.35)

    #     print(f"图片已保存为: {output_path}")

    # # 饱和度 1.2
    # output_path = "D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V3)\\images\\001.tumor\\" + dcm_name[:-4] + "-08-000000.png" 
    # adjust_random_saturation(input_path, output_path)
    # print(f"图片已保存为: {output_path}")


    # # 小噪声
    # output_path = "D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V3)\\images\\001.tumor\\" + dcm_name[:-4] + "-09-000000.png" 
    # add_small_noise(input_path, output_path)
    # print(f"图片已保存为: {output_path}")




# # 补充val 和 test 的有肿瘤案例
# import pandas as pd
# from PIL import Image
# import os
# from PIL import ImageEnhance


# df = pd.read_excel("D:\\project\\MOE_WS-DAN\\train_adjust.xlsx")
# # all_names = df['dcm_name'].values.tolist()
# dcm_names = df[df['tumor_nature'] == 1]['dcm_name'].tolist()
# print(len(dcm_names))  # 89


# # dcm_names = ['C19004A_0003627-Anonymized-202205231529-D-L.dcm', 'C20001D_0007253-Anonymized-202304271547-D-R.dcm', 'C20005B_0000194-Anonymized-202207181606-D-R.dcm', 'C19007A_0000109-Anonymized-202009030920-D-L.dcm', 'C19004A_0001700-Anonymized-202105211426-D-R.dcm', 'C20001D_0005899-Anonymized-202302280958-D-L.dcm', 'C20001D_0002870-Anonymized-202108061002-D-L.dcm', 'C20001D_0002593-Anonymized-202107271523-D-L.dcm', 'C20001D_0000121-Anonymized-202104221406-D-L.dcm', 'C19004A_0003144-Anonymized-202108241619-D-L.dcm', 'C20001D_0000068-Anonymized-202104211358-D-L.dcm', 'C19005B_0000346-Anonymized-202105281616-D-R.dcm', 'C19004A_0004252-Anonymized-202209211537-D-R.dcm', 'C19005C_0000280-Anonymized-202107211519-D-L.dcm', 'C20001D_0003443-Anonymized-202109061715-D-L.dcm', 'C20001D_0002651-Anonymized-202107290844-D-L.dcm', 'C20001D_0001174-Anonymized-202106041457-D-R.dcm', 'C16004_0000157-Anonymized-202106031202-D-L.dcm', 'C19004A_0004570-Anonymized-202211041453-D-R.dcm', 'C18005D_0000540-Anonymized-202208101459-D-L.dcm', 'C19004A_0004335-Anonymized-202209281624-D-L.dcm', 'C20001D_0003937-Anonymized-202110191404-D-L.dcm', 'C18005B_0000130-Anonymized-202206060825-D-L.dcm', 'C19005B_0000492-Anonymized-202106081343-D-R.dcm', 'C20001D_0002963-Anonymized-202108110941-D-R.dcm', 'C19004A_0002584-Anonymized-202107081641-D-L.dcm', 'C19004A_0004299-Anonymized-202209261637-D-R.dcm', 'C19004A_0002990-Anonymized-202108091614-D-R.dcm', 'C19007_0002338-Anonymized-202008201456-D-L.dcm', 'C16002A_0000462-Anonymized-202107190924-D-L.dcm', 'C20004C_0000080-Anonymized-202007081530-D-R.dcm', 'C16004B_0000284-Anonymized-202108201052-D-L.dcm', 'C16004B_0000201-Anonymized-202108170910-D-L.dcm', 'C16004B_0000099-Anonymized-202108111320-D-L.dcm', 'C18002A_0000108-Anonymized-202007091723-D-L.dcm', 'C20004_0000913-Anonymized-202005301058-D-L.dcm', 'C20003_0001064-Anonymized-202007201406-D-R.dcm', 'C16004_0000031-Anonymized-202105311122-D-R.dcm', 'C20004C_0000048-Anonymized-202007080919-D-L.dcm', 'C16002A_0000462-Anonymized-202107190927-D-R.dcm', 'C20004D_0000137-Anonymized-202007241640-D-L.dcm', 'C20004D_0000345-Anonymized-202007301753-D-L.dcm', 'C20004C_0000312-Anonymized-202007161121-D-L.dcm', 'C20004C_0000223-Anonymized-202007140851-D-L.dcm', 'C20004_0000208-Anonymized-202005201003-D-L.dcm', 'C20004B_0000467-Anonymized-202007041040-D-L.dcm', 'C20004_0000377-Anonymized-202005221602-D-R.dcm', 'C20004B_0000412-Anonymized-202007031050-D-R.dcm', 'C16002A_0000478-Anonymized-202107200840-D-R.dcm', 'C20003_0000814-Anonymized-202007071435-D-L.dcm', 'C20004D_0000119-Anonymized-202007241434-D-R.dcm', 'C20003_0000779-Anonymized-202007061434-D-R.dcm', 'C20003_0001064-Anonymized-202007201403-D-L.dcm', 'C20004D_0000321-Anonymized-202007301530-D-L.dcm', 'C20004_0000423-Anonymized-202005231141-D-L.dcm', 'C16004_0000548-Anonymized-202106141001-D-R.dcm', 'C19002B_0000354-Anonymized-202106211136-D-R.dcm', 'C20001_0000245-Anonymized-202006091231-D-L.dcm', 'C20004B_0000123-Anonymized-202006251004-D-R.dcm', 'C18004B_0000119-Anonymized-202108281153-D-R.dcm', 'C18006B_0000056-Anonymized-202007290824-D-R.dcm', 'C20001_0000277-Anonymized-202006101332-D-R.dcm']

# # dcm_names = ['']

# for dcm_name in dcm_names:
#     from0 = "D:\\data\\pngs-all(V3)\\" + dcm_name[:-4] + ".png"
#     to0 = "D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V3)\\images\\ttt\\" + dcm_name[:-4] + ".png"

#     img = Image.open(from0)
#     img.save(to0)


df = pd.read_excel("D:\\project\\MOE_WS-DAN\\total_adjust.xlsx")
all_names = df['dcm_name'].values.tolist()
print(len(all_names))  # 89
names0 = os.listdir('D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V3)\\images\\tumor_mask')

for name in all_names:
    name = name[:-4] + '.png'
    if name in names0:
        continue
    
    else:
        input_path = 'D:\\data\\masks-all-0(V3)\\' + name
        output_path = 'D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V3)\\images\\222\\' + name
        # 读取mask并转换为数组
        img = Image.open(input_path)
        mask = np.array(img)

        # 创建全零数组并转换回图像
        zero_mask = Image.fromarray(np.zeros_like(mask))

        # 保存结果
        zero_mask.save(output_path)


