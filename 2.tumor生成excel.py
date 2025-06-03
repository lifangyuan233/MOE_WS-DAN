import os
import pandas as pd

df = pd.read_excel("D:\\project\\MOE_WS-DAN\\train_adjust.xlsx")
tumor_names = df[df['tumor_nature'] == 1]['dcm_name'].tolist()
positive_names = df[df['is_positive'] == 1]['dcm_name'].tolist()

# print(len(tumor_names))
# print(len(positive_names))


def list_files_to_dataframe(folder_path):
    """
    将指定文件夹中的所有文件按名称顺序写入到 DataFrame 中。

    :param folder_path: 文件夹路径
    :return: 包含文件名的 DataFrame
    """
    # 获取文件夹中的所有文件名，并按名称排序
    files = sorted(os.listdir(folder_path))
    tumors = [0] * len(files)
    positives = [0] * len(files)

    for index, f in enumerate(files):
        if f[:-14] + ".dcm" in tumor_names:
            tumors[index] = 1
        if f[:-14] + ".dcm" in positive_names:
            positives[index] = 1
    
    df = pd.DataFrame({'dcm_name': files, 'tumor_nature': tumors, 'is_positive': positives})

    df.to_excel("D:\\project\\MOE_WS-DAN\\train_tumor_pro.xlsx",index=False)
    
    return df

# 示例用法
folder_path = "D:\\project\\MOE_WS-DAN\\datasets\\CUB_200_2011(V3)\\images\\001.tumor" 
df = list_files_to_dataframe(folder_path)

