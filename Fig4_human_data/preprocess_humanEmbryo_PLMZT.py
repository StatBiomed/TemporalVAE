# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：preprocess_humanEmbryo_PLMZT.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/8/22 13:41 
"""
import scanpy as sc
import pandas as pd
import anndata as ad
from collections import Counter
from utils.utils_DandanProject import calHVG_adata as calHVG
from utils.utils_DandanProject import series_matrix2csv
from utils.utils_Dandan_plot import draw_venn
import h5py
import os
import gzip
import pandas as pd

def main():
    file_path = "../data/240405_preimplantation_Melania/"
    p_raw = pd.read_csv(f"{file_path}/P_raw_count.txt", sep="\t", index_col=0)
    m_raw = pd.read_csv(f"{file_path}/M_raw_count.csv", sep=",", index_col=0)
    z_raw = pd.read_csv(f"{file_path}/Z_raw_count_TPM.txt", sep="\t", index_col=0)

    # 定义文件夹路径
    folder_path = f'{file_path}/L_raw_count'

    # 获取所有以 .txt.gz 结尾的文件名
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt.gz')]

    # 创建一个空的列表用于存储每个文件的DataFrame
    dfs = []

    # 逐个读取文件并存储在DataFrame中
    for file in files:
        file_path = os.path.join(folder_path, file)
        with gzip.open(file_path, 'rt') as f:
            df = pd.read_csv(f, sep='\t')  # 假设文件是以制表符分隔的
            dfs.append(df)

    # 合并所有DataFrame为一个
    combined_df = pd.concat(dfs, ignore_index=True)
    return


if __name__ == '__main__':
    main()
