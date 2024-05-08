# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：test_orginal_download_stereo_data.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/2/19 10:53 
"""


import scanpy as sc
import numpy as np
import pandas as pd
import pyreadr
import os
import anndata
from collections import Counter
def main():
    file_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/mouse_embryo_stereo/"
    sc_file_name = "/Mouse_brain.h5ad"
    print("check for {}".format(sc_file_name))

    adata = sc.read_h5ad(filename=file_path + sc_file_name)
    print("Import data, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    print("Annotation information of data includes: {}".format(adata.obs_keys()))  # 胞注釋信息的keys
    print("Cell id first 5: {}".format(adata.obs_names[:5]))  # 返回胞ID 数据类型是object
    print("Gene id first 5: {}".format(adata.var_names.to_list()[:5]))  # 返回基因数据类型是list adata.obs.head()# 査看前5行的数据

    print("Finish all")




if __name__ == '__main__':
    main()
