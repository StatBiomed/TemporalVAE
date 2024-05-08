# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：compare_manyResult_df.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/4/2 16:11 
"""
import pandas as pd

import sys
import os

print(os.getcwd())
# sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/CNNC-master/utils")
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/model_master")
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan")

import anndata as ad
from collections import Counter
from utils.utils_DandanProject import *
from utils.utils_Dandan_plot import *
import time
import logging
from utils.logging_system import LogHelper
from sklearn.linear_model import LinearRegression


def main():
    result = pd.DataFrame(columns=["epoch", "spearman", "pearson"])
    miss_epoch = []
    for epoch in range(90, 200):
        file_name = f"/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/240409_human_2data_inter/240322Human_embryo/xiang2019/hvg500/supervise_vae_regressionclfdecoder_adversarial0121212_240407_humanEmbryo_dim50_timeembryoneg5to5_epoch{epoch}_externalDataembryo_minGeneNum50/k_fold_test_result.csv"
        # file_name = f"/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/240408_human_2data_inter/240322Human_embryo/xiang2019/hvg500/supervise_vae_regressionclfdecoder_adversarial0121212_240407_humanEmbryo_dim50_timeembryoneg5to5_epoch{epoch}_externalDataembryo_minGeneNum50/k_fold_test_result.csv"
        # file_name = f"/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/240402_humanEmbryo/240322Human_embryo/xiang2019/hvg500/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch{epoch}_minGeneNum50/k_fold_test_result.csv"
        try:
            _df = pd.read_csv(file_name)
            _r = calculate_real_predict_corrlation_score(_df["time"], _df["pseudotime"], only_str=False)
            result.loc[epoch] = {"epoch": epoch, "spearman": _r[1]['spearman'].correlation, "pearson": _r[1]['pearson'].correlation}
        except:
            miss_epoch.append(epoch)

    return


if __name__ == '__main__':
    main()
