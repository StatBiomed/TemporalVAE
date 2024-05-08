# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE
@IDE     ：PyCharm
@Author  ：awa1212
@Date    ：2023-11-16 16:33:55

"""
import pandas as pd
import json
import scanpy as sc
import matplotlib.pyplot as plt
# 2023-11-03 11:38:31
from sklearn.decomposition import PCA
from collections import Counter
from utils.utils_Dandan_plot import colors_tuple_hexadecimalColorCode
import numpy as np

from utils.utils_Dandan_plot import plot_on_each_test_donor_violin_fromDF
import pandas as pd
# ----------------------- k-fold on human embryo xiang2019 data, for temporalVAE compare with LR, PCA, RF -----------------------
file_name = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/240413_preimplantation_Melania/240405_preimplantation_Melania/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch100_minGeneNum50/k_fold_test_result.csv"
data_pd = pd.read_csv(file_name)
temp_pd=data_pd.loc[data_pd["dataset_label"]!="t"]

plot_on_each_test_donor_violin_fromDF(temp_pd, "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/240413_preimplantation_Melania/240405_preimplantation_Melania/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch100_minGeneNum50/", "predicted_time", x_str="time", special_file_name_str="without_Tyser_",cmap_color="viridis")
print(1)