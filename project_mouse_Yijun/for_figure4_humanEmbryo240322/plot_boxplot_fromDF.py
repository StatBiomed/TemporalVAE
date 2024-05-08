# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：plot_boxplot_fromDF.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/3/24 15:46 
"""
import sys

sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan")

from utils.utils_Dandan_plot import calculate_real_predict_corrlation_score,plot_boxplot_from_dic


def main():
    import pandas as pd
    # ----------------------- k-fold on human embryo xiang2019 data, for temporalVAE compare with LR, PCA, RF -----------------------
    file_name = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/240402_humanEmbryo/240322Human_embryo/xiang2019/hvg500/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch70_minGeneNum50/k_fold_test_result.csv"
    data_pd = pd.read_csv(file_name)
    VAE = calculate_real_predict_corrlation_score(data_pd["time"], data_pd["predicted_time"])

    file_name = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/240403_forFig4_compareWithBaseLine/240322Human_embryo/xiang2019/hvg500/linearRegression/result_df.csv"
    data_pd = pd.read_csv(file_name)
    LR = calculate_real_predict_corrlation_score(data_pd["time"], data_pd["pseudotime"])

    file_name = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/240403_forFig4_compareWithBaseLine/240322Human_embryo/xiang2019/hvg500/PCA/result_df.csv"
    data_pd = pd.read_csv(file_name)
    PCA = calculate_real_predict_corrlation_score(data_pd["time"], data_pd["pseudotime"])

    file_name = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/240403_forFig4_compareWithBaseLine/240322Human_embryo/xiang2019/hvg500/randomForest/result_df.csv"
    data_pd = pd.read_csv(file_name)
    RF = calculate_real_predict_corrlation_score(data_pd["time"], data_pd["pseudotime"])
    # 构建数据，确保按照VAE、LR、PCA的顺序
    data = {
        'Method': ['TemporalVAE', 'TemporalVAE', 'LR', 'LR', 'PCA', 'PCA', 'RF', 'RF'],
        'Correlation Type': ['Spearman', 'Pearson', 'Spearman', 'Pearson', 'Spearman', 'Pearson', 'Spearman', 'Pearson'],
        'Value': [0.91631, 0.91689, 0.86697, 0.86489, 0.24692, 0.22120, 0.65485, 0.65186]
    }
    plot_boxplot_from_dic(data)



if __name__ == '__main__':
    main()
