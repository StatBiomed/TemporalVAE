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

from utils.utils_Dandan_plot import calculate_real_predict_corrlation_score


def main():
    import pandas as pd
    # ----------------------- k-fold on mouse atlas data, for temporalVAE compare with LR, PCA, RF -----------------------
    file_name = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/230827_trainOn_mouse_embryonic_development_kFold_testOnYZdata0809/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/supervise_vae_regressionclfdecoder_dim50_timeembryoneg5to5_epoch100_dropDonorno_mouseEmbryonicDevelopment_embryoneg5to5/result_df.csv"
    data_pd = pd.read_csv(file_name)
    VAE = calculate_real_predict_corrlation_score(data_pd["time"], data_pd["predicted_time"])

    file_name = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/240322_forFig3_compareWithBaseLine/linearRegression_atlas/result_df.csv"
    data_pd = pd.read_csv(file_name)
    LR = calculate_real_predict_corrlation_score(data_pd["time"], data_pd["pseudotime"])

    file_name = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/240322_forFig3_compareWithBaseLine/PCA_atlas/result_df.csv"
    data_pd = pd.read_csv(file_name)
    PCA = calculate_real_predict_corrlation_score(data_pd["time"], data_pd["pseudotime"])

    file_name = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/240322_forFig3_compareWithBaseLine/randomForest_atlas/result_df.csv"
    data_pd = pd.read_csv(file_name)
    RF = calculate_real_predict_corrlation_score(data_pd["time"], data_pd["pseudotime"])
    data = {
        'Method': ['TemporalVAE', 'TemporalVAE', 'LR', 'LR', 'PCA', 'PCA', "RF", "RF"],
        'Correlation Type': ['Spearman', 'Pearson', 'Spearman', 'Pearson', 'Spearman', 'Pearson', 'Spearman', 'Pearson'],
        'Value': [0.89082, 0.91370, 0.78595, 0.79398, 0.25161, 0.14779, 0.07168, 0.36478]
    }

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns


    # for "embryo_1"
    # time_dic={"TemporalVAE":2161.1639816761017,}
    df = pd.DataFrame(data)

    # 设置绘图风格
    sns.set(style="whitegrid")

    # 创建条形图
    plt.figure(figsize=(8, 6))
    barplot = sns.barplot(x='Method', y='Value', hue='Correlation Type', data=df, palette=["#B0E0E6", "#D8BFD8"])

    # 添加标题和坐标轴标签
    plt.title('Method Performance: Spearman vs Pearson Correlation', fontsize=16)
    plt.ylabel('Correlation Value', fontsize=14)
    plt.xlabel('Method', fontsize=14)

    # 调整图例
    plt.legend(title='Correlation Type', title_fontsize='13', fontsize='12')

    # 在每个条形上显示数值
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.3f'),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center',
                         xytext=(0, 10),
                         textcoords='offset points')

    # 设置Y轴范围以清晰展示负相关性，添加Y=0参考线强调正负相关性
    plt.ylim(0, 1)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')

    # 美化图表
    sns.despine(offset=10, trim=True)  # 减少边框
    plt.tight_layout()  # 自动调整子图参数,使之填充整个图像区域

    # 显示图表
    plt.show()

    return

if __name__ == '__main__':
    main()
