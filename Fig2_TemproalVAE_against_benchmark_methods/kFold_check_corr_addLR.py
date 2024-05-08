# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：kFold_check_corr.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/9/26 16:27

pseudotime
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
import random
import numpy as np

import pandas as pd
import seaborn as sns



def main():
    # method_list = ["pca", "randomForest", "psupertime", "vae"]
    dataset_list = ["acinarHVG", "acinarHVG", "embryoBeta", "humanGermline"]
    dataset_dic = {"acinarHVG": "Acinar", "humanGermline": "Human female germ line", "embryoBeta": "Embryo beta"}
    # dataset_list = ["acinarHVG", "humanGermline", "embryoBeta"]

    for dataset in dataset_list:
        pca_df = get_method_result("pca", dataset)
        randomForest_df = get_method_result("randomForest", dataset)
        psupertime_df = get_method_result("psupertime", dataset)
        vae_df = get_method_result("vae", dataset)
        lr_df = get_method_result("LR", dataset)
        # 初始化标签列表
        all_labels = ['No removal']+list(np.sort(np.unique(pca_df["time"])))
        plot_kFold_corr(pca_df.copy(), randomForest_df.copy(), lr_df.copy(), psupertime_df.copy(), vae_df.copy(), dataset, all_labels, dataset_dic)
        plot_boxAndDot_on_allData(pca_df.copy(), randomForest_df.copy(), lr_df.copy(), psupertime_df.copy(), vae_df.copy(), dataset, dataset_dic)
        # plot boxplot on all data


# 创建一个函数来计算Spearman相关系数


def calculate_corr(df, label_to_remove, corr_method, neg_bool=1):
    if label_to_remove != "No removal":
        # 删除指定标签
        # df["time"] = df["time"].astype(int)  # take care here as psupertime and vae can have continues time as input
        df_filtered = df[df['time'] != label_to_remove]

    else:
        df_filtered = df.copy()
    # 计算Spearman相关系数
    if corr_method == "spearmanr":
        try:
            spearman_corr, _ = spearmanr(df_filtered['time'], df_filtered['pseudotime'])
        except:
            print("error?")
    elif corr_method == "kendalltau":
        spearman_corr, _ = kendalltau(df_filtered['time'], df_filtered['pseudotime'])
    return abs(spearman_corr) * neg_bool


def get_method_result(method, dataset): #

    file = f"{os.getcwd()}/{method}_results/{dataset}_{method}_result.csv"
    data = pd.read_csv(file, index_col=0)
    return data
def preprocess_parameters(dataset): #"acinarHVG", "embryoBeta", "humanGermline"
    print(f"for dataset {dataset}.")
    # ------------ for Mouse embryonic beta cells dataset:
    if dataset=="embryoBeta":
        data_x_df = pd.read_csv(f'data_fromPsupertime/{dataset}_X.csv', index_col=0).T
        hvg_gene_list = pd.read_csv(f'{os.getcwd()}/data_fromPsupertime/{dataset}_gene_list.csv', index_col=0)
        data_x_df = data_x_df[hvg_gene_list["gene_name"]]
        data_y_df = pd.read_csv(f'data_fromPsupertime/{dataset}_Y.csv', index_col=0)
        data_y_df = data_y_df["time"]
        preprocessing_params = {"select_genes": "all", "log": True}
    # ------------ for Human Germline dataset:
    elif dataset == "humanGermline":
        data_x_df = pd.read_csv('data_fromPsupertime/humanGermline_X.csv', index_col=0).T
        hvg_gene_list = pd.read_csv(f'{os.getcwd()}/data_fromPsupertime/{dataset}_gene_list.csv', index_col=0)
        data_x_df = data_x_df[hvg_gene_list["gene_name"]]
        data_y_df = pd.read_csv('data_fromPsupertime/humanGermline_Y.csv', index_col=0)
        data_y_df = data_y_df["time"]
        preprocessing_params = {"select_genes": "all", "log": True}
    # ------------ for Acinar dataset, in acinar data set total 8 donors with 8 ages:
    elif dataset == "acinarHVG":
        data_x_df = pd.read_csv('data_fromPsupertime/acinar_hvg_sce_X.csv', index_col=0).T
        data_y_df = pd.read_csv('data_fromPsupertime/acinar_hvg_sce_Y.csv')
        data_y_df = np.array(data_y_df['x'])
        preprocessing_params = {"select_genes": "all", "log": False}
    import anndata
    from pypsupertime import Psupertime
    # START HERE
    adata_org = anndata.AnnData(data_x_df)
    adata_org.obs["time"] = data_y_df
    print(f"Input Data: n_genes={adata_org.n_vars}, n_cells={adata_org.n_obs}")
    # ------------------preprocess adata here
    tp = Psupertime(n_jobs=5, n_folds=5,
                    preprocessing_params=preprocessing_params
                    )  # if for Acinar cell "select_genes": "all"

    adata = tp.preprocessing.fit_transform(adata_org.copy())
    del tp
    return adata,data_x_df,data_y_df
def add_norCol_df(dfa: pd.DataFrame):
    df = dfa.copy()
    df["normalized"] = (df["pseudotime"] - df["pseudotime"].min()) / (
            df["pseudotime"].max() - df["pseudotime"].min())
    return df


def plot_boxAndDot_on_allData(pca_df, randomForest_df, lr_df, psupertime_df, vae_df, dataset, dataset_dic):
    # 初始化相关系数列表
    # 合并四个DataFrame以便使用FacetGrid
    # plt.figure(figsize=(10, 8))
    pca_df = add_norCol_df(pca_df)
    randomForest_df = add_norCol_df(randomForest_df)
    lr_df = add_norCol_df(lr_df)
    psupertime_df = add_norCol_df(psupertime_df)
    vae_df = add_norCol_df(vae_df)
    df_concat = pd.concat([pca_df, randomForest_df, lr_df, psupertime_df, vae_df], keys=['PCA', 'RF', 'LR', 'Psupertime', 'TemporalVAE'])
    label_num = len(np.unique(pca_df["time"]))
    # 设置Seaborn的样式

    sns.set(style="whitegrid")

    # 使用Seaborn的FacetGrid创建子图
    g = sns.FacetGrid(df_concat.reset_index(level=0), col='level_0', col_wrap=1, sharex=True, sharey=True,
                      aspect=label_num * 0.4, height=3)
    g.map_dataframe(sns.boxplot, x='time', y='normalized', palette="Set3")
    g.map_dataframe(sns.stripplot, x='time', y='normalized', jitter=True, alpha=0.9, palette="Dark2", s=3)
    g.set_axis_labels('Time', 'Pseudotime')
    g.set_titles(col_template='{col_name}')
    g.add_legend()

    # 调整子图布局

    plt.suptitle(f'{dataset_dic[dataset]}: normalized pseudo-time on each donor', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 设置字体大小
    plt.rc('axes', titlesize=14)
    plt.rc('axes', labelsize=12)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    plt.savefig(f"{os.getcwd()}/{dataset}_methods_boxAndDot_results.pdf")
    plt.savefig(f"{os.getcwd()}/{dataset}_methods_boxAndDot_results.png", dpi=200)

    # 显示图形
    plt.show()
    plt.close()


def plot_kFold_corr(pca_df, randomForest_df, lr_df, psupertime_df, vae_df, dataset, all_labels, dataset_dic):
    plt.close()
    # 初始化相关系数列表
    pca_spearman_correlations = []
    randomForest_spearman_correlations = []
    lr_spearman_correlations = []
    psupertime_spearman_correlations = []
    vae_spearman_correlations = []

    # pca_kendalltau_correlations = []
    # randomForest_kendalltau_correlations = []
    # lr_kendalltau_correlations = []
    # psupertime_kendalltau_correlations = []
    # vae_kendalltau_correlations = []

    # 循环计算四个DataFrame的相关系数
    for label_to_remove in all_labels:
        # spearmanr
        pca_spearman_correlations.append(calculate_corr(pca_df, label_to_remove, corr_method="spearmanr"))
        randomForest_spearman_correlations.append(calculate_corr(randomForest_df, label_to_remove, corr_method="spearmanr"))
        lr_spearman_correlations.append(calculate_corr(lr_df, label_to_remove, corr_method="spearmanr"))
        psupertime_spearman_correlations.append(calculate_corr(psupertime_df, label_to_remove, corr_method="spearmanr"))
        vae_spearman_correlations.append(calculate_corr(vae_df, label_to_remove, corr_method="spearmanr"))
        # kendalltau
        # pca_kendalltau_correlations.append(calculate_corr(pca_df, label_to_remove, corr_method="kendalltau"))
        # randomForest_kendalltau_correlations.append(calculate_corr(randomForest_df, label_to_remove, corr_method="kendalltau"))
        # lr_kendalltau_correlations.append(calculate_corr(lr_df, label_to_remove, corr_method="kendalltau"))
        # psupertime_kendalltau_correlations.append(
        #     calculate_corr(psupertime_df, label_to_remove, corr_method="kendalltau"))
        # vae_kendalltau_correlations.append(calculate_corr(vae_df, label_to_remove, corr_method="kendalltau"))

    # 创建标签位置
    x = np.arange(len(all_labels))

    # 设置柱状图宽度
    width = 0.15

    # 创建两个子图，一个显示Spearman相关性，另一个显示Kendall Tau相关性
    fig, ax1 = plt.subplots(1, 1, figsize=(len(all_labels) * 1.2, 5), sharex=False)
    # fig, ax1 = plt.subplot(figsize=(len(all_labels), 4), sharex=False)
    # fig, (ax1, ax2) = plt.subplots(1, 1, figsize=(10, 8), sharex=True)

    # 子图1：Spearman相关性
    ax1.bar(x - 2 * width, pca_spearman_correlations, width, label='PCA', color="#00f5d4")
    ax1.bar(x - 1 * width, randomForest_spearman_correlations, width, label='RF', color="#00bbf9")
    ax1.bar(x, lr_spearman_correlations, width, label='LR', color="#fee440")
    ax1.bar(x + 1 * width, psupertime_spearman_correlations, width, label='Psupertime', color="#f15bb5")
    ax1.bar(x + 2 * width, vae_spearman_correlations, width, label='TemporalVAE', color="#9b5de5")
    ax1.set_ylabel('Spearman Correlation')
    # 移除外围黑框
    # for spine in ax1.spines.values():
    #     spine.set_visible(False)

    ax1.grid(False)  # 完全关闭网格线
    # 或者只关闭垂直方向的网格线
    ax1.grid(True, which='both', axis='y')  # 仅开启水平网格线
    # # 子图2：Kendall Tau相关性
    # ax2.bar(x - 1.5 * width, pca_kendalltau_correlations, width, label='pca', color="#ECEE81")
    # ax2.bar(x - 0.5 * width, randomForest_kendalltau_correlations, width, label='random forest', color="#8DDFCB")
    # ax2.bar(x + 0.5 * width, psupertime_kendalltau_correlations, width, label='psupertime', color="#82A0D8")
    # ax2.bar(x + 1.5 * width, vae_kendalltau_correlations, width, label='temporalVAE', color="#EDB7ED")

    # ax2.set_ylabel('Kendall Tau Correlation')
    ax1.set_xlabel('Remove donor')
    plt.xticks(x, all_labels)
    plt.legend(loc='lower right')
    # 调整子图布局
    plt.suptitle(f'{dataset_dic[dataset]}: Correlation for Each Deleted donor', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{os.getcwd()}/{dataset}_methods_results.pdf")
    plt.savefig(f"{os.getcwd()}/{dataset}_methods_results.png", dpi=200)
    print(f"figure save at {os.getcwd()}")
    # 显示图形
    plt.show()
    plt.close()
def corr(x1, x2, special_str=""):
    from scipy.stats import spearmanr, kendalltau
    sp_correlation, sp_p_value = spearmanr(x1, x2)
    ke_correlation, ke_p_value = kendalltau(x1, x2)

    sp = f"{special_str} spearman correlation score: {sp_correlation}, p-value: {sp_p_value}."
    print(sp)
    ke = f"{special_str} kendalltau correlation score: {ke_correlation},p-value: {ke_p_value}."
    print(ke)

    return sp, ke

if __name__ == '__main__':
    main()
