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
from utils.utils_DandanProject import denormalize


def main():
    # method_list = ["pca", "randomForest", "psupertime", "vae"]
    dataset_list = ["acinarHVG", "acinarHVG", "embryoBeta", "humanGermline"]
    dataset_dic = {"acinarHVG": "Acinar", "humanGermline": "Human female germ line", "embryoBeta": "Embryo beta"}
    # dataset_list = ["acinarHVG", "humanGermline", "embryoBeta"]

    for dataset in dataset_list:
        pca_df = get_method_result("pca", dataset)
        rf_df = get_method_result("randomForest", dataset)
        lr_df = get_method_result("LR", dataset)
        psupertime_df = get_method_result("psupertime", dataset)
        vae_df = get_method_result("vae", dataset)
        # add 2024-08-05 23:14:33
        science_df = get_method_result("science2022", dataset)
        seurat_df = get_method_result("seurat", dataset)
        ot_df = get_method_result("ot", dataset)
        # 初始化标签列表
        all_labels = ['No removal'] + list(np.sort(np.unique(pca_df["time"])))
        plot_kFold_corr(pca_df.copy(),
                        rf_df.copy(),
                        lr_df.copy(),
                        psupertime_df.copy(),
                        vae_df.copy(),
                        science_df.copy(),
                        seurat_df.copy(),
                        ot_df.copy(),
                        dataset, all_labels, dataset_dic)
        plot_boxAndDot_on_allData(pca_df.copy(),
                                  rf_df.copy(),
                                  lr_df.copy(),
                                  psupertime_df.copy(),
                                  vae_df.copy(),
                                  science_df.copy(),
                                  seurat_df.copy(),
                                  ot_df.copy(),
                                  dataset, dataset_dic)
        # 2024-08-07 11:09:40 add
        multi_corr_df = pd.DataFrame(columns=["method", "Spearman", "Pearson", "EMD", "MMD", "R-squared"])
        pca_df=add_norCol_df(pca_df)
        multi_corr_df.loc[len(multi_corr_df.index)] = (["PCA"] +
                                                       list(corr(pca_df['time'], pca_df['pseudotime'], only_value=True)) +
                                                       list(distribution_metric(pca_df['time'], pca_df['pseudotime']))
                                                       )
        rf_df=add_norCol_df(rf_df)
        multi_corr_df.loc[len(multi_corr_df.index)] = (["RF"] +
                                                       list(corr(rf_df['time'], rf_df['pseudotime'], only_value=True)) +
                                                       list(distribution_metric(rf_df['time'], rf_df['pseudotime']))
                                                       )
        lr_df=add_norCol_df(lr_df)
        multi_corr_df.loc[len(multi_corr_df.index)] = (["LR"] +
                                                       list(corr(lr_df['time'], lr_df['pseudotime'], only_value=True)) +
                                                       list(distribution_metric(lr_df['time'], lr_df['pseudotime']))
                                                       )
        science_df=add_norCol_df(science_df)
        multi_corr_df.loc[len(multi_corr_df.index)] = (["Science2022"] +
                                                       list(corr(science_df['time'], science_df['pseudotime'], only_value=True)) +
                                                       list(distribution_metric(science_df['time'], science_df['pseudotime']))
                                                       )
        seurat_df=add_norCol_df(seurat_df)
        multi_corr_df.loc[len(multi_corr_df.index)] = (["Seurat"] +
                                                       list(corr(seurat_df['time'], seurat_df['pseudotime'], only_value=True)) +
                                                       list(distribution_metric(seurat_df['time'], seurat_df['pseudotime']))
                                                       )
        ot_df=add_norCol_df(ot_df)
        multi_corr_df.loc[len(multi_corr_df.index)] = (["OT-Regressor"] +
                                                       list(corr(ot_df['time'], ot_df['pseudotime'], only_value=True)) +
                                                       list(distribution_metric(ot_df['time'], ot_df['pseudotime']))
                                                       )
        psupertime_df=add_norCol_df(psupertime_df)
        multi_corr_df.loc[len(multi_corr_df.index)] = (["Psupertime"] +
                                                       list(corr(psupertime_df['time'], psupertime_df['pseudotime'], only_value=True)) +
                                                       list(distribution_metric(psupertime_df['time'], psupertime_df['pseudotime']))
                                                       )
        vae_df=add_norCol_df(vae_df)
        vae_df['predicted_time'] = vae_df['pseudotime'].apply(denormalize, args=(min(vae_df["time"]),
                                                                                 max(vae_df["time"]),
                                                                                 min(vae_df["trans_label"]),
                                                                                 max(vae_df["trans_label"])))
        multi_corr_df.loc[len(multi_corr_df.index)] = (["TemporalVAE"] +
                                                       list(corr(vae_df['time'], vae_df['predicted_time'], only_value=True)) +
                                                       list(distribution_metric(vae_df['time'], vae_df['predicted_time']))
                                                       )
        print(f"*** {dataset}")

        print( multi_corr_df)

    return


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


def get_method_result(method, dataset):  #

    file = f"{os.getcwd()}/{method}_results/{dataset}_{method}_result.csv"
    data = pd.read_csv(file, index_col=0)
    return data


def preprocess_parameters(dataset, additional_path=""):  # "acinarHVG", "embryoBeta", "humanGermline"
    print(f"for dataset {dataset}.")
    # ------------ for Mouse embryonic beta cells dataset and Human Germline dataset:
    if dataset in ["embryoBeta", "humanGermline"]:
        data_x_df = pd.read_csv(f'data_fromPsupertime/{dataset}_X.csv', index_col=0).T
        hvg_gene_list = pd.read_csv(f'data_fromPsupertime/{dataset}_gene_list.csv', index_col=0)
        data_x_df = data_x_df[hvg_gene_list["gene_name"]]
        data_y_df = pd.read_csv(f'data_fromPsupertime/{dataset}_Y.csv', index_col=0)
        data_y_df = data_y_df["time"]
        preprocessing_params = {"select_genes": "all", "log": True}
    # # ------------ for Human Germline dataset:
    # elif dataset == "humanGermline":
    #     data_x_df = pd.read_csv('data_fromPsupertime/humanGermline_X.csv', index_col=0).T
    #     hvg_gene_list = pd.read_csv(f'{os.getcwd()}/data_fromPsupertime/{dataset}_gene_list.csv', index_col=0)
    #     data_x_df = data_x_df[hvg_gene_list["gene_name"]]
    #     data_y_df = pd.read_csv('data_fromPsupertime/humanGermline_Y.csv', index_col=0)
    #     data_y_df = data_y_df["time"]
    #     preprocessing_params = {"select_genes": "all", "log": True}
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
    return adata, data_x_df, data_y_df


def add_norCol_df(dfa: pd.DataFrame):
    df = dfa.copy()
    df["normalized"] = (df["pseudotime"] - df["pseudotime"].min()) / (
            df["pseudotime"].max() - df["pseudotime"].min())
    return df


def plot_boxAndDot_on_allData(pca_df, randomForest_df, lr_df, psupertime_df, vae_df,
                              science_df,
                              seurat_df,
                              ot_df,
                              dataset, dataset_dic):
    # 初始化相关系数列表
    # 合并四个DataFrame以便使用FacetGrid
    # plt.figure(figsize=(10, 8))
    pca_df = add_norCol_df(pca_df)
    randomForest_df = add_norCol_df(randomForest_df)
    lr_df = add_norCol_df(lr_df)
    psupertime_df = add_norCol_df(psupertime_df)
    vae_df = add_norCol_df(vae_df)

    # 2024-08-06 14:07:12 add
    science_df = add_norCol_df(science_df)
    seurat_df = add_norCol_df(seurat_df)
    ot_df = add_norCol_df(ot_df)

    # df_concat = pd.concat([pca_df, randomForest_df, lr_df, psupertime_df, vae_df], keys=['PCA', 'RF', 'LR', 'Psupertime', 'TemporalVAE'])
    #  2024-08-06 14:09:35 add
    df_concat = pd.concat([science_df, seurat_df, ot_df, psupertime_df, vae_df], keys=['Science2022', 'Seurat', "OT-Regressor", 'Psupertime', 'TemporalVAE'])
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

    plt.suptitle(f'{dataset_dic[dataset]}: normalized pseudo-time on each donor', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 设置字体大小
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)

    plt.savefig(f"{os.getcwd()}/{dataset}_methods_boxAndDot_results.pdf")
    plt.savefig(f"{os.getcwd()}/{dataset}_methods_boxAndDot_results.png", dpi=200)

    # 显示图形
    plt.show()
    plt.close()


def plot_kFold_corr(pca_df, randomForest_df, lr_df, psupertime_df, vae_df,
                    science_df,
                    seurat_df,
                    ot_df,
                    dataset, all_labels, dataset_dic,
                    corr_matric="spearmanr"):
    plt.close()
    # init
    pca_spearman_correlations = []
    randomForest_spearman_correlations = []
    lr_spearman_correlations = []
    psupertime_spearman_correlations = []
    vae_spearman_correlations = []

    # 2024-08-06 13:17:02 add
    science_spearman_correlations = []
    seurat_spearman_correlations = []
    ot_spearman_correlations = []

    # 循环计算四个DataFrame的相关系数
    for label_to_remove in all_labels:
        pca_spearman_correlations.append(calculate_corr(pca_df, label_to_remove, corr_method=corr_matric))
        randomForest_spearman_correlations.append(calculate_corr(randomForest_df, label_to_remove, corr_method=corr_matric))
        lr_spearman_correlations.append(calculate_corr(lr_df, label_to_remove, corr_method=corr_matric))
        psupertime_spearman_correlations.append(calculate_corr(psupertime_df, label_to_remove, corr_method=corr_matric))
        vae_spearman_correlations.append(calculate_corr(vae_df, label_to_remove, corr_method=corr_matric))

        # 2024-08-06 13:17:12 add
        science_spearman_correlations.append(calculate_corr(science_df, label_to_remove, corr_method=corr_matric))
        seurat_spearman_correlations.append(calculate_corr(seurat_df, label_to_remove, corr_method=corr_matric))
        ot_spearman_correlations.append(calculate_corr(ot_df, label_to_remove, corr_method=corr_matric))

    # 创建标签位置
    x = np.arange(len(all_labels))

    # 设置柱状图宽度
    width = 0.15

    # 创建两个子图，一个显示Spearman相关性，另一个显示Kendall Tau相关性
    fig, ax1 = plt.subplots(1, 1, figsize=(len(all_labels) * 1.2, 5), sharex=False)
    # fig, ax1 = plt.subplot(figsize=(len(all_labels), 4), sharex=False)
    # fig, (ax1, ax2) = plt.subplots(1, 1, figsize=(10, 8), sharex=True)

    # 子图1：Spearman相关性
    # ax1.bar(x - 2 * width, pca_spearman_correlations, width, label='PCA', color="#00f5d4")
    # ax1.bar(x - 1 * width, randomForest_spearman_correlations, width, label='RF', color="#00bbf9")
    # ax1.bar(x, lr_spearman_correlations, width, label='LR', color="#fee440")
    ax1.bar(x - 2 * width, science_spearman_correlations, width, label='Science2022', color="#00f5d4")
    ax1.bar(x - 1 * width, seurat_spearman_correlations, width, label='Seurat', color="#00bbf9")
    ax1.bar(x, ot_spearman_correlations, width, label='OT-Regressor', color="#fee440")
    ax1.bar(x + 1 * width, psupertime_spearman_correlations, width, label='Psupertime', color="#f15bb5")
    ax1.bar(x + 2 * width, vae_spearman_correlations, width, label='TemporalVAE', color="#9b5de5")

    corr_matric_dic = {"spearmanr": "Spearman"}
    ax1.set_ylabel(f'{corr_matric_dic[corr_matric]} Correlation')
    # 移除外围黑框
    # for spine in ax1.spines.values():
    #     spine.set_visible(False)

    ax1.grid(False)  # 完全关闭网格线
    # 或者只关闭垂直方向的网格线
    ax1.grid(True, which='both', axis='y')  # 仅开启水平网格线

    ax1.set_xlabel('Remove donor')
    plt.xticks(x, all_labels)
    plt.legend(loc='lower right', fontsize=14)
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=14)
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)
    # 调整子图布局
    plt.suptitle(f'{dataset_dic[dataset]}: Correlation for Each Deleted donor', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{os.getcwd()}/{dataset}_methods_results.pdf")
    plt.savefig(f"{os.getcwd()}/{dataset}_methods_results.png", dpi=200)
    print(f"figure save at {os.getcwd()}")
    # 显示图形
    plt.show()
    plt.close()


def corr(x1, x2, special_str="", only_value=False):
    from scipy.stats import spearmanr, kendalltau
    sp_correlation, sp_p_value = spearmanr(x1, x2)
    ke_correlation, ke_p_value = kendalltau(x1, x2)
    if only_value:
        return sp_correlation if sp_p_value < 0.05 else 0, ke_correlation if ke_p_value < 0.05 else 0
    sp = f"{special_str} spearman correlation score: {sp_correlation}, p-value: {sp_p_value}."
    print(sp)
    ke = f"{special_str} kendalltau correlation score: {ke_correlation},p-value: {ke_p_value}."
    print(ke)

    return sp, ke


def distribution_metric(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    from scipy.stats import wasserstein_distance
    emd = wasserstein_distance(y_true, y_pred)

    from sklearn.metrics.pairwise import rbf_kernel

    def compute_mmd(x, y, kernel=rbf_kernel):
        """计算 MMD 距离"""
        xx = kernel(x, x)
        yy = kernel(y, y)
        xy = kernel(x, y)
        return xx.mean() + yy.mean() - 2 * xy.mean()

    mmd = compute_mmd(y_true.reshape(len(y_true), 1), y_pred.reshape(len(y_pred), 1))
    # 计算 R^2
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"EMD is {emd}; MMD is {mmd}; R2 is {r_squared}")
    return emd, mmd, r_squared


if __name__ == '__main__':
    main()
