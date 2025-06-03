# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：plotSupplementary_ablation.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2025-04-15 13:51:06

ablation exps for Temporal and Ablated-TemporalVAE (VAE get low-dim data representation and then use LR to predict pseudo-time)
"""
import os
import time

import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau
import random
import numpy as np

import pandas as pd
import seaborn as sns

from model_master.experiment_temporalVAE import temporalVAEExperiment
from utils.utils_project import denormalize


def main():
    # method_list = ["pca", "randomForest", "psupertime", "vae"]
    dataset_list = ["acinarHVG", "acinarHVG", "embryoBeta", "humanGermline"]
    dataset_dic = {"acinarHVG": "Acinar", "humanGermline": "Human female germ line", "embryoBeta": "Embryo beta"}
    # dataset_list = ["acinarHVG", "humanGermline", "embryoBeta"]

    for dataset in dataset_list:
        temporalVAE_df = get_method_result("temporalVAE", dataset)
        vaeLR_df = get_method_result("vae", dataset)
        all_labels = ['No removal'] + list(np.sort(np.unique(vaeLR_df["time"])))
        plot_kFold_corr(temporalVAE_df.copy(),
                        vaeLR_df.copy(),
                        dataset, all_labels, dataset_dic)
        plot_boxAndDot_on_allData(temporalVAE_df.copy(),
                        vaeLR_df.copy(),
                                  dataset, dataset_dic)

        # 2024-08-07 11:09:40 add
        multi_corr_df = pd.DataFrame(columns=["method", "Spearman", "Pearson", "kendalltau"])
        # multi_corr_df = pd.DataFrame(columns=["method", "Spearman", "Pearson", "kendalltau","EMD", "MMD", "R-squared"])

        temporalVAE_df = add_norCol_df(temporalVAE_df)
        temporalVAE_df['predicted_time'] = temporalVAE_df['pseudotime'].apply(denormalize, args=(min(temporalVAE_df["time"]),
                                                                                 max(temporalVAE_df["time"]),
                                                                                 min(temporalVAE_df["trans_label"]),
                                                                                 max(temporalVAE_df["trans_label"])))
        multi_corr_df.loc[len(multi_corr_df.index)] = (["TemporalVAE"]
                                                       + list(corr(temporalVAE_df['time'], temporalVAE_df['predicted_time'], as_str=True))
                                                       # + list(distribution_metric(temporalVAE_df['time'], temporalVAE_df['predicted_time']))
                                                       )
        vaeLR_df = add_norCol_df(vaeLR_df)
        vaeLR_df['predicted_time'] = vaeLR_df['pseudotime'].apply(denormalize, args=(min(vaeLR_df["time"]),
                                                                                 max(vaeLR_df["time"]),
                                                                                 min(vaeLR_df["trans_label"]),
                                                                                 max(vaeLR_df["trans_label"])))
        multi_corr_df.loc[len(multi_corr_df.index)] = (["Ablated-TemporalVAE"]
                                                       + list(corr(vaeLR_df['time'], vaeLR_df['predicted_time'], as_str=True))
                                                       # + list(distribution_metric(temporalVAE_df['time'], temporalVAE_df['predicted_time']))
                                                       )
        print(f"*** {dataset}")
        print(multi_corr_df)
        multi_corr_df.set_index("method", inplace=True)
        df = multi_corr_df.applymap(lambda x: f"&{x}")
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(df.to_string(index=True, header=True))
            print('\\\\')
    return


def corr_withRemoveDonor(df, label_to_remove, corr_method,):
    if label_to_remove != "No removal":
        # df["time"] = df["time"].astype(int)  # take care here as psupertime and vae can have continues time as input
        df_filtered = df[df['time'] != label_to_remove]
    else:
        df_filtered = df.copy()
    if corr_method == "spearmanr":
        try:
            corr, _ = spearmanr(df_filtered['time'], df_filtered['pseudotime'])
        except:
            print("error?")
    elif corr_method == "kendalltau":
        corr, _ = kendalltau(df_filtered['time'], df_filtered['pseudotime'])
    return abs(corr)


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


def plot_boxAndDot_on_allData(temporalVAE_df,vae_df,
                              dataset, dataset_dic):
    # 初始化相关系数列表
    # 合并四个DataFrame以便使用FacetGrid
    # plt.figure(figsize=(10, 8))
    vae_df = add_norCol_df(vae_df)
    temporalVAE_df = add_norCol_df(temporalVAE_df)

    #  2024-08-06 14:09:35 add
    df_concat = pd.concat([vae_df, temporalVAE_df], keys=['Ablated-TemporalVAE', 'TemporalVAE'])
    label_num = len(np.unique(temporalVAE_df["time"]))
    # 设置Seaborn的样式

    sns.set(style="whitegrid")

    # 使用Seaborn的FacetGrid创建子图
    g = sns.FacetGrid(df_concat.reset_index(level=0), col='level_0', col_wrap=1, sharex=True, sharey=True,
                      aspect=label_num * 0.4, height=3)
    g.map_dataframe(sns.boxplot, x='time', y='normalized', palette="Set3")
    g.map_dataframe(sns.stripplot, x='time', y='normalized', jitter=True, alpha=0.9, palette="Dark2", s=3)
    g.set_axis_labels('Time', 'Normalized pseudo-time')
    g.set_titles(col_template='{col_name}')
    g.add_legend()

    # 调整子图布局

    plt.suptitle(f'{dataset_dic[dataset]}: normalized pseudo-time on each donor', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 设置字体大小
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    plt.savefig(f"{os.getcwd()}/{dataset}_temporalVAEvsVAE_boxAndDot_results.png", dpi=200)

    # 显示图形
    plt.show()
    plt.close()


def plot_kFold_corr(temporalVAE_df,vae_df,
                    dataset, all_labels, dataset_dic,
                    corr_matric="spearmanr"):
    plt.close()
    # init
    temporalVAE_spearman_correlations = []
    vae_spearman_correlations = []



    # 循环计算四个DataFrame的相关系数
    for label_to_remove in all_labels:
        vae_spearman_correlations.append(corr_withRemoveDonor(vae_df, label_to_remove, corr_method=corr_matric))
        temporalVAE_spearman_correlations.append(corr_withRemoveDonor(temporalVAE_df, label_to_remove, corr_method=corr_matric))

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
    ax1.bar(x-width, vae_spearman_correlations, width, label='Ablated-TemporalVAE', color="#7cbf2a")
    ax1.bar(x, temporalVAE_spearman_correlations, width, label='TemporalVAE', color="#9b5de5")

    corr_matric_dic = {"spearmanr": "Spearman"}
    ax1.set_ylabel(f'{corr_matric_dic[corr_matric]} correlation')
    # 移除外围黑框
    # for spine in ax1.spines.values():
    #     spine.set_visible(False)

    ax1.grid(False)  # 完全关闭网格线
    # 或者只关闭垂直方向的网格线
    ax1.grid(True, which='both', axis='y')  # 仅开启水平网格线

    ax1.set_xlabel('Remove donor')
    plt.xticks(x, all_labels)
    plt.legend(loc='lower right', fontsize=16)
    plt.rc('axes', titlesize=16)
    plt.rc('axes', labelsize=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)
    # 调整子图布局
    plt.suptitle(f'{dataset_dic[dataset]}: Correlation for Each Deleted donor', fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f"{os.getcwd()}/{dataset}_temporalVAEvsVAE_results.png", dpi=200)
    print(f"figure save at {os.getcwd()}")
    # 显示图形
    plt.show()
    plt.close()


def corr(x1, x2, special_str="", as_str=False):
    from scipy.stats import spearmanr, kendalltau, pearsonr
    sp_correlation, sp_p_value = spearmanr(x1, x2)
    ke_correlation, ke_p_value = kendalltau(x1, x2)
    pe_correlation, pe_p_value = pearsonr(x1, x2)
    if as_str:
        # return sp_correlation if sp_p_value < 0.05 else 0, ke_correlation if ke_p_value < 0.05 else 0
        return (f"{np.round(sp_correlation, 3)}; \\textit{{P}}={np.round(sp_p_value, 3)}",
                f"{np.round(pe_correlation, 3)}; \\textit{{P}}={np.round(pe_p_value, 3)}",
                f"{np.round(ke_correlation, 3)}; \\textit{{P}}={np.round(ke_p_value, 3)}")

        # return str(np.round(sp_correlation,3) )+";\textit{P}="if sp_p_value < 0.05 else 0, ke_correlation if ke_p_value < 0.05 else 0
    sp = f"{special_str} spearman correlation score: {sp_correlation}, p-value: {sp_p_value}."
    print(sp)
    pe = f"{special_str} pearson correlation score: {pe_correlation}, p-value: {pe_p_value}."
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
