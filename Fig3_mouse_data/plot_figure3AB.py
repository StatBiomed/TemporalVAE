# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：plot_figure3AB.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/9/6 22:04 
"""
# -*-coding:utf-8 -*-
import os
import sys

if os.getcwd().split("/")[-1] != "TemporalVAE":
    os.chdir("..")
sys.path.append(os.getcwd())

from TemporalVAE.utils import calculate_real_predict_corrlation_score
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
# 2023-11-03 11:38:31
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from collections import Counter
from TemporalVAE.utils import plt_umap_byScanpy
import numpy as np

# global_folder_path = "results/230827_trainOn_mouse_embryonic_development_kFold_testOnYZdata0809/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/supervise_vae_regressionclfdecoder_dim50_timeembryoneg5to5_epoch100_dropDonorno_mouseEmbryonicDevelopment_embryoneg5to5"
global_folder_path = "results/Fig3_TemporalVAE_kfoldOn_mouseAtlas_240901/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch100_minGeneNum100"
print(f"The k-fold test result is from {global_folder_path}")

def main():

    adata = sc.read_h5ad(f"{global_folder_path}/latent_mu.h5ad")
    plt_latentSpace_mouseAtlas(adata, global_folder_path,random_seed=0)

    kfold_result_df = pd.read_csv(f"{global_folder_path}/k_fold_test_result.csv", index_col=0)
    plot_kfold_mouseAtlas_fromCSV(kfold_result_df, save_as=f"{global_folder_path}/k_fold_result_violin.png")

def plot_kfold_mouseAtlas_fromCSV(boxPlot_df, x_axis_attr="time", y_axis_attr="predicted_time", save_as=""):
    if (x_axis_attr not in boxPlot_df.columns) or (y_axis_attr not in boxPlot_df.columns):
        print(f"Error: {x_axis_attr} or {y_axis_attr} not in kfold result csv file. Please check.")
        return
    corr_stats = calculate_real_predict_corrlation_score(boxPlot_df[y_axis_attr], boxPlot_df[x_axis_attr])
    print(f"=== data correlation: \n{corr_stats}")
    #
    time_counts = boxPlot_df.groupby(x_axis_attr)[y_axis_attr].count().reset_index()
    donor_counts = boxPlot_df.groupby(x_axis_attr)["donor"].nunique().reset_index()
    #
    unique_times = list(time_counts[x_axis_attr])
    print(f"time point include: {unique_times}")
    # 使用颜色映射创建一个颜色字典
    # colors = colors_tuple()
    # cmap = colors[:len(unique_times)]
    cmap = plt.get_cmap('turbo')(np.linspace(0, 1, len(unique_times)))
    color_dict = dict(zip(unique_times, cmap))
    color_dict = {str(key): value for key, value in color_dict.items()}

    # 根据 "time" 值从颜色字典中获取颜色
    boxPlot_df["color"] = boxPlot_df[x_axis_attr].map(color_dict)

    # fig, ax = plt.subplots(1, 1, figsize = (20, 7))
    plt.figure(figsize=(23, 11))  # 设置宽度为10，高度为6
    sns.set_theme(style="whitegrid")
    sns.violinplot(x=x_axis_attr, y=y_axis_attr, data=boxPlot_df, palette=color_dict, bw=0.2, scale='width', saturation=1)
    plt.title("K-fold test of mouse atlas data.", fontsize=18)
    plt.xlabel("Embryo stage", fontsize=16)
    plt.ylabel("Predicted biological time", fontsize=16)
    # 添加颜色的图例
    marksize_dic = {time: np.log(time_counts[time_counts[x_axis_attr] == time][y_axis_attr].values[0]) for time in unique_times}

    _temp = {time: donor_counts[donor_counts[x_axis_attr] == time]['donor'].values[0] for time in unique_times}
    _temp_cmap = plt.get_cmap('RdPu')(np.linspace(0, 1, 1 + len(np.unique(list(_temp.values())))))
    _temp_makerfacecolor = {1: _temp_cmap[1], 2: _temp_cmap[2], 3: _temp_cmap[3], 4: _temp_cmap[4], 5: _temp_cmap[5], 12: _temp_cmap[6]}
    makerfacecolor_dic = {time: _temp_makerfacecolor[_temp[time]] for time in unique_times}

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  label=f"{time_counts[time_counts[x_axis_attr] == time][y_axis_attr].values[0]} from {donor_counts[donor_counts[x_axis_attr] == time]['donor'].values[0]} embryo",
                                  markerfacecolor=makerfacecolor_dic[time], markersize=marksize_dic[time]) for time in unique_times]
    #  2024-09-08 16:25:59 add corr
    plt.text(0.9, 0.08, 'Correlation between x and y\nSpearman: 0.890\nPearson: 0.914\nKendall’s τ: 0.729',
             verticalalignment='center', horizontalalignment='center',
             transform=plt.gca().transAxes,  # This makes the coordinates relative to the axes
             color='black', fontsize=18,
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.5'))


    plt.legend(handles=legend_elements, title="Cell and embryo Num", bbox_to_anchor=(1.01, -0.10), loc="lower left", borderaxespad=0, prop={'size': 10})
    plt.xticks(size=16, rotation=90)
    plt.yticks(size=16)
    plt.tight_layout()
    try:
        plt.savefig(f"{save_as}", dpi=350)
        print(f"Finish save images as: {save_as}")
    except:
        pass

    plt.show()
    plt.close()

    return


def cal_pc1pc2(adata, attr, save_path, grey_indexes=None, ncol=1, custom_palette_bool=False):
    X_standardized = StandardScaler().fit_transform(adata.X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_standardized)
    data_df_all = pd.DataFrame(data=X_pca, index=adata.obs_names, columns=["pc1", "pc2"])
    data_df_all[attr] = adata.obs[attr].values
    data_df_all.to_csv(f"{save_path}/latentSpace_pc1pc2_{attr}.csv")
    if grey_indexes is not None:
        # 将索引列表转换为布尔数组
        grey_mask = data_df_all.index.isin(grey_indexes)
        # 绘制灰色点
        # df_grey = data_df_all[grey_mask].copy(deep=True)
        # sns.scatterplot(x="pc1", y="pc2", data=df_grey, s=3, alpha=0.4,color="grey")
        # plt.scatter(df.loc[grey_mask, "pc1"], df.loc[grey_mask, "pc2"], color="grey", s=3, alpha=0.5)
        # 绘制其他点
        df_normal = data_df_all[~grey_mask]
        df_normal_copy = pd.DataFrame(data=df_normal.values, columns=["pc1", "pc2", attr])
        # if custom_palette_bool:
        #     custom_palette = colors_tuple_hexadecimalColorCode()[:len(np.unique(df_normal[attr]))]
        #     plot_pc1pc2(df_normal_copy, attr, save_path, pca, ncol=ncol, custom_palette=custom_palette)
        # else:
        # df_normal_copy = df_normal_copy.sort_values(by=attr)
        plot_pc1pc2(df_normal_copy, attr, save_path, pca, ncol=ncol)
    else:
        plot_pc1pc2(data_df_all.copy(deep=True), attr, save_path, pca, ncol=ncol)


def plot_pc1pc2(data_df, attr, save_path, pca, ncol=1, custom_palette=None):
    plt.close()
    plt.figure(figsize=(8, 8))
    if custom_palette is not None:
        sns.scatterplot(x="pc1", y="pc2", data=data_df, hue=attr, s=5, alpha=0., palette=list(custom_palette))
    else:
        sns.scatterplot(x="pc1", y="pc2", data=data_df, hue=attr, s=5, alpha=0.7, palette="turbo")
    # 添加图例和标签
    plt.xlabel(f'pc1: {round(pca.explained_variance_ratio_[0], 3) * 100}%')
    plt.ylabel(f'pc2: {round(pca.explained_variance_ratio_[1], 3) * 100}%')
    plt.title(f'Scatter Plot with {attr.capitalize().split("_")[0]} Labels.', y=-0.12)
    plt.legend(ncol=ncol, title=attr.capitalize().split("_")[0], loc="upper left", markerscale=0.8, prop={'size': 7})
    # plt.legend(ncol=ncol, title=attr.capitalize(), loc="lower right", bbox_to_anchor=(1.0, 0.1),
    #            bbox_transform=plt.gcf().transFigure, markerscale=0.8, prop={'size': 5})
    # save
    plt.savefig(f"{save_path}/latentSpace_pc1pc2_{attr}.png", format="png", transparent=True, dpi=400)
    plt.show()
    plt.close()


def plt_latentSpace_mouseAtlas(adata, save_path,random_seed=123):
    # adata_atlas = adata[adata.obs["batch"] != -1]
    # ------- plot for mouse atlas, but too many cell types and cells, so select 1/10 cells and top 10 cell types.----------
    random.seed(random_seed)
    random_indices = random.sample(range(adata.shape[0]), int(adata.n_obs / 5), )
    adata_subCells = adata[random_indices, :].copy()
    celltype_dic = Counter(adata_subCells.obs["celltype_update"])
    top_10_celltype = sorted(celltype_dic, key=celltype_dic.get, reverse=True)[:10]

    adata_subCells_top10 = adata_subCells[adata_subCells.obs.loc[adata_subCells.obs["celltype_update"].isin(list(top_10_celltype))].index].copy()
    # cal_pc1pc2(adata_subCells2.copy(), "day", save_path, custom_palette_bool=True, ncol=3)  # ["day", "embryo_id","experimental_batch","batch"]
    # cal_pc1pc2(adata_subCells2.copy(), "celltype_update", save_path, custom_palette_bool=True)  # ["day", "embryo_id","experimental_batch","batch"]
    plt_umap_byScanpy(adata_subCells_top10.copy(),
                      ["time", "celltype_update"],
                      save_path,
                      # mode="read",
                      figure_size=(7, 6),
                      special_file_name_str="top10CellType_subcell_",
                      color_map="turbo",
                      n_neighbors=50,n_pcs=40,)

    return


if __name__ == '__main__':
    main()
