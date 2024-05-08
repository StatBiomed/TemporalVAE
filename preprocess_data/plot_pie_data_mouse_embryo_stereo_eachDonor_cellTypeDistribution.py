# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：plot_pie_data_mouse_embryo_stereo_eachDonor_cellTypeDistribution.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/8/30 12:22 
"""

import scanpy as sc
import numpy as np
import pandas as pd
import pyreadr
import os
import anndata
from collections import Counter
import matplotlib.pyplot as plt
import math

def main():
    folder_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/mouse_embryo_stereo/mouse_embryo_stereo_detail_split/"

    sc_file_name_list = sorted([os.path.join(folder_path, filename) for filename in os.listdir(folder_path) if
                                filename.endswith('.h5ad')])

    gene_used_in_mouse_embryonic_development_file = \
        "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/mouse_embryonic_development/" \
        "preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/gene_info.csv"
    print("use hvg gene list from {} to filter the mouse_embryo_stereo sc data .h5ad".format(
        gene_used_in_mouse_embryonic_development_file))
    gene_used_in_mouse_embryonic_development_data = pd.read_csv(gene_used_in_mouse_embryonic_development_file,
                                                                index_col=0,
                                                                sep="\t")
    result_dic = dict()
    for _index, _file_name in enumerate(sc_file_name_list):
        print("Start {},{}/{}".format(_file_name, _index, len(sc_file_name_list)))
        try:
            result_dic[_file_name.split("/")[-1].replace(".MOSTA.h5ad", "")] = preprocessData_scData_h5ad_detail_split(_file_name,
                                                                                                                   min_gene_num=50,
                                                                                                                   required_gene_info=gene_used_in_mouse_embryonic_development_data)
        except:
            continue
        # # save result as csv file
        # _path = '{}/counter/'.format(folder_path)
        # if not os.path.exists(_path):
        #     os.makedirs(_path)
    print("plot pie image.")

    # 示例主字典
    # data = {
    #     'Category1': {'A': 30, 'B': 50, 'C': 20},
    #     'Category2': {'X': 40, 'Y': 25, 'Z': 35}
    # }
    # 获取主字典的键和子字典的值
    categories = list(result_dic.keys())
    sub_data_values = list(result_dic.values())

    # 计算需要多少行子图和列
    num_rows = math.ceil(len(sub_data_values) / 4)
    num_cols = min(4, len(sub_data_values))

    # 创建具有多个子图的图表
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 5 * num_rows))

    # 展平子图数组
    flat_axes = np.ravel(axes)
    # 确定标签对应的颜色
    unique_labels = set(label for sub_data in sub_data_values for label in sub_data.keys())
    label_colors = tuple(list(plt.get_cmap("Set1").colors[::-1]) +
                   list(plt.get_cmap("tab10").colors[::-1]) +
                   list(plt.get_cmap("Dark2").colors[::-1]) +
                   list(plt.get_cmap("Set2").colors[::-1]) +
                   list(plt.get_cmap("Set3").colors[::-1]) +
                   list(plt.get_cmap("tab20").colors[::-1]) +
                   list(plt.get_cmap("Accent").colors[::-1]))
    # label_colors = plt.cm.tab20.colors
    # 创建颜色字典，将相同标签映射到相同颜色
    color_dict = {label: color for label, color in zip(unique_labels, label_colors)}

    # 遍历子图，并为每个子图绘制饼图
    for i, (category, sub_data) in enumerate(result_dic.items()):
        ax = flat_axes[i]
        labels = list(sub_data.keys())
        values = list(sub_data.values())
        # 为每个标签指定颜色
        colors = [color_dict[label] for label in labels]
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors,
               textprops={
                   # 'color':'r',#文本颜色
                   'fontsize': 5,  # 文本大小
                   # 'fontfamily':'Microsoft JhengHei',#设置微软雅黑字体
               })
        # ax.pie(sub_data.values(), labels=sub_data.keys(), autopct='%1.1f%%', startangle=140,
        #        textprops={
        #            # 'color':'r',#文本颜色
        #            'fontsize': 5,  # 文本大小
        #            # 'fontfamily':'Microsoft JhengHei',#设置微软雅黑字体
        #        })
        ax.axis('equal')
        ax.set_title(f' {category}: total {sum(sub_data.values())} cells')

    # 调整子图之间的间距
    plt.tight_layout()

    # 显示图表

    plt.savefig("{}/detail_split.png".format(folder_path))
    plt.show()
    plt.close()
    # # 获取主字典的键和子字典的值
    # categories = list(result_dic.keys())
    # sub_data_values = list(result_dic.values())
    # # 计算需要多少行子图
    # num_rows = math.ceil(len(sub_data_values) / 4)
    #
    # # 创建具有多个子图的图表
    # fig, axes = plt.subplots(nrows=num_rows, ncols=4, figsize=(15, 5 * num_rows))
    #
    # # 创建一个具有多个子图的图表
    # # fig, axes = plt.subplots(nrows=1, ncols=len(sub_data_values), figsize=(10, 5))
    # # for i, (ax_row, sub_data, category) in enumerate(zip(axes, sub_data_values, categories)):
    # #     for ax, (label, value) in zip(ax_row, sub_data.items()):
    # #         ax.pie([value], labels=[label], autopct='%1.1f%%', startangle=140,
    # #         textprops={
    # #                # 'color':'r',#文本颜色
    # #                'fontsize': 5,  # 文本大小
    # #                # 'fontfamily':'Microsoft JhengHei',#设置微软雅黑字体
    # #            })
    # #         ax.axis('equal')
    # #         ax.set_title(f'Pie Chart for {category}-{label}')
    # # 遍历子图，并为每个子图绘制饼图
    # for ax, sub_data, category in zip(axes, sub_data_values, categories):
    #     labels = list(sub_data.keys())
    #     values = list(sub_data.values())
    #
    #     ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=140,
    #            textprops={
    #                # 'color':'r',#文本颜色
    #                'fontsize': 5,  # 文本大小
    #                # 'fontfamily':'Microsoft JhengHei',#设置微软雅黑字体
    #            })
    #     ax.axis('equal')
    #     ax.set_title(f' {category}: total {sum(values)} cells')
    #
    # # 调整子图之间的间距
    # plt.tight_layout()
    #
    # # 显示图表
    # plt.show()
    print("Finish all")


def preprocessData_scData_h5ad_detail_split(sc_file_name, required_gene_info=None,
                                            min_gene_num=10,
                                            min_cell_num=10):
    print("check for {}".format(sc_file_name))

    adata = sc.read_h5ad(filename=sc_file_name)
    print("Import data, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    print("Cell number: {}".format(adata.n_obs))
    print("Gene number: {}".format(adata.n_vars))
    print("Annotation information of data includes: {}".format(adata.obs_keys()))  # 胞注釋信息的keys
    print("Cell id first 5: {}".format(adata.obs_names[:5]))  # 返回胞ID 数据类型是object
    print("Gene id first 5: {}".format(adata.var_names.to_list()[:5]))  # 返回基因数据类型是list adata.obs.head()# 査看前5行的数据
    print("Gene id first 5: {}".format(adata.var.index.to_list()[:5]))  # 返回基因数据类型是list adata.obs.head()# 査看前5行的数据

    required_gene_shortName_list = list(set(required_gene_info["gene_short_name"]) & set(adata.var_names))
    # required_gene_ENS_list = list(
    #     required_gene_info[required_gene_info["gene_short_name"].isin(required_gene_shortName_list)].index)
    adata_filted = adata[:, required_gene_shortName_list].copy()
    print("Use gene list from mouse_embryo,\n"
          "Anndata with cell number: {}, gene number: {}".format(adata_filted.n_obs, adata_filted.n_vars))

    _shape = adata_filted.shape
    _new_shape = (0, 0)
    while _new_shape != _shape:  # make sure drop samples and genes
        _shape = adata_filted.shape
        sc.pp.filter_cells(adata_filted, min_genes=min_gene_num)  # drop samples with less than 20 gene expression
        sc.pp.filter_genes(adata_filted, min_cells=min_cell_num)  # drop genes which none expression in 3 samples
        _new_shape = adata_filted.shape
    print("Drop cells with less than {} gene expression, "
          "drop genes which none expression in {} samples".format(min_gene_num, min_cell_num))
    print("After filter, get cell number: {}, gene number: {}".format(adata_filted.n_obs, adata_filted.n_vars))

    return dict(Counter(adata.obs["annotation"]))


if __name__ == '__main__':
    main()
