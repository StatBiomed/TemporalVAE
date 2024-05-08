# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：preprocess_data_mouse_embryo_stereo_subOrgan.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/8/29 17:22

2023-08-29 17:22:12
preprocess sc data on each organ of organ_list
min gene num is 50
use gene list provided by mouse embryonic development combined data

"""

import scanpy as sc
import numpy as np
import pandas as pd
import pyreadr
import os
import anndata
min_gene_num = 50

def main():
    file_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/mouse_embryo_stereo/"
    sc_file_name = "/Mouse_embryo_all_stage.h5ad"
    # organ_list = list(
    #     {'Spinal cord': 8918, 'Muscle': 8808, 'Liver': 7236, 'Brain': 6499, 'Cavity': 5491, 'Connective tissue': 3838,
    #      'Heart': 2801,
    #      'Mesothelium': 1763, 'Adipose tissue': 1628, 'Dorsal root ganglion': 1485, 'Meninges': 1410, 'Lung': 1409,
    #      'Jaw and tooth': 1397, 'Epidermis': 1250, 'Smooth muscle': 1124, 'Cartilage': 1114, 'Kidney': 605,
    #      'GI tract': 511}.keys())
    organ_list=['Brain','Liver','Heart']
    # mouse_embryonic_development/data_count_hvg.csv use gene name as 'ENSMUSG00000051951'
    gene_used_in_mouse_embryonic_development_file = \
        "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/mouse_embryonic_development/" \
        "preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/gene_info.csv"
    print("use hvg gene list from {} to filter the mouse_embryo_stereo sc data .h5ad".format(
        gene_used_in_mouse_embryonic_development_file))
    gene_used_in_mouse_embryonic_development_data = pd.read_csv(gene_used_in_mouse_embryonic_development_file,
                                                                index_col=0,
                                                                sep="\t")

    for index, _organ in enumerate(organ_list):
        print("--- Start {},{}/{}".format(_organ, index, len(organ_list)))
        sc_data_df, cell_info, gene_info = preprocessData_scData_h5ad_subType(file_path,
                                                                              sc_file_name,
                                                                              min_gene_num=min_gene_num, subOrgan=_organ,
                                                                              required_gene_info=gene_used_in_mouse_embryonic_development_data)
        # save result as csv file
        _path = '{}/preprocess_{}_minGene50_of{}/'. \
            format(file_path, sc_file_name.replace("/", "").replace(".h5ad", "").replace(" ", "_"),
                   _organ.replace(" ", "").replace("/", ""))
        if not os.path.exists(_path):
            os.makedirs(_path)
        from collections import Counter
        print("info of cell type info:{}".format(Counter(cell_info["celltype_update"])))
        print("info of time info:{}".format(Counter(cell_info["time"])))
        sc_data_df = pd.DataFrame(data=sc_data_df.values.T, index=sc_data_df.columns, columns=sc_data_df.index)
        print("the original sc expression anndata should be gene as row, cell as column")
        sc_data_df.to_csv("{}/data_count_hvg.csv".format(_path), sep="\t")
        cell_info.to_csv("{}/cell_with_time.csv".format(_path), sep="\t")
        gene_info.to_csv("{}/gene_info.csv".format(_path), sep="\t")
        print("--- Finish {}, save at {}".format(_organ, _path))
    print("Finish all")


def preprocessData_scData_h5ad_subType(file_path, sc_file_name, required_gene_info=None,
                                       min_gene_num=10,
                                       min_cell_num=10, subOrgan=""):
    print("check for {}".format(sc_file_name))

    adata = sc.read_h5ad(filename=file_path + sc_file_name)
    print("Import data, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    print("Annotation information of data includes: {}".format(adata.obs_keys()))  # 胞注釋信息的keys
    print("Cell id first 5: {}".format(adata.obs_names[:5]))  # 返回胞ID 数据类型是object
    print("Gene id first 5: {}".format(adata.var_names.to_list()[:5]))  # 返回基因数据类型是list adata.obs.head()# 査看前5行的数据

    print("-only save cell of organ {}.".format(subOrgan))
    save_cell = adata.obs["annotation"] == subOrgan
    adata = adata[save_cell].copy()
    print("-After filter, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))

    required_gene_shortName_list = list(set(required_gene_info["gene_short_name"]) & set(adata.var_names))
    # required_gene_ENS_list = list(
    #     required_gene_info[required_gene_info["gene_short_name"].isin(required_gene_shortName_list)].index)
    adata_filted = adata[:, required_gene_shortName_list].copy()
    print("-Use gene list from mouse_embryo,\n"
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

    gene_info = required_gene_info[required_gene_info["gene_short_name"].isin(adata_filted.var.index)]
    geneShortKey_ensValue_dic = {row["gene_short_name"]: row["gene_id"] for ens, row in gene_info.iterrows()}
    # plot each time point, cell expressed gene number distribution
    import pandas as pd
    import matplotlib.pyplot as plt

    # 为了绘制每个时间点n_gene值发生的次数，我们可以使用直方图的形式来表示，但是以线图的形式展示
    # 定义time_points变量并重新绘制图像
    df=adata_filted.obs
    time_points = df['timepoint'].unique()

    plt.figure(figsize=(10, 6))

    # 定义bins
    bins = np.linspace(df['n_genes'].min(), df['n_genes'].max(), 20)

    for time_point in time_points:
        subset = df[df['timepoint'] == time_point]['n_genes']
        # 计算每个bin的计数
        counts, _ = np.histogram(subset, bins)
        # 用线图代替直方图绘图
        plt.plot(bins[:-1], counts, marker='o', label=f'Time Point: {time_point}')

    plt.title('Frequency of n_gene Values Across Time Points')
    plt.xlabel('n_gene')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    ####--------
    sc_expression_df = pd.DataFrame(data=adata_filted.X.toarray(), columns=adata_filted.var.index,
                                    index=adata_filted.obs.index)
    sc_expression_df_ensGene = sc_expression_df.rename(columns=geneShortKey_ensValue_dic)
    cell_info = pd.DataFrame(index=adata.obs_names)
    cell_info["donor"] = cell_info.index.map(lambda idx: "embryo_stereo_" + str(idx).split('-')[-1])
    cell_info["celltype_update"] = adata.obs["annotation"]
    cell_info["day"] = adata.obs["timepoint"]
    cell_info["time"] = cell_info['day'].str.replace(r'[A-Za-z]', '', regex=True)
    cell_info["cell_id"] = adata.obs_names
    cell_info = cell_info.loc[sc_expression_df.index]
    print("Get expression dataframe with shape (cell, gene): {}, and cell time info with shape: {}.".format(
        sc_expression_df.shape, cell_info.shape))

    return sc_expression_df_ensGene, cell_info, gene_info


if __name__ == '__main__':
    main()
