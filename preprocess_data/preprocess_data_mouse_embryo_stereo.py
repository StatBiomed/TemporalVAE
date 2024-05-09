# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：preprocess_data_mouse_embryo_stereo.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/8/24 13:07

*** References: Chen, Ao, et al. "Spatiotemporal transcriptomic atlas of mouse organogenesis using DNA nanoball-patterned arrays." Cell 185.10 (2022): 1777-1792.
    1. directly download from: https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/Mouse_embryo_all_stage.h5ad
    2. use genes from {mouse atlas hvg gene list} to filter the mouse_embryo_stereo sc data .h5ad
    3. filter cells with less than 50 gene expressed.
    4. finally save at "data/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50/": "data_count_hvg.csv", "cell_with_time.csv", "gene_info.csv"
"""

import scanpy as sc
import numpy as np
import pandas as pd
import pyreadr
import os
import anndata


def main():
    file_path = "data/mouse_embryo_stereo/"
    sc_file_name_list = ["/Mouse_embryo_all_stage.h5ad"]
    # mouse_embryonic_development/data_count_hvg.csv use gene name as 'ENSMUSG00000051951'
    gene_used_in_mouse_embryonic_development_file = "data/mouse_embryonic_development/" \
                                                    "preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/gene_info.csv"
    print("use hvg gene list from {} to filter the mouse_embryo_stereo sc data .h5ad".format(
        gene_used_in_mouse_embryonic_development_file))
    gene_used_in_mouse_embryonic_development_data = pd.read_csv(gene_used_in_mouse_embryonic_development_file,
                                                                index_col=0,
                                                                sep="\t")

    for _file_index in range(len(sc_file_name_list)):
        print("Start {}".format(sc_file_name_list[_file_index]))
        sc_data_df, cell_info, gene_info = preprocessData_scData_h5ad(file_path,
                                                                      sc_file_name_list[_file_index],
                                                                      min_gene_num=50,
                                                                      required_gene_info=gene_used_in_mouse_embryonic_development_data)
        # save result as csv file
        _path = '{}/{}_minGene50/'.format(file_path, "preprocess_" + sc_file_name_list[_file_index].replace("/", "").replace(".h5ad", "").replace(" ", "_"))
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
        print("Finish {}".format(sc_file_name_list[_file_index]))
    print("Finish all")


def preprocessData_scData_h5ad(file_path, sc_file_name, required_gene_info=None,
                               min_gene_num=10,
                               min_cell_num=10):
    print("check for {}".format(sc_file_name))

    adata = sc.read_h5ad(filename=file_path + sc_file_name)
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

    gene_info = required_gene_info[required_gene_info["gene_short_name"].isin(adata_filted.var.index)]
    geneShortKey_ensValue_dic = {row["gene_short_name"]: row["gene_id"] for ens, row in gene_info.iterrows()}
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
