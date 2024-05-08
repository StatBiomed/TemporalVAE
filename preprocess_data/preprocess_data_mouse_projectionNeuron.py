# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE
@File    ：preprocess_data_mouse_projectionNeuron.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/1/2 10:45
"""
import sys

sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan")

import scanpy as sc
import pandas as pd
import os
from utils.utils_DandanProject import calHVG_adata as calHVG
from utils.utils_Dandan_plot import draw_venn


def main():
    file_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/240102mouse_projectionNeuron/"
    cell_info = f"{file_path}/GSE161690_cell_info.csv"
    cell_info = pd.read_csv(cell_info)

    gene_info = f"{file_path}/GSE161690_gene_names.csv"
    gene_info = pd.read_csv(gene_info)

    data_raw_count = f"{file_path}/GSE161690_count.mtx"
    data_raw_count = sc.read(data_raw_count)

    gene_used_in_mouse_embryonic_development_file = \
        "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/mouse_embryonic_development/" \
        "preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/gene_info.csv"
    print("use hvg gene list from {} to filter the mouse_embryo_stereo sc data .h5ad".format(
        gene_used_in_mouse_embryonic_development_file))
    gene_used_in_mouse_embryonic_development_data = pd.read_csv(gene_used_in_mouse_embryonic_development_file,
                                                                index_col=0,
                                                                sep="\t")

    print(f"Start {file_path}/{sc_file_name}")
    sc_data_df, cell_info, gene_info, adata_hvg = preprocessData_Yimin_scData_h5ad(file_path, sc_file_name,
                                                                                   required_gene_info=gene_used_in_mouse_embryonic_development_data,
                                                                                   min_gene_num=50, min_cell_num=50
                                                                                   )
    # save result as csv file
    _path = f'{file_path}/preprocess_231205YimingData_3hvgmethod/{sc_file_name.replace("/", "").replace(".h5ad", "").replace(" ", "_")}/'
    if not os.path.exists(_path):
        os.makedirs(_path)
        print(f"makdir {_path}")
    from collections import Counter
    print("info of cell type info:{}".format(Counter(cell_info["celltype"])))
    print("info of time info:{}".format(Counter(cell_info["time"])))
    sc_data_df = pd.DataFrame(data=sc_data_df.values.T, index=sc_data_df.columns, columns=sc_data_df.index)
    print("the original sc expression anndata should be gene as row, cell as column")
    sc_data_df.to_csv("{}/data_count_hvg.csv".format(_path), sep="\t")
    cell_info.to_csv("{}/cell_with_time.csv".format(_path), sep="\t")
    gene_info.to_csv("{}/gene_info.csv".format(_path), sep="\t")
    adata_hvg.write_h5ad(f"{_path}/adata_hvg.h5ad")
    print(f"Finish {file_path}/{sc_file_name}, save at {_path}")

    print("Finish all")


def preprocessData_Yimin_scData_h5ad(file_path, sc_file_name, required_gene_info=None,
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
    print("raw count in adata.raw.X, so set adata.X=adata.raw.X")
    adata.X = adata.raw.X
    import matplotlib.pyplot as plt
    with plt.rc_context({'figure.figsize': (20, 6)}):
        sc.pl.umap(adata, color=['celltype', "day"])

    required_gene_info["gene_short_name"] = required_gene_info["gene_short_name"].str.upper()
    adata.var_names = adata.var_names.str.upper()
    draw_venn({"atlas hvg": required_gene_info["gene_short_name"].values, "adata all gene": adata.var_names})
    overlap_gene_atlas_adata = list(set(required_gene_info["gene_short_name"]) & set(adata.var_names))
    # cal hvg gene list
    hvg_cellRanger_list = calHVG(adata.copy(), gene_num=1000, method="cell_ranger")
    hvg_seurat_list = calHVG(adata.copy(), gene_num=1000, method="seurat")
    hvg_seurat_v3_list = calHVG(adata.copy(), gene_num=1000, method="seurat_v3")
    draw_venn({"cell ranger": hvg_cellRanger_list, "seurat": hvg_seurat_list, "seurat v3": hvg_seurat_v3_list})
    draw_venn({"cell ranger": hvg_cellRanger_list, "atlas hvg": required_gene_info["gene_short_name"].values})
    draw_venn({"seurat": hvg_seurat_list, "atlas hvg": required_gene_info["gene_short_name"].values})
    draw_venn({"seurat v3": hvg_seurat_v3_list, "atlas hvg": required_gene_info["gene_short_name"].values})

    print(f"concat all hvg calculated")
    import itertools
    combined_hvg_list = list(set(itertools.chain(hvg_cellRanger_list, hvg_seurat_list, hvg_seurat_v3_list)))
    # combined_hvg_list = list(set(itertools.chain(hvg_cellRanger_list, hvg_seurat_list, hvg_seurat_v3_list, overlap_gene_atlas_adata)))
    adata_hvg = adata[:, combined_hvg_list].copy()
    print("Filter the sc count data with hvg gene, "
          "get cell number: {}, gene number: {}".format(adata_hvg.n_obs, adata_hvg.n_vars))

    _shape = adata_hvg.shape
    _new_shape = (0, 0)
    while _new_shape != _shape:  # make sure drop samples and genes
        _shape = adata_hvg.shape
        sc.pp.filter_cells(adata_hvg, min_genes=min_gene_num)  # drop samples with less than 20 gene expression
        sc.pp.filter_genes(adata_hvg, min_cells=min_cell_num)  # drop genes which none expression in 3 samples
        _new_shape = adata_hvg.shape
    print("Drop cells with less than {} gene expression, "
          "drop genes which none expression in {} samples".format(min_gene_num, min_cell_num))
    print("After filter, get cell number: {}, gene number: {}".format(adata_hvg.n_obs, adata_hvg.n_vars))

    sc_expression_df = pd.DataFrame(data=adata_hvg.X.toarray(), columns=adata_hvg.var.index,
                                    index=adata_hvg.obs.index)

    cell_info = pd.DataFrame(adata.obs)
    cell_info["donor"] = cell_info["sample"]
    cell_info["time"] = cell_info["day_numerical"]
    cell_info["cell_id"] = cell_info.index
    print("Get expression dataframe with shape (cell, gene): {}, and cell time info with shape: {}.".format(
        sc_expression_df.shape, cell_info.shape))
    gene_info = list(adata_hvg.var_names)
    gene_info = pd.DataFrame(data=gene_info)
    return sc_expression_df, cell_info, gene_info, adata_hvg


if __name__ == '__main__':
    main()
