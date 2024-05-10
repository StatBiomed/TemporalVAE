# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：preprocess_humanEmbryo_CS7_Tyser.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/4/9 23:03 
"""

import os
import sys

if os.getcwd().split("/")[-1] != "TemporalVAE":
    os.chdir("..")
sys.path.append(os.getcwd())
import scanpy as sc
import pandas as pd
import anndata as ad
from collections import Counter
from utils.utils_DandanProject import calHVG_adata as calHVG
from utils.utils_DandanProject import read_rds_file
from utils.utils_DandanProject import series_matrix2csv
from utils.utils_Dandan_plot import draw_venn


def main():
    select_hvg_bool = True  # 2024-04-03 17:51:44 add here
    hvg_num = 1000
    file_path = "data/240322Human_embryo/Tyser2021/"

    exp_count_pd = read_rds_file(f"{file_path}/raw_reads.rds")
    cell_info_pd = read_rds_file(f"{file_path}/annot_umap.rds")
    cell_info_pd.index = cell_info_pd["cell_name"]
    exp_count_pd.index = cell_info_pd["cell_name"]
    adata = ad.AnnData(X=exp_count_pd, obs=cell_info_pd)
    adata.write_h5ad(f"{file_path}/raw_count.h5ad")
    # ------------ print and plot for show the structure of dataset
    print("Import data, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    print("Cell number: {}".format(adata.n_obs))
    print("Gene number: {}".format(adata.n_vars))
    print("Annotation information of data includes: {}".format(adata.obs_keys()))  # 胞注釋信息的keys
    print("Cell id first 5: {}".format(adata.obs_names[:5]))  # 返回胞ID 数据类型是object
    print("Gene id first 5: {}".format(adata.var_names.to_list()[:5]))  # 返回基因数据类型是list adata.obs.head()# 査看前5行的数据

    sc.pl.highest_expr_genes(adata, n_top=20)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    sc.pl.violin(adata,
                 ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
                 jitter=0.4,
                 multi_panel=True, )
    sc.pl.scatter(adata, x="total_counts", y="pct_counts_mt")
    sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts")

    # ------------
    adata.obs["time"] = 15

    if select_hvg_bool:
        sc.pp.filter_genes(adata, min_cells=50)  # drop genes which none expression in 3 samples
        hvg_cellRanger_list = calHVG(adata.copy(), gene_num=hvg_num, method="cell_ranger")
        hvg_seurat_list = calHVG(adata.copy(), gene_num=hvg_num, method="seurat")
        hvg_seurat_v3_list = calHVG(adata.copy(), gene_num=hvg_num, method="seurat_v3")
        draw_venn({"cell ranger": hvg_cellRanger_list, "seurat": hvg_seurat_list, "seurat v3": hvg_seurat_v3_list})

        print(f"concat all hvg calculated")
        import itertools
        combined_hvg_list = list(set(itertools.chain(hvg_cellRanger_list, hvg_seurat_list, hvg_seurat_v3_list)))
        filtered_adata_hvg = adata[:, combined_hvg_list].copy()
    else:
        filtered_adata_hvg = adata.copy()

    _shape = filtered_adata_hvg.shape
    _new_shape = (0, 0)
    min_gene_num = 50
    min_cell_num = 50
    while _new_shape != _shape:  # make sure drop samples and genes
        _shape = filtered_adata_hvg.shape
        sc.pp.filter_cells(filtered_adata_hvg, min_genes=min_gene_num)  # drop samples with less than 20 gene expression
        sc.pp.filter_genes(filtered_adata_hvg, min_cells=min_cell_num)  # drop genes which none expression in min_cell_num cells
        _new_shape = filtered_adata_hvg.shape
    print("Drop cells with less than {} gene expression, "
          "drop genes which none expression in {} samples".format(min_gene_num, min_cell_num))
    print("After filter, get cell number: {}, gene number: {}".format(filtered_adata_hvg.n_obs, filtered_adata_hvg.n_vars))
    print("the original sc expression anndata should be gene as row, cell as column")
    filtered_adata_hvg.write_h5ad(f"{file_path}/raw_count_hvg{hvg_num}.h5ad")

    sc_expression_df = pd.DataFrame(data=filtered_adata_hvg.X.T,
                                    columns=filtered_adata_hvg.obs.index,
                                    index=filtered_adata_hvg.var.index)

    sc_expression_df.to_csv(f"{file_path}/data_count_hvg.csv", sep="\t")

    cell_info = filtered_adata_hvg.obs
    cell_info["day"] = "day15"
    cell_info["time"] = 15
    cell_info["cell_id"] = filtered_adata_hvg.obs.index
    cell_info["dataset_label"] = "Tyser"
    cell_info["cell_type"] = cell_info["cluster_id"]
    cell_info["title"] = cell_info.index
    cell_info = cell_info.loc[sc_expression_df.columns]
    cell_info.to_csv(f"{file_path}/cell_with_time.csv", sep="\t")
    gene_info = filtered_adata_hvg.var
    gene_info.to_csv(f"{file_path}/gene_info.csv", sep="\t")

    print("Finish all")

    return


if __name__ == '__main__':
    main()
