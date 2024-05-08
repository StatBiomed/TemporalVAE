# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：preprocess_humanEmbryo_okubo2023.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/3/27 12:25 
"""
import os
from typing import Tuple, Union
import tempfile
from collections import Counter
import urllib.request as request
from contextlib import closing
import gzip
from utils.utils_DandanProject import calHVG_adata as calHVG
from utils.utils_DandanProject import series_matrix2csv
from utils.utils_Dandan_plot import draw_venn
import click
import pandas as pd
import anndata as ad
import scanpy as sc
def main():
    file_path="/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/240322Human_embryo/okubo2023/"
    raw_fpkm_pd = pd.read_csv(f"{file_path}/GSE131747_Normalized_RPKM_matrix_20230518.txt", sep="\t", header=0, index_col=0)
    raw_fpkm_pd.index=raw_fpkm_pd["gene_short_name"]
    raw_fpkm_pd = raw_fpkm_pd[~raw_fpkm_pd.index.duplicated(keep='first')]
    adata = sc.AnnData(raw_fpkm_pd.iloc[:, 2:])  # 或者根据你的实际列号调整
    adata.obs["gene_short_name"]=raw_fpkm_pd["gene_short_name"]
    adata.obs["locus"]=raw_fpkm_pd["locus"]

    cell_info1 = f"{file_path}/GSE131747-GPL18573_series_matrix.txt.gz"
    cell_info1 = series_matrix2csv(cell_info1)

    # cell_info2 = f"{file_path}/GSE131747-GPL24676_series_matrix.txt.gz"
    # cell_info2 = series_matrix2csv(cell_info2)
    #
    # cell_info3 = f"{file_path}/GSE131747-GPL30173_series_matrix.txt"
    # cell_info3 = series_matrix2csv(cell_info3)
    #
    # cell_info = pd.concat([cell_info1[1], cell_info2[1],cell_info3[1]], axis=0)
    cell_info=cell_info1[1]
    cell_info['cell_id'] = adata.var_names
    cell_info.index=adata.var_names
    cell_info["time"]=6
    cell_info["day"] ="day6"

    adata.var=cell_info
    adata=adata.transpose()
    print("Import data, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    print("Cell number: {}".format(adata.n_obs))
    print("Gene number: {}".format(adata.n_vars))
    print("Annotation information of data includes: {}".format(adata.obs_keys()))  # 胞注釋信息的keys
    print("Cell id first 5: {}".format(adata.obs_names[:5]))  # 返回胞ID 数据类型是object
    print("Gene id first 5: {}".format(adata.var_names.to_list()[:5]))  # 返回基因数据类型是list adata.obs.head()# 査看前5行的数据
    print("Gene id first 5: {}".format(adata.var.index.to_list()[:5]))

    min_gene_num = 50
    min_cell_num = 10
    sc.pp.filter_cells(adata, min_genes=min_gene_num)  # drop samples with less than 50 gene expression
    sc.pp.filter_genes(adata, min_cells=min_cell_num)

    hvg_cellRanger_list = calHVG(adata.copy(), gene_num=1000, method="cell_ranger")
    hvg_seurat_list = calHVG(adata.copy(), gene_num=1000, method="seurat")
    hvg_seurat_v3_list = calHVG(adata.copy(), gene_num=1000, method="seurat_v3")
    draw_venn({"cell ranger": hvg_cellRanger_list, "seurat": hvg_seurat_list, "seurat v3": hvg_seurat_v3_list})
    print(f"concat all hvg calculated")
    import itertools
    combined_hvg_list = list(set(itertools.chain(hvg_cellRanger_list, hvg_seurat_list, hvg_seurat_v3_list)))
    filtered_adata_hvg = adata[:, combined_hvg_list].copy()

    _shape = filtered_adata_hvg.shape
    _new_shape = (0, 0)
    min_gene_num = 50
    min_cell_num = 10
    while _new_shape != _shape:  # make sure drop samples and genes
        _shape = filtered_adata_hvg.shape
        sc.pp.filter_cells(filtered_adata_hvg, min_genes=min_gene_num)  # drop samples with less than 20 gene expression
        sc.pp.filter_genes(filtered_adata_hvg, min_cells=min_cell_num)  # drop genes which none expression in 3 samples
        _new_shape = filtered_adata_hvg.shape
    print("Drop cells with less than {} gene expression, "
          "drop genes which none expression in {} samples".format(min_gene_num, min_cell_num))
    print("After filter, get cell number: {}, gene number: {}".format(filtered_adata_hvg.n_obs, filtered_adata_hvg.n_vars))
    print("the original sc expression anndata should be gene as row, cell as column")


    filtered_adata_hvg.write_h5ad(f"{file_path}/adata_hvg.h5ad")
    sc_expression_df = pd.DataFrame(data=filtered_adata_hvg.X.T,
                                    columns=filtered_adata_hvg.obs.index,
                                    index=filtered_adata_hvg.var_names)

    sc_expression_df.to_csv(f"{file_path}/data_count_hvg.csv", sep="\t")

    filtered_adata_hvg.obs.to_csv(f"{file_path}/cell_with_time.csv", sep="\t")
    filtered_adata_hvg.var.to_csv(f"{file_path}/gene_info.csv", sep="\t")
    print("After filter, get cell number: {}, gene number: {}".format(filtered_adata_hvg.n_obs, filtered_adata_hvg.n_vars))

    print("Finish all.")
    return




# 定义一个函数，用于从字符串中提取数字
def extract_number(s):
    import re
    match = re.search(r'D(\d+)', s)
    if match:
        return int(match.group(1))
    return None
if __name__ == '__main__':
    main()
