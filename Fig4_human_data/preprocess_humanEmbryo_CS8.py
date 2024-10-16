# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：preprocess_humanEmbryo_CS8.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/8/5 23:10

data from Xiao, Zhenyu, et al. "3D reconstruction of a gastrulating human embryo." Cell 187.11 (2024): 2855-2874.
count matrix download from https://cs8.3dembryo.com/#/download
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
from utils.utils_Dandan_plot import draw_venn,plot_data_quality


def main():
    select_hvg_bool = False  # 2024-04-03 17:51:44 add here
    hvg_num = 1000
    file_path = "data/240322Human_embryo/xiaoCellCS8/"
    human_hvg_gene_list=pd.read_csv("data/240405_preimplantation_Melania/gene_info.csv",
                                    sep='\t',index_col=0)
    human_hvg_gene_list=list(human_hvg_gene_list.index)

    exp_count_pd = pd.read_csv(f"{file_path}/data_count.csv", header=None)
    exp_count_pd.columns = ['data']

    column_names = exp_count_pd.iloc[0, 0].split('\t')
    column_names = [name.strip('"').strip("'") for name in column_names]
    exp_count_pd = exp_count_pd.iloc[1:]
    exp_count_pd = exp_count_pd['data'].str.split('\t', expand=True)

    row_names = exp_count_pd.iloc[:, 0]
    exp_count_pd = exp_count_pd.iloc[:, 1:]

    exp_count_pd.columns = column_names
    exp_count_pd.index=row_names
    exp_count_pd = exp_count_pd.astype(int)

    cell_info_pd = pd.read_csv(f"{file_path}/cell_info.csv",sep="\t")
    cell_info_pd["cell_name"]=cell_info_pd["cell"]
    cell_info_pd.index = cell_info_pd["cell_name"]
    exp_count_pd=exp_count_pd.T
    adata = ad.AnnData(X=exp_count_pd, obs=cell_info_pd)
    adata.var.index.name=str(adata.var.index.name)
    # adata.write_h5ad(f"{file_path}/raw_count.h5ad")
    # ------------ print and plot for show the structure of dataset
    print("Import data, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    print("Cell number: {}".format(adata.n_obs))
    print("Gene number: {}".format(adata.n_vars))
    print("Annotation information of data includes: {}".format(adata.obs_keys()))  # 胞注釋信息的keys
    print("Cell id first 5: {}".format(adata.obs_names[:5]))  # 返回胞ID 数据类型是object
    print("Gene id first 5: {}".format(adata.var_names.to_list()[:5]))  # 返回基因数据类型是list adata.obs.head()# 査看前5行的数据

    # mitochondrial genes, "MT-" for human, "Mt-" for mouse
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    # ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    # hemoglobin genes
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")



    # ------------
    adata.obs["time"] = 18.5

    if select_hvg_bool:
        plot_data_quality(adata)
        for _var in ["mt", "ribo", "hb"]:
            adata = adata[:, ~adata.var[_var]]
        print(f"Data (cell,gene) {adata.shape} after remove mitochondrial({adata.var['mt'].sum()}), "
              f"ribosomal({adata.var['ribo'].sum()}), "
              f"and hemoglobin genes({adata.var['hb'].sum()})")
        plot_data_quality(adata)
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

        filtered_adata_hvg = adata[:,list(set(adata.var_names) & set(human_hvg_gene_list))]

    sc.pl.highest_expr_genes(filtered_adata_hvg, n_top=10)
    plot_data_quality(filtered_adata_hvg)

    _shape = filtered_adata_hvg.shape
    print(f"After filter by hvg gene: (cell, gene){_shape}")
    _new_shape = (0, 0)
    min_gene_num = 200 # 2024-09-10 11:41:29 add
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
    plot_data_quality(adata)
    # filtered_adata_hvg.var.index.name = str(filtered_adata_hvg.var.index.name)
    filtered_adata_hvg.write_h5ad(f"{file_path}/raw_count_hvg{hvg_num}.h5ad")

    sc_expression_df = pd.DataFrame(data=filtered_adata_hvg.X.T,
                                    columns=filtered_adata_hvg.obs.index,
                                    index=filtered_adata_hvg.var.index)

    sc_expression_df.to_csv(f"{file_path}/data_count_hvg.csv", sep="\t")

    cell_info = filtered_adata_hvg.obs
    cell_info["day"] = "day18.5"
    cell_info["time"] = 18.5
    cell_info["cell_id"] = filtered_adata_hvg.obs.index
    cell_info["dataset_label"] = "Xiao"
    cell_info["cell_type"] = cell_info["clusters"]
    cell_info["title"] = cell_info.index
    cell_info = cell_info.loc[sc_expression_df.columns]
    cell_info.to_csv(f"{file_path}/cell_with_time.csv", sep="\t")
    gene_info = filtered_adata_hvg.var
    gene_info.to_csv(f"{file_path}/gene_info.csv", sep="\t")

    print("Finish all")

    return


if __name__ == '__main__':
    main()
