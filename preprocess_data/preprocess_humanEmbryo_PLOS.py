# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：preprocess_humanEmbryo_PLOS.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/3/3 21:06 
"""
import scanpy as sc
import pandas as pd
import anndata as ad
from collections import Counter
from utils.utils_DandanProject import calHVG_adata as calHVG
from utils.utils_DandanProject import series_matrix2csv
from utils.utils_Dandan_plot import draw_venn


def main():
    select_hvg_bool = False  # 2024-04-03 17:51:44 add here
    file_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/240322Human_embryo/PLOS2019/"
    count_data = pd.read_csv(f"{file_path}/data_count.csv", index_col=0, sep="\t")
    _cell_temp = pd.read_csv(f"{file_path}/sample.csv", index_col=0, sep=",")
    _cell_temp2 = pd.read_csv(f"{file_path}/SraRunTable.txt", index_col=0, sep=",")
    _cell_temp["GEO_Accession (exp)"] = _cell_temp.index
    _cell_temp2["Run"] = _cell_temp2.index
    cell_info_pd = pd.merge(_cell_temp, _cell_temp2, on="GEO_Accession (exp)", how="inner")
    cell_info_pd.index = cell_info_pd["Title"]

    # cell_infotemp = series_matrix2csv(f"{file_path}/GSE125616_series_matrix.txt.gz")

    Counter(cell_info_pd["development_day"])
    cell_info_pd = cell_info_pd.loc[count_data.columns]
    Counter(cell_info_pd["development_day"])
    adata = ad.AnnData(X=count_data.values.T, obs=cell_info_pd, var=pd.DataFrame(index=count_data.index))
    print("Import data, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    print("Cell number: {}".format(adata.n_obs))
    print("Gene number: {}".format(adata.n_vars))
    print("Annotation information of data includes: {}".format(adata.obs_keys()))  # 胞注釋信息的keys
    print("Cell id first 5: {}".format(adata.obs_names[:5]))  # 返回胞ID 数据类型是object
    print("Gene id first 5: {}".format(adata.var_names.to_list()[:5]))  # 返回基因数据类型是list adata.obs.head()# 査看前5行的数据
    print("Gene id first 5: {}".format(adata.var.index.to_list()[:5]))  # 返回基因数据类型是list adata.obs.head()# 査看前5行的数据
    # 定义你想要保留的 development_day 的值
    # days_to_keep = ["day6", "day7", "day8", "day9","day10","day13","day14","day0"]
    days_to_keep = ["day6", "day7", "day8", "day9","day10","day14"]
    # days_to_keep = ["day6", "day7", "day8", "day9", "day10"]
    # adata.obs['development_day'] = adata.obs['development_day'].replace('Endometrial', 'day0')

    # 使用布尔索引来筛选 those observations in anndata.obs that meet the condition
    filtered_adata = adata[adata.obs["development_day"].isin(days_to_keep)].copy()
    filtered_adata.obs["time"] = filtered_adata.obs["development_day"].str.replace("day", "").astype(int)
    print("Use gene list from mouse_embryo,Anndata with cell number: {}, gene number: {}".format(filtered_adata.n_obs, filtered_adata.n_vars))

    if select_hvg_bool:
        sc.pp.filter_genes(filtered_adata, min_cells=50)  # drop genes which none expression in 3 samples
        hvg_cellRanger_list = calHVG(filtered_adata.copy(), gene_num=1000, method="cell_ranger")
        hvg_seurat_list = calHVG(filtered_adata.copy(), gene_num=1000, method="seurat")
        hvg_seurat_v3_list = calHVG(filtered_adata.copy(), gene_num=1000, method="seurat_v3")
        draw_venn({"cell ranger": hvg_cellRanger_list, "seurat": hvg_seurat_list, "seurat v3": hvg_seurat_v3_list})

        print(f"concat all hvg calculated")
        import itertools
        combined_hvg_list = list(set(itertools.chain(hvg_cellRanger_list, hvg_seurat_list, hvg_seurat_v3_list)))
        filtered_adata_hvg = filtered_adata[:, combined_hvg_list].copy()
    else:
        filtered_adata_hvg = filtered_adata.copy()

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
    filtered_adata_hvg.write_h5ad(f"{file_path}/adata_hvg.h5ad")

    sc_expression_df = pd.DataFrame(data=filtered_adata_hvg.X.T,
                                    columns=filtered_adata_hvg.obs.index,
                                    index=filtered_adata_hvg.var.index)

    sc_expression_df.to_csv(f"{file_path}/data_count_hvg.csv", sep="\t")

    cell_info = filtered_adata_hvg.obs
    cell_info["day"] = filtered_adata_hvg.obs["development_day"]
    cell_info["time"] = cell_info['day'].str.replace(r'[A-Za-z]', '', regex=True)
    cell_info["cell_id"] = filtered_adata_hvg.obs.index
    cell_info["dataset_label"]="PLOS"
    cell_info["cell_type"]=cell_info["Stage"]
    cell_info["title"]=cell_info.index
    cell_info = cell_info.loc[sc_expression_df.columns]
    cell_info.to_csv(f"{file_path}/cell_with_time.csv", sep="\t")
    gene_info = filtered_adata_hvg.var
    gene_info.to_csv(f"{file_path}/gene_info.csv", sep="\t")

    print("Finish all")


if __name__ == '__main__':
    main()
