# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：preprocess_humanEmbryo_xiang2019data.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/3/23 14:12 
"""

import os
import sys

if os.getcwd().split("/")[-1] != "TemporalVAE":
    os.chdir("..")
sys.path.append(os.getcwd())
from utils.utils_DandanProject import calHVG_adata as calHVG
from utils.utils_DandanProject import series_matrix2csv
from utils.utils_Dandan_plot import draw_venn
import pandas as pd
import anndata as ad
import scanpy as sc


def main():
    hvg_num = 500
    file_path = "data/human_embryo_preimplantation/Xiang2019/"
    raw_count = f"{file_path}/GSE136447_555sample_gene_count_matrix.txt"
    raw_count = pd.read_csv(raw_count, sep="\t", header=0, index_col=0)

    cell_info1 = f"{file_path}/GSE136447-GPL20795_series_matrix.txt"
    cell_info1 = series_matrix2csv(cell_info1)

    cell_info2 = f"{file_path}/GSE136447-GPL23227_series_matrix.txt"
    cell_info2 = series_matrix2csv(cell_info2)

    cell_info = pd.concat([cell_info1[1], cell_info2[1]], axis=0)
    cell_info['cell_id'] = cell_info['title'].apply(lambda x: x.replace('Embryo_', ""))
    # cell_info["development_day"] = cell_info.index.map(extract_number)
    # cell_info["time"]=cell_info["development_day"]
    # necessary change D13.5 to D14, although their annotation is 13.5 in dataset, they are marked as D14 in the Paper.
    cell_info["time"] = cell_info["characteristics_ch1"].apply(lambda x: eval(x.replace(" ", "").replace('age:embryoinvitroday', "").replace("13.5","14")))
    cell_info["day"] = cell_info["time"].apply(lambda x: "day" + str(x))
    cell_info["cell_type"] = cell_info["characteristics_ch1_2"].apply(lambda x: x.replace(" ", "").replace('celltype:', ""))
    cell_info["Stage"] = cell_info["cell_type"]
    cell_info = cell_info.set_index("cell_id")
    # "day","Stage","n_genes","predicted_time"

    if set(cell_info.index) == set(raw_count.columns):
        print("sample id is complete same.")
        raw_count = raw_count.T
        cell_info = cell_info.loc[raw_count.index]
    else:
        exit(0)
    adata = ad.AnnData(X=raw_count.values, obs=cell_info, var=pd.DataFrame(index=raw_count.columns))
    print("Import data, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    print("Cell number: {}".format(adata.n_obs))
    print("Gene number: {}".format(adata.n_vars))
    print("Annotation information of data includes: {}".format(adata.obs_keys()))  # 胞注釋信息的keys
    print("Cell id first 5: {}".format(adata.obs_names[:5]))  # 返回胞ID 数据类型是object
    print("Gene id first 5: {}".format(adata.var_names.to_list()[:5]))  # 返回基因数据类型是list adata.obs.head()# 査看前5行的数据
    print("Gene id first 5: {}".format(adata.var.index.to_list()[:5]))

    min_gene_num = 50
    min_cell_num = 50
    sc.pp.filter_cells(adata, min_genes=min_gene_num)  # drop samples with less than 20 gene expression
    sc.pp.filter_genes(adata, min_cells=min_cell_num)

    hvg_cellRanger_list = calHVG(adata.copy(), gene_num=hvg_num, method="cell_ranger")
    hvg_seurat_list = calHVG(adata.copy(), gene_num=hvg_num, method="seurat")
    hvg_seurat_v3_list = calHVG(adata.copy(), gene_num=hvg_num, method="seurat_v3")

    draw_venn({"cell ranger": hvg_cellRanger_list, "seurat": hvg_seurat_list, "seurat v3": hvg_seurat_v3_list})
    print(f"concat all hvg calculated")
    import itertools
    combined_hvg_list = list(set(itertools.chain(hvg_cellRanger_list, hvg_seurat_list, hvg_seurat_v3_list)))
    filtered_adata_hvg = adata[:, combined_hvg_list].copy()

    _shape = filtered_adata_hvg.shape
    _new_shape = (0, 0)
    min_gene_num = 50
    min_cell_num = 50
    while _new_shape != _shape:  # make sure drop samples and genes
        _shape = filtered_adata_hvg.shape
        sc.pp.filter_cells(filtered_adata_hvg, min_genes=min_gene_num)  # drop samples with less than 20 gene expression
        sc.pp.filter_genes(filtered_adata_hvg, min_cells=min_cell_num)  # drop genes which none expression in 3 samples
        _new_shape = filtered_adata_hvg.shape
    print("Drop cells with less than {} gene expression, "
          "drop genes which none expression in {} samples".format(min_gene_num, min_cell_num))
    print("After filter, get cell number: {}, gene number: {}".format(filtered_adata_hvg.n_obs, filtered_adata_hvg.n_vars))
    print("the original sc expression anndata should be gene as row, cell as column")

    filtered_adata_hvg.var_names = filtered_adata_hvg.var_names.str.split("|").str[-1]
    if filtered_adata_hvg.var_names.duplicated().any():
        # 转换为DataFrame，以便使用pandas的groupby功能
        df = pd.DataFrame(filtered_adata_hvg.X, columns=filtered_adata_hvg.var_names)

        # 求每个基因的平均表达量
        df_mean = df.groupby(df.columns, axis=1).mean()

        # 创建一个新的AnnData对象，使用处理后的数据
        filtered_adata_hvg2 = sc.AnnData(X=df_mean.values, obs=filtered_adata_hvg.obs, var=pd.DataFrame(index=df_mean.columns))
        # filtered_adata_hvg2.var_names_make_unique()
    filtered_adata_hvg2.write_h5ad(f"{file_path}/hvg{hvg_num}/adata_hvg.h5ad")
    filtered_adata_hvg2.obs["dataset_label"] = "xiang2019"
    sc_expression_df = pd.DataFrame(data=filtered_adata_hvg2.X.T,
                                    columns=filtered_adata_hvg2.obs.index,
                                    index=filtered_adata_hvg2.var_names)

    sc_expression_df.to_csv(f"{file_path}/hvg{hvg_num}/data_count_hvg.csv", sep="\t")

    filtered_adata_hvg2.obs.to_csv(f"{file_path}/hvg{hvg_num}/cell_with_time.csv", sep="\t")
    filtered_adata_hvg2.var.to_csv(f"{file_path}/hvg{hvg_num}/gene_info.csv", sep="\t")
    print("After filter, get cell number: {}, gene number: {}".format(filtered_adata_hvg2.n_obs, filtered_adata_hvg2.n_vars))

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
