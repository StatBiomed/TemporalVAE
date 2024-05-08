# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：preprocess_data_mouse_embryo_stereo_calHvg_subOrgan.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/10/19 18:18 

2023-10-19 18:18:36
preprocess sc data on each organ of organ_list
min gene num is 50
for each organ calculate hvgs for biological analysis.

"""
import sys

sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan")
import scanpy as sc
import pandas as pd
import os
import re
from collections import Counter


def main(min_gene_num=50,
         min_cell_num=50,
         hvg_num=1000,
         gene_pattern="hvg+atlas_hvg"):  # gene_pattern: "only_hvg" "all" "hvg+atlas_hvg" "atlas_hvg"
    print(f"gene pattern is {gene_pattern}; min gene num {min_gene_num}; min cell num {min_cell_num}; hvg num {hvg_num}.")
    file_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/mouse_embryo_stereo/"
    sc_file_name = "/Mouse_embryo_all_stage.h5ad"

    organ_list = ['Liver', 'Heart', 'Brain']

    for index, organ in enumerate(organ_list):
        print("--- Start {},{}/{}".format(organ, index, len(organ_list)))
        sc_data_df, cell_info, gene_info = preprocessData_scData_h5ad_subType_calHvgs(file_path,
                                                                                      sc_file_name,
                                                                                      min_gene_num=min_gene_num,
                                                                                      min_cell_num=min_cell_num,
                                                                                      organ=organ,
                                                                                      hvg_num=hvg_num,
                                                                                      gene_pattern=gene_pattern)
        # save result as csv file
        save_path = '{}/preprocess_{}_minGene{}_hvg{}_pattern{}_of{}/'. \
            format(file_path,
                   sc_file_name.replace("/", "").replace(".h5ad", "").replace(" ", "_"),
                   min_gene_num,
                   hvg_num,
                   gene_pattern,
                   organ.replace(" ", "").replace("/", ""))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        from collections import Counter
        print("info of cell type info:{}".format(Counter(cell_info["celltype_update"])))
        print("info of time info:{}".format(Counter(cell_info["time"])))
        sc_data_df = pd.DataFrame(data=sc_data_df.values.T, index=sc_data_df.columns, columns=sc_data_df.index)
        print("the original sc expression anndata should be gene as row, cell as column")

        cell_info.to_csv("{}/cell_with_time.csv".format(save_path), sep="\t")
        gene_info.to_csv("{}/gene_info.csv".format(save_path), sep="\t")
        sc_data_df.to_csv("{}/data_count_hvg.csv".format(save_path), sep="\t")
        print("--- Finish {}, save at {}".format(organ, save_path))
    print("Finish all")


def preprocessData_scData_h5ad_subType_calHvgs(file_path, sc_file_name, required_gene_info=None,
                                               min_gene_num=10,
                                               min_cell_num=10, organ="", hvg_num=1000, gene_pattern="only_hvg"):
    """

    :param file_path:
    :param sc_file_name:
    :param required_gene_info:
    :param min_gene_num:
    :param min_cell_num:
    :param organ:
    :param hvg_num:
    :param gene_pattern: "only_hvg" "all" "hvg+atlas_hvg" "atlas_hvg"
    :return:
    """
    print("check for {}".format(sc_file_name))

    adata = sc.read_h5ad(filename=file_path + sc_file_name)
    print("Import data, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    print("Annotation information of data includes: {}".format(adata.obs_keys()))  # 胞注釋信息的keys
    print("Cell id first 5: {}".format(adata.obs_names[:5]))  # 返回胞ID 数据类型是object
    print("Gene id first 5: {}".format(adata.var_names.to_list()[:5]))  # 返回基因数据类型是list adata.obs.head()# 査看前5行的数据

    print("-only save cell of organ {}.".format(organ))
    save_cell = adata.obs["annotation"] == organ
    adata = adata[save_cell].copy()
    print("-After filter, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    # ------------------------ remove gene start with mt- and hbb and hba ------------------------
    gene_list = list(adata.var_names)
    pattern = r'^Rp[sl]\d+|^Rplp\d+|^Rpsa'
    compiled_pattern = re.compile(pattern, re.IGNORECASE)
    rp_gene_list = [s for s in gene_list if compiled_pattern.search(s)]
    mt_gene_list = [s for s in gene_list if s.lower().startswith("mt-")]
    hbb_gene_list = [s for s in gene_list if
                     s.lower().startswith("hba") or s.lower().startswith("hbb") or s.lower().startswith("hbq")]
    from utils.utils_DandanProject import geneId_geneName_dic
    gene_dic, gene_total_pd = geneId_geneName_dic(return_total_gene_pd_bool=True)
    print("drop out lincRNA and pesudogenes.")
    linc_pseudo_gene_list = list(gene_total_pd.loc[gene_total_pd["gene_type"] != 'protein_coding']["gene_short_name"])
    delete_gene_list = rp_gene_list + mt_gene_list + hbb_gene_list + linc_pseudo_gene_list
    print(f"delete genes: {delete_gene_list}")
    # ---------------------------------------------------------------------------------------
    hvg_organ_list = cal_hvg_list(adata, delete_gene_list, min_gene_num, min_cell_num, hvg_num)
    hvg_atlas_df = hvg_atlasData_df()
    hvg_atalas_list = list(set(hvg_atlas_df["gene_short_name"]) & set(adata.var_names))
    # ---------------------------------------------------------------------------------------
    if gene_pattern == "only_hvg":  # "only_hvg" "all" "hvg+atlas_hvg" "atlas_hvg"
        adata_filted = adata[:, hvg_organ_list].copy()
    elif gene_pattern == "atlas_hvg":
        adata_filted = adata[:, hvg_atalas_list].copy()
    elif gene_pattern == "hvg+atlas_hvg":
        adata_filted = adata[:, list(set(list(hvg_atalas_list) +list( hvg_organ_list)))].copy()
    elif gene_pattern == "all":
        print(f"use all genes.")
        adata_filted = adata[:, ~adata.var_names.isin(delete_gene_list)].copy()
    # ---------------------------------------------------------------------------------------
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

    sc_expression_df = pd.DataFrame(data=adata_filted.X.toarray(), columns=adata_filted.var.index,
                                    index=adata_filted.obs.index)
    cell_info = pd.DataFrame(index=adata.obs_names)
    cell_info["donor"] = cell_info.index.map(lambda idx: "embryo_stereo_" + str(idx).split('-')[-1])
    cell_info["celltype_update"] = adata.obs["annotation"]
    cell_info["day"] = adata.obs["timepoint"]
    cell_info["time"] = cell_info['day'].str.replace(r'[A-Za-z]', '', regex=True)
    cell_info["cell_id"] = adata.obs_names
    cell_info = cell_info.loc[sc_expression_df.index]
    print("Get expression dataframe with shape (cell, gene): {}, and cell time info with shape: {}.".format(
        sc_expression_df.shape, cell_info.shape))
    print(Counter(cell_info["time"]))
    gene_list = pd.DataFrame(sc_expression_df.columns)
    return sc_expression_df, cell_info, gene_list


def hvg_atlasData_df():
    gene_used_in_mouse_embryonic_development_file = \
        "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/mouse_embryonic_development/" \
        "preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/gene_info.csv"
    print("use hvg gene list from {} to filter the mouse_embryo_stereo sc data .h5ad".format(
        gene_used_in_mouse_embryonic_development_file))
    gene_used_in_mouse_embryonic_development_data = pd.read_csv(gene_used_in_mouse_embryonic_development_file,
                                                                index_col=0,
                                                                sep="\t")
    return gene_used_in_mouse_embryonic_development_data


def cal_hvg_list(adata, delete_gene_list, min_gene_num, min_cell_num, hvg_num):
    print(f"calculate {hvg_num} hvgs.")
    adata_temp = adata[:, ~adata.var_names.isin(delete_gene_list)].copy()
    _shape = adata_temp.shape
    _new_shape = (0, 0)
    while _new_shape != _shape:  # make sure drop samples and genes
        _shape = adata_temp.shape
        sc.pp.filter_cells(adata_temp, min_genes=min_gene_num + 20)  # drop samples with less than 20 gene expression
        sc.pp.filter_genes(adata_temp, min_cells=min_cell_num + 20)  # drop genes which none expression in 3 samples
        _new_shape = adata_temp.shape
    sc.pp.normalize_total(adata_temp, target_sum=1e6)
    sc.pp.log1p(adata_temp)
    hvg_cellRanger = sc.pp.highly_variable_genes(adata_temp, flavor="cell_ranger", n_top_genes=hvg_num, inplace=False)
    hvg_cellRanger_list = adata_temp.var.index[hvg_cellRanger["highly_variable"]]
    return hvg_cellRanger_list


if __name__ == '__main__':
    main()
