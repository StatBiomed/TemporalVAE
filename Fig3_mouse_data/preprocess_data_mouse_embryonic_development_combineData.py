# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE
@File    ：preprocess_data_mouse_embryonic_development_combineData.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/8/25 10:12

*** References is Qiu, C., Martin, B.K., Welsh, I.C. et al. A single-cell time-lapse of mouse prenatal development from gastrula to birth. Nature 626, 1084–1093 (2024).
*** Data source is https://shendure-web.gs.washington.edu/content/members/cxqiu/public/backup/jax/download/adata/
    1. directly download:
        adata_JAX_dataset_1.h5ad	2023-04-02 13:15	11G
        adata_JAX_dataset_2.h5ad	2023-04-02 13:15	8.9G
        adata_JAX_dataset_3.h5ad	2023-04-02 13:15	9.9G
        adata_JAX_dataset_4.h5ad	2023-04-02 13:15	11G
        df_cell.csv	2023-07-27 09:59	1.5G
        df_gene.csv
    2. combine four .h5ad files, remove cells with "P0" day annotation.
    3. random select 1/10 cells to calculate hvgs.
    4. filter cells with less than 100 gene expressed and genes with less than 50 cells expressed.
    5. finally save at "data/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/"
        "data_count_hvg.csv", "cell_with_time.csv", "gene_info.csv"
"""
import os
import scanpy as sc
import numpy as np
import pandas as pd
import pyreadr
import anndata
import logging
from utils.logging_system import LogHelper

from collections import Counter
import gc


def main():
    hvg_num = 1000
    min_gene_num = 100
    min_cell_num = 50
    # --------------------- set logger --------------------------
    save_path = f"data/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene{min_gene_num}_minCell{min_cell_num}_hvg{hvg_num}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger_file = '{}/dataset_combine.log'.format(save_path)
    LogHelper.setup(log_path=logger_file, level='INFO')
    _logger = logging.getLogger(__name__)
    # --------------------- set sub dataset location and read total cell and gene info file --------------------------
    sc_file_name_list = ["mouse_embryonic_development//adata_JAX_dataset_1.h5ad",
                         "mouse_embryonic_development//adata_JAX_dataset_2.h5ad",
                         "mouse_embryonic_development//adata_JAX_dataset_3.h5ad",
                         "mouse_embryonic_development//adata_JAX_dataset_4.h5ad"
                         ]
    sc_file_batch_dic = {"mouse_embryonic_development//adata_JAX_dataset_1.h5ad": 0,
                         "mouse_embryonic_development//adata_JAX_dataset_2.h5ad": 1,
                         "mouse_embryonic_development//adata_JAX_dataset_3.h5ad": 2,
                         "mouse_embryonic_development//adata_JAX_dataset_4.h5ad": 3
                         }
    df_cell_rds = pyreadr.read_r("data/mouse_embryonic_development//df_cell.rds")[None]
    # cell_data.index[0]: 'run_4_P2-01A.ATTCAAGCATGTTACGCAAG-0-0' last digit is the batch id, check the sc_file_batch_dic
    cell_data = pd.read_csv("data/mouse_embryonic_development//df_cell.csv", index_col=0)
    gene_data = pd.read_csv("data/mouse_embryonic_development//df_gene.csv", index_col=0)
    _logger.info("the shape of total cell data: {}".format(cell_data.shape))
    _logger.info("the shape of total gene data: {}".format(gene_data.shape))
    _logger.info("types of total gene: {}".format(np.unique(gene_data["gene_type"])))

    # ------------------- filter some gene ---------------------------
    _logger.info("After drop out lincRNA and pesudogenes.")
    gene_data_filted = gene_data.loc[gene_data["gene_type"] != 'lincRNA']
    gene_data_filted = gene_data_filted.loc[gene_data_filted["gene_type"] != 'pseudogene']
    _logger.info("types of gene: {}".format(np.unique(gene_data_filted["gene_type"])))
    _logger.info("the shape of gene data: {}".format(gene_data_filted.shape))

    # ------------------- add celltype_updata column from df_cell_rds data to cell_data------------------
    cell_id_to_sample_rds = dict(zip(df_cell_rds['cell_id'], df_cell_rds['celltype_update']))
    cell_data['celltype_update'] = cell_data['cell_id'].map(cell_id_to_sample_rds)
    _logger.info("add 'celltype_updata' annotation to the cell info data")

    # --------------------- start merged sub dataset to one file --------------------------
    sc_data_df, cell_info, gene_info = preprocessData_hvg_scData_list_combine(_logger,
                                                                              "data/", sc_file_name_list,
                                                                              cell_data.copy(), gene_data_filted.copy(),
                                                                              sc_file_batch_dic=sc_file_batch_dic,
                                                                              hvg_num=hvg_num,
                                                                              min_gene_num=min_gene_num,
                                                                              min_cell_num=min_cell_num)

    # --------------------- save result as csv file --------------------------
    sc_data_df_geneRow_cellCol = pd.DataFrame(data=sc_data_df.values.T, index=sc_data_df.columns, columns=sc_data_df.index)

    cell_info.to_csv("{}/cell_with_time.csv".format(save_path), sep="\t")
    gene_info.to_csv("{}/gene_info.csv".format(save_path), sep="\t")
    sc_data_df_geneRow_cellCol.to_csv("{}/data_count_hvg.csv".format(save_path), sep="\t")

    _logger.info("Finish {}".format(save_path))
    _logger.info("Finish all")


def read_large_h5ad_file(_logger, file_path, sc_file_name_list, _file_index, cell_data, gene_data, sc_file_batch_dic):
    adata = sc.read_h5ad(filename=file_path + sc_file_name_list[_file_index])
    _logger.info("Import data, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    _logger.info("Cell number: {}".format(adata.n_obs))
    _logger.info("Gene number: {}".format(adata.n_vars))
    _logger.info("Annotation information of cell(obs) includes: {}".format(adata.obs_keys()))  # 胞注釋信息的keys
    for attribute_to_delete in adata.obs_keys():
        adata.obs.drop(attribute_to_delete, axis=1, inplace=True)

    _logger.info("Annotation information of gene(var) includes: {}".format(adata.var_keys()))  # 胞注釋信息的keys
    _logger.info("Cell id first 5: {}".format(adata.obs_names[:5]))  # 返回胞ID 数据类型是object
    _logger.info("Gene id first 5: {}".format(adata.var_names.to_list()[:5]))  # 返回基因数据类型是list adata.obs.head()# 査看前5行的数据
    _logger.info("Gene id first 5: {}".format(adata.var.index.to_list()[:5]))  # 返回基因数据类型是list adata.obs.head()# 査看前5行的数据
    batch_idx = str(sc_file_batch_dic[sc_file_name_list[_file_index]])
    adata.obs_names = [_i + "-" + batch_idx for _i in adata.obs_names]
    adata = adata[:, gene_data["gene_id"]]
    _logger.info("After drop out lincRNA and pesudogenes in sc data, "
                 "Anndata with cell number: {}, gene number: {}".format(adata.n_obs,
                                                                        adata.n_vars))
    if batch_idx == "3":
        _logger.info(
            "dropout donor with time is 0, which annotation as 'day'== 'P0' in {}".format(sc_file_name_list[_file_index]))
        drop_out_cell = list(set(cell_data.loc[cell_data["day"] == "P0"].index) & set(adata.obs.index))
        adata = adata[adata.obs_names.drop(drop_out_cell)].copy()

    _logger.info("adata cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    return adata


def preprocessData_hvg_scData_list_combine(_logger,
                                           file_path, sc_file_name_list, cell_data, gene_data, sc_file_batch_dic={},
                                           hvg_num=1000,
                                           min_gene_num=100,
                                           min_cell_num=50):
    merged_adata = pd.DataFrame()
    for _file_index in range(len(sc_file_name_list)):
        _logger.info(
            "--- start read {}/{}, {} h5ad file".format(_file_index + 1, len(sc_file_name_list), sc_file_name_list[_file_index]))
        adata = read_large_h5ad_file(_logger, file_path, sc_file_name_list, _file_index, cell_data, gene_data, sc_file_batch_dic)
        if isinstance(merged_adata, pd.DataFrame):
            merged_adata = adata.copy()
        else:
            merged_adata = anndata.concat([merged_adata, adata], join="inner", axis=0)
        # merged_adata = anndata.concat([merged_adata.copy(), adata.copy()], join="inner", axis=0)
        _logger.info(
            "After combine, merged adata cell number: {}, gene number: {}".format(merged_adata.n_obs, merged_adata.n_vars))
        del adata
        gc.collect()
    _logger.info("Finial, merged anndata with cell number: {}, gene number: {}".format(merged_adata.n_obs,
                                                                                       merged_adata.n_vars))
    # ----------------------------------calculate hvgs--------------------------------------------------
    import random
    _logger.info("Calculate hvg gene list use cell_ranger method from scanpy.")
    _logger.info("Due to leak of memory, random select 1/10 samples to calculate hvgs.")
    random.seed(123)
    # 随机生成不重复的样本索引
    random_indices = random.sample(range(merged_adata.shape[0]), int(merged_adata.n_obs / 10), )
    _logger.info(f"Random select {len(random_indices)} cells")
    # 从 anndata 对象中获取选定的样本数据
    merged_adata_temp = merged_adata[random_indices, :].copy()
    # merged_adata_temp = merged_adata.copy()
    # hvg_seuratV3 = sc.pp.highly_variable_genes(adata_filted, n_top_genes=2000, flavor='seurat_v3',inplace=False)
    sc.pp.normalize_total(merged_adata_temp, target_sum=1e6)
    sc.pp.log1p(merged_adata_temp)
    hvg_cellRanger = sc.pp.highly_variable_genes(merged_adata_temp, flavor="cell_ranger", n_top_genes=hvg_num, inplace=False)
    hvg_cellRanger_list = merged_adata_temp.var.index[hvg_cellRanger["highly_variable"]]
    del merged_adata_temp
    gc.collect()
    # ---------------------------------------------------------------------------------------------

    merged_adata_hvg = merged_adata[:, hvg_cellRanger_list].copy()
    _logger.info("Filter the sc count data with hvg gene, "
                 "get cell number: {}, gene number: {}".format(merged_adata_hvg.n_obs, merged_adata_hvg.n_vars))
    del merged_adata
    gc.collect()

    _shape = merged_adata_hvg.shape

    _new_shape = (0, 0)
    while _new_shape != _shape:  # make sure drop samples and genes
        _shape = merged_adata_hvg.shape
        sc.pp.filter_cells(merged_adata_hvg, min_genes=min_gene_num)  # drop samples with less than 20 gene expression
        sc.pp.filter_genes(merged_adata_hvg, min_cells=min_cell_num)  # drop genes which none expression in 3 samples
        _new_shape = merged_adata_hvg.shape
    _logger.info("Drop cells with less than {} gene expression, "
                 "drop genes which none expression in {} samples".format(min_gene_num, min_cell_num))
    _logger.info(
        "After filter, get cell number: {}, gene number: {}".format(merged_adata_hvg.n_obs, merged_adata_hvg.n_vars))

    sc_expression_df = pd.DataFrame(data=merged_adata_hvg.X.toarray(), columns=merged_adata_hvg.var.index,
                                    index=merged_adata_hvg.obs.index)
    del merged_adata_hvg

    cell_info = cell_data.loc[sc_expression_df.index]
    cell_info["cell"] = cell_info["cell_id"]
    cell_info["time"] = cell_info['day'].str.replace(r'[A-Za-z]', '', regex=True)
    cell_info["donor"] = cell_info["embryo_id"]
    cell_info["cell_type"] = cell_info["celltype_update"]

    _logger.info("Get expression dataframe with shape (cell, gene): {}, and cell time info with shape: {}.".format(
        sc_expression_df.shape, cell_info.shape))
    _logger.info("Detail time info of merged cell info: {}".format(Counter(cell_info["time"])))
    _logger.info("Detail batch (sub dataset) info of merged cell info: {}".format(Counter(cell_info["batch"])))
    gene_info = gene_data.loc[sc_expression_df.columns]

    return sc_expression_df, cell_info, gene_info


if __name__ == '__main__':
    main()
