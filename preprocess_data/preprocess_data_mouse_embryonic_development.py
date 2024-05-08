# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：preprocess_data_mouse_embryonic_development.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/7/18 16:27

for sc_file_name_list = ["/adata_JAX_dataset_1.h5ad",
                         "/adata_JAX_dataset_2.h5ad",
                         "/adata_JAX_dataset_3.h5ad",
                         "/adata_JAX_dataset_4.h5ad"],
generate preprocess datasets respectively
saved at "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/mouse_embryonic_development/"
"""
import scanpy as sc
import numpy as np
import pandas as pd
import pyreadr
import os


def main():
    file_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/mouse_embryonic_development/"
    sc_file_name_list = ["/adata_JAX_dataset_1.h5ad",
                         "/adata_JAX_dataset_2.h5ad",
                         "/adata_JAX_dataset_3.h5ad",
                         "/adata_JAX_dataset_4.h5ad"]
    df_cell_rds = pyreadr.read_r(file_path + '/df_cell.rds')[None]

    df_cell_file = "/df_cell.csv"
    df_gene_file = "/df_gene.csv"
    cell_data = pd.read_csv(file_path + df_cell_file, index_col=0)  # cell_data.index[0]: 'run_4_P2-01A.ATTCAAGCATGTTACGCAAG-0-0'
    gene_data = pd.read_csv(file_path + df_gene_file, index_col=0)
    print("the shape of cell data: {}".format(cell_data.shape))
    print("the shape of gene data: {}".format(gene_data.shape))
    print("types of gene: {}".format(np.unique(gene_data["gene_type"])))

    print("After drop out lincRNA and pesudogenes.")
    gene_data_filted = gene_data.loc[gene_data["gene_type"] != 'lincRNA']
    gene_data_filted = gene_data_filted.loc[gene_data_filted["gene_type"] != 'pseudogene']
    print("types of gene: {}".format(np.unique(gene_data_filted["gene_type"])))
    print("the shape of gene data: {}".format(gene_data_filted.shape))

    # ------ some confused .rds file in https://shendure-web.gs.washington.edu/content/members/cxqiu/public/nobackup/jax/download/meta_data/
    # df_cell_birth_series_rds = pyreadr.read_r(file_path + '/df_cell.birth_series.rds')[None]
    # df_cell_somitogenesis_validation_rds = pyreadr.read_r(file_path + '/df_cell.somitogenesis_validation.rds')[None]
    # print(cell_data.index[0])
    # print(df_cell_rds.index[0])
    # print(cell_data.iloc[0])
    # print(df_cell_rds.iloc[0])
    # --- add celltype_updata column from df_cell_rds data to cell_data
    cell_id_to_sample_rds = dict(zip(df_cell_rds['cell_id'], df_cell_rds['celltype_update']))
    cell_data['celltype_update'] = cell_data['cell_id'].map(cell_id_to_sample_rds)

    for _file_index in range(len(sc_file_name_list)):
        print("Start {}".format(sc_file_name_list[_file_index]))
        sc_data_df, cell_info, gene_info = preprocessData_hvg_scData(file_path, sc_file_name_list[_file_index], _file_index,
                                                                     cell_data, gene_data_filted,hvg_num=1000)
        # save result as csv file
        _path = '{}/{}/'.format(file_path, "preprocess_" + sc_file_name_list[_file_index].replace("/", "").replace(".h5ad", "").replace(" ","_"))
        if not os.path.exists(_path):
            os.makedirs(_path)
        sc_data_df_geneRow_cellCol=  pd.DataFrame(data=sc_data_df.values.T, index=sc_data_df.columns, columns=sc_data_df.index)

        sc_data_df_geneRow_cellCol.to_csv("{}/data_count_hvg.csv".format(_path),sep="\t")
        cell_info.to_csv("{}/cell_with_time.csv".format(_path),sep="\t")
        gene_info.to_csv("{}/gene_info.csv".format(_path),sep="\t")
        print("Finish {}".format(sc_file_name_list[_file_index]))
    print("Finish all")


def preprocessData_hvg_scData(file_path, sc_file_name, batch_id, cell_data, gene_data, hvg_num=1000, min_gene_num=10,
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

    adata_filted = adata[:, gene_data["gene_id"]].copy()
    print("After drop out lincRNA and pesudogenes in sc data, "
          "Anndata with cell number: {}, gene number: {}".format(adata_filted.n_obs, adata_filted.n_vars))

    print("Calculate hvg gene list use cell_ranger method from scanpy.")
    # hvg_seuratV3 = sc.pp.highly_variable_genes(adata_filted, n_top_genes=2000, flavor='seurat_v3',inplace=False)
    sc.pp.normalize_total(adata_filted, target_sum=1e6)
    sc.pp.log1p(adata_filted)
    hvg_cellRanger = sc.pp.highly_variable_genes(adata_filted, flavor="cell_ranger", n_top_genes=hvg_num, inplace=False)
    hvg_cellRanger_list = adata_filted.var.index[hvg_cellRanger["highly_variable"]]
    adata_count_hvg = adata[:, hvg_cellRanger_list].copy()
    print("Filter the sc count data with hvg gene, "
          "get cell number: {}, gene number: {}".format(adata_count_hvg.n_obs, adata_count_hvg.n_vars))

    _shape = adata_count_hvg.shape

    _new_shape = (0, 0)
    while _new_shape != _shape:  # make sure drop samples and genes
        _shape = adata_count_hvg.shape
        sc.pp.filter_cells(adata_count_hvg, min_genes=min_gene_num)  # drop samples with less than 20 gene expression
        sc.pp.filter_genes(adata_count_hvg, min_cells=min_cell_num)  # drop genes which none expression in 3 samples
        _new_shape = adata_count_hvg.shape
    print("Drop cells with less than {} gene expression, "
          "drop genes which none expression in {} samples".format(min_gene_num, min_cell_num))
    print("After filter, get cell number: {}, gene number: {}".format(adata_count_hvg.n_obs, adata_count_hvg.n_vars))

    # sc.pp.normalize_total(adata_count_hvg, target_sum=1e6)
    # sc.pp.log1p(adata_count_hvg)
    # print("Finish normalize per cell, so that every cell has the same total count after normalization.")

    sc_expression_df = pd.DataFrame(data=adata_count_hvg.X.toarray(), columns=adata_count_hvg.var.index,
                                    index=adata_count_hvg.obs.index)
    # denseM = sc_expression_df.values
    # from sklearn.preprocessing import scale
    # denseM = scale(denseM.astype(float), axis=0, with_mean=True, with_std=True)
    # print("Finish normalize per gene as Gaussian-dist (0, 1).")

    # sc_expression_df = pd.DataFrame(data=denseM, columns=sc_expression_df.columns, index=sc_expression_df.index)
    sc_expression_df.rename(lambda x: f'{x}-' + str(batch_id), inplace=True)
    cell_info = cell_data.loc[sc_expression_df.index]
    cell_info["cell"] = cell_info["cell_id"]
    cell_info["time"] = cell_info['day'].str.replace(r'[A-Za-z]', '', regex=True)
    cell_info["donor"] = cell_info["embryo_id"]
    cell_info["cell_type"] = cell_info["celltype_update"]

    print("Get expression dataframe with shape (cell, gene): {}, and cell time info with shape: {}.".format(
        sc_expression_df.shape, cell_info.shape))

    gene_info = gene_data.loc[sc_expression_df.columns]

    return sc_expression_df, cell_info, gene_info


if __name__ == '__main__':
    main()
