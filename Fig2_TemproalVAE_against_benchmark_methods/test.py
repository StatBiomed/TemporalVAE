# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：test.py
@IDE     ：PyCharm
@Author  ：awa121
@Date    ：2023/9/24 21:05

"""

from pypsupertime import Psupertime
import anndata
import pandas as pd
import os
import numpy as np

os.chdir('/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/Fig2_TemproalVAE_against_benchmark_methods')


def main():
    # for mouse embryonic beta cells dataset:
    result_file_name = "embryoBeta"
    data_org = pd.read_csv('data_fromPsupertime/GSE87375_Single_Cell_RNA-seq_Gene_TPM.txt', index_col=0, sep="\t").T
    # get beta cell
    cell_id = data_org.index[1:]
    cell_beta_id = [i for i in cell_id if i[0] == "b"]
    data_org.columns = data_org.iloc[0]
    data_org = data_org[1:]
    data_org = data_org.loc[:, ~data_org.columns.duplicated()]

    adata = anndata.AnnData(data_org)
    adata.var_names_make_unique()
    adata = adata[cell_beta_id].copy()
    temp_time = np.array(adata.obs_names)

    temp_time = [eval(i.split("_")[0].replace("bP", "").replace("bE17.5", "-1")) for i in temp_time]

    adata.obs["time"] = temp_time
    import scanpy as sc
    # sc.pl.highest_expr_genes(adata, n_top=20, )
    sc.pp.filter_genes(adata, min_cells=25)

    adata.var['ERCC'] = adata.var_names.str.startswith('ERCC-')  # annotate the group of mitochondrial genes as 'ERCC'
    adata = adata[:, ~adata.var.ERCC]
    adata.var['RP'] = adata.var_names.str.startswith('RP')  # annotate the group of mitochondrial genes as 'RP'
    adata = adata[:, ~adata.var.RP]

    # sc.pp.calculate_qc_metrics(adata, qc_vars=['ERCC'], percent_top=None, log1p=False, inplace=True)
    # sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_ERCC'],jitter=0.4, multi_panel=True)
    # sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
    # sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')
    preprocessing_params = {"select_genes": "hvg", "log": True}
    # to get hvg gene of humanGermline dataset
    tp = Psupertime(n_jobs=1, n_folds=5,preprocessing_params=preprocessing_params)
    adata_hvg = tp.preprocessing.fit_transform(adata.copy())
    del tp

    hvg_gene_df = pd.DataFrame(adata_hvg.var_names)
    hvg_gene_df = hvg_gene_df.rename(columns={'Symbol': 'gene_name'})
    hvg_gene_df.to_csv(f'{os.getcwd()}/data_fromPsupertime/{result_file_name}_gene_list.csv', index=True)

    x_df = data_org.loc[adata_hvg.obs_names]
    x_df = x_df[hvg_gene_df["gene_name"]]
    x_df = x_df.T
    x_df.to_csv(f'{os.getcwd()}/data_fromPsupertime/{result_file_name}_X.csv', index=True)

    y_df = pd.DataFrame(adata_hvg.obs.time)
    y_df.to_csv(f'{os.getcwd()}/data_fromPsupertime/{result_file_name}_Y.csv', index=True)

    print("Finish save files.")


if __name__ == '__main__':
    main()