# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：integration_humanEmbryo_7dataset.py
@Author  ：awa121
@Date    ：2025/3/16 11:32

Dataset	Reference	n_cells	n_embryo or donor	platforms	Condition	time range  location
Xiang19	Xiang et al, Nature 2020	555	42	Smart-seq2	in vitro (3D culcture)	E6, 7, 8, 9, 10, 12 data/240322Human_embryo/xiang2019
P	Petropoulos et al, Cell 2016	1529	88	Smart-seq2	ex vivo (?), IVF	E3, E4, E5, E6, E7  data/240405_preimplantation_Melania/P_raw_count
M	Molè et al, Nat Comm 2021	4820	16	10x Genomics	ex vivo (?), IVF	E5, 6, 7, 9, 11 data/240405_preimplantation_Melania/M_raw_count
Z	Zhou et al, Nature 2019	5911	65	STRT-Seq	in vitro, IVF	E6, 8, 10, 12   data/240405_preimplantation_Melania/Z_raw_count
L	Liu et al, Sci Adv 2022	1719	25	STRT-Seq	in vitro, IVF	E5, 6   data/240405_preimplantation_Melania/L_raw_count

T	Tyser et al, Nature 2021	1195	1	Smart-seq2	in vivo, medical termination	CS7 data/240405_preimplantation_Melania/Tyser
Xiao	Xiao et al, Cell 2024	38562	1	Stereo-seq	in vivo, elective termination	CS8	62 slides, bin50=25um data/240405_preimplantation_Melania/xiaoCellCS8
Cui	Cui et al, Nat CB 2025	28804	1	Stereo-seq	in vivo, elective termination	CS7	82 slides, bin50=25um, NEW***   data/240405_preimplantation_Melania/Cui_raw_count


"""
import os
import sys

if os.getcwd().split("/")[-1] != "TemporalVAE":
    os.chdir("..")
sys.path.append(os.getcwd())
import pandas as pd
import anndata as ad
import scanpy as sc
from utils.utils_DandanProject import read_rds_file
from collections import Counter
import anndata as ad
from utils.utils_Dandan_plot import plot_data_quality

def main():
    # Counter({'z': 5911, 'm': 4820, 'p': 1529, 't': 1195, 'l': 989})
    adata = read_trans_Melania()

    # new ones
    adata_cui = read_trans_Cui(adata.copy())
    adata_xiao = read_trans_xiao_data(adata.copy())

    # old datasets
    # ---T
    adata_T = read_trans_Tyser_data(adata.copy())
    # ---L
    adata_L = read_trans_Liu_data(adata.copy())
    # ----Z
    adata_Z = read_trans_Zhou_data(adata.copy())
    # ---M
    adata_M = read_trans_Mole_data(adata.copy())
    # ---P
    adata_P = read_trans_Petropoulos_data(adata.copy())

    # gene list reset as overlap set
    col_list_filtered = [col for col in adata.var_names if (col in adata_cui.var_names)&(col in adata_xiao.var_names)]
    print(f"reserved {len(col_list_filtered)} genes: {col_list_filtered}")
    adata_new = ad.concat([adata_cui[:,col_list_filtered],
                           adata_xiao[:,col_list_filtered],
                           adata_T[:, col_list_filtered],
                           adata_L[:, col_list_filtered],
                           adata_Z[:, col_list_filtered],
                           adata_M[:, col_list_filtered],
                           adata_P[:, col_list_filtered], ], axis=0)
    adata_new.obs['dataset_label'] = adata_new.obs['dataset_label'].str.upper()
    print(adata_new)
    print(f"{Counter(adata_new.obs['dataset_label'])}")
    # save integration results
    adata_new.write_h5ad(f"data/240405_preimplantation_Melania/integration_7dataset/rawCount_7dataset.h5ad")
    sc_expression_df = pd.DataFrame(data=adata_new.X.T, columns=adata_new.obs.index, index=adata_new.var.index)
    sc_expression_df.to_csv(f"data/240405_preimplantation_Melania/integration_7dataset/data_count_hvg.csv", sep="\t")
    adata_new.obs.to_csv(f"data/240405_preimplantation_Melania/integration_7dataset/cell_with_time.csv", sep="\t")
    adata_new.var.to_csv(f"data/240405_preimplantation_Melania/integration_7dataset/gene_info.csv", sep="\t")

    print("finished")
    return
def read_trans_Cui(adata,min_gene_num = 200):
    # rawCount_cui = pd.read_csv(f"data/240405_preimplantation_Melania/Cui_raw_count/data_count.csv", sep="\t", header=0, index_col=0)
    rawCount_cui = ad.read_csv(f"data/240405_preimplantation_Melania/Cui_raw_count/data_count.csv", delimiter='\t')  # raw count, 25833gene/28804cell
    rawCount_cui = rawCount_cui.T
    cell_info_cui = pd.read_csv(f"data/240405_preimplantation_Melania/Cui_raw_count/cell_info.csv", sep="\t", header=0, index_col=0)
    rawCount_cui.obs = cell_info_cui
    rawCount_cui.var_names = rawCount_cui.var_names.str.split(".").str[0]
    print(rawCount_cui)
    col_list_filtered = [col for col in adata.var_names if col in rawCount_cui.var_names]
    rawCount_cui = rawCount_cui[:, col_list_filtered]

    # filter low-quality cell
    _shape = rawCount_cui.shape
    print(f"After filter by hvg gene: (cell, gene){_shape}")
    _new_shape = (0, 0)
      # 2024-09-10 11:41:29 add
    # min_cell_num = 50
    while _new_shape != _shape:  # make sure drop samples and genes
        _shape = rawCount_cui.shape
        sc.pp.filter_cells(rawCount_cui, min_genes=min_gene_num)  # drop samples with less than 20 gene expression
        # sc.pp.filter_genes(rawCount_cui, min_cells=min_cell_num)  # drop genes which none expression in min_cell_num cells
        _new_shape = rawCount_cui.shape
    print(f"Drop cells with less than {min_gene_num} gene expression, ")
    print("After filter, get cell number: {}, gene number: {}".format(rawCount_cui.n_obs, rawCount_cui.n_vars))
    print("the original sc expression anndata should be gene as row, cell as column")
    plot_data_quality(rawCount_cui)

    # add annotation
    rawCount_cui.obs['time'] = 17.5
    rawCount_cui.obs["day"] = "D17.5_cui"
    rawCount_cui.obs["cell_id"] = rawCount_cui.obs_names
    rawCount_cui.obs["dataset_label"] = "c"
    rawCount_cui.obs["donor"] = rawCount_cui.obs["day"]

    rawCount_cui.obs["cell_type"] = rawCount_cui.obs["clusters"]
    categories = rawCount_cui.obs["cell_type"].cat.categories
    new_categories = [f"{cat}_cui" for cat in categories]
    rawCount_cui.obs["cell_type"] = rawCount_cui.obs["cell_type"].cat.rename_categories(new_categories)

    rawCount_cui.obs["title"] = rawCount_cui.obs.index
    rawCount_cui.obs["species"] = "human"
    return rawCount_cui
def read_trans_Melania():
    adata = sc.read_h5ad(f"data/240405_preimplantation_Melania/Melania_5datasets/adata_human_preimplantation_for_degong.h5")
    non_nan_attr_list = []
    for _f in list(adata.obs.columns):
        if len(Counter(adata.obs[_f])) < 100:
            print(f"***{_f}\t{Counter(adata.obs[_f])}")
            if not any(pd.isna(key) for key in Counter(adata.obs[_f]).keys()):
                non_nan_attr_list.append(_f)
    for _f in non_nan_attr_list:
        print(f"***{_f}\t{Counter(adata.obs[_f])}")
    print("Import data, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    print("Cell number: {}".format(adata.n_obs))
    print("Gene number: {}".format(adata.n_vars))
    print("Annotation information of data includes: {}".format(adata.obs_keys()))  # 胞注釋信息的keys
    print("Cell id first 5: {}".format(adata.obs_names[:5]))  # 返回胞ID 数据类型是object
    print("Gene id first 5: {}".format(adata.var_names.to_list()[:5]))  # 返回基因数据类型是list adata.obs.head()# 査看前5行的数据

    sc.pl.highest_expr_genes(adata, n_top=20)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    sc.pl.violin(adata,
                 ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
                 jitter=0.4,
                 multi_panel=True, )
    import re
    import numpy as np
    def extract_time(day_cat):
        # 处理特殊格式如 'D_14_21_t'
        # if day_cat=="D_14_21_t":
        #     return 16.5
        if '_' in day_cat:
            numbers = list(map(int, re.findall(r'\d+', day_cat)))
            if len(numbers) > 1:
                # return numbers[1]
                return round(np.mean(numbers), 2)
            return numbers[0]  # 如果只有一个数字，直接返回
        # 通常格式如 'D5_p'
        return int(re.search(r'\d+', day_cat).group())

    # 应用这个函数到 day_cat 列，创建新的 time 列
    adata.obs['time'] = adata.obs['day_cat'].apply(extract_time)
    temp_dic = pd.Series(adata.obs['time'].values, index=adata.obs['day_cat']).to_dict()
    print(f"Time trans dic: {temp_dic}")
    print("After filter, get cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))

    adata.obs["day"] = adata.obs["day_cat"]
    adata.obs["cell_id"] = adata.obs.index
    adata.obs["dataset_label"] = adata.obs["batch"]
    adata.obs["donor"] = adata.obs["day_cat"]
    adata.obs["cell_type"] = adata.obs["lineage"]
    adata.obs["title"] = adata.obs.index
    adata.obs["species"] = "human"
    print(adata)
    return adata


def read_trans_Zhou_data(adata):
    rawCount_Z = pd.read_csv("data/240405_preimplantation_Melania/Z_raw_count/GSE109555_All_Embryo_TPM.txt",
                             sep="\t", header=0, index_col=0)  # not raw count, TPM, 22359gene/5911cell
    adata_Z = adata[adata.obs['batch'] == 'z'].copy()
    adata_Z_cellName_list = ["-".join(i.split("-")[:-2]) for i in adata_Z.obs_names]
    if set(rawCount_Z.columns) == set(adata_Z_cellName_list):
        adata_Z.obs["orginal_cell_name"] = adata_Z_cellName_list
    else:
        print("***ERROR***")
    rawCount_Z.index = rawCount_Z.index.str.split(".").str[0]
    rawCount_Z_hvg = rawCount_Z[rawCount_Z.index.isin(adata_Z.var_names)]
    rawCount_Z_hvg = rawCount_Z_hvg.groupby(level=0).mean()
    rawCount_Z_hvg = rawCount_Z_hvg.T
    rawCount_Z_hvg = rawCount_Z_hvg.reindex(index=adata_Z.obs["orginal_cell_name"], columns=adata_Z.var_names)
    adata_Z.X = rawCount_Z_hvg.to_numpy().astype(float)
    return adata_Z


def read_trans_Mole_data(adata):
    rawCount_M = pd.read_csv("data/240405_preimplantation_Melania/M_raw_count/data_count.csv",
                             sep="\t", header=0, index_col=0)  # raw count 45068gene/4820cell
    adata_M = adata[adata.obs['batch'] == 'm'].copy()
    adata_M_cellName_list = ["-".join(i.split("-")[:-2]) for i in adata_M.obs_names]
    if set(rawCount_M.columns) == set(adata_M_cellName_list):
        adata_M.obs["orginal_cell_name"] = adata_M_cellName_list
    else:
        print("***ERROR***")
    rawCount_M.index = rawCount_M.index.str.split(".").str[0]
    rawCount_M_hvg = rawCount_M[rawCount_M.index.isin(adata_M.var_names)]
    rawCount_M_hvg = rawCount_M_hvg.groupby(level=0).mean()
    rawCount_M_hvg = rawCount_M_hvg.T
    rawCount_M_hvg = rawCount_M_hvg.reindex(index=adata_M.obs["orginal_cell_name"], columns=adata_M.var_names)
    adata_M.X = rawCount_M_hvg.to_numpy().astype(float)
    return adata_M


def read_trans_Petropoulos_data(adata):
    rawCount_P = pd.read_csv("data/240405_preimplantation_Melania/P_raw_count/counts.txt",
                             sep="\t", header=0, index_col=0)  # raw count 26178gene/1529cell
    adata_P = adata[adata.obs['batch'] == 'p'].copy()
    adata_P_cellName_list = ["-".join(i.split("-")[:-2]) for i in adata_P.obs_names]
    if set(rawCount_P.columns) == set(adata_P_cellName_list):
        adata_P.obs["orginal_cell_name"] = adata_P_cellName_list
    else:
        print("***ERROR***")
    rawCount_P_hvg = rawCount_P[rawCount_P.index.isin(adata_P.var_names)]
    rawCount_P_hvg = rawCount_P_hvg.groupby(level=0).mean()
    rawCount_P_hvg = rawCount_P_hvg.T
    rawCount_P_hvg = rawCount_P_hvg.reindex(index=adata_P.obs["orginal_cell_name"], columns=adata_P.var_names)
    adata_P.X = rawCount_P_hvg.to_numpy().astype(float)
    return adata_P


def read_trans_Tyser_data(adata):
    rawCount_T = sc.read_h5ad("data/240405_preimplantation_Melania/Tyser2021/raw_count.h5ad")  # raw count, anndata, 1195cell/57490gene
    adata_T = adata[adata.obs['batch'] == 't'].copy()
    adata_T_cellName_list = ["-".join(i.split("-")[:-2]).replace("_", ".") for i in adata_T.obs_names]
    if set(rawCount_T.obs_names) == set(adata_T_cellName_list):
        adata_T.obs["orginal_cell_name"] = adata_T_cellName_list
    else:
        print("***ERROR***")
    rawCount_T.var_names = rawCount_T.var_names.str.split(".").str[0]
    # 合并相同 var_names 并计算平均值
    df = pd.DataFrame(rawCount_T.X, index=rawCount_T.obs_names, columns=rawCount_T.var_names)
    df_avg = df.groupby(level=0, axis=1).mean()

    rawCount_T_avg = sc.AnnData(
        X=df_avg.values,
        obs=rawCount_T.obs,
        var=pd.DataFrame(index=df_avg.columns),
        uns=rawCount_T.uns,
        obsm=rawCount_T.obsm,
        varm=rawCount_T.varm
    )
    rawCount_T_avg = rawCount_T_avg[:, list(adata_T.var_names)]
    rawCount_T_avg = rawCount_T_avg[adata_T.obs["orginal_cell_name"], :]
    rawCount_T_avg = rawCount_T_avg[:, adata_T.var_names]
    adata_T.X = rawCount_T_avg.X
    return adata_T


def read_trans_Liu_data(adata):
    rawCount_L = sc.read_h5ad("data/240405_preimplantation_Melania/L_raw_count/adata_liu.h5")  # not raw count, anndata 989cell/24153gene
    adata_L = adata[adata.obs['batch'] == 'l'].copy()
    adata_L_cellName_list = ["-".join(i.split("-")[:-1]).replace(".", "_") for i in adata_L.obs_names]
    if set(rawCount_L.obs_names) == set(adata_L_cellName_list):
        adata_L.obs["orginal_cell_name"] = adata_L_cellName_list
    else:
        print("***ERROR***")
    rawCount_L.var_names = rawCount_L.var_names.str.split(".").str[0]
    # 合并相同 var_names 并计算平均值
    df = pd.DataFrame(rawCount_L.X, index=rawCount_L.obs_names, columns=rawCount_L.var_names)
    df_avg = df.groupby(level=0, axis=1).mean()

    rawCount_L_avg = sc.AnnData(
        X=df_avg.values,
        obs=rawCount_L.obs,
        var=pd.DataFrame(index=df_avg.columns),
        uns=rawCount_L.uns,
        obsm=rawCount_L.obsm,
        varm=rawCount_L.varm
    )
    rawCount_L_avg = rawCount_L_avg[:, list(adata_L.var_names)]
    rawCount_L_avg = rawCount_L_avg[adata_L.obs["orginal_cell_name"], :]
    rawCount_L_avg = rawCount_L_avg[:, adata_L.var_names]
    adata_L.X = rawCount_L_avg.X
    return adata_L


def read_trans_xiao_data(adata,min_gene_num = 200):
    rawCount_xiao = sc.read_h5ad("data/240405_preimplantation_Melania/xiaoCellCS8/raw_count.h5ad")  # raw count, 25958gene/38562cell
    rawCount_xiao.var_names = rawCount_xiao.var_names.str.split(".").str[0]
    print(rawCount_xiao)
    col_list_filtered = [col for col in adata.var_names if col in rawCount_xiao.var_names]
    rawCount_xiao = rawCount_xiao[:, col_list_filtered]

    # filter low-quality cell
    _shape = rawCount_xiao.shape
    print(f"After filter by hvg gene: (cell, gene){_shape}")
    _new_shape = (0, 0)
      # 2024-09-10 11:41:29 add
    # min_cell_num = 50
    while _new_shape != _shape:  # make sure drop samples and genes
        _shape = rawCount_xiao.shape
        sc.pp.filter_cells(rawCount_xiao, min_genes=min_gene_num)  # drop samples with less than 20 gene expression
        # sc.pp.filter_genes(rawCount_xiao, min_cells=min_cell_num)  # drop genes which none expression in min_cell_num cells
        _new_shape = rawCount_xiao.shape
    print(f"Drop cells with less than {min_gene_num} gene expression, ")
    print("After filter, get cell number: {}, gene number: {}".format(rawCount_xiao.n_obs, rawCount_xiao.n_vars))
    print("the original sc expression anndata should be gene as row, cell as column")
    plot_data_quality(rawCount_xiao)

    # add annotation
    rawCount_xiao.obs['time'] = 18.5
    rawCount_xiao.obs["day"] = "D18.5_xiao"
    rawCount_xiao.obs["cell_id"] = rawCount_xiao.obs_names
    rawCount_xiao.obs["dataset_label"] = "x"
    rawCount_xiao.obs["donor"] = rawCount_xiao.obs["day"]

    rawCount_xiao.obs["cell_type"] = rawCount_xiao.obs["clusters"]
    categories = rawCount_xiao.obs["cell_type"].cat.categories
    new_categories = [f"{cat}_xiao" for cat in categories]
    rawCount_xiao.obs["cell_type"] = rawCount_xiao.obs["cell_type"].cat.rename_categories(new_categories)

    rawCount_xiao.obs["title"] = rawCount_xiao.obs.index
    rawCount_xiao.obs["species"] = "human"
    return rawCount_xiao


if __name__ == '__main__':
    main()
