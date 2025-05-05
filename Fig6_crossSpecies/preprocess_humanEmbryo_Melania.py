# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：preprocess_humanEmbryo_Melania.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/8/19 15:09 
"""
import scanpy as sc
import pandas as pd
import anndata as ad
from collections import Counter
from utils.utils_DandanProject import calHVG_adata as calHVG
from utils.utils_DandanProject import series_matrix2csv
from utils.utils_Dandan_plot import draw_venn
import h5py


def main():
    select_hvg_bool = False  # 2024-04-03 17:51:44 add here
    file_path = "../data//human_embryo_preimplantation/Melania_5datasets/"
    adata = sc.read_h5ad(f"{file_path}/adata_human_preimplantation_for_degong.h5")
    non_nan_attr_list = []
    for _f in list(adata.obs.columns):
        if len(Counter(adata.obs[_f])) < 100:
            print(f"***{_f}\t{Counter(adata.obs[_f])}")
            if not any(pd.isna(key) for key in Counter(adata.obs[_f]).keys()):
                non_nan_attr_list.append(_f)
    for _f in non_nan_attr_list:
        print(f"***{_f}\t{Counter(adata.obs[_f])}")

    # "batch", "lineage"

    # tyser_df = adata.obs.loc[adata.obs["batch"] == "t"]
    # petropoulos_df = adata.obs.loc[adata.obs["batch"] == "p"]
    # petropoulos_one = adata[adata.obs_names == "E5.5.101-p-b4", :]
    # import numpy as np

    # data_x=adata.X.toarray()
    # data_raw = np.array(adata.raw.X)
    # data_raw_recover = np.expm1(data_raw)
    # hvg_genes = list(adata.var_names)
    # adata_p=adata[adata.obs["batch"]=="p"]
    # adata_m=adata[adata.obs["batch"]=="m"]

    # p_cell_names =list(adata_p.obs_names)
    # adata_raw_pd=pd.DataFrame(data=adata.raw.X,index=adata.raw.obs_names,columns=adata.raw.var_names)
    # adata_raw_pd=adata_raw_pd.loc[p_cell_names]
    # adata_raw_pd=adata_raw_pd[hvg_genes]
    #
    # P_raw_count=pd.read_csv(f"{file_path}/P_raw_count.txt",sep="\t",index_col=0)
    # P_raw_count=P_raw_count.loc[hvg_genes].T
    #
    # recover_log=pd.DataFrame(data=np.expm1(adata_raw_pd))
    # ------------ print and plot for show the structure of dataset
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
    # sc.pl.scatter(adata, x="total_counts", y="pct_counts_mt")
    # sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts")

    # ------------
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

    adata.write_h5ad(f"{file_path}/normalized_exp_gene{adata.shape[1]}.h5ad")

    # 2024-08-20 18:22:19 change data_count from normalized version to raw count version.
    raw_data=pd.DataFrame(data=adata.raw.X,index=adata.raw.obs_names,columns=adata.raw.var_names)
    raw_data=raw_data[list(adata.var_names)].T
    raw_data.to_csv(f"{file_path}/data_count_hvg_raw.csv", sep="\t")
    sc_expression_df = pd.DataFrame(data=adata.X.T.toarray(), columns=adata.obs.index, index=adata.var.index)
    sc_expression_df.to_csv(f"{file_path}/data_count_hvg.csv", sep="\t")

    cell_info = adata.obs
    cell_info["day"] = adata.obs["day_cat"]
    cell_info["cell_id"] = adata.obs.index
    cell_info["dataset_label"] = adata.obs["batch"]
    cell_info["donor"] = adata.obs["day_cat"]
    cell_info["cell_type"] = cell_info["lineage"]
    cell_info["title"] = cell_info.index
    cell_info["species"]="human"
    cell_info = cell_info.loc[sc_expression_df.columns]
    cell_info.to_csv(f"{file_path}/cell_with_time.csv", sep="\t")
    gene_info = adata.var
    gene_info.to_csv(f"{file_path}/gene_info.csv", sep="\t")

    print("Finish all")


if __name__ == '__main__':
    main()
