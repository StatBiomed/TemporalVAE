# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：preprocess_data_marmoset_inVivo.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/9/12 23:14 
"""
import os, sys

if os.getcwd().split("/")[-1] != "TemporalVAE":
    os.chdir("..")
sys.path.append(os.getcwd())
import pandas as pd
import gc
from collections import Counter
import anndata as ad


def main():
    # data_4dataset=pd.read_csv("data/240910_marmoset_nature2022/Four_dataset_allTissues.csv",index_col=0)
    save_path = "data/240910_marmoset_nature2022/inVivo_rawCounts/"
    inVivo_rawCounts = pd.read_csv("data/240910_marmoset_nature2022/RawCounts.csv", index_col=0)
    _temp = pd.DataFrame(inVivo_rawCounts.iloc[:2]).T
    _temp["Stage"] = _temp['Lineage'].apply(lambda x: x.split('_')[1])
    _temp["Lab"] = _temp['Lineage'].apply(lambda x: x.split('_')[0])
    _temp["cell_name"] = _temp.index.map(lambda x: x.split('.s_')[0])
    _temp["species"] = "marmoset"
    # CS7 to 17.5, CS6 to 14, CS5 to 12, CS3 to 5, CS2 to 3, CS1 to 1
    stage_mapping = {'CS7': 17.5,
                     'CS6': 14,
                     'CS5': 12,
                     'CS3': 5,
                     'CS2': 3,
                     'CS1': 1}
    _temp['time'] = _temp['Stage'].map(stage_mapping)
    Counter(_temp["Stage"])

    inVivo_rawCounts_adata = ad.AnnData(X=inVivo_rawCounts.iloc[2:].astype(int).T, obs=_temp)

    from utils.utils_Dandan_plot import plot_data_quality
    plot_data_quality(inVivo_rawCounts_adata)

    inVivo_rawCounts_adata.write_h5ad(f"{save_path}/RawCounts.h5ad")

    human_gene = pd.read_csv("data/240405_preimplantation_Melania/Melania_5datasets/gene_info.csv", sep="\t", index_col=0)
    print(f"Intersection Genes in marmoset and Melania is "
          f"{len(set(human_gene.index) & set(inVivo_rawCounts_adata.var_names))}")  # 1556 genes
    sc_expression_df = pd.DataFrame(data=inVivo_rawCounts_adata.X.T,
                                    columns=inVivo_rawCounts_adata.obs.index,
                                    index=inVivo_rawCounts_adata.var.index)

    sc_expression_df.to_csv(f"{save_path}/data_count_hvg.csv", sep="\t")
    cell_info = inVivo_rawCounts_adata.obs
    cell_info.to_csv(f"{save_path}/cell_with_time.csv", sep="\t")
    gene_info = inVivo_rawCounts_adata.var
    gene_info.to_csv(f"{save_path}/gene_info.csv", sep="\t")
    # len(set(_temp["cell_name"]))
    # Counter(inVivo_rawCounts_adata.obs["Stage"])

    # inVivo_cellInfo_pd=pd.read_csv("data/240910_marmoset_nature2022/E-MTAB-9367.sdrf.txt",sep="\t")
    # len(set(inVivo_rawCounts_adata.obs["cell_name"])-set(inVivo_cellInfo_pd["Source Name"]))
    # selected_row = inVivo_rawCounts_adata.obs[inVivo_rawCounts_adata.obs["cell_name"] == "CME25A7619362"]
    # selected_row = inVivo_rawCounts_adata.obs[inVivo_rawCounts_adata.obs["cell_name"] == "SLX.20308.i710_i511.H3FT3DRXY"]
    # Counter(inVivo_cellInfo_pd["Characteristics[embryo stage]"])

    # len(set(inVivo_rawCounts_adata.obs_names)&set(inVivo_cellInfo_pd['Factor Value[single cell identifier]']))
    # for _c in inVivo_cellInfo_pd.columns:
    #     print(f"{_c}: {len(set(inVivo_rawCounts_adata.obs['cell_name']) & set(inVivo_cellInfo_pd[_c]))}")
    # len(set(inVivo_rawCounts_adata.obs_names))
    # inVitro_rawCounts=pd.read_csv("data/240910_marmoset_nature2022/RawCountsInVitro.csv",index_col=0)
    # inVitro_rawCounts_adata=ad.AnnData(X=inVitro_rawCounts.iloc[1:].astype(int).T,obs=pd.DataFrame(inVitro_rawCounts.iloc[0]))

    gc.collect()
    print(f"Files save at {save_path}")

    return


if __name__ == '__main__':
    main()
