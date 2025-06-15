# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE
@File    ：preprocess_data_Cyno.py
@IDE     ：PyCharm
@Author  ：awa121
@Date    ：2024/9/12 23:14

raw .rds data is download from https://drive.google.com/file/d/1C8MJ4u8djLqs212CllT0cH0aDgn6OWAV/view

R language to read .rds file:
file_path="data/240910_marmoset_nature2022/"
Cyno = readRDS(paste(file_path,"Cyno_data.rds",sep = ""))
Cyno <- UpdateSeuratObject(Cyno) # 30792 features across 1453 samples

Cyno_data_count = Cyno[['RNA']]@counts
Cyno_data_count <- as.data.frame(Cyno_data_count)
Cyno_cellInfo=Cyno@meta.data
table(Idents(Cyno))
Cyno_cellInfo$"celltype_time"=Idents(Cyno)
write.table(
  Cyno_data_count,
  paste(file_path, "/Cyno_rawCounts/data_count.csv", sep = ""),
  row.names = TRUE,
  col.names = TRUE,
  sep = "\t"
)
write.table(
  Cyno_cellInfo,
  paste(file_path, "/Cyno_rawCounts/cell_info.csv", sep = ""),
  row.names = TRUE,
  col.names = TRUE,
  sep = "\t"
)
"""
import os, sys

if os.getcwd().split("/")[-1] != "TemporalVAE":
    os.chdir("../..")
sys.path.append(os.getcwd())
import pandas as pd
import gc
from collections import Counter
import anndata as ad


def main():
    save_path = "data/240910_marmoset_nature2022/Cyno_rawCounts/"
    Cyno_rawCounts = pd.read_csv(f"{save_path}/data_count.csv", index_col=0,sep="\t")
    Cyno_cell_info = pd.read_csv(f"{save_path}/cell_info.csv",index_col=0,sep="\t")
    Cyno_cell_info["Stage"] = Cyno_cell_info['celltype_time'].apply(lambda x: x.split('_')[1])
    Cyno_cell_info["celltype"] = Cyno_cell_info['celltype_time'].apply(lambda x: x.split('_')[0])
    Cyno_cell_info["cell_name"] = Cyno_cell_info.index
    Cyno_cell_info["species"] = "cynomolgus"
    print(Counter(Cyno_cell_info["Stage"]))
    # CS7 to 17.5, CS6 to 14, CS5 to 12, CS3 to 5, CS2 to 3, CS1 to 1
    stage_mapping = {'CS7': 17.5,
                     'CS6': 14,
                     'CS5': 12,
                     'CS3': 5,
                     'CS2': 3,
                     'CS1': 1}
    Cyno_cell_info['time'] = Cyno_cell_info['Stage'].map(stage_mapping)
    print(Counter(Cyno_cell_info["time"]))

    Cyno_rawCounts_adata = ad.AnnData(X=Cyno_rawCounts.T, obs=Cyno_cell_info)

    from TemporalVAE.utils import plot_data_quality
    plot_data_quality(Cyno_rawCounts_adata)

    Cyno_rawCounts_adata.write_h5ad(f"{save_path}/RawCounts.h5ad")

    human_gene = pd.read_csv("data//human_embryo_preimplantation/Melania_5datasets/gene_info.csv", sep="\t", index_col=0)
    print(f"Intersection Genes in marmoset and Melania is "
          f"{len(set(human_gene.index) & set(Cyno_rawCounts_adata.var_names))}")  # 1556 genes
    sc_expression_df = pd.DataFrame(data=Cyno_rawCounts_adata.X.T,
                                    columns=Cyno_rawCounts_adata.obs.index,
                                    index=Cyno_rawCounts_adata.var.index)

    sc_expression_df.to_csv(f"{save_path}/data_count_hvg.csv", sep="\t")
    cell_info = Cyno_rawCounts_adata.obs
    cell_info.to_csv(f"{save_path}/cell_with_time.csv", sep="\t")
    gene_info = Cyno_rawCounts_adata.var
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
