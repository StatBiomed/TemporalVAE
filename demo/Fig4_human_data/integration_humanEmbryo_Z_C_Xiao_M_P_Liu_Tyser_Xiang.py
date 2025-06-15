# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE
@File    ：integration_humanEmbryo_Z_C_Xiao_M_P_Liu_Tyser_Xiang.py
@Author  ：awa121
@Date    ：2025/3/16 11:32

Dataset	Reference	n_cells	n_embryo or donor	platforms	Condition	time range  location
Xiang19	Xiang et al, Nature 2020	555	42	Smart-seq2	in vitro (3D culcture)	E6, 7, 8, 9, 10, 12 data/human_embryo_preimplantation/Xiang2019
P	Petropoulos et al, Cell 2016	1529	88	Smart-seq2	ex vivo (?), IVF	E3, E4, E5, E6, E7  data/human_embryo_preimplantation/P_raw_count
M	Molè et al, Nat Comm 2021	4820	16	10x Genomics	ex vivo (?), IVF	E5, 6, 7, 9, 11 data/human_embryo_preimplantation/M_raw_count
Z	Zhou et al, Nature 2019	5911	65	STRT-Seq	in vitro, IVF	E6, 8, 10, 12   data/human_embryo_preimplantation/Zhou_raw_count
L	Liu et al, Sci Adv 2022	1719	25	STRT-Seq	in vitro, IVF	E5, 6   data/human_embryo_preimplantation/Liu_raw_count

T	Tyser et al, Nature 2021	1195	1	Smart-seq2	in vivo, medical termination	CS7 data/human_embryo_preimplantation/Tyser
Xiao	Xiao et al, Cell 2024	38562	1	Stereo-seq	in vivo, elective termination	CS8	62 slides, bin50=25um data/human_embryo_preimplantation/XiaoCS8
Cui	Cui et al, Nat CB 2025	28804	1	Stereo-seq	in vivo, elective termination	CS7	82 slides, bin50=25um, NEW***   data/human_embryo_preimplantation/Cui_raw_count 'Spatial transcriptomic characterization of a Carnegie stage 7 human embryo'

Xiang19 555 E6,7,8,9,10,12,13.5,14 in vitro
Lv2019

integration results: {'Z': 5911, 'C': 5082, 'Xiao': 4938, 'M': 4820, 'P': 1529, 'L': 989, 'T': 1195,  'Xiang': 555}
reference: z, c, xiao, m, p, l, 6 datasets, in total 23269 cells.
"""
import os
import sys

if os.getcwd().split("/")[-1] != "TemporalVAE":
    os.chdir("../..")
sys.path.append(os.getcwd())
import pandas as pd
import scanpy as sc
from collections import Counter
import anndata as ad
from TemporalVAE.utils import plot_data_quality


def main():
    # 2025-04-01 10:10:34 Counter({'z': 5911, 'm': 4820, 'p': 1529, 't': 1195, 'l': 989})
    # 2025-04-20 16:54:49 add p annotation, so remove some p cell
    # {'Z': 5911, 'C': 5082, 'Xiao': 4938, 'M': 4820, 'P': 1529, 'T': 1195, 'L': 989, 'Xiang': 555}
    # 2025-04-21 20:49:25
    # temp=sc.read_h5ad(f"data/human_embryo_preimplantation/integration_8dataset/rawCount_Z_C_Xiao_M_P_Liu_Tyser_Xiang.h5ad")
    # temp.obs.to_csv("data/human_embryo_preimplantation/integration_8dataset/annotation_Z_C_Xiao_M_P_Liu_Tyser_Xiang.csv")
    # for dataset in ['C', 'Xiao', 'T', 'L', 'Z', 'M', 'P', 'Xiang']:
    #     _ad=temp[temp.obs['dataset_label']==dataset]
    #     print(dataset,Counter(_ad.obs["cell_type"]))
    adata_Melania = read_trans_Melania(file_path="data/human_embryo_preimplantation/Melania_5datasets")
    geneList_Melania=adata_Melania.var_names
    # new ones
    adata_xiang19 = read_trans_Xiang19(file_path="data/human_embryo_preimplantation/Xiang2019/",
                                       gene_list_ref=geneList_Melania)
    # ---P
    adata_P = read_trans_Petropoulos_data(adata_Melania.copy())
    # ---T
    adata_T = read_trans_Tyser_data(adata_Melania.copy())
    # ---L
    adata_Liu = read_trans_Liu_data(adata_Melania.copy())
    # ---M
    adata_M = read_trans_Mole_data(adata_Melania.copy())
    # ----Z
    adata_Z = read_trans_Zhou_data(adata_Melania.copy())


    adata_cui = read_trans_Cui(file_path="data/human_embryo_preimplantation/Cui_raw_count/", gene_list_ref=geneList_Melania)
    # adata_Lv = read_trans_Lv_data(file_path="data/human_embryo_preimplantation/Lv2019/", gene_list_ref=geneList_Melania)
    adata_xiao = read_trans_xiao_data(file_path="data/human_embryo_preimplantation/XiaoCS8", gene_list_ref=geneList_Melania)


    # old datasets

    # gene list reset as overlap set
    col_list_filtered = [col for col in adata_Melania.var_names if (col in adata_cui.var_names) & (col in adata_xiao.var_names) & (col in adata_xiang19.var_names)]
    # col_list_filtered = [col for col in adata_Melania.var_names if (col in adata_Lv.var_names) &(col in adata_cui.var_names) & (col in adata_xiao.var_names) & (col in adata_xiang19.var_names)]
    print(f"reserved {len(col_list_filtered)} genes number: {len(col_list_filtered)}")
    adata_mulitDatasets_Final = ad.concat([adata_cui[:, col_list_filtered],
                           adata_xiao[:, col_list_filtered],
                           adata_T[:, col_list_filtered],
                           adata_Liu[:, col_list_filtered],
                           adata_Z[:, col_list_filtered],
                           adata_M[:, col_list_filtered],
                           adata_P[:, col_list_filtered],
                           # adata_Lv[:, col_list_filtered],
                           adata_xiang19[:, col_list_filtered]], axis=0)
    adata_mulitDatasets_Final.obs['dataset_label'] = adata_mulitDatasets_Final.obs['dataset_label'].str.capitalize()
    print(Counter(adata_mulitDatasets_Final.obs['cell_type']))



    adata_mulitDatasets_Final.obs["cell_type"]= cellType_reAnnotation(adata_mulitDatasets_Final.obs["cell_type"])
    # _adata=adata_mulitDatasets_Final[adata_mulitDatasets_Final.obs["cell_type"]=='MIX'].copy()
    # filter low-quality cell
    _shape = adata_mulitDatasets_Final.shape
    print(f"before filter by hvg gene: (cell, gene){_shape}")
    print(f"\tDataset:{Counter(adata_mulitDatasets_Final.obs['dataset_label'])}\n"
          f"\tCell Type number: {len(Counter(adata_mulitDatasets_Final.obs['cell_type']))}\n"
          f"\tCell Type detail: {Counter(adata_mulitDatasets_Final.obs['cell_type'])}.")
    _new_shape = (0, 0)
    while _new_shape != _shape:  # make sure drop samples and genes
        _shape = adata_mulitDatasets_Final.shape
        sc.pp.filter_cells(adata_mulitDatasets_Final, min_genes=50)  # drop samples with less than 20 gene expression
        sc.pp.filter_genes(adata_mulitDatasets_Final, min_cells=50)  # drop genes which none expression in min_cell_num cells
        _new_shape = adata_mulitDatasets_Final.shape
    print(f"Drop cells with less than 50 gene expression, ")
    print(f"After filter, get (cell, gene): {adata_mulitDatasets_Final.shape}")
    print(f"\tDataset:{Counter(adata_mulitDatasets_Final.obs['dataset_label'])}\n"
          f"\tCell Type number: {len(Counter(adata_mulitDatasets_Final.obs['cell_type']))}\n"
          f"\tCell Type detail: {Counter(adata_mulitDatasets_Final.obs['cell_type'])}.")

    # save integration results
    adata_mulitDatasets_Final.write_h5ad(f"data/human_embryo_preimplantation/integration_8dataset/rawCount_Z_C_Xiao_M_P_Liu_Tyser_Xiang.h5ad")
    print(f"save h5ad file as data/human_embryo_preimplantation/integration_8dataset/rawCount_Z_C_Xiao_M_P_Liu_Tyser_Xiang.h5ad")


    return
def read_trans_Lv_data(file_path,gene_list_ref=None, min_gene_num=200,save_file_bool=False):
    # rawCount_adata = sc.read_h5ad("data/human_embryo_preimplantation/Lv2019/adata_hvg.h5ad")
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
    days_to_keep = ["day6", "day7", "day8", "day9", "day10", "day14"]
    # days_to_keep = ["day6", "day7", "day8", "day9", "day10"]
    # adata.obs['development_day'] = adata.obs['development_day'].replace('Endometrial', 'day0')

    # 使用布尔索引来筛选 those observations in anndata.obs that meet the condition
    rawCount_adata = adata[adata.obs["development_day"].isin(days_to_keep)].copy()



    # raw count, 56924gene/593cell
    print(rawCount_adata)
    col_list_filtered = [col for col in gene_list_ref if col in rawCount_adata.var_names]
    rawCount_adata = rawCount_adata[:, col_list_filtered]

    # filter low-quality cell
    _shape = rawCount_adata.shape
    print(f"before filter : (cell, gene){_shape}")
    _new_shape = (0, 0)
    # 2024-09-10 11:41:29 add
    # min_cell_num = 50
    while _new_shape != _shape:  # make sure drop samples and genes
        _shape = rawCount_adata.shape
        sc.pp.filter_cells(rawCount_adata, min_genes=min_gene_num)  # drop samples with less than 20 gene expression
        # sc.pp.filter_genes(rawCount_adata, min_cells=min_cell_num)  # drop genes which none expression in min_cell_num cells
        _new_shape = rawCount_adata.shape
    print(f"Drop cells with less than {min_gene_num} gene expression, ")
    print(f"After filter, get cell number: {rawCount_adata.n_obs}, gene number: {rawCount_adata.n_vars}")

    # add annotation
    rawCount_adata.obs["time"] = rawCount_adata.obs["development_day"].str.replace("day", "").astype(int)
    rawCount_adata.obs["day"] =rawCount_adata.obs["time"].apply(lambda x: "D" + str(x) + "_lv")
    rawCount_adata.obs["cell_id"] = rawCount_adata.obs_names
    rawCount_adata.obs["dataset_label"] = "Lv"
    rawCount_adata.obs["donor"] = rawCount_adata.obs["day"]

    rawCount_adata.obs["cell_type"] = rawCount_adata.obs["Stage"]
    rawCount_adata.obs["cell_type"]=    rawCount_adata.obs["cell_type"].astype("category")
    categories = rawCount_adata.obs["cell_type"].cat.categories
    new_categories = [f"{cat}_lv" for cat in categories]
    rawCount_adata.obs["cell_type"] = rawCount_adata.obs["cell_type"].cat.rename_categories(new_categories)

    rawCount_adata.obs["title"] = rawCount_adata.obs.index
    rawCount_adata.obs["species"] = "human"
    if save_file_bool:
        rawCount_adata.write_h5ad(f"{file_path}/adata_hvg.h5ad")
        sc_expression_df = pd.DataFrame(data=rawCount_adata.X.T,
                                        columns=rawCount_adata.obs.index,
                                        index=rawCount_adata.var.index)
        sc_expression_df.to_csv(f"{file_path}/data_count_hvg.csv", sep="\t")
        rawCount_adata.obs.to_csv(f"{file_path}/cell_with_time.csv", sep="\t")
        rawCount_adata.var.to_csv(f"{file_path}/gene_info.csv", sep="\t")
    return rawCount_adata

def cellType_reAnnotation(orginal_cell_type,):
    temp = orginal_cell_type.apply(lambda x: x.split("_")[0].replace("S1.", "").replace("S2.", "").replace("S3.", ""))
    print(sorted(Counter(temp).items(), key=lambda x: x[0]))

    temp = temp.replace("AM", "Amnion")
    temp = temp.replace("CTBs", "Cytotrophoblast")
    temp = temp.replace("Cytotrophoblasts", "Cytotrophoblast")
    temp = temp.replace("EPI", "Epiblast")
    temp = temp.replace("Epi", "Epiblast")
    temp = temp.replace("Epiblasts", "Epiblast")
    temp = temp.replace("Ery", "Erythroblast ")
    temp = temp.replace("Erythroblasts", "Erythroblast")
    temp = temp.replace("Endo", "Endoderm")
    # temp = temp.apply(lambda x: re.sub(r'^EPI$', 'Epiblast', x, flags=re.IGNORECASE))

    temp = temp.replace('Epiblasts', 'Epiblast')
    temp = temp.replace('epiblast', 'Epiblast')
    temp = temp.replace('EVTs', 'Extravillous Cytotrophoblast')
    temp = temp.replace("Hypoblasts", "Hypoblast")
    temp = temp.replace("HEP", "Haemato-endothelial Progenitor")
    temp = temp.replace("ICM", "Inner cell mass")
    temp = temp.replace("Meso", "Mesoderm")
    temp = temp.replace("Noto", "Notochord")
    temp = temp.replace("Hemogenic Endothelial Progenitors", "Haemato-endothelial Progenitor")
    temp = temp.replace("PE", "Primitive Endoderm")
    temp = temp.replace("primitive endoderm", "Primitive Endoderm")
    temp = temp.replace("PTE", "Polar TE")
    temp = temp.replace("PS", "Primitive Streak")
    temp = temp.replace("Syncytiotrophoblasts", "Syncytiotrophoblast")
    temp = temp.replace("STBs", "Syncytiotrophoblast")
    temp = temp.replace("TE", "Trophectoderm")
    temp = temp.replace("trophectoderm", "Trophectoderm")
    temp = temp.replace("Visceral.Endo", "Visceral Endoderm")
    temp = temp.replace("coISK", "Cocultured Ishikawa cell")
    temp = temp.replace("blastocyst", "Blastocyst")
    temp = temp.replace("implantation", "Implantation")
    temp = temp.replace("post-implantation", "Post-implantation")
    temp = temp.replace("not applicable", "Not applicable")

    print(sorted(Counter(temp).keys(), key=lambda x: x[0]))
    print(Counter(temp))

    # amnion extra-embryonic mesoderm (AM.EXE.Meso),
    # definitive/visceral endoderm (DE/VE)
    # embryonic/extra-embryonic mesoderm (Em/EXE.Meso)
    # epiblast/ectoderm (Epi/Ecto)
    # extra-embryonic mesoderm (ExE Mesoderm)
    #  gastrulating cell/primitive streak (Gast/PS)
    #  mural TE (MTE)
    #  primitive streak anlage epiblast (PSA-EPI)
    #  yolk sac extra-embryonic embryonic/mesoderm (YS.EXE.Em/ExE.Meso),
    #  yolk sac endoderm (YS.Endo),
    return temp
def read_trans_Xiang19(file_path,gene_list_ref=None,min_gene_num=50,save_file_bool=False):
    from TemporalVAE.utils import series_matrix2csv

    raw_count = pd.read_csv(f"{file_path}/GSE136447_555sample_gene_count_matrix.txt", sep="\t", header=0, index_col=0)
    raw_count=raw_count.T
    raw_count.columns = raw_count.columns.str.split("|").str[-1]
    raw_count.columns = raw_count.columns.str.split(".").str[0]
    if len(set(raw_count.columns.to_list())) != raw_count.shape[1]:
        raw_count = raw_count.groupby(level=0, axis=1).mean()
    cell_info1 = f"{file_path}/GSE136447-GPL20795_series_matrix.txt"
    cell_info1 = series_matrix2csv(cell_info1)

    cell_info2 = f"{file_path}/GSE136447-GPL23227_series_matrix.txt"
    cell_info2 = series_matrix2csv(cell_info2)

    cell_info = pd.concat([cell_info1[1], cell_info2[1]], axis=0)

    # cell_info["development_day"] = cell_info.index.map(extract_number)
    # cell_info["time"]=cell_info["development_day"]
    cell_info["time"] = cell_info["characteristics_ch1"].apply(lambda x: eval(x.replace(" ", "").replace('age:embryoinvitroday', "").replace("13.5","14")))
    cell_info["day"] = cell_info["time"].apply(lambda x: "D" + str(x) + "_xiang19")
    cell_info['cell_id'] = cell_info['title'].apply(lambda x: x.replace('Embryo_', ""))
    cell_info["dataset_label"] = "Xiang"
    cell_info["donor"] = cell_info["day"]
    cell_info["cell_type"] = cell_info["characteristics_ch1_2"].apply(lambda x: x.replace(" ", "").replace('celltype:', ""))

    cell_info["Stage"] = cell_info["cell_type"]
    # cell_info['title'] = cell_info.index
    cell_info['species'] = "human"
    cell_info = cell_info.set_index("cell_id")
    # "day","Stage","n_genes","predicted_time"

    if set(cell_info.index) == set(raw_count.index):
        print("sample id is complete same.")

        cell_info = cell_info.loc[raw_count.index]
    else:
        exit(0)
    raw_count_xiang19 = ad.AnnData(X=raw_count.values, obs=cell_info, var=pd.DataFrame(index=raw_count.columns))
    print("Import data, cell number: {}, gene number: {}".format(raw_count_xiang19.n_obs, raw_count_xiang19.n_vars))
    print("Cell number: {}".format(raw_count_xiang19.n_obs))
    print("Gene number: {}".format(raw_count_xiang19.n_vars))
    print("Annotation information of data includes: {}".format(raw_count_xiang19.obs_keys()))  # 胞注釋信息的keys
    print("Cell id first 5: {}".format(raw_count_xiang19.obs_names[:5]))  # 返回胞ID 数据类型是object
    print("Gene id first 5: {}".format(raw_count_xiang19.var_names.to_list()[:5]))  # 返回基因数据类型是list adata.obs.head()# 査看前5行的数据
    print("Gene id first 5: {}".format(raw_count_xiang19.var.index.to_list()[:5]))

    # min_cell_num = 50
    sc.pp.filter_cells(raw_count_xiang19, min_genes=min_gene_num)  # drop samples with less than 20 gene expression

    # sc.pp.filter_genes(adata, min_cells=min_cell_num)

    # hvg_cellRanger_list = calHVG(adata.copy(), gene_num=hvg_num, method="cell_ranger")
    # hvg_seurat_list = calHVG(adata.copy(), gene_num=hvg_num, method="seurat")
    # hvg_seurat_v3_list = calHVG(adata.copy(), gene_num=hvg_num, method="seurat_v3")

    # draw_venn({"cell ranger": hvg_cellRanger_list, "seurat": hvg_seurat_list, "seurat v3": hvg_seurat_v3_list})
    # print(f"concat all hvg calculated")
    # import itertools
    # combined_hvg_list = list(set(itertools.chain(hvg_cellRanger_list, hvg_seurat_list, hvg_seurat_v3_list)))
    col_list_filtered = [col for col in gene_list_ref if col in raw_count_xiang19.var_names]
    raw_count_xiang19 = raw_count_xiang19[:, col_list_filtered]

    _shape = raw_count_xiang19.shape
    print(f"After filter by hvg gene: (cell, gene){_shape}")
    _new_shape = (0, 0)
    # 2024-09-10 11:41:29 add
    # min_cell_num = 50
    while _new_shape != _shape:  # make sure drop samples and genes
        _shape = raw_count_xiang19.shape
        sc.pp.filter_cells(raw_count_xiang19, min_genes=min_gene_num)  # drop samples with less than 20 gene expression
        # sc.pp.filter_genes(rawCount_cui, min_cells=min_cell_num)  # drop genes which none expression in min_cell_num cells
        _new_shape = raw_count_xiang19.shape
    print(f"Drop cells with less than {min_gene_num} gene expression, ")
    print("After filter, get cell number: {}, gene number: {}".format(raw_count_xiang19.n_obs, raw_count_xiang19.n_vars))
    print("the original sc expression anndata should be gene as row, cell as column")

    return raw_count_xiang19


# 定义一个函数，用于从字符串中提取数字
def extract_number(s):
    import re
    match = re.search(r'D(\d+)', s)
    if match:
        return int(match.group(1))
    return None


def read_trans_Cui(file_path,gene_list_ref=None, min_gene_num=200,save_file_bool=False):
    # rawCount_cui = pd.read_csv(f"data/human_embryo_preimplantation/Cui_raw_count/data_count.csv", sep="\t", header=0, index_col=0)
    rawCount_cui = ad.read_csv(f"{file_path}/data_count.csv", delimiter='\t')  # raw count, 25833gene/28804cell
    rawCount_cui = rawCount_cui.T
    cell_info_cui = pd.read_csv(f"{file_path}/cell_info.csv", sep="\t", header=0, index_col=0)
    rawCount_cui.obs = cell_info_cui
    rawCount_cui.var_names = rawCount_cui.var_names.str.split(".").str[0]
    print(rawCount_cui)
    col_list_filtered = [col for col in gene_list_ref if col in rawCount_cui.var_names]
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


def read_trans_Melania(file_path,save_file_bool=False):

    adata = sc.read_h5ad(f"{file_path}/adata_human_preimplantation_for_degong.h5")
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

    #
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

    if save_file_bool:
        print("Save preprocess Melania file")
        adata.write_h5ad(f"{file_path}/normalized_exp_gene{adata.shape[1]}.h5ad")
        raw_data = pd.DataFrame(data=adata.raw.X,
                                index=adata.raw.obs_names,
                                columns=adata.raw.var_names)
        raw_data = raw_data[list(adata.var_names)].T
        raw_data.to_csv(f"{file_path}/data_count_hvg_raw.csv", sep="\t")
        sc_expression_df = pd.DataFrame(data=adata.X.T.toarray(),
                                        columns=adata.obs.index,
                                        index=adata.var.index)
        sc_expression_df.to_csv(f"{file_path}/data_count_hvg.csv", sep="\t")
        adata.obs.to_csv(f"{file_path}/cell_with_time.csv", sep="\t")
        adata.var.to_csv(f"{file_path}/gene_info.csv", sep="\t")
    print(adata)
    return adata


def read_trans_Zhou_data(adata,save_file_bool=False):
    rawCount_Z = pd.read_csv("data/human_embryo_preimplantation/Zhou_raw_count/GSE109555_All_Embryo_TPM.txt",
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
    #  remove "MIX" cell
    # adata_Z = adata_Z[adata_Z.obs['cell_type'] != "MIX_z", :]
    return adata_Z


def read_trans_Mole_data(adata,save_file_bool=False):
    rawCount_M = pd.read_csv("data/human_embryo_preimplantation/M_raw_count/data_count.csv",
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


def read_trans_Petropoulos_data(adata,save_file_bool=False):
    rawCount_P = pd.read_csv("data/human_embryo_preimplantation/P_raw_count/counts.txt",
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
    print(Counter(adata_P.obs["cell_type"]))
    cell_type_mapping_Melaina=dict(zip(adata_P.obs["title"], adata_P.obs["cell_type"]))
    # align cell annotation
    # P dataset annotation is from data/human_embryo_preimplantation/cell_annotation_fromZhaoNature.xls
    all_sheets = pd.read_excel("data/human_embryo_preimplantation/cell_annotation_fromZhaoNature.xls", sheet_name=None)
    second_sheet_name = list(all_sheets.keys())[1]  # 第二个 Sheet 的名称
    df = all_sheets[second_sheet_name]
    P_df = df[df['paper'] == 'Petropoulos et al']
    P_df["title"]=P_df["cell"]+"-p-b4"
    # adata_P=adata_P[P_df["title"]]
    # print(Counter(adata_P.obs["cell_type"]))
    # temp=adata_P[adata_P.obs["cell_type"]=="not applicable_p"]
    # temp_df=P_df[P_df["title"].isin(temp.obs["title"])]
    # Counter(temp_df["annotation"])
    # Counter(P_df["annotation"])
    print(Counter(P_df["annotation"]))
    cell_type_mapping_nature = dict(zip(P_df["title"], P_df["annotation"]))
    merged_mapping = {**cell_type_mapping_Melaina, **cell_type_mapping_nature}
    # adata_P.obs["cell_type"] = adata_P.obs.index.map(cell_type_mapping)
    adata_P.obs["cell_type"] = adata_P.obs.index.map(merged_mapping)

    # adata_P = adata_P[adata_P.obs['cell_type'] != "EPI.PrE.INT", :]
    # adata_P = adata_P[adata_P.obs['cell_type'] != "not applicable_p", :]
    print(Counter(adata_P.obs["cell_type"]))
    return adata_P


def read_trans_Tyser_data(adata,save_file_bool=False):
    rawCount_T = sc.read_h5ad("data/human_embryo_preimplantation/Tyser2021/raw_count.h5ad")  # raw count, anndata, 1195cell/57490gene
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


def read_trans_Liu_data(adata,save_file_bool=False):
    rawCount_L = sc.read_h5ad("data/human_embryo_preimplantation/Liu_raw_count/adata_liu.h5")  # not raw count, anndata 989cell/24153gene
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
    adata_L.obs["dataset_label"]="L"
    return adata_L


def read_trans_xiao_data(file_path,gene_list_ref=None, min_gene_num=200,save_file_bool=False):
    rawCount_xiao = sc.read_h5ad(f"{file_path}/raw_count.h5ad")  # raw count, 25958gene/38562cell
    rawCount_xiao.var_names = rawCount_xiao.var_names.str.split(".").str[0]
    print(rawCount_xiao)
    col_list_filtered = [col for col in gene_list_ref if col in rawCount_xiao.var_names]
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
    rawCount_xiao.obs["dataset_label"] = "xiao"
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
