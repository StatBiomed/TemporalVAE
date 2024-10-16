# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：plot_perturbution_results.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/10/1 20:30 
"""
import gc
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import gseapy

if os.getcwd().split("/")[-1] != "TemporalVAE":
    os.chdir("..")
sys.path.append(os.getcwd())
import pandas as pd
from utils.utils_DandanProject import denormalize, get_top_gene_perturb_data
from utils.utils_Dandan_plot import plt_enrichmentResult, plt_violin_topGene_inWhole_stage, plt_perturb_xTime_yDetT
import numpy as np


def main():
    species = "human"  # human from Melania
    file_path = "results/Fig7_TemporalVAE_identify_keyGenes_humanMelania_240902/240405_preimplantation_Melania/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch100_minGeneNum50/"
    calculate(species, file_path)

    species = "mouse"
    file_path = "results/Fig7_TemporalVAE_identify_keyGenes_mouseAtlas_240901/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch5_minGeneNum100/"
    calculate(species, file_path)


def calculate(species, file_path):
    print(f"For {species} use perturb result from {file_path}")
    print(f"All library names for {species}: {gseapy.get_library_name(organism=species)}")
    # ----
    bin_dic = {"mouse": [0, 10.25, 14, 20], 'human': [0, 7, 14, 20]}  # mark one
    perturb_show_gene_num = 2
    env_gene_num = 30
    gene_set_dic = {"mouse": ['GO_Biological_Process_2023', 'GO_Molecular_Function_2018',
                              'KEGG_2019_Mouse',
                              'Reactome_2022',
                              'GO_Cellular_Component_2023',
                              ],  # 'WikiPathways_2024_Mouse','MSigDB_Hallmark_2020','Aging_Perturbations_from_GEO_down',
                    "human": ['MSigDB_Hallmark_2020',
                              'KEGG_2021_Human',
                              'Reactome_2022',
                              'GO_Molecular_Function_2018',
                              'GO_Biological_Process_2023',
                              'GO_Cellular_Component_2023',
                              ]}  # # 'MSigDB_Oncogenic_Signatures','MSigDB_Computational','GWAS_Catalog_2023','HDSigDB_Human_2021', 'Genome_Browser_PWMs','Enrichr_Libraries_Most_Popular_Genes', 'Enrichr_Submissions_TF-Gene_Coocurrence','Enrichr_Users_Contributed_Lists_2020', 'WikiPathways_2024_Human','Aging_Perturbations_from_GEO_down'
    top_metric = "vote_samples"  # or "abs_mean"
    # ----
    cell_info = pd.read_csv(f"{file_path}/eachSample_info.csv", index_col=0)
    perturb_data = pd.read_csv(f"{file_path}/After_perturb_singleGene_eachSample_predcitedTime.csv", index_col=0)
    pertrub_cor_data = pd.read_csv(f"{file_path}/SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial_timeCorGene_predTimeChange.csv", index_col=0)
    # ---
    gene_mapping = dict(zip(pertrub_cor_data['gene'], pertrub_cor_data['gene_short_name']))
    perturb_data = perturb_data.rename(columns=gene_mapping)
    cell_info['predicted_time_denor'] = cell_info['predicted_time'].apply(denormalize, args=(cell_info['time'].min(), cell_info['time'].max(), -5, 5))
    perturb_data_denor = perturb_data.apply(lambda col: col.apply(denormalize, args=(cell_info['time'].min(), cell_info['time'].max(), -5, 5)))
    # ---
    bins = bin_dic[species]
    stage_list = ['early', 'middle', 'late']
    cell_info['3stage'] = pd.cut(cell_info['time'], bins=bins, labels=stage_list, right=False)
    stage_groups = cell_info.groupby('3stage')['time'].apply(list)
    print(f"samples includes time point {cell_info['time'].unique()}, \n"
          f"split as stage:{stage_list}")
    for _s in stage_list:
        print(f"{_s} includes: {set(stage_groups[_s])} with {len(stage_groups[_s])} cells.")

    # ---
    perturb_top_gene_dic = {}
    for _s in stage_list:
        plot_pd, top_gene_list = get_top_gene_perturb_data(cell_info.copy(),
                                                           _s,
                                                           perturb_data_denor,
                                                           top_gene_num=perturb_show_gene_num,
                                                           top_metric=top_metric, )
        perturb_top_gene_dic[_s] = list(top_gene_list)
        plt_perturb_xTime_yDetT(plot_pd, top_gene_list, save_path=file_path, stage=_s)

        # go_mf = gp.get_library(name='GO_Molecular_Function_2018', organism=species)

        _, env_gene_list = get_top_gene_perturb_data(cell_info.copy(),
                                                     _s,
                                                     perturb_data_denor,
                                                     top_gene_num=env_gene_num,
                                                     top_metric=top_metric, )

        gene_set = gene_set_dic[species]
        enr = plt_enrichmentResult(species, gene_set, env_gene_list, _s, file_path, top_term=5)

    print(f"In each stage, with {top_metric}, top {perturb_show_gene_num}: {perturb_top_gene_dic}")
    # ----
    plt_violin_topGene_inWhole_stage(perturb_top_gene_dic, perturb_data_denor,
                                     cell_info, perturb_show_gene_num,
                                     species,
                                     file_path)

    print("Finish all.")
    # ---
    return


if __name__ == '__main__':
    main()
