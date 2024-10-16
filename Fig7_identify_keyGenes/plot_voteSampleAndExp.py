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
from utils.utils_DandanProject import denormalize, geneId_geneName_dic
import numpy as np
from utils.utils_Dandan_plot import plot_detTandExp


def main():
    # ---
    species = "human"  # human from Melania
    file_path = "results/Fig7_TemporalVAE_identify_keyGenes_humanMelania_240902/240405_preimplantation_Melania/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch100_minGeneNum50/"
    calculate(species, file_path)
    # ---
    species = "mouse"
    file_path = "results/Fig7_TemporalVAE_identify_keyGenes_mouseAtlas_240901/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch5_minGeneNum100/"
    calculate(species, file_path)


def calculate(species, file_path, ):
    print(f"For {species} use perturb result from {file_path}")
    print(f"All library names for {species}: {gseapy.get_library_name(organism=species)}")
    # ----
    pertrub_cor_data = pd.read_csv(f"{file_path}/SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial_timeCorGene_predTimeChange.csv", index_col=0)
    cell_info = pd.read_csv(f"{file_path}/eachSample_info.csv", index_col=0)
    perturb_data = pd.read_csv(f"{file_path}/After_perturb_singleGene_eachSample_predcitedTime.csv", index_col=0)

    # ---
    gene_mapping = dict(zip(pertrub_cor_data['gene'], pertrub_cor_data['gene_short_name']))
    perturb_data = perturb_data.rename(columns=gene_mapping)  #
    perturb_data_denor = perturb_data.apply(lambda col: col.apply(denormalize, args=(cell_info['time'].min(), cell_info['time'].max(), -5, 5)))
    cell_info['predicted_time_denor'] = cell_info['predicted_time'].apply(denormalize, args=(cell_info['time'].min(), cell_info['time'].max(), -5, 5))

    # ---
    plot_detTandExp(pertrub_cor_data.copy(), "", y_str="median",
                    y_legend_str=f'Median △t: pseudo-time after perturb and without perturb.',
                    special_filename_str=f"medianDetT", save_path=file_path,
                    )
    plot_detTandExp(pertrub_cor_data.copy(), "", y_str="mean",
                    y_legend_str=f'Mean △t: pseudo-time after perturb and without perturb.',
                    special_filename_str=f"meanDetT", save_path=file_path,
                    )
    plt_topGene_voteNumAndExp(cell_info, perturb_data_denor, pertrub_cor_data, file_path, top_gene_num=10)
    plt_topGene_voteNumAndExp(cell_info, perturb_data_denor, pertrub_cor_data, file_path, top_gene_num=20)
    plt_topGene_voteNumAndExp(cell_info, perturb_data_denor, pertrub_cor_data, file_path, top_gene_num=30)

    print("Finish all.")
    # ---
    return


def plt_topGene_voteNumAndExp(cell_info, perturb_data_denor, pertrub_cor_data, file_path, top_gene_num=10):
    _temp = perturb_data_denor - np.array(cell_info["predicted_time_denor"])[:, np.newaxis]
    _temp = abs(_temp)
    top_columns_per_row = _temp.apply(lambda row: row.nlargest(top_gene_num).index.tolist(), axis=1)
    all_top_columns = [col for sublist in top_columns_per_row for col in sublist]
    column_counts = pd.Series(all_top_columns).value_counts()

    pertrub_cor_data[f"top{top_gene_num}VoteNum"] = pertrub_cor_data['gene_short_name'].map(column_counts).fillna(0)
    pertrub_cor_data[f"top{top_gene_num}VoteNum"] = pertrub_cor_data[f"top{top_gene_num}VoteNum"].astype(int)

    plot_detTandExp(pertrub_cor_data.copy(), "", y_str=f"top{top_gene_num}VoteNum",
                    y_legend_str=f'Total Votes for Top {top_gene_num} Genes per Sample',
                    special_filename_str=f"voteTop{top_gene_num}", save_path=file_path,
                    )


if __name__ == '__main__':
    main()
