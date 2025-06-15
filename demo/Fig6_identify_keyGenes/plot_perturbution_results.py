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
import gseapy

if os.getcwd().split("/")[-1] != "TemporalVAE":
    os.chdir("../..")
sys.path.append(os.getcwd())
import pandas as pd
from TemporalVAE.utils import denormalize, get_top_gene_perturb_data,preprocessData_and_dropout_some_donor_or_gene
from TemporalVAE.utils import plt_enrichmentResult
from TemporalVAE.utils import plt_lineChart_stageGeneDic_inStages, plt_allGene_dot_voteNum_meanDetT_Exp, plt_venn_fromDict,plt_muiltViolin_forGenes_xRawCount


def main():


    bin_dic = {"mouse": [0, 10.25, 14, 20], 'human': [0, 7, 14, 20]}

    species = "human"  # human from Melania
    file_path = "results/Fig7_TemporalVAE_identify_keyGenes_humanMelania_240902/240405_preimplantation_Melania/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch100_minGeneNum50/"
    calculate(species, file_path, bin_dic)

    species = "mouse"
    file_path = "results/Fig7_TemporalVAE_identify_keyGenes_mouseAtlas_240901/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch5_minGeneNum100/"
    calculate(species, file_path, bin_dic)


def calculate(species,
              file_path,
              bin_dic,
              perturb_show_gene_num=2,
              enr_gene_num=50,
              top_metric="vote_samples"):
    print(f"For {species} use perturb result from {file_path}")
    print(f"All library names for {species}: {gseapy.get_library_name(organism=species)}")
    # ----
    gene_set_dic = {"mouse": ['GO_Biological_Process_2023',
                              'GO_Molecular_Function_2018',
                              # 'KEGG_2019_Mouse',
                              # 'Reactome_2022',
                              'GO_Cellular_Component_2023',
                              ],  # 'WikiPathways_2024_Mouse','MSigDB_Hallmark_2020','Aging_Perturbations_from_GEO_down',
                    "human": ['MSigDB_Hallmark_2020',
                              'KEGG_2021_Human',
                              'Reactome_2022',
                              'GO_Molecular_Function_2018',
                              'GO_Biological_Process_2023',
                              'GO_Cellular_Component_2023',
                              ]}  # # 'MSigDB_Oncogenic_Signatures','MSigDB_Computational','GWAS_Catalog_2023','HDSigDB_Human_2021', 'Genome_Browser_PWMs','Enrichr_Libraries_Most_Popular_Genes', 'Enrichr_Submissions_TF-Gene_Coocurrence','Enrichr_Users_Contributed_Lists_2020', 'WikiPathways_2024_Human','Aging_Perturbations_from_GEO_down'
    # or "abs_mean"
    # ----
    cell_info = pd.read_csv(f"{file_path}/eachSample_info.csv", index_col=0)
    perturb_data = pd.read_csv(f"{file_path}/After_perturb_singleGene_eachSample_predcitedTime.csv", index_col=0)
    perturb_gene_cor_data = pd.read_csv(f"{file_path}/SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial_timeCorGene_predTimeChange.csv", index_col=0)
    perturb_gene_cor_data.index = perturb_gene_cor_data["gene_short_name"]
    # ---
    gene_mapping = dict(zip(perturb_gene_cor_data['gene'], perturb_gene_cor_data['gene_short_name']))
    perturb_data = perturb_data.rename(columns=gene_mapping)
    cell_info['predicted_time_denor'] = cell_info['predicted_time'].apply(denormalize, args=(cell_info['time'].min(), cell_info['time'].max(), -5, 5))
    perturb_data_denor = perturb_data.apply(lambda col: col.apply(denormalize, args=(cell_info['time'].min(), cell_info['time'].max(), -5, 5)))
    print(f"if index equal: {cell_info.index.equals(perturb_data_denor.index)}")
    # ---
    plt_allGene_dot_voteNum_meanDetT_Exp(cell_info.copy(), perturb_data_denor.copy(), perturb_gene_cor_data.copy(), file_path,
                                         top_gene_num=enr_gene_num, x_str="mean", stage_str="allStage", species=species)
    # ---
    bins = bin_dic[species]
    stage_list = ['early', 'middle', 'late']
    cell_info['3stage'] = pd.cut(cell_info['time'], bins=bins, labels=stage_list, right=False)
    stage_groups = cell_info.groupby('3stage')['time'].apply(list)
    print(f"samples includes time point {cell_info['time'].unique()}, \n"
          f"split as stage:{stage_list}")
    stage_timePoint_dic = dict()
    for _s in stage_list:
        print(f"{_s} includes: {set(stage_groups[_s])} with {len(stage_groups[_s])} cells.")
        stage_timePoint_dic[_s] = list(set(stage_groups[_s]))
    # ---
    perturb_top_gene_dic = {}
    enr_top_gene_dic = {}
    used_genes = set()
    for _s in stage_list:
        plot_pd, top_gene_list, stage_pert_data = get_top_gene_perturb_data(cell_info.copy(),
                                                                            _s,
                                                                            perturb_data_denor.copy(),
                                                                            top_gene_num=perturb_show_gene_num,
                                                                            top_metric=top_metric, )
        plt_allGene_dot_voteNum_meanDetT_Exp(cell_info.copy(), stage_pert_data.copy(), perturb_gene_cor_data.copy(), file_path,
                                             top_gene_num=enr_gene_num, stage_str=_s, x_str="mean", species=species)
        # plt_perturb_xTime_yDetT(plot_pd, top_gene_list, save_path=file_path, stage=_s)

        _, enr_gene_list, _ = get_top_gene_perturb_data(cell_info.copy(),
                                                        _s,
                                                        perturb_data_denor.copy(),
                                                        top_gene_num=enr_gene_num,
                                                        top_metric=top_metric, )
        enr_top_gene_dic[_s] = enr_gene_list
        gene_set = gene_set_dic[species]
        enr = plt_enrichmentResult(species, gene_set, enr_gene_list, _s, file_path, top_term=5)

        # Combine top genes with checks for already used genes
        _genes = []
        i = 0
        _all = top_gene_list + enr_gene_list
        while len(_genes) < 4:
            _g = _all[i]
            if _g not in used_genes:
                _genes.append(_g)
                used_genes.add(_g)  # 添加到已使用集合中
            i = i + 1

        perturb_top_gene_dic[_s] = _genes
        # perturb_top_gene_dic[_s] =list(set(list(top_gene_list)+list(enr_gene_list)[:2]))

    print(f"In each stage, with {top_metric}, top {perturb_show_gene_num}: {perturb_top_gene_dic}")
    print(f"In each stage, with {top_metric}, top {enr_gene_num}: {enr_top_gene_dic}")

    # ---- 2024-11-05 18:37:40 plot in manuscript Fig6A
    # --- 2024-11-05 19:44:09 plot Venn of enr_top_gene_dic
    intersection = plt_venn_fromDict(enr_top_gene_dic, file_path, perturb_show_gene_num, species)
    # --- 2024-11-05 23:23:42 plot some special gene of mouse
    if species=="mouse":
        special_gene_for_mouse_dic = {'early': ['Bmp5', 'Bmp4', 'Hapln1', 'Hba-x'],
                                  'middle': ['Hecw1', 'Onecut2', 'Csmd1', 'Lingo2'],
                                  'late': ['Snhg11', 'Chl1', 'Slc4a1', 'Syt4']}
        plt_lineChart_stageGeneDic_inStages(special_gene_for_mouse_dic.copy(), perturb_data_denor.copy(),
                                            cell_info.copy(), perturb_show_gene_num,
                                            species, stage_timePoint_dic,
                                            file_path, cal_detT_str="mean", plt_stage="whole", plt_timePoint="whole",
                                            special_filename_tail_str="_specialGeneForMouse")
        plt_lineChart_stageGeneDic_inStages(special_gene_for_mouse_dic.copy(), perturb_data_denor.copy(),
                                            cell_info.copy(), perturb_show_gene_num,
                                            species, stage_timePoint_dic,
                                            file_path, cal_detT_str="mean", plt_stage="early", plt_timePoint="early",
                                            special_filename_tail_str="_specialGeneForMouse")
        intersection
    elif species=="human":
        plt_lineChart_stageGeneDic_inStages(perturb_top_gene_dic.copy(), perturb_data_denor.copy(),
                                            cell_info.copy(), perturb_show_gene_num,
                                            species, stage_timePoint_dic,
                                            file_path, cal_detT_str="mean", plt_stage="whole", plt_timePoint="whole")
        plt_lineChart_stageGeneDic_inStages(perturb_top_gene_dic.copy(), perturb_data_denor.copy(),
                                            cell_info.copy(), perturb_show_gene_num,
                                            species, stage_timePoint_dic,
                                            file_path, cal_detT_str="mean", plt_stage="early", plt_timePoint="early")


    # plt intersection genes whole line
    intersection_dic={"intersection": intersection}
    plt_lineChart_stageGeneDic_inStages(intersection_dic.copy(), perturb_data_denor.copy(),
                                        cell_info.copy(), perturb_show_gene_num,
                                        species, stage_timePoint_dic,
                                        file_path, cal_detT_str="mean", plt_stage="intersection", plt_timePoint="whole",
                                        figsize_hight_weight=len(intersection)/2+4)
    # plt expression of genes
    if species=="mouse":
        sc_expression_df, cell_time = preprocessData_and_dropout_some_donor_or_gene('data/',
                                                                                    '/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000///data_count_hvg.csv',
                                                                                    '/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000///cell_with_time.csv',
                                                                                    min_cell_num=50,
                                                                                    min_gene_num=100)
        sc_expression_df = sc_expression_df.rename(columns=gene_mapping)
        plt_muiltViolin_forGenes_xRawCount(sc_expression_df, intersection, cell_time, file_path,
                                           perturb_show_gene_num, species, special_filename_str="_intersection")

        combined_list = [item for sublist in special_gene_for_mouse_dic.values() for item in sublist]
        print(combined_list)
        plt_muiltViolin_forGenes_xRawCount(sc_expression_df, combined_list, cell_time, file_path,
                                           perturb_show_gene_num, species, special_filename_str="_perturbTopGenes")
    elif species=="human":
        sc_expression_df, cell_time = preprocessData_and_dropout_some_donor_or_gene('data/',
                                                                                    '/human_embryo_preimplantation/Melania_5datasets//data_count_hvg.csv',
                                                                                    '/human_embryo_preimplantation/Melania_5datasets//cell_with_time.csv',
                                                                                    min_cell_num=50,
                                                                                    min_gene_num=50,
                                                                                    data_raw_count_bool=False)
        sc_expression_df = sc_expression_df.rename(columns=gene_mapping)
        plt_muiltViolin_forGenes_xRawCount(sc_expression_df, intersection, cell_time, file_path,
                                           perturb_show_gene_num, species,special_filename_str="_intersection")

        combined_list = [item for sublist in perturb_top_gene_dic.values() for item in sublist]
        print(combined_list)
        plt_muiltViolin_forGenes_xRawCount(sc_expression_df, combined_list, cell_time, file_path,
                                           perturb_show_gene_num, species,special_filename_str="_perturbTopGenes")


    # ------- more gene for supplementary
    more_perturb_top_gene_dic = dict()
    for _s in stage_list:
        _, enr_gene_list, _ = get_top_gene_perturb_data(cell_info.copy(),
                                                        _s,
                                                        perturb_data_denor.copy(),
                                                        top_gene_num=enr_gene_num,
                                                        top_metric=top_metric, )
        # Combine top genes with checks for already used genes
        _genes = []
        i = 0
        while len(_genes) < 8:
            _g = enr_gene_list[i]
            if _g not in used_genes:
                _genes.append(_g)
                used_genes.add(_g)  # 添加到已使用集合中
            i = i + 1
        more_perturb_top_gene_dic[_s] = _genes
    plt_lineChart_stageGeneDic_inStages(more_perturb_top_gene_dic.copy(), perturb_data_denor.copy(),
                                        cell_info.copy(), perturb_show_gene_num,
                                        species, stage_timePoint_dic,
                                        file_path, cal_detT_str="mean", plt_stage="whole",
                                        special_filename_tail_str="_moreGenesForSup",
                                        figsize_hight_weight=6)

    # --- 2024-11-05 00:00:15 plot for all 50 for test
    n_parts = 5
    split_dicts = [{} for _ in range(n_parts)]
    for key, value in enr_top_gene_dic.items():
        parts = [value[i:i + len(value) // n_parts] for i in range(0, len(value), len(value) // n_parts)]
        for idx, part in enumerate(parts):
            split_dicts[idx][key] = part
    for i in range(n_parts):
        plt_lineChart_stageGeneDic_inStages(split_dicts[i].copy(), perturb_data_denor.copy(),
                                            cell_info.copy(), perturb_show_gene_num,
                                            species, stage_timePoint_dic,
                                            file_path, cal_detT_str="mean", plt_stage="early", plt_timePoint="whole",
                                            special_filename_head_str="test/",
                                            special_filename_tail_str=f"_moreGenesForTest_{i}",
                                            figsize_hight_weight=10)
        plt_lineChart_stageGeneDic_inStages(split_dicts[i].copy(), perturb_data_denor.copy(),
                                            cell_info.copy(), perturb_show_gene_num,
                                            species, stage_timePoint_dic,
                                            file_path, cal_detT_str="mean", plt_stage="middle", plt_timePoint="whole",
                                            special_filename_head_str="test/",
                                            special_filename_tail_str=f"_moreGenesForTest_{i}",
                                            figsize_hight_weight=10)
        plt_lineChart_stageGeneDic_inStages(split_dicts[i].copy(), perturb_data_denor.copy(),
                                            cell_info.copy(), perturb_show_gene_num,
                                            species, stage_timePoint_dic,
                                            file_path, cal_detT_str="mean", plt_stage="late", plt_timePoint="whole",
                                            special_filename_head_str="test/",
                                            special_filename_tail_str=f"_moreGenesForTest_{i}",
                                            figsize_hight_weight=10)

    gc.collect()
    # plt_line_topGene_inWhole_stage(perturb_top_gene_dic, perturb_data_denor,
    #                                cell_info, perturb_show_gene_num,
    #                                species,
    #                                file_path, cal_detT_str="median")
    # plt_violinAndDot_topGene_inWhole_stage(perturb_top_gene_dic, perturb_data_denor,
    #                                        cell_info, perturb_show_gene_num,
    #                                        species,
    #                                        file_path)

    print("Finish all.")
    # ---
    return


if __name__ == '__main__':
    main()
