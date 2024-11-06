# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE
@File    ：TemporalVAE_mouse_fineTune_Train_on_U_pairs_S.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/1/8 15:25 



"""

import os
import sys

if os.getcwd().split("/")[-1] != "TemporalVAE":
    os.chdir("..")
sys.path.append(os.getcwd())

from utils.utils_DandanProject import *

import pandas as pd

import argparse


def main():
    print("train (fine tune) on moment u concat s; and test on u and s, respectively.")
    parser = argparse.ArgumentParser(description="fine tune model")
    parser.add_argument('--special_path_str', type=str,  # 2023-07-13 17:40:22
                        default="Fig5_RNAvelocity_240901/",
                        help="results all save here")
    parser.add_argument('--fine_tune_mode', type=str,  # 2023-07-13 17:40:22
                        default="focusEncoder",
                        help="fine_tune_mode")
    parser.add_argument('--sc_file_name', type=str,
                        # default="240108mouse_embryogenesis/hematopoiesis",
                        default="240108mouse_embryogenesis/neuron",
                        help="sc file folder path.")
    parser.add_argument('--clf_weight', type=float,
                        # default=0.2,  # for hematopoiesis
                        default=0.1, # for neuron
                        help="clf_weight.")
    parser.add_argument('--detT', type=float,
                        default=0.001,
                        help="detT value.")
    args = parser.parse_args()
    sc_file_name = f"data/{args.sc_file_name}.h5ad"

    fine_tune_mode = args.fine_tune_mode
    clf_weight = args.clf_weight

    print(f"for sc file {sc_file_name} with fine tune mode {fine_tune_mode} with clf weight {clf_weight}.")
    # ----------------- imput parameters  -----------------
    mouse_atlas_file = "data/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/data_count_hvg.csv"
    config_file = "vae_model_configs/supervise_vae_regressionclfdecoder_mouse_atlas_finetune_u_s_focusEncoder.yaml"
    checkpoint_file = "checkpoint_files/mouse_atlas.ckpt"
    _temp = sc_file_name.split("/")[-1].replace(".h5ad", "")
    save_result_path = f"results/{args.special_path_str}/{_temp}/{fine_tune_mode}_clfWeight{clf_weight}_detT{args.detT}/"

    # save parameters used now 2023-12-01 16:30:17 useless
    # _local_variables = locals().copy()
    # _df = get_parameters_df(_local_variables)
    # _df.to_csv(f"{save_result_path}/parameters_use_in_this_results.csv")

    # ----------------- read test adata -----------------
    adata_concated, adata, gene_dic, hvg_dic = read_mouse_embryogenesisData(sc_file_name)

    result_tuple = cal_new_adata(adata_concated,
                                 gene_dic, mouse_atlas_file,
                                 f"{save_result_path}/all/",
                                 config_file, checkpoint_file,
                                 fine_tune_mode=fine_tune_mode,
                                 clf_weight=clf_weight, hvg_dic=hvg_dic, detT=args.detT)
    v, fine_tune_result_adata, plt_attr, fine_tune_test_result_dic, original_result = result_tuple
    plot_RNAvelocity(v, fine_tune_result_adata, save_result_path, plt_attr, fine_tune_test_result_dic, original_result)
    print("finish all")
    return


def read_mouse_embryogenesisData(sc_file_name):
    _temp = "data/240108mouse_embryogenesis/"
    adata = sc.read(sc_file_name, cache=True)
    print("Import data, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    # 2024-04-18 23:58:47 filter only use cao dataset.
    # adata=adata[adata.obs["group"] !="beth"]
    # print("filter data from beth dataset, get cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))

    print("Annotation information of data includes: {}".format(adata.obs_keys()))  # 胞注釋信息的keys
    print("Cell id first 5: {}".format(adata.obs_names[:5]))  # 返回胞ID 数据类型是object
    print("Gene id first 5: {}".format(adata.var_names[:5]))  # 返回基因数据类型是list adata.obs.head()# 査看前5行的数据
    try:
        adata.obs['cell_type'] = adata.obs['celltype']
    except:
        pass
    adata.obs_names_make_unique()
    adata.var_names_make_unique()
    hvg_dic = None

    gene_dic = geneId_geneName_dic()

    def temp_process_day(day_str):

        return float(day_str.replace('E', '').replace('b', ''))

    # Apply the custom function to create the "time_label" column
    adata.obs["time_label"] = adata.obs["day"].apply(temp_process_day)
    adata.obs['cell_type_encoded'] = adata.obs['cell_type'].astype('category').cat.codes
    # Note-- unsplice data is more sparse than splice data
    adata_unspliced = adata.copy()
    adata_unspliced.X = adata.layers["Mu"]
    adata_unspliced.obs["cell_id"] = adata_unspliced.obs_names
    adata_unspliced.obs_names = "unspliced_" + adata_unspliced.obs_names
    adata_unspliced.obs["s_or_mrna"] = "unspliced"

    adata_spliced = adata.copy()
    adata_spliced.X = adata.layers["Ms"]
    adata_spliced.obs["cell_id"] = adata_spliced.obs_names
    adata_spliced.obs_names = "spliced_" + adata_spliced.obs_names
    adata_spliced.obs["s_or_mrna"] = "spliced"
    adata_concated = anndata.concat([adata_spliced.copy(), adata_unspliced.copy()], axis=0)
    return adata_concated, adata, gene_dic, hvg_dic


def cal_new_adata(adata_train, gene_dic, mouse_atlas_file,
                  save_result_path,
                  config_file, checkpoint_file,
                  fine_tune_mode, clf_weight, hvg_dic=None, detT=0.001):
    if not os.path.exists(f"{save_result_path}"):
        os.makedirs(save_result_path)
    print(f"make result save at {save_result_path}")
    # ----------------- get renormalized test data -----------------
    print(adata_train.obs_names[:5])
    if hvg_dic is None:
        trainData_renormalized_df, loss_gene_shortName_list, train_cell_info_df = predict_newData_preprocess_df(gene_dic,
                                                                                                                adata_train,
                                                                                                                min_gene_num=0,
                                                                                                                reference_file=mouse_atlas_file)
    else:
        trainData_renormalized_df, loss_gene_shortName_list, train_cell_info_df, trainData_renormalized_hvg_df = predict_newData_preprocess_df(gene_dic,
                                                                                                                                               adata_train,
                                                                                                                                               min_gene_num=0,
                                                                                                                                               reference_file=mouse_atlas_file,
                                                                                                                                               hvg_dic=hvg_dic)
    # remain_hvg_list = list(trainData_renormalized_hvg_df.columns)
    # for spliced as test
    spliced_rownames = train_cell_info_df[train_cell_info_df["s_or_mrna"] == "spliced"].index
    test_spliced = {"df": trainData_renormalized_df.loc[spliced_rownames],
                    "cell_info": train_cell_info_df.loc[spliced_rownames],
                    "hvg_df": trainData_renormalized_hvg_df.loc[spliced_rownames] if hvg_dic is not None else None}
    # for unspliced as test
    unspliced_rownames = train_cell_info_df[train_cell_info_df["s_or_mrna"] == "unspliced"].index
    test_unspliced = {"df": trainData_renormalized_df.loc[unspliced_rownames],
                      "cell_info": train_cell_info_df.loc[unspliced_rownames],
                      "hvg_df": trainData_renormalized_hvg_df.loc[unspliced_rownames] if hvg_dic is not None else None}
    testData_dic = {"spliced": test_spliced, "unspliced": test_unspliced}

    print("mrna cell num:", len(trainData_renormalized_df))
    trainData_renormalized_df.to_csv(f"{save_result_path}/preprocessed_data.csv")
    try:
        trainData_renormalized_hvg_df.to_csv(f"{save_result_path}/preprocessed_data_hvg.csv")
    except:
        pass

    # ------------------ predict latent from a trained model  -------------------------------------------------

    # plt_attr = list(set(adata_train.obs.columns) & {"ClusterName", "specific_type", "cell_type", "celltype", "theiler", "stage", "sample", "sequencing.batch",
    #                                                 "s_or_mrna", "time_label", "age", "day"})
    plt_attr = ["day", "physical_pseudotime_by_preTrained_mouseAtlas_model", "physical_pseudotime_by_finetune_model", "cell_type", "s_or_mrna", "sample"]
    # # 2023-12-13 12:34:12 finetuning on mrna data, and test on spliced data
    trainData_renormalized_hvg_df = trainData_renormalized_hvg_df if hvg_dic is not None else None
    original_result, fine_tune_result_adata, fine_tune_test_result_dic, predict_detT, v = fineTuning_calRNAvelocity(trainData_renormalized_df,
                                                                                                                    config_file, checkpoint_file,
                                                                                                                    save_result_path=save_result_path,
                                                                                                                    cell_time_info=train_cell_info_df,
                                                                                                                    fine_tune_mode=fine_tune_mode,
                                                                                                                    clf_weight=clf_weight,
                                                                                                                    sc_expression_df_add=trainData_renormalized_hvg_df,
                                                                                                                    plt_attr=plt_attr,
                                                                                                                    testData_dic=testData_dic, detT=detT,
                                                                                                                    batch_size=50000)
    # save gene list
    used_gene_pd = pd.DataFrame(data=trainData_renormalized_df.columns, columns=["used_gene_shortName"])
    used_gene_pd.to_csv(f"{save_result_path}/used_geneShortName.csv")
    loss_gene_mrna_shortName_pd = pd.DataFrame(data=loss_gene_shortName_list, columns=["miss_gene_shortName"])
    loss_gene_mrna_shortName_pd.to_csv(f"{save_result_path}/miss_gene_shortName.csv")

    original_result = original_result[1].cpu().numpy()
    return (v, fine_tune_result_adata, plt_attr, fine_tune_test_result_dic, original_result)


def color_plate_for_hematopoiesis_neuron(save_result_path):
    if "hematopoiesis" in save_result_path:
        celltype_plate = None
        day_color_plate = {'day': {'E6.5': 'red', 'E6.75': 'yellow', 'E7': 'green', 'E7.25': 'blue', 'E7.5': 'tab:orange',
                                   'E7.75': 'tab:purple', 'E8': 'tab:gray', 'E8.25': 'tab:cyan', 'E8.5a': 'tab:brown',
                                   'E8.5b': 'cyan', 'E9.5': 'tab:red', 'E10.5': 'tab:olive', 'E11.5': 'tab:green',
                                   'E12.5': 'tab:blue', 'E13.5': 'tab:pink'},
                           'cell_type': "Set1",
                           'celltype': "Set1"}

    elif 'neuron' in save_result_path:
        celltype_plate = {"Intermediate progenitor cells": 'tab:orange',
                          "Inhibitory interneurons": 'tab:purple',
                          "Di/mesencephalon inhibitory neurons": 'tab:gray',
                          "Spinal cord inhibitory neurons": 'tab:cyan',
                          "Di/mesencephalon excitatory neurons": 'tab:brown',
                          "Spinal cord excitatory neurons": 'tab:red',
                          "Neuron progenitor cells": 'tab:olive',
                          "Noradrenergic neurons": 'tab:green',
                          "Motor neurons": 'tab:blue',
                          "Forebrain/midbrain": 'tab:pink'}
        day_color_plate = {'day': {'E6.5': 'red', 'E6.75': 'yellow', 'E7': 'green', 'E7.25': 'blue', 'E7.5': 'tab:orange',
                                   'E7.75': 'tab:purple', 'E8': 'tab:gray', 'E8.25': 'tab:cyan', 'E8.5a': 'tab:brown',
                                   'E8.5b': 'cyan', 'E9.5': 'tab:red', 'E10.5': 'tab:olive', 'E11.5': 'tab:green',
                                   'E12.5': 'tab:blue', 'E13.5': 'tab:pink'},
                           'cell_type': celltype_plate,
                           'celltype': celltype_plate}
    return day_color_plate, celltype_plate


def plot_RNAvelocity(v, fine_tune_result_adata, save_result_path, plt_attr, fine_tune_test_result_dic, original_result):
    fine_tune_result_adata_spliced = fine_tune_result_adata[fine_tune_result_adata.obs["s_or_mrna"] == "spliced"].copy()
    fine_tune_result_adata_unspliced = fine_tune_result_adata[fine_tune_result_adata.obs["s_or_mrna"] == "unspliced"].copy()

    day_color_plate, celltype_plate = color_plate_for_hematopoiesis_neuron(save_result_path)

    result_adata_spliced = plt_umap_byScanpy(fine_tune_result_adata_spliced,
                                             attr_list=plt_attr, save_path=save_result_path,
                                             show_in_row=False, figure_size=(7, 4.5),
                                             special_file_name_str="spliced_",
                                             palette_dic=day_color_plate)
    result_adata_unspliced = plt_umap_byScanpy(fine_tune_result_adata_unspliced,
                                               attr_list=plt_attr, save_path=save_result_path,
                                               show_in_row=False, figure_size=(7, 4.5),
                                               special_file_name_str="unspliced_",
                                               palette_dic=day_color_plate)
    fine_tune_result_adata = plt_umap_byScanpy(fine_tune_result_adata, attr_list=plt_attr,
                                               save_path=save_result_path,
                                               show_in_row=False, figure_size=(7, 4.5),
                                               special_file_name_str="splicedAndUnspliced",
                                               palette_dic=day_color_plate)

    # -------2024-01-16 23:07:03
    import scvelo as scv
    # adata_velocity = result_adata_spliced
    adata_velocity = fine_tune_test_result_dic["spliced"]
    adata_velocity.layers["unspliced"] = fine_tune_test_result_dic["unspliced"].X
    adata_velocity.layers["spliced"] = fine_tune_test_result_dic["spliced"].X
    adata_velocity.layers["velocity"] = v
    # adata_velocity.uns["pca"] = adata.uns["pca"]
    # adata_velocity.obsm["X_pca"] = adata.obsm["X_pca"]
    scv.tl.velocity_graph(adata_velocity)
    adata_velocity.uns["umap"] = result_adata_spliced.uns["umap"]
    adata_velocity.obsm["X_umap"] = result_adata_spliced.obsm["X_umap"]
    # scv.tl.umap(adata_velocity, min_dist=0.75)

    method_name="TemporalVAE"
    if "hematopoiesis" in save_result_path:
        dataset_name = "hematopoiesis"
        scv.pl.velocity_embedding_stream(adata_velocity,
                                         color="celltype",
                                         legend_loc='on data',
                                         legend_fontoutline=2,
                                         figsize=(5, 4.5),
                                         legend_fontsize=8,
                                         linewidth=0.5, s=20,
                                         sort_order=False,
                                         legend_fontweight="normal",
                                         title=f"{dataset_name} with cell types",
                                         palette="Set1",
                                         save=f"{save_result_path}/{method_name}_{dataset_name}_celltype.png", dpi=200
                                         )
        scv.pl.velocity_embedding_stream(adata_velocity,
                                         color="celltype",
                                         legend_loc='right margin',
                                         legend_fontoutline=2,
                                         figsize=(5, 4.5),
                                         legend_fontsize=8,
                                         linewidth=0.5, s=20,
                                         sort_order=False,
                                         legend_fontweight="normal",
                                         title=f"{dataset_name} with cell types",
                                         palette="Set1",
                                         save=f"{save_result_path}/{method_name}_{dataset_name}_celltype_legend.png", dpi=200
                                         )
        try:
            adata_velocity.obs['day'].cat.reorder_categories(["E8.5b", "E9.5", "E10.5", "E11.5", "E12.5", "E13.5"])
        except:
            adata_velocity.obs['day'].cat.reorder_categories(["E9.5", "E10.5", "E11.5", "E12.5", "E13.5"])

        scv.pl.velocity_embedding_stream(adata_velocity,
                                         color="day",
                                         legend_loc='right margin',
                                         # legend_fontoutline=2,
                                         figsize=(5, 4.5),
                                         legend_fontsize=8,
                                         linewidth=0.5,
                                         legend_fontweight="normal",
                                         s=20,
                                         title=f"{dataset_name} with day",
                                         palette=day_color_plate["day"],
                                         save=f"{save_result_path}/{method_name}_{dataset_name}_day.png", dpi=200
                                         )
    elif 'neuron' in save_result_path:
        dataset_name = "neuron"
        scv.pl.velocity_embedding_stream(adata_velocity,
                                         color="celltype",
                                         legend_loc='on data',
                                         legend_fontoutline=2,
                                         figsize=(5, 4.5), s=20,
                                         legend_fontsize=8,
                                         linewidth=0.5,
                                         legend_fontweight="normal",
                                         title=f"{dataset_name} with cell types",
                                         palette=celltype_plate,
                                         save=f"{save_result_path}/{method_name}_{dataset_name}_celltype.png", dpi=200
                                         )
        scv.pl.velocity_embedding_stream(adata_velocity,
                                         color="celltype",
                                         legend_loc='right margin',
                                         legend_fontoutline=2,
                                         figsize=(5, 4.5), s=20,
                                         legend_fontsize=8,
                                         linewidth=0.5,
                                         legend_fontweight="normal",
                                         title=f"{dataset_name} with cell types",
                                         palette=celltype_plate,
                                         save=f"{save_result_path}/{method_name}_{dataset_name}_celltype_legend.png", dpi=200
                                         )
        adata_velocity.obs['day'].cat.reorder_categories(["E9.5", "E10.5", "E11.5", "E12.5", "E13.5"])

        scv.pl.velocity_embedding_stream(adata_velocity,
                                         color="day",
                                         legend_loc='right margin',
                                         legend_fontoutline=2,
                                         figsize=(5, 4.5), s=20,
                                         legend_fontsize=8,
                                         linewidth=0.5,
                                         legend_fontweight="normal",
                                         title=f"{dataset_name} with day",
                                         palette=day_color_plate["day"],
                                         save=f"{save_result_path}/{method_name}_{dataset_name}_day.png", dpi=200
                                         )
    # ----- save results
    adata_velocity.write_h5ad(f"{save_result_path}/velocity_result.h5ad")

    fine_tune_result_adata.layers["original_latent_mu_result"] = original_result
    fine_tune_result_adata.layers["fine_tune_latent_mu_result"] = fine_tune_result_adata.X
    fine_tune_result_adata.write_h5ad(f"{save_result_path}/fine_tune_result.h5ad")

    spliced_fine_tune_result_data = fine_tune_test_result_dic["spliced"].copy()
    spliced_fine_tune_result_data.write_h5ad(f"{save_result_path}/spliced_latent_mu.h5ad")
    unspliced_fine_tune_result_data = fine_tune_test_result_dic["unspliced"].copy()
    unspliced_fine_tune_result_data.write_h5ad(f"{save_result_path}/unspliced_latent_mu.h5ad")

    from utils.utils_Dandan_plot import plt_latentDim
    plt_latentDim(spliced_fine_tune_result_data, unspliced_fine_tune_result_data, save_result_path)


if __name__ == '__main__':
    main()

    print("Finish all.")
