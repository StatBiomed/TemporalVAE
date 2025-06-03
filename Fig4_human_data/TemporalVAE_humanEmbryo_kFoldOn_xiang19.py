# -*-coding:utf-8 -*-
"""
@Project ：pairsRegulatePrediction
@File    ：TemporalVAE_humanEmbryo_kFoldOn_xiang19.py
@IDE     ：PyCharm
@Author  ：awa121
@Date    ：2024/3/3 21:02


"""

import os
import sys

if os.getcwd().split("/")[-1] != "TemporalVAE":
    os.chdir("..")
sys.path.append(os.getcwd())

import torch

torch.set_float32_matmul_precision('high')
import pyro
import logging
from utils.logging_system import LogHelper

smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
assert pyro.__version__.startswith('1.8.5')
pyro.set_rng_seed(1)
from utils.utils_project import auto_select_gpu_and_cpu, preprocessData_and_dropout_some_donor_or_gene, Embryodonor_resort_key, onlyTrain_model
from utils.utils_project import denormalize, task_kFoldTest
from utils.utils_plot import plt_umap_byScanpy
from collections import Counter
import os
import yaml
import argparse
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


def main():
    parser = argparse.ArgumentParser(description="CNN model for prediction of gene paris' regulatory relationship")
    parser.add_argument('--result_save_path', type=str,  # 2023-07-13 17:40:22
                        default="Fig4_TemporalVAE_kFoldOn_humanEmbryo_xiang2019_250428",
                        # default="Fig4_TemporalVAE_kFoldOn_humanEmbryo_xiang2019_240901",
                        help="results all save here")
    parser.add_argument('--file_path', type=str,
                        default="/human_embryo_preimplantation/integration_8dataset/",
                        # default="/human_embryo_preimplantation/Xiang2019/hvg500/",
                        help="sc file folder path.")
    # ------------------ preprocess sc data setting ------------------
    parser.add_argument('--min_gene_num', type=int,
                        default="50",
                        help="filter cell with min gene num, default 50")
    parser.add_argument('--min_cell_num', type=int,
                        default="50",
                        help="filter gene with min cell num, default 50")
    # ------------------ model training setting ------------------
    parser.add_argument('--train_epoch_num', type=int,
                        default="50",  # orginal is 70
                        help="Train epoch num")
    parser.add_argument('--batch_size', type=int,
                        default=100000,
                        help="batch size")
    parser.add_argument('--time_standard_type', type=str,
                        # default="embryoneg1to1",
                        default="embryoneg5to5",
                        help="y_time_nor_train standard type may cause different latent space: log2, 0to1, neg1to1, labeldic,sigmoid,logit")
    # supervise_vae            supervise_vae_regressionclfdecoder
    parser.add_argument('--vae_param_file', type=str,
                        default="supervise_vae_regressionclfdecoder_mouse_stereo",
                        help="vae model parameters file.")
    # ------------------ task setting ------------------
    parser.add_argument('--kfold_test', action="store_true", help="(Optional) make the task k fold test on dataset.", default=True)
    parser.add_argument('--train_whole_model', action="store_true", help="(Optional) use all data to train a model.", default=True)
    # parser.add_argument('--identify_time_cor_gene', action="store_true", help="(Optional) identify time-cor gene by model trained by all.", default=False)

    # Todo, useless, wait to delete "KNN_smooth_type"
    parser.add_argument('--KNN_smooth_type', type=str,
                        default="mingze",
                        help="KNN smooth method")  # don't use 2023-06-26 14:04:25

    args = parser.parse_args()

    data_golbal_path = "data/"
    result_save_path = "results/" + args.result_save_path + "/"
    data_path = args.file_path + "/"
    yaml_path = "vae_model_configs/"
    # --------------------------------------- import vae model parameters from yaml file----------------------------------------------
    with open(yaml_path + "/" + args.vae_param_file + ".yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # ---------------------------------------set logger and parameters, creat result save path and folder----------------------------------------------
    latent_dim = config['model_params']['latent_dim']
    time_standard_type = args.time_standard_type
    # filter xiang data from integrated dataset
    temp_adata = ad.read_h5ad(f"data/{data_path}/rawCount_Z_C_Xiao_M_P_Liu_Tyser_Xiang.h5ad")
    temp_adata = temp_adata[temp_adata.obs['dataset_label'] == "Xiang"]
    temp_adata_raw_count = pd.DataFrame(data=temp_adata.X.T,
                                        columns=temp_adata.obs.index,
                                        index=temp_adata.var_names)
    temp_adata_raw_count.to_csv(f"data/{data_path}/Xiang_rawCount.csv", sep="\t")
    temp_adata.obs.to_csv(f"data/{data_path}/Xiang_cellAnnotation.csv", sep="\t")
    sc_data_file_csv = f"{data_path}/Xiang_rawCount.csv"
    cell_info_file_csv = f"{data_path}/Xiang_cellAnnotation.csv"

    _path = '{}/{}/'.format(result_save_path, data_path)
    if not os.path.exists(_path):
        os.makedirs(_path)

    logger_file = '{}/{}_dim{}_time{}_epoch{}_minGeneNum{}.log'.format(_path, args.vae_param_file, latent_dim,
                                                                       time_standard_type, args.train_epoch_num,
                                                                       args.min_gene_num)
    LogHelper.setup(log_path=logger_file, level='INFO')
    _logger = logging.getLogger(__name__)
    _logger.info("Finished setting up the logger at: {}.".format(logger_file))
    _logger.info("Train on dataset: {}.".format(data_golbal_path + data_path))
    device = auto_select_gpu_and_cpu()
    _logger.info("Auto select run on {}".format(device))
    _logger.info("load vae model parameters from file: {}".format(yaml_path + args.vae_param_file + ".yaml"))
    # ------------ Preprocess data, with hvg gene from preprocess_data_mouse_embryonic_development.py------------------------
    sc_expression_df, cell_time = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path,
                                                                                sc_data_file_csv,
                                                                                cell_info_file_csv,
                                                                                # donor_attr=donor_attr, drop_out_donor=drop_out_donor,
                                                                                min_cell_num=args.min_cell_num,
                                                                                min_gene_num=args.min_gene_num,
                                                                                data_raw_count_bool=True)  # 2024-04-20 15:38:58

    special_path_str = ""
    # ---------------------------------------- set donor list and dictionary -----------------------------------------------------
    donor_list = np.unique(cell_time["day"])
    donor_list = sorted(donor_list, key=Embryodonor_resort_key)
    donor_dic = dict()
    for i in range(len(donor_list)):
        donor_dic[donor_list[i]] = i
    batch_dic = donor_dic.copy()
    _logger.info("Consider donor as batch effect, donor use label: {}".format(donor_dic))
    _logger.info("For each donor (donor_id, cell_num):{} ".format(Counter(cell_time["day"])))
    save_file_name = f"{_logger.root.handlers[0].baseFilename.replace('.log', '')}/"
    #  ---------------------------------------------- TASK: use all data to train a model  ----------------------------------------------
    if args.train_whole_model:
        drop_out_donor = "t"
        print(f"drop the donor: {drop_out_donor}")
        cell_drop_index_list = cell_time.loc[cell_time["dataset_label"] == drop_out_donor].index
        sc_expression_df_filter = sc_expression_df.drop(cell_drop_index_list, axis=0)
        cell_time_filter = cell_time.drop(cell_drop_index_list, axis=0)
        cell_time_filter = cell_time_filter.loc[sc_expression_df_filter.index]
        sc_expression_train, y_time_nor_train, donor_index_train, runner, experiment, _m, train_clf_result, label_dic, total_result = onlyTrain_model(
            sc_expression_df_filter, donor_dic,
            special_path_str,
            cell_time_filter,
            time_standard_type, config, args,
            device=device, plot_latentSpaceUmap=False, plot_trainingLossLine=True, time_saved_asFloat=True, batch_dic=batch_dic, donor_str="day",
            batch_size=int(args.batch_size))  # 2023-10-24 17:44:31 batch as 10,000 due to overfit, batch size as 100,000 may be have different result
        predict_donors_df = pd.DataFrame(train_clf_result, columns=["pseudotime"], index=cell_time_filter.index)
        predict_donors_df['predicted_time'] = predict_donors_df['pseudotime'].apply(denormalize, args=(min(label_dic.keys()) / 100, max(label_dic.keys()) / 100,
                                                                                                       min(label_dic.values()), max(label_dic.values())))
        cell_time_filter = pd.concat([cell_time_filter, predict_donors_df], axis=1)

        plt_image_adata = ad.AnnData(X=total_result["mu"].cpu().numpy())
        plt_image_adata.obs = cell_time_filter[["time", "predicted_time", "dataset_label", "cell_type", "day"]]

        plt_umap_byScanpy(plt_image_adata.copy(),
                          ["time", "predicted_time", "cell_type"],
                          save_path=save_file_name, mode=None, figure_size=(5, 4),
                          color_map="turbo",
                          n_neighbors=80, n_pcs=10, special_file_name_str="n50_")
        # color_map="viridis"

    ## # # ----------------------------------TASK 1: K-FOLD TEST--------------------------------------
    if args.kfold_test:
        predict_donors_dic, label_dic = task_kFoldTest(donor_list, sc_expression_df, donor_dic, batch_dic, special_path_str, cell_time, time_standard_type,
                                                       config, args.train_epoch_num, _logger, donor_str="day", batch_size=args.batch_size, cmap_color="viridis")

        _logger.info("Finish fold-test.")

    _logger.info("Finish all.")


if __name__ == '__main__':
    main()
