# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：science2022_kFoldOn_mouseAtlas.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/8/31 14:05 
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
from utils.utils_DandanProject import str2bool, auto_select_gpu_and_cpu, preprocessData_and_dropout_some_donor_or_gene, trans_time
from utils.utils_DandanProject import denormalize
from collections import Counter

import yaml
import argparse
from utils.utils_Dandan_plot import Embryodonor_resort_key,plot_on_each_test_donor_violin_fromDF
import numpy as np
from benchmarking_methods.benchmarking_methods import science2022
import pandas as pd
import gc

def main():
    parser = argparse.ArgumentParser(description="CNN model for prediction of gene paris' regulatory relationship")
    parser.add_argument('--result_save_path', type=str,  # 2023-07-13 17:40:22
                        default="/Fig3_science2022_LR_PCA_RF_kFoldOn_mouseAtlas_240901/",
                        help="results all save here")
    parser.add_argument('--file_path', type=str,
                        default="/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/",
                        help="sc file folder path.")
    # ------------------ preprocess sc data setting ------------------
    parser.add_argument('--min_gene_num', type=int,
                        default="100",
                        help="filter cell with min gene num, default 50")
    parser.add_argument('--min_cell_num', type=int,
                        default="50",
                        help="filter gene with min cell num, default 50")
    # ------------------ model training setting ------------------
    # parser.add_argument('--train_epoch_num', type=int,
    #                     default="100",
    #                     help="Train epoch num")
    parser.add_argument('--batch_size', type=int,
                        default=100000,
                        help="batch size")
    parser.add_argument('--time_standard_type', type=str,
                        default="embryoneg5to5",
                        help="y_time_nor_train standard type may cause different latent space: log2, 0to1, neg1to1, labeldic,sigmoid,logit")
    # parser.add_argument('--vae_param_file', type=str,
    #                     default="supervise_vae_regressionclfdecoder_mouse_stereo",
    #                     help="vae model parameters file.")
    # ------------------ task setting ------------------
    # parser.add_argument('--kfold_test', action="store_true",
    #                     help="(Optional) make the task k fold test on dataset.", default=True)
    # # parser.add_argument('--train_whole_model', action="store_true",
    # #                     help="(Optional) use all data to train a model.", default=True)
    # parser.add_argument('--identify_time_cor_gene', action="store_true",
    #                     help="(Optional) identify time-cor gene by model trained by all.", default=False)

    # Todo, useless, wait to delete "KNN_smooth_type"
    # parser.add_argument('--KNN_smooth_type', type=str,
    #                     default="mingze",
    #                     help="KNN smooth method")  # don't use 2023-06-26 14:04:25
    # parser.add_argument('--use_checkpoint_bool', type=str2bool,
    #                     default="False",
    #                     help="use checkpoint file as pre-trained model or not.")
    args = parser.parse_args()

    data_golbal_path = "data/"
    result_save_path = "results/" + args.result_save_path + "/"
    data_path = args.file_path + "/"
    yaml_path = "vae_model_configs/"

    # --------------------------set logger and parameters, creat result save path and folder----------------------------------

    time_standard_type = args.time_standard_type
    sc_data_file_csv = data_path + "/data_count_hvg.csv"
    cell_info_file_csv = data_path + "/cell_with_time.csv"

    _path = '{}/{}/'.format(result_save_path, data_path)
    if not os.path.exists(_path):
        os.makedirs(_path)

    logger_file = f'{_path}/science_time{time_standard_type}_minGeneNum{args.min_gene_num}.log'
    LogHelper.setup(log_path=logger_file, level='INFO')
    _logger = logging.getLogger(__name__)
    _logger.info("Finished setting up the logger at: {}.".format(logger_file))
    _logger.info("Train on dataset: {}.".format(data_golbal_path + data_path))
    device = auto_select_gpu_and_cpu()
    _logger.info("Auto select run on {}".format(device))
    # ------------ Preprocess data, with hvg gene from preprocess_data_mouse_embryonic_development.py------------------------
    sc_expression_df, cell_time = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path, sc_data_file_csv, cell_info_file_csv,
                                                                                min_cell_num=args.min_cell_num,
                                                                                min_gene_num=args.min_gene_num)

    special_path_str = ""
    # ---------------------------------------- set donor list and dictionary -----------------------------------------------------
    donor_list = np.unique(cell_time["donor"])
    donor_list = sorted(donor_list, key=Embryodonor_resort_key)
    donor_dic = dict()
    for i in range(len(donor_list)):
        donor_dic[donor_list[i]] = i
    batch_dic = donor_dic.copy()
    _logger.info("Consider donor as batch effect, donor use label: {}".format(donor_dic))
    _logger.info("For each donor (donor_id, cell_num):{} ".format(Counter(cell_time["donor"])))

    # # # ----------------------------------TASK 1: K-FOLD TEST--------------------------------------
    gc.collect()
    save_path = _logger.root.handlers[0].baseFilename.replace('.log', '')
    kFold_test_result_df = pd.DataFrame(columns=['time', 'pseudotime'])
    predict_donors_dic = dict()
    for fold in range(len(donor_list)):
        _logger.info(f'--- Start fold {(fold+1)}/{len(donor_list)}:')
        sc_expression_train = sc_expression_df.loc[cell_time.index[cell_time["donor"] != donor_list[fold]]]
        sc_expression_test = sc_expression_df.loc[cell_time.index[cell_time["donor"] == donor_list[fold]]]
        x_sc_train = torch.tensor(sc_expression_train.values, dtype=torch.get_default_dtype()).t()
        x_sc_test = torch.tensor(sc_expression_test.values, dtype=torch.get_default_dtype()).t()
        _logger.info("Set x_sc_train data with shape (gene, cells): {}".format(x_sc_train.shape))
        _logger.info("Set x_sc_test data with shape (gene, cells): {}".format(x_sc_test.shape))
        cell_time_dic = dict(zip(cell_time.index, cell_time['time']))
        y_time_train = x_sc_train.new_tensor(np.array(sc_expression_train.index.map(cell_time_dic) * 100).astype(int))
        y_time_test = x_sc_test.new_tensor(np.array(sc_expression_test.index.map(cell_time_dic) * 100).astype(int))

        # donor_index_train = x_sc_train.new_tensor([int(batch_dic[cell_time.loc[_cell_name]['donor']]) for _cell_name in sc_expression_train.index.values])
        # donor_index_test = x_sc_test.new_tensor([int(batch_dic[cell_time.loc[_cell_name]['donor']]) for _cell_name in sc_expression_test.index.values])

        # for classification model with discrete time cannot use sigmoid and logit time type
        y_time_nor_train, label_dic = trans_time(y_time_train, time_standard_type, capture_time_other=y_time_test, min_max_val=None)
        y_time_nor_test, label_dic = trans_time(y_time_test, time_standard_type, label_dic_train=label_dic)
        query_predictions = science2022(train_x=sc_expression_train, train_y=y_time_nor_train, test_df=sc_expression_test)

        test_result_df = pd.DataFrame(index=sc_expression_test.index)
        test_result_df['time'] = y_time_nor_test
        test_result_df["pseudotime"] = query_predictions
        test_result_df['stage'] = cell_time['time']
        kFold_test_result_df = pd.concat([kFold_test_result_df, test_result_df], axis=0)
        gc.collect()

    kFold_test_result_df['predicted_time'] = kFold_test_result_df['pseudotime'].apply(denormalize,
                                                                                      args=(min(label_dic.keys()) / 100,
                                                                                            max(label_dic.keys()) / 100,
                                                                                            min(label_dic.values()),
                                                                                            max(label_dic.values())))

    kFold_test_result_df.to_csv(f"{save_path}/k_fold_test_result.csv")
    from utils.utils_DandanProject import calculate_real_predict_corrlation_score
    corr_stats = calculate_real_predict_corrlation_score(kFold_test_result_df['predicted_time'], kFold_test_result_df['stage'])
    _logger.info(f'k-fold test results of corrs: {corr_stats}, \n result save at {save_path}/k_fold_test_result.csv')
    # color_dic = plot_on_each_test_donor_violin_fromDF(kFold_test_result_df.copy(), save_path, physical_str="predicted_time", x_str="stage", cmap_color="viridis")

    # print("k-fold test final result:")
    # corr(kFold_test_result_df["time"], kFold_test_result_df["pseudotime"])

    _logger.info("Finish all.")


if __name__ == '__main__':
    main()
