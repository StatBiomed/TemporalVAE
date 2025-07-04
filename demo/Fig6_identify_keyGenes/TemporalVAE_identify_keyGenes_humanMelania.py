# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE
@File    ：TemporalVAE_identify_keyGenes_humanMelania.py
@IDE     ：PyCharm
@Author  ：awa121
@Date    ：2024-09-28 23:36:59

2023-07-31 11:38:33
train on mouse_embryonic_development data,

cd /mnt/yijun/nfs_share/awa_project/awa_github/TemporalVAE/
source ~/.bashrc
nohup python -u Fig3_mouse_data/TemporalVAE_kFoldOn_mouseAtlas.py --result_save_path 240611_mouseAtlas_test --kfold_test --train_whole_model >> logs/VAE_mouse_kFoldOn_mouseAtlas.log 2>&1 &
"""
import gc
import os
import sys

if os.getcwd().split("/")[-1] != "TemporalVAE":
    os.chdir("../..")
sys.path.append(os.getcwd())

import torch

torch.set_float32_matmul_precision('high')
import pyro
import logging
from TemporalVAE.utils import LogHelper

smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
assert pyro.__version__.startswith('1.8.5')
pyro.set_rng_seed(1)
from TemporalVAE.utils import str2bool, preprocessData_and_dropout_some_donor_or_gene
from TemporalVAE.utils import onlyTrain_model, identify_timeCorGene
from collections import Counter

import yaml
import argparse
from TemporalVAE.utils import Embryodonor_resort_key
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="CNN model for prediction of gene paris' regulatory relationship")
    parser.add_argument('--result_save_path', type=str,  # 2023-07-13 17:40:22
                        default="test/Fig7_TemporalVAE_identify_keyGenes_humanMelania_240902/",
                        help="results all save here")
    parser.add_argument('--file_path', type=str,
                        default="/human_embryo_preimplantation/Melania_5datasets/",
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
                        default="100",
                        help="Train epoch num")
    parser.add_argument('--batch_size', type=int,
                        default=200000,
                        help="batch size")
    parser.add_argument('--time_standard_type', type=str,
                        default="embryoneg5to5",
                        help="y_time_nor_train standard type may cause different latent space: log2, 0to1, neg1to1, labeldic,sigmoid,logit")
    parser.add_argument('--vae_param_file', type=str,
                        default="supervise_vae_regressionclfdecoder_mouse_stereo",
                        help="vae model parameters file.")
    # ------------------ task setting ------------------
    parser.add_argument('--train_whole_model', action="store_true",
                        help="(Optional) use all data to train a model.", default=True)
    parser.add_argument('--identify_time_cor_gene', action="store_true",
                        help="(Optional) identify time-cor gene by model trained by all.", default=True)

    # Todo, useless, wait to delete "KNN_smooth_type"
    parser.add_argument('--KNN_smooth_type', type=str,
                        default="mingze",
                        help="KNN smooth method")  # don't use 2023-06-26 14:04:25
    parser.add_argument('--use_checkpoint_bool', type=str2bool,
                        default="False",
                        help="use checkpoint file as pre-trained model or not.")
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
    # --------------------------set logger and parameters, creat result save path and folder----------------------------------
    latent_dim = config['model_params']['latent_dim']

    time_standard_type = args.time_standard_type
    sc_data_file_csv = data_path + "/data_count_hvg.csv"
    cell_info_file_csv = data_path + "/cell_with_time.csv"

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
    _logger.info("load vae model parameters from file: {}".format(yaml_path + args.vae_param_file + ".yaml"))
    # ------------ Preprocess data, with hvg gene from preprocess_data_mouse_embryonic_development.py------------------------
    sc_expression_df, cell_time = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path, sc_data_file_csv,
                                                                                cell_info_file_csv,
                                                                                min_cell_num=args.min_cell_num,
                                                                                min_gene_num=args.min_gene_num,
                                                                                data_raw_count_bool=False)

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

    #  ------------------ TASK: use all data to train a model  ----------------------------------------------

    if args.train_whole_model:
        sc_expression_train, y_time_nor_train, donor_index_train, runner, experiment, _m, train_clf_result, label_dic, total_result = onlyTrain_model(
            sc_expression_df, donor_dic,
            special_path_str,
            cell_time,
            time_standard_type, config, args,
            plot_latentSpaceUmap=False, plot_trainingLossLine=True, time_saved_asFloat=True, batch_dic=donor_dic,
            batch_size=int(args.batch_size),
            donor_str='day')  # 2023-10-24 17:44:31 batch as 10,000 due to overfit, batch size as 100,000 may be have different result
        import anndata as ad
        save_path = f"{_logger.root.handlers[0].baseFilename.replace('.log', '')}/"
        plt_image_adata = ad.AnnData(X=total_result["mu"].cpu().numpy())
        plt_image_adata.obs = cell_time
        plt_image_adata.write_h5ad(f"{save_path}/latent_mu.h5ad")
        # from utils.utils_Dandan_plot import plt_umap_byScanpy
        # plt_umap_byScanpy(plt_image_adata.copy(), ["time"],
        #                   save_path=save_path, mode=None, figure_size=(7, 6), color_map="viridis")
        #  -------------- TASK: and identify time-cor gene  ----------------------------------------------
        gc.collect()
        if args.identify_time_cor_gene:
            identify_timeCorGene(sc_expression_train, cell_time, y_time_nor_train,
                                 donor_index_train, runner, experiment, total_result,
                                 special_path_str, config, parallel_bool=False)

    _logger.info("Finish all.")


if __name__ == '__main__':
    main()
