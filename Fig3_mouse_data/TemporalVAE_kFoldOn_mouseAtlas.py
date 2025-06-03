# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：TemporalVAE_kFoldOn_mouseAtlas.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/7/31 11:37

2023-07-31 11:38:33
train on mouse_embryonic_development data,

cd /mnt/yijun/nfs_share/awa_project/awa_github/TemporalVAE/
source ~/.bashrc
nohup python -u Fig3_mouse_data/TemporalVAE_kFoldOn_mouseAtlas.py --result_save_path 240611_mouseAtlas_test --kfold_test --train_whole_model >> logs/VAE_mouse_kFoldOn_mouseAtlas.log 2>&1 &
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
from utils.utils_project import str2bool, auto_select_gpu_and_cpu, preprocessData_and_dropout_some_donor_or_gene
from utils.utils_project import task_kFoldTest, onlyTrain_model, identify_timeCorGene
from collections import Counter

import yaml
import argparse
from utils.utils_plot import Embryodonor_resort_key
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="CNN model for prediction of gene paris' regulatory relationship")
    parser.add_argument('--result_save_path', type=str,  # 2023-07-13 17:40:22
                        default="Fig3_TemporalVAE_kfoldOn_mouseAtlas_240901/",
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
    parser.add_argument('--kfold_test', action="store_true",
                        help="(Optional) make the task k fold test on dataset.", default=True)
    parser.add_argument('--train_whole_model', action="store_true",
                        help="(Optional) use all data to train a model.", default=True)
    parser.add_argument('--identify_time_cor_gene', action="store_true",
                        help="(Optional) identify time-cor gene by model trained by all.", default=False)

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
    if args.use_checkpoint_bool:
        checkpoint_file = 'checkpoint_files/mouse_atlas.ckpt'
    else:
        checkpoint_file = None
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
    device = auto_select_gpu_and_cpu()
    _logger.info("Auto select run on {}".format(device))
    _logger.info("load vae model parameters from file: {}".format(yaml_path + args.vae_param_file + ".yaml"))
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

    #  ------------------ TASK: use all data to train a model  ----------------------------------------------

    if args.train_whole_model:
        sc_expression_train, y_time_nor_train, donor_index_train, runner, experiment, _m, train_clf_result, label_dic, total_result = onlyTrain_model(
            sc_expression_df, donor_dic,
            special_path_str,
            cell_time,
            time_standard_type, config, args,
            device=device, plot_latentSpaceUmap=False, time_saved_asFloat=True,
            batch_size=int(args.batch_size),
            checkpoint_file=checkpoint_file)  # 2023-10-24 17:44:31 batch as 10,000 due to overfit, batch size as 100,000 may be have different result
        import anndata as ad
        save_path = f"{_logger.root.handlers[0].baseFilename.replace('.log', '')}/"
        plt_image_adata = ad.AnnData(X=total_result["mu"].cpu().numpy())
        plt_image_adata.obs = cell_time
        plt_image_adata.write_h5ad(f"{save_path}/latent_mu.h5ad")
        from utils.utils_plot import plt_umap_byScanpy
        # plt_umap_byScanpy(plt_image_adata.copy(), ["time", "celltype_update"], save_path=save_path,mode=None)
        plt_umap_byScanpy(plt_image_adata.copy(), ["time"],
                          save_path=save_path, mode=None, figure_size=(7, 6), color_map="viridis")
        #  -------------- TASK: and identify time-cor gene  ----------------------------------------------
        if args.identify_time_cor_gene:
            identify_timeCorGene(sc_expression_train, cell_time,y_time_nor_train, donor_index_train, runner, experiment, total_result,
                                 special_path_str, config, parallel_bool=False)


    # ---------------TASK : K-FOLD TEST--------------------------------------
    if args.kfold_test:
        predict_donors_dic, label_dic = task_kFoldTest(donor_list, sc_expression_df,
                                                           donor_dic, batch_dic, special_path_str, cell_time, time_standard_type,
                                                           config, args.train_epoch_num, _logger, checkpoint_file=checkpoint_file, batch_size=args.batch_size)

        _logger.info("Finish plot image and fold-test.")


    _logger.info("Finish all.")


if __name__ == '__main__':
    main()
