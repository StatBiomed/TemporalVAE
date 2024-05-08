# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：VAE_mouse_plotLatentSpace_trainOnStereoAndAtlas_withOrWithoutAdversarial.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/10/10 18:48 
"""

import os

import pandas as pd

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import sys

# sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/CNNC-master/utils")
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/model_master")
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan")
from utils.GPU_manager_pytorch import check_memory

import torch

torch.set_float32_matmul_precision('high')
import pyro

import logging
from utils.logging_system import LogHelper

smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
assert pyro.__version__.startswith('1.8.5')
pyro.set_rng_seed(1)
# from args_config import save_yaml_config
from utils.utils_DandanProject import *
from collections import Counter
import os
import yaml
import argparse

# from models import *
from utils.utils_Dandan_plot import *
import numpy as np
from multiprocessing import Queue
import anndata as ad


def main():
    parser = argparse.ArgumentParser(description="CNN model for prediction of gene paris' regulatory relationship")
    parser.add_argument('--result_save_path', type=str,  # 2023-07-13 17:40:22
                        default="test",
                        help="results all save here")
    parser.add_argument('--file_path', type=str,
                        # default="/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene100/",
                        default="/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/",
                        help="first sc file folder path.")
    # 2023-08-29 15:35:42 here external test data is the second dataset to combine
    parser.add_argument('--external_test_path', type=str,
                        default="/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofBrain/",
                        help="second sc file path.")
    # ------------------ preprocess sc data setting ------------------
    parser.add_argument('--min_gene_num', type=int,
                        default="100",
                        help="filter cell with min gene num, default 50")
    parser.add_argument('--min_cell_num', type=int,
                        default="50",
                        help="filter gene with min cell num, default 50")
    parser.add_argument('--KNN_smooth_type', type=str,
                        default="mingze",
                        help="KNN smooth method")  # don't use 2023-06-26 14:04:25
    # ------------------ model training setting ------------------
    parser.add_argument('--train_epoch_num', type=int,
                        default="3",
                        help="Train epoch num")
    parser.add_argument('--time_standard_type', type=str,
                        default="embryoneg5to5",
                        help="y_time_nor_train standard type may cause different latent space: log2, 0to1, neg1to1, labeldic,sigmoid,logit")

    # supervise_vae supervise_vae_regressionclfdecoder   supervise_vae_regressionclfdecoder_adversarial_0816_012
    parser.add_argument('--vae_param_file', type=str,
                        default="supervise_vae_regressionclfdecoder_adversarial0121212_231001_06",
                        # default="supervise_vae_regressionclfdecoder_mouse_atlas_stereo_noAdversarial",
                        help="vae model parameters file.")
    parser.add_argument('--adversarial_train_bool', type=str2bool,
                        default="True",
                        help="if or not use adversarial training.")
    # 2023-08-31 10:25:04 add batch effect source
    parser.add_argument('--batch_effect_source', type=str,
                        default="dataset",
                        help="can select donor or dataset.")
    parser.add_argument('--parallel_bool', type=str2bool,
                        default="False",
                        help="parallel version or not.")
    parser.add_argument('--works', type=int,
                        default=3,
                        help="parallel use 3 works.")

    args = parser.parse_args()

    data_golbal_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/"
    result_save_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/" + args.result_save_path + "/"
    data_path = args.file_path + "/"
    yaml_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/vae_model_configs/"
    # --------------------------------------- import vae model parameters from yaml file----------------------------------------------
    with open(yaml_path + "/" + args.vae_param_file + ".yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # ---------------------------------------set logger and parameters, creat result save path and folder-----------------------
    latent_dim = config['model_params']['latent_dim']
    KNN_smooth_type = args.KNN_smooth_type

    time_standard_type = args.time_standard_type
    sc_data_file_csv = data_path + "/data_count_hvg.csv"
    cell_info_file_csv = data_path + "/cell_with_time.csv"

    _path = '{}/{}/'.format(result_save_path, data_path)
    if not os.path.exists(_path):
        os.makedirs(_path)
    logger_file = '{}/{}_dim{}_time{}_epoch{}_externalData{}.log'.format(_path, args.vae_param_file, latent_dim,
                                                                         time_standard_type,
                                                                         args.train_epoch_num,
                                                                         args.external_test_path.split("_")[-1].replace("/", ""))
    LogHelper.setup(log_path=logger_file, level='INFO')
    _logger = logging.getLogger(__name__)
    # 2023-09-02 16:29:00

    # 创建一个 Queue 用于日志消息
    log_queue = Queue()
    # 创建一个 QueueHandler 并将其添加到 logger 中
    queue_handler = logging.handlers.QueueHandler(log_queue)
    _logger.addHandler(queue_handler)

    _logger.info("Finished setting up the logger at: {}.".format(logger_file))
    _logger.info("Train on dataset: {}.".format(data_golbal_path + data_path))

    _logger.info("load vae model parameters from file: {}".format(yaml_path + args.vae_param_file + ".yaml"))

    # ------------ 2023-08-29 15:48:49 Preprocess data, combine first and second sc data ------------------------
    # get the donor info of external dataset
    external_sc_data_file_csv = args.external_test_path + "/data_count_hvg.csv"
    external_cell_info_file_csv = args.external_test_path + "/cell_with_time.csv"
    check_memory()
    # 2023-09-12 19:39:08 to fast check performance only use 1/10 cells,
    # 2023-09-12 20:17:14 change to new .py VAE_trainOn_mouse_data_adversarial_lessCell_forFastCheck
    # 2023-10-05 20:00:55 use a stereo-seq cell list to filter the stereo data
    if "Brain" in args.external_test_path:
        organ = "Brain"
    elif "Heart" in args.external_test_path:
        organ = "Heart"
    elif "Liver" in args.external_test_path:
        organ = "Liver"

    # stereo_cell_df = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/" \
    #                  "231004_trainOn_mouse_embryo_stereo_organs_kFold_minGene50_75_100/mouse_embryo_stereo/" \
    #                  f"preprocess_Mouse_embryo_all_stage_minGene50_of{organ}/" \
    #                  "supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch100_minGeneNum100_mouseEmbryonicDevelopment_embryoneg5to5/" \
    #                  "preprocessed_cell_info.csv"
    # _logger.info(f"use stereo cell file {stereo_cell_df}")
    # stereo_cellId = list(pd.read_csv(stereo_cell_df, index_col=0).index)
    sc_expression_df, cell_time = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path, sc_data_file_csv, cell_info_file_csv,
                                                                                donor_attr="donor",
                                                                                min_cell_num=args.min_cell_num,
                                                                                min_gene_num=args.min_gene_num,
                                                                                external_file_name=external_sc_data_file_csv,
                                                                                external_cell_info_file=external_cell_info_file_csv)

    special_path_str = ""  # LH7_1
    # special_path_str = "_mouseEmbryonicDevelopment" + "_" + args.time_standard_type  # LH7_1
    # ---------------------------------------- set donor list and dictionary ----------------------------------

    donor_list = np.unique(cell_time["donor"])
    donor_list = sorted(donor_list, key=Embryodonor_resort_key)
    donor_dic = dict()
    for i in range(len(donor_list)):
        donor_dic[donor_list[i]] = i
    _logger.info("donor dic: {}".format(args.batch_effect_source, donor_dic))
    _logger.info("For each donor (donor_id, cell_num):{} ".format(Counter(cell_time["donor"])))

    # ----------------------------------------- set batch effect id -----------------------------------------------------
    # 2023-08-31 10:21:17 add variable batch_dic
    if args.batch_effect_source == "donor":
        batch_dic = donor_dic.copy()
    elif args.batch_effect_source == "dataset":
        batch_dic = dict()
        for _d in donor_list:
            batch_dic[_d] = 1 if "stereo" in _d else 0
    _logger.info("Consider {} as batch effect, use label: {}".format(args.batch_effect_source, batch_dic))

    # ---------------------------------------- set test donor list and dictionary ----------------------------------
    _logger.info(f"to illustrate, integration with mouse development atlas can improve prediction performance on stereo-seq data")
    _logger.info(f"Only use K-Fold test on stero-seq data embryo,"
                 f": one embryo from stereo-seq data as test data, other embryos from stereo and all atlas as train.")
    test_donor_list = [_ for _ in donor_list if "stereo" in _]
    test_donor_list = sorted(test_donor_list, key=Embryodonor_resort_key)
    _logger.info(f"k-fold test embryo: {test_donor_list}")

    # # # ------------------- train model by atlas and stereo data, plot latent of stereo data---------------------------------
    sc_expression_train, y_time_nor_train, donor_index_train, runner, experiment, _m, train_clf_result, label_dic, total_result = onlyTrain_model(
        sc_expression_df, donor_dic,
        special_path_str,
        cell_time,
        time_standard_type, config, args, batch_size=100000,
        plot_latentSpaceUmap=False, time_saved_asFloat=True,
        adversarial_bool=args.adversarial_train_bool, batch_dic=batch_dic)
    ''' predict from checkpoint
    ck="/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/231119_plotLatentSpace_trainOnStereoAndAtlas_minGeneNum100_withoutAdversarial/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch300_externalDataofBrain/wholeData/SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial/version_0/checkpoints/last.ckpt"
    spliced_result = read_model_parameters_fromCkpt(sc_expression_df, yaml_path + "/" + args.vae_param_file + ".yaml", ck,adversarial_bool=args.adversarial_train_bool)
    spliced_clf_result, spliced_latent_mu_result = spliced_result[0][0], spliced_result[0][1]
    latent_mu_result_adata = ad.AnnData(X=spliced_latent_mu_result.cpu().numpy())
    '''
    latent_mu_result_adata = ad.AnnData(X=total_result["mu"].cpu().numpy())
    # for stereo data: cell time have nan on columns
    cell_time_atlas = cell_time.loc[~cell_time["donor"].isin(test_donor_list)].copy()
    cell_time_stereo = cell_time.loc[cell_time["donor"].isin(test_donor_list)].copy()
    cell_time_stereo["embryo_id"] = cell_time_stereo["donor"]
    cell_time_stereo["experimental_batch"] = "stereo_batch"
    cell_time_stereo["batch"] = -1
    cell_time_stereo["keep"] = "yes"
    cell_time_stereo["cell"] = cell_time_stereo["cell_id"]
    cell_time_stereo["cell_type"] = cell_time_stereo["celltype_update"]
    cell_time_all = pd.concat([cell_time_atlas, cell_time_stereo])
    if not (cell_time_all.index == cell_time.index).all():
        print(f" concat wrong, please check!")
        exit(1)
    latent_mu_result_adata.obs = cell_time_all

    latent_mu_result_adata.obs["y_time_nor"] = y_time_nor_train
    latent_mu_result_adata.obs["atlas_stereo"] = donor_index_train
    save_file_name = f"{_logger.root.handlers[0].baseFilename.replace('.log', '')}/"
    latent_mu_result_adata.write_h5ad(f"{save_file_name}/latent_mu.h5ad")
    # plot pc1 pc2
    _logger.info("plot pca, pc1 pc2")
    from draw_images.read_json_plotViolin_oneTimeMulitDonor import cal_pc1pc2
    latent_mu_result_adata_stereo = latent_mu_result_adata[latent_mu_result_adata.obs["donor"].isin(test_donor_list)].copy()
    cal_pc1pc2(latent_mu_result_adata_stereo, "day", save_path=save_file_name, ncol=2)
    latent_mu_result_adata_atlas = latent_mu_result_adata[~latent_mu_result_adata.obs["donor"].isin(test_donor_list)].copy()
    cal_pc1pc2(latent_mu_result_adata_atlas, "day", save_path=save_file_name, ncol=2)
    cal_pc1pc2(latent_mu_result_adata, "atlas_stereo", save_path=save_file_name, ncol=2)

    # plot umap
    # _logger.info("plot umap")
    # umap_vae_latent_space( latent_mu_result_adata_stereo.X, latent_mu_result_adata_stereo.obs["y_time_nor"], label_dic, special_path_str, config,
    #                           special_str="trainData_mu_time", drop_batch_dim=0)
    # umap_vae_latent_space(latent_mu_result_adata_stereo.X, latent_mu_result_adata_stereo.obs["donor"], donor_dic, special_path_str, config,
    #                           special_str="trainData_mu_donor", drop_batch_dim=0)

    _logger.info("Finish all.")


if __name__ == '__main__':
    main()
