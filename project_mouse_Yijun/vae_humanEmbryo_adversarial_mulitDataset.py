# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：vae_humanEmbryo_adversarial_mulitDataset.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/4/5 17:55 

cd /mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/
source ~/.bashrc
nohup python -u project_mouse_Yijun/vae_humanEmbryo_adversarial_mulitDataset.py --result_save_path 240407_human_testExponentialLR --vae_param_file supervise_vae_regressionclfdecoder_adversarial0121212_240407_humanEmbryo >> logs/vae_humanEmbryo_adversarial_mulitDataset_240407_human_testExponentialLR2.log 2>&1 &



2024-04-05 17:56:21 code copy from VAE_mouse_kFoldOnStereo_trainOnAtlasAndSubStereo_withOrWithoutAdversarial.py
"""

import os

import pandas as pd

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import sys

# sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/CNNC-master/utils")
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/PyTorch-VAE-master")
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan")
from utils.GPU_manager_pytorch import auto_select_gpu_and_cpu, check_memory

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
import multiprocessing
from multiprocessing import Queue


def main():
    parser = argparse.ArgumentParser(description="CNN model for prediction of gene paris' regulatory relationship")
    parser.add_argument('--result_save_path', type=str,  # 2023-07-13 17:40:22
                        default="test",
                        # default="240405Human_embryo_adversarial_try",
                        help="results all save here")
    parser.add_argument('--file_path', type=str,
                        # default="/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene100/",
                        default="/240322Human_embryo/xiang2019/hvg1000/",
                        help="first sc file folder path.")
    # 2023-08-29 15:35:42 here external test data is the second dataset to combine
    parser.add_argument('--external_test_path', type=str,
                        default="/240322Human_embryo/Tyser2021/",
                        help="second sc file path.")
    # ------------------ preprocess sc data setting ------------------
    parser.add_argument('--min_gene_num', type=int,
                        default="50",
                        help="filter cell with min gene num, default 50")
    parser.add_argument('--min_cell_num', type=int,
                        default="50",
                        help="filter gene with min cell num, default 50")
    # ------------------ model training setting ------------------
    parser.add_argument('--train_epoch_num', type=int,
                        default="80",
                        help="Train epoch num")
    parser.add_argument('--time_standard_type', type=str,
                        default="embryoneg5to5",
                        help="y_time_nor_train standard type may cause different latent space: log2, 0to1, neg1to1, labeldic,sigmoid,logit")
    parser.add_argument('--vae_param_file', type=str,
                        # default="supervise_vae_regressionclfdecoder_adversarial0121212_240405_humanEmbryo_epoch200",
                        # default="supervise_vae_regressionclfdecoder_batchDecoder_240405_humanEmbryo",
                        # default="supervise_vae_regressionclfdecoder_adversarial0121212_240407_humanEmbryo",
                        default="supervise_vae_regressionclfdecoder_mouse_stereo",
                        help="vae model parameters file.")
    parser.add_argument('--adversarial_train_bool', type=str2bool,
                        default="False",
                        help="if or not use adversarial training.")
    # ------------------ task setting ------------------
    parser.add_argument('--kfold_test', action="store_true", help="(Optional) make the task k fold test on dataset.", default=False)
    parser.add_argument('--train_whole_model', action="store_true", help="(Optional) use all data to train a model.", default=True)

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

    time_standard_type = args.time_standard_type
    sc_data_file_csv = data_path + "/data_count_hvg.csv"
    cell_info_file_csv = data_path + "/cell_with_time.csv"

    _path = '{}/{}/'.format(result_save_path, data_path)
    if not os.path.exists(_path):
        os.makedirs(_path)
    logger_file = '{}/{}_dim{}_time{}_epoch{}_externalData{}_minGeneNum{}.log'.format(_path,
                                                                                      args.vae_param_file,
                                                                                      latent_dim,
                                                                                      time_standard_type,
                                                                                      args.train_epoch_num,
                                                                                      args.external_test_path.split("_")[-1].replace("/", ""),
                                                                                      args.min_gene_num)
    LogHelper.setup(log_path=logger_file, level='INFO')
    _logger = logging.getLogger(__name__)
    # 2023-09-02 16:29:00

    # 创建一个 Queue 用于日志消息
    log_queue = Queue()
    # 创建一个 QueueHandler 并将其添加到 logger 中
    queue_handler = logging.handlers.QueueHandler(log_queue)
    _logger.addHandler(queue_handler)
    _logger.info(f"parameters used: \n {' '.join([f'--{key}={value}' for key, value in vars(args).items()])}")
    _logger.info("Finished setting up the logger at: {}.".format(logger_file))
    _logger.info("Train on dataset: {}.".format(data_golbal_path + data_path))

    _logger.info("load vae model parameters from file: {}".format(yaml_path + args.vae_param_file + ".yaml"))

    # ------------ 2023-08-29 15:48:49 Preprocess data, combine first and second sc data ------------------------
    # get the donor info of external dataset
    external_sc_data_file_csv = args.external_test_path + "/data_count_hvg.csv"
    external_cell_info_file_csv = args.external_test_path + "/cell_with_time.csv"
    check_memory()
    sc_expression_df, cell_time = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path, sc_data_file_csv, cell_info_file_csv,
                                                                                donor_attr="donor",
                                                                                min_cell_num=args.min_cell_num,
                                                                                min_gene_num=args.min_gene_num,
                                                                                external_file_name=external_sc_data_file_csv,
                                                                                external_cell_info_file=external_cell_info_file_csv,
                                                                                )
    special_path_str = ""
    # ---------------------------------------- set donor list and dictionary ----------------------------------
    cell_time["donor"] = cell_time["dataset_label"] + "_" + cell_time["day"]
    donor_list = np.unique(cell_time["donor"])
    # donor_list = sorted(donor_list, key=Embryodonor_resort_key)
    # donor_dic = dict()
    # for i in range(len(donor_list)):
    #     donor_dic[donor_list[i]] = i
    _logger.info(f"donor list: {donor_list}")
    _logger.info("For each donor (donor_id, cell_num):{} ".format(Counter(cell_time["donor"])))

    # ----------------------------------------- set batch effect id -----------------------------------------------------
    donor_dic = dict()
    test_donor_list = []
    for _d in donor_list:
        if "xiang2019" in _d:
            donor_dic[_d] = 1
        elif "PLOS" in _d:
            donor_dic[_d] = 0
            test_donor_list.append(_d)
        elif "Tyser" in _d:
            donor_dic[_d]=0
            test_donor_list.append(_d)
    _logger.info(f"donor to batchId dictionary: {donor_dic}")
    batch_dic = donor_dic
    cell_time_test = cell_time.loc[cell_time["dataset_label"] == "PLOS"]
    # ----------------------------------TASK 1: K-FOLD TEST--------------------------------------
    if args.kfold_test:
        predict_donors_dic, label_dic = task_kFoldTest(test_donor_list, sc_expression_df, donor_dic, batch_dic, special_path_str, cell_time_test, time_standard_type,
                                                       config, args.train_epoch_num, _logger, donor_str="donor", batch_size=100000,
                                                       adversarial_bool=args.adversarial_train_bool)

        _logger.info("Finish fold-test.")
    # ----------------------------------test some thing--------------------------------------
    # temp_file_name="/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/test/240322Human_embryo/xiang2019/hvg500/supervise_vae_regressionclfdecoder_adversarial0121212_240407_humanEmbryo_dim50_timeembryoneg5to5_epoch200_externalDataembryo_minGeneNum50//k_fold_test_result.csv"
    # temp_df=pd.read_csv(temp_file_name)
    # temp_plos_df=temp_df.loc[temp_df["dataset_label"]=="PLOS"]
    # temp_color_dic = plot_on_each_test_donor_violin_fromDF(temp_plos_df.copy(), "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/240405Human_embryo_adversarial_try/240322Human_embryo/xiang2019/hvg500/supervise_vae_regressionclfdecoder_adversarial0121212_240405_humanEmbryo_dim50_timeembryoneg5to5_epoch100_externalDataembryo_minGeneNum50",physical_str="predicted_time", x_str="time",special_file_name_str="only_PLOS_")

    #  ---------------------------------------------- TASK: use all data to train a model  ----------------------------------------------
    if args.train_whole_model:
        sc_expression_train, y_time_nor_train, donor_index_train, runner, experiment, _m, train_clf_result, label_dic, total_result = onlyTrain_model(
            sc_expression_df, donor_dic,
            special_path_str,
            cell_time,
            time_standard_type, config, args, batch_size=100000,
            plot_latentSpaceUmap=False, time_saved_asFloat=True,
            adversarial_bool=args.adversarial_train_bool, batch_dic=batch_dic, plot_trainingLossLine=True)
        import anndata as ad
        latent_mu_result_adata = ad.AnnData(X=total_result["mu"].cpu().numpy())
        predict_donors_df = pd.DataFrame(train_clf_result, columns=["pseudotime"], index=cell_time.index)
        predict_donors_df['predicted_time'] = predict_donors_df['pseudotime'].apply(denormalize, args=(min(label_dic.keys()) / 100, max(label_dic.keys()) / 100,
                                                                                                       min(label_dic.values()), max(label_dic.values())))
        cell_time = pd.concat([cell_time, predict_donors_df], axis=1)
        cell_time2 = cell_time.copy(deep=True)
        cell_time2.loc[cell_time2['dataset_label'] == 'PLOS', 'cell_type'] = np.nan
        latent_mu_result_adata.obs = cell_time2
        save_path = _logger.root.handlers[0].baseFilename.replace('.log', '')
        latent_mu_result_adata.write_h5ad(f"{save_path}/latent_mu.h5ad")

        plt_umap_byScanpy(latent_mu_result_adata.copy(), ["time", "predicted_time", "cell_type", "dataset_label", "donor"], save_path=save_path, mode=None, figure_size=(9, 4),
                          color_map="turbo", show_in_row=False, n_neighbors=10, n_pcs=40, special_file_name_str="nN_10_")  # color_map="viridis"
        plt_umap_byScanpy(latent_mu_result_adata.copy(), ["time", "predicted_time", "cell_type", "dataset_label", "donor"], save_path=save_path, mode=None, figure_size=(9, 4),
                          color_map="turbo", show_in_row=False, n_neighbors=20, n_pcs=40, special_file_name_str="nN_20_")  # color_map="viridis"
        plt_umap_byScanpy(latent_mu_result_adata.copy(), ["time", "predicted_time", "cell_type", "dataset_label", "donor"], save_path=save_path, mode=None, figure_size=(9, 4),
                          color_map="turbo", show_in_row=False, n_neighbors=30, n_pcs=40, special_file_name_str="nN_30_")  # color_map="viridis"
        plt_umap_byScanpy(latent_mu_result_adata.copy(), ["time", "predicted_time", "cell_type", "dataset_label", "donor"], save_path=save_path, mode=None, figure_size=(9, 4),
                          color_map="turbo", show_in_row=False, n_neighbors=40, n_pcs=40, special_file_name_str="nN_40_")  # color_map="viridis"
        plt_umap_byScanpy(latent_mu_result_adata.copy(), ["time", "predicted_time", "cell_type", "dataset_label", "donor"], save_path=save_path, mode=None, figure_size=(9, 4),
                          color_map="turbo", show_in_row=False, n_neighbors=50, n_pcs=40, special_file_name_str="nN_50_")  # color_map="viridis"
        # latent_mu_result_adata_PlOS=latent_mu_result_adata[latent_mu_result_adata.obs["dataset_label"] == "PLOS", :]
        # latent_mu_result_adata_xiang=latent_mu_result_adata[latent_mu_result_adata.obs["dataset_label"] == "xiang2019", :]
        # plt_umap_byScanpy(latent_mu_result_adata_PlOS.copy(), ["time", "predicted_time", "cell_type", "dataset_label","donor"], save_path=save_path, mode=None, figure_size=(5, 4), color_map="turbo",show_in_row=False,n_neighbors=10, n_pcs=40)  # color_map="viridis"
        # plt_umap_byScanpy(latent_mu_result_adata_xiang.copy(), ["time", "predicted_time", "cell_type", "dataset_label","donor"], save_path=save_path, mode=None, figure_size=(5, 4), color_map="turbo",show_in_row=False,n_neighbors=10, n_pcs=40)  # color_map="viridis"

    # plot pc1 pc2
    # _logger.info("plot pca, pc1 pc2")
    # from draw_images.read_json_plotViolin_oneTimeMulitDonor import cal_pc1pc2
    # cal_pc1pc2(latent_mu_result_adata_stereo, "day", save_path=save_path, ncol=2)
    # cal_pc1pc2(latent_mu_result_adata_atlas, "day", save_path=save_path, ncol=2)
    # cal_pc1pc2(latent_mu_result_adata, "atlas_stereo", save_path=save_path, ncol=2)

    _logger.info("finish all.")


if __name__ == '__main__':
    main()
