# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：VAE_mouse_plotLatentSpace_oneDataset.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/10/10 09:28 

train on all data, plot the train set latent space
illustrate the latent space with time information
"""

import os
import sys

# sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/CNNC-master/utils")
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/model_master")
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan")
from utils.GPU_manager_pytorch import auto_select_gpu_and_cpu

import torch

torch.set_float32_matmul_precision('high')
import pyro

import logging
from utils.logging_system import LogHelper

smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
assert pyro.__version__.startswith('1.8.5')
pyro.set_rng_seed(1)
from utils.utils_DandanProject import *
from collections import Counter
import os
import yaml
import argparse
from models import *
from utils.utils_Dandan_plot import *
from multiprocessing import Queue

def main():
    parser = argparse.ArgumentParser(description="CNN model for prediction of gene paris' regulatory relationship")
    parser.add_argument('--result_save_path', type=str,  # 2023-07-13 17:40:22
                        default="test",
                        help="results all save here")
    parser.add_argument('--file_path', type=str,
                        # default="/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene100/",
                        default="/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofLiver/",
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
                        default="4",
                        help="Train epoch num")
    parser.add_argument('--batch_size', type=int,
                        default=100000,
                        help="batch size")
    parser.add_argument('--time_standard_type', type=str,
                        default="embryoneg5to5",
                        help="y_time_nor_train standard type may cause different latent space: log2, 0to1, neg1to1, labeldic,sigmoid,logit")
    # supervise_vae            supervise_vae_regressionclfdecoder
    parser.add_argument('--vae_param_file', type=str,
                        default="supervise_vae_regressionclfdecoder_mouse_stereo",
                        help="vae model parameters file.")

    # Todo, useless, wait to delete "KNN_smooth_type"
    parser.add_argument('--KNN_smooth_type', type=str,
                        default="mingze",
                        help="KNN smooth method")  # don't use 2023-06-26 14:04:25
    # parser.add_argument('--dropout_donor', type=str,
    #                     default="no",  # LH7_1  or no
    #                     help="dropout a donor.")

    # parser.add_argument('--dropout_batch_effect_dim', type=int, default=5,  # LH7_1  or no
    #                     help="Consider donor id as batch effect, set dropout batch effect dim.")

    args = parser.parse_args()

    data_golbal_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/"
    result_save_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/" + args.result_save_path + "/"
    data_path = args.file_path + "/"
    # save_yaml_config(vars(args), path='{}/config.yaml'.format(data_golbal_path + data_path))
    yaml_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/vae_model_configs/"
    # --------------------------------------- import vae model parameters from yaml file----------------------------------------------
    with open(yaml_path + "/" + args.vae_param_file + ".yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # ---------------------------------------set logger and parameters, creat result save path and folder----------------------------------------------

    latent_dim = config['model_params']['latent_dim']
    KNN_smooth_type = args.KNN_smooth_type

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
    # 创建一个 Queue 用于日志消息
    log_queue = Queue()
    # 创建一个 QueueHandler 并将其添加到 logger 中
    queue_handler = logging.handlers.QueueHandler(log_queue)
    _logger.addHandler(queue_handler)

    _logger.info("Finished setting up the logger at: {}.".format(logger_file))
    _logger.info("Train on dataset: {}.".format(data_golbal_path + data_path))
    device = auto_select_gpu_and_cpu()
    _logger.info("Auto select run on {}".format(device))
    _logger.info("load vae model parameters from file: {}".format(yaml_path + args.vae_param_file + ".yaml"))

    # ------------ Preprocess data, with hvg gene from preprocess_data_mouse_embryonic_development.py------------------------
    sc_expression_df, cell_time = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path, sc_data_file_csv,cell_info_file_csv,
                                                                                min_cell_num=args.min_cell_num,
                                                                                min_gene_num=args.min_gene_num,
                                                                                normalized_cellTotalCount=1e6)  # test totalcount to 1e4

    special_path_str = ""
    # special_path_str = "_mouseEmbryonicDevelopment" + "_" + args.time_standard_type  # LH7_1
    # ---------------------------------------- set donor list and dictionary -----------------------------------------------------
    donor_list = np.unique(cell_time["donor"])
    donor_list = sorted(donor_list, key=Embryodonor_resort_key)
    # donor_list = sorted(donor_list, key=Embryodonor_resort_key)
    donor_dic = dict()
    for i in range(len(donor_list)):
        donor_dic[donor_list[i]] = i
    _logger.info("Consider donor as batch effect, donor use label: {}".format(donor_dic))
    _logger.info("For each donor (donor_id, cell_num):{} ".format(Counter(cell_time["donor"])))

    # # ---------------------------------------------- use all data to train a model and identify time-cor gene --------------------------------------------------
    sc_expression_train, y_time_nor_train, donor_index_train, runner, experiment, _m, train_clf_result, label_dic, total_result = onlyTrain_model(
        sc_expression_df, donor_dic,
        special_path_str,
        cell_time,
        time_standard_type, config, args,
        device=device, plot_latentSpaceUmap=False, time_saved_asFloat=True,
        batch_size=int(args.batch_size)) # 2023-10-24 17:44:31 batch as 10,000 due to overfit, batch size as 100,000 may be have different result
    # plot pc1pc2 images
    import anndata as ad
    save_file_name = f"{_logger.root.handlers[0].baseFilename.replace('.log', '')}/"
    plt_image_adata = ad.AnnData(X=total_result["mu"].cpu().numpy())
    plt_image_adata.obs = cell_time
    plt_image_adata.write_h5ad(f"{save_file_name}/latent_mu.h5ad")

    _logger.info("Finish all.")
    # -------------------------------------------------- identify time cor gene --------------------------------------
    # time_cor_gene_pd=identify_timeCorGene(sc_expression_train, y_time_nor_train, donor_index_train, runner, experiment, total_result,
    #                      special_path_str, config,parallel_bool=False)
    #

    _logger.info("Finish all.")


if __name__ == '__main__':
    main()
