# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：VAE_human_directPredict.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/3/25 17:11
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import sys

# sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/CNNC-master/utils")
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/model_master")
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
import os
import argparse

from models import vae_models
from utils.utils_Dandan_plot import *
import numpy as np
from multiprocessing import Queue
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning import seed_everything
from experiment import VAEXperiment
from dataset import SupervisedVAEDataset_onlyPredict
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="CNN model for prediction of gene paris' regulatory relationship")
    parser.add_argument('--result_save_path', type=str,  # 2023-07-13 17:40:22
                        default="240327_humanEmbryo/noTimeLabel_directPredict_240401/",
                        help="results all save here")
    parser.add_argument('--file_path', type=str,
                        # default="/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene100/",
                        # default="/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/",
                        default="/240322Human_embryo/xiang2019/hvg500/",
                        help="first sc file folder path.")
    # 2023-08-29 15:35:42 here external test data is the second dataset to combine
    parser.add_argument('--external_test_path', type=str,
                        # default="/240322Human_embryo/okubo2023/",
                        default="/240322Human_embryo/PLOS2019/",
                        # default="/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50/",
                        # default="/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofBrain/",
                        help="second sc file path.")
    # ------------------ preprocess sc data setting ------------------
    parser.add_argument('--min_gene_num', type=int,
                        default="50",
                        help="filter cell with min gene num, default 50")
    parser.add_argument('--min_cell_num', type=int,
                        default="50",
                        help="filter gene with min cell num, default 50")

    # ------------------ model training setting ------------------

    parser.add_argument('--time_standard_type', type=str,
                        default="embryoneg5to5",
                        help="y_time_nor_train standard type may cause different latent space: log2, 0to1, neg1to1, labeldic,sigmoid,logit")

    parser.add_argument('--vae_param_file', type=str,
                        default="supervise_vae_regressionclfdecoder_mouse_stereo",
                        help="vae model parameters file.")

    args = parser.parse_args()

    data_golbal_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/"
    result_save_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/" + args.result_save_path + "/"
    data_path = args.file_path + "/" + args.external_test_path + "/"
    yaml_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/vae_model_configs/"
    checkpoint_file = '/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/240402_humanEmbryo/240322Human_embryo/xiang2019/hvg500/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch70_minGeneNum50/wholeData/SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial/version_0/checkpoints/last.ckpt'
    # checkpoint_file = '/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/240323_humanEmbryo/240322Human_embryo/xiang2019/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch100_minGeneNum50/wholeData/SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial/version_5/checkpoints/last.ckpt'
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
    logger_file = '{}/{}_dim{}_time{}_externalData{}_minGeneNum{}.log'.format(_path,
                                                                              args.vae_param_file,
                                                                              latent_dim,
                                                                              time_standard_type,
                                                                              args.external_test_path.split("_")[-1].replace("/", ""),
                                                                              args.min_gene_num)
    if not os.path.exists(logger_file.replace(".log", "")):
        os.makedirs(logger_file.replace(".log", ""))
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
    # 2023-09-12 19:39:08 to fast check performance only use 1/10 cells,
    # 2023-09-12 20:17:14 change to new .py VAE_trainOn_mouse_data_adversarial_lessCell_forFastCheck
    # 2023-10-05 20:00:55 use a stereo-seq cell list to filter the stereo data
    # if "Brain" in args.external_test_path:
    #     organ = "Brain"
    # elif "Heart" in args.external_test_path:
    #     organ = "Heart"
    # elif "Liver" in args.external_test_path:
    #     organ = "Liver"
    #
    # stereo_cell_df = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/" \
    #                  "231004_trainOn_mouse_embryo_stereo_organs_kFold_minGene50_75_100/mouse_embryo_stereo/" \
    #                  f"preprocess_Mouse_embryo_all_stage_minGene50_of{organ}/" \
    #                  f"supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch100_minGeneNum{args.min_gene_num}_mouseEmbryonicDevelopment_embryoneg5to5/" \
    #                  "preprocessed_cell_info.csv"
    # _logger.info(f"use stereo cell file {stereo_cell_df}")
    # stereo_cellId = list(pd.read_csv(stereo_cell_df, index_col=0).index)

    gene_dic = geneId_geneName_dic()

    try:
        adata_stereo = anndata.read_csv(f"{data_golbal_path}/{external_sc_data_file_csv}", delimiter='\t')
    except:
        adata_stereo = anndata.read_csv(f"{data_golbal_path}/{external_sc_data_file_csv}", delimiter=',')
    cell_time_stereo = pd.read_csv(f"{data_golbal_path}/{external_cell_info_file_csv}", sep="\t", index_col=0)
    # cell_time_stereo = cell_time_stereo.loc[stereo_cellId]
    # adata_stereo = adata_stereo[:, stereo_cellId]
    mouse_atlas_file = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/240322Human_embryo/xiang2019/hvg500/data_count_hvg.csv"
    adata_stereo = adata_stereo.copy().T

    adata_stereo.obs = cell_time_stereo
    # 计算每个细胞类型的数量
    trainData_renormalized_df, loss_gene_shortName_list, train_cell_info_df = predict_newData_preprocess_df(gene_dic, adata_stereo,
                                                                                                            min_gene_num=0,
                                                                                                            mouse_atlas_file=mouse_atlas_file,
                                                                                                            bool_change_geneID_to_geneShortName=False
                                                                                                            )

    special_path_str = ""
    # ---------------------------------------- set donor list and dictionary ----------------------------------

    # 2024-02-23 14:26:17 add only predict on stereo data
    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    # 去掉每层名字前面的 "model."
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        # 去掉前缀 "model."
        if key.startswith('model.'):
            key = key[6:]
        new_state_dict[key] = value
    # MyVAEModel = vae_models[config['model_params']['name']](**config['model_params'])
    config['model_params']['in_channels'] = trainData_renormalized_df.values.shape[1]
    MyVAEModel = vae_models["SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial"](**config['model_params'])
    MyVAEModel.load_state_dict(new_state_dict)
    MyVAEModel.eval()
    check_memory()
    # device = auto_select_gpu_and_cpu()
    device = auto_select_gpu_and_cpu(free_thre=5, max_attempts=100000000)  # device: e.g. "cuda:0"
    runner = Trainer(devices=[int(device.split(":")[-1])])
    seed_everything(config['exp_params']['manual_seed'], True)
    #
    x_sc = torch.tensor(trainData_renormalized_df.values, dtype=torch.get_default_dtype()).t()
    data_x = [[x_sc[:, i], 0, 0] for i in range(x_sc.shape[1])]

    # predict batch size will not influence the training
    data_predict = SupervisedVAEDataset_onlyPredict(predict_data=data_x, predict_batch_size=len(data_x))

    experiment = VAEXperiment(MyVAEModel, config['exp_params'])
    # z=experiment.predict_step(data_predict,1)
    train_result = runner.predict(experiment, data_predict)
    pseudoTime_directly_predict_by_pretrained_model = train_result[0][0]
    pseudoTime_directly_predict_by_pretrained_model_df = pd.DataFrame(pseudoTime_directly_predict_by_pretrained_model, columns=["pseudotime_by_preTrained_mouseAtlas_model"])
    pseudoTime_directly_predict_by_pretrained_model_df.index = trainData_renormalized_df.index
    pseudoTime_directly_predict_by_pretrained_model_df["physical_pseudotime_by_preTrained_mouseAtlas_model"] = pseudoTime_directly_predict_by_pretrained_model_df[
        "pseudotime_by_preTrained_mouseAtlas_model"].apply(denormalize, args=(6, 14, -5, 5))
    mu_predict_by_pretrained_model = train_result[0][1].cpu().numpy()

    save_path = _logger.root.handlers[0].baseFilename.replace('.log', '')
    _logger.info(f"plot result save at {save_path}")
    cell_time_stereo = pd.concat([cell_time_stereo, pseudoTime_directly_predict_by_pretrained_model_df], axis=1)
    color_dic = plot_violin_240223(cell_time_stereo, save_path)
    # print(color_dic)
    color_dic = {f"day{str(key)}": value for key, value in color_dic.items()}

    plot_umap_240223(mu_predict_by_pretrained_model, cell_time_stereo, color_dic=color_dic, save_path=save_path, attr_str="day")
    plot_umap_240223(mu_predict_by_pretrained_model, cell_time_stereo, save_path=save_path, attr_str="Stage")

    _logger.info("Finish all.")


if __name__ == '__main__':
    main()
