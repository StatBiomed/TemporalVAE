# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE
@File    ：VAE_mouse_kFoldOnStereo_trainOnAtlasAndSubStereo_withOrWithoutAdversarial.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/10/2 15:18 


2023-10-02 15:22:18
for mouse embryo stereo, use Brain organ data only
to illustrate, integration with mouse development atlas can improve prediction performance on stereo-seq data.
k-fold on stereo-seq data: one embryo from stereo-seq data as test data, other embryos from stereo and all atlas as train.

"""

import os

import pandas as pd

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import sys

# sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/CNNC-master/utils")
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/model_master")
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan")

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
    # parser.add_argument('--KNN_smooth_type', type=str,
    #                     default="mingze",
    #                     help="KNN smooth method")  # don't use 2023-06-26 14:04:25
    # ------------------ model training setting ------------------
    parser.add_argument('--train_epoch_num', type=int,
                        default="300",
                        help="Train epoch num")
    parser.add_argument('--batch_size', type=int,
                        default=200000,
                        help="batch size")
    parser.add_argument('--time_standard_type', type=str,
                        default="embryoneg5to5",
                        help="y_time_nor_train standard type may cause different latent space: log2, 0to1, neg1to1, labeldic,sigmoid,logit")

    parser.add_argument('--vae_param_file', type=str,
                        default="supervise_vae_regressionclfdecoder_adversarial0121212_231001_06",
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
    parser.add_argument('--use_checkpoint_bool', type=str2bool,
                        default="False",
                        help="use checkpoint file as pre-trained model or not.")

    args = parser.parse_args()

    data_golbal_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/"
    result_save_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/" + args.result_save_path + "/"
    data_path = args.file_path + "/"
    yaml_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/vae_model_configs/"
    if args.use_checkpoint_bool:
        checkpoint_file = '/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/' \
                          '231020_plotLatentSpace_mouse_data_minGene50_hvg1000CalByEachOrgan_timeCorGene/' \
                          'mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/' \
                          'supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch200_minGeneNum100/' \
                          'wholeData/SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial/version_0/checkpoints/last.ckpt'
    else:
        checkpoint_file = None
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
    # 2023-09-12 19:39:08 to fast check performance only use 1/10 cells,
    # 2023-09-12 20:17:14 change to new .py VAE_trainOn_mouse_data_adversarial_lessCell_forFastCheck
    # 2023-10-05 20:00:55 use a stereo-seq cell list to filter the stereo data
    if "Brain" in args.external_test_path:
        organ = "Brain"
    elif "Heart" in args.external_test_path:
        organ = "Heart"
    elif "Liver" in args.external_test_path:
        organ = "Liver"

    stereo_cell_df = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/" \
                     "231004_trainOn_mouse_embryo_stereo_organs_kFold_minGene50_75_100/mouse_embryo_stereo/" \
                     f"preprocess_Mouse_embryo_all_stage_minGene50_of{organ}/" \
                     f"supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch100_minGeneNum{args.min_gene_num}_mouseEmbryonicDevelopment_embryoneg5to5/" \
                     "preprocessed_cell_info.csv"
    _logger.info(f"use stereo cell file {stereo_cell_df}")
    stereo_cellId = list(pd.read_csv(stereo_cell_df, index_col=0).index)
    sc_expression_df, cell_time = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path, sc_data_file_csv, cell_info_file_csv,
                                                                                donor_attr="donor",
                                                                                min_cell_num=args.min_cell_num,
                                                                                min_gene_num=args.min_gene_num,
                                                                                external_file_name=external_sc_data_file_csv,
                                                                                external_cell_info_file=external_cell_info_file_csv,
                                                                                external_cellId_list=stereo_cellId)
    special_path_str = ""
    # ---------------------------------------- set donor list and dictionary ----------------------------------
    donor_list = np.unique(cell_time["donor"])
    donor_list = sorted(donor_list, key=Embryodonor_resort_key)
    donor_dic = dict()
    for i in range(len(donor_list)):
        donor_dic[donor_list[i]] = i
    _logger.info("donor dic: {}".format(args.batch_effect_source, donor_dic))
    _logger.info("For each donor (donor_id, cell_num):{} ".format(Counter(cell_time["donor"])))

    # ----------------------------------------- set batch effect id -----------------------------------------------------
    # 2023-08-31 10:21:17 add variable batch_dic: atlas is 0, stereo is 1
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

    # # # ----------------------------------TASK 1: K-FOLD TEST--------------------------------------
    if args.parallel_bool:
        # 2023-08-31 17:21:16 parallel version
        _logger.info("use parallel version and with works {}".format(args.works))

        with multiprocessing.Pool(processes=args.works) as pool:
            processed_results = pool.starmap(process_fold, [
                (fold, test_donor_list, sc_expression_df, donor_dic, batch_dic, special_path_str, cell_time,
                 time_standard_type, config, args, args.batch_size, args.adversarial_train_bool, checkpoint_file) for fold in range(len(test_donor_list))])
        try:
            predict_donors_dic = dict()
            for fold, result in enumerate(processed_results):
                _predict_donors_dic, test_clf_result, label_dic = result
                predict_donors_dic.update(_predict_donors_dic)
        except:
            # time.sleep(60 * random.randint(5, 20))
            while True:
                _logger.info("some embryo don't be calculated. recover from temp file.")
                temp_save_file_name = "{}{}/{}_plot_on_all_test_donor_time{}{}".format(
                    _logger.root.handlers[0].baseFilename.replace(".log", ""),
                    special_path_str,
                    config['model_params']['name'], time_standard_type, "_temp.json")
                _logger.info("read saved temp json from".format(temp_save_file_name))
                if os.path.exists(temp_save_file_name):
                    predict_donors_dic = read_saved_temp_result_json(temp_save_file_name)
                    recalculate_donor_list = list(set(test_donor_list) - set(predict_donors_dic.keys()))
                else:
                    recalculate_donor_list = test_donor_list
                _logger.info("need to recalculate {}".format(recalculate_donor_list))
                if len(recalculate_donor_list) == 0:
                    _logger.info("All donor has been calculated.")
                    break
                # 创建一个空列表来存储每个元素在第二个列表中的索引
                re_fold_list = [test_donor_list.index(_e) for _e in recalculate_donor_list if _e in test_donor_list]
                with multiprocessing.Pool(processes=args.works) as pool:
                    processed_results = pool.starmap(process_fold, [
                        (fold, test_donor_list, sc_expression_df, donor_dic, batch_dic, special_path_str, cell_time,
                         time_standard_type, config, args, args.batch_size, args.adversarial_train_bool, checkpoint_file) for fold in re_fold_list])
                # 2023-09-20 00:02:01
                for _in, _re in enumerate(processed_results):
                    if _re is not None:
                        _p, test_clf_result, label_dic = _re
    else:
        # no parallel version
        predict_donors_dic = dict()
        for fold in range(len(test_donor_list)):
            check_memory()
            _logger.info(
                "the {}/{} fold train, use donor-{} as test set".format(fold + 1, len(test_donor_list), test_donor_list[fold]))
            device = auto_select_gpu_and_cpu()
            _logger.info("Auto select run on {}".format(device))
            if args.adversarial_train_bool:
                predict_donor_dic, test_clf_result, label_dic = one_fold_test_adversarialTrain(fold, test_donor_list,
                                                                                               sc_expression_df,
                                                                                               donor_dic, batch_dic,
                                                                                               special_path_str, cell_time,
                                                                                               time_standard_type,
                                                                                               config, args.train_epoch_num,
                                                                                               plot_trainingLossLine=True,
                                                                                               plot_latentSpaceUmap=False,
                                                                                               time_saved_asFloat=True,
                                                                                               batch_size=args.batch_size,
                                                                                               checkpoint_file=checkpoint_file)

            else:
                predict_donor_dic, test_clf_result, label_dic = one_fold_test(fold, test_donor_list, sc_expression_df, donor_dic,
                                                                              batch_dic,
                                                                              special_path_str, cell_time, time_standard_type,
                                                                              config, args.train_epoch_num,
                                                                              device,
                                                                              plot_trainingLossLine=False,
                                                                              plot_latentSpaceUmap=False,
                                                                              time_saved_asFloat=True, batch_size=args.batch_size,
                                                                              checkpoint_file=checkpoint_file)
            temp_save_dic(special_path_str, config, time_standard_type, predict_donor_dic.copy(), label_dic)
            predict_donors_dic.update(predict_donor_dic)
    # # ---------------------------------------------- plot total result  --------------------------------------------------
    #  2023-09-09 00:10:01 for fast get result
    # label_dic = {850: -5.0, 875: -4.756, 900: -4.512, 925: -4.268, 950: -4.024, 975: -3.78, 1000: -3.537, 1025: -3.293,
    #              1050: -3.049, 1075: -2.805, 1100: -2.561, 1125: -2.317, 1150: -2.073, 1175: -1.829, 1200: -1.585, 1225: -1.341,
    #              1250: -1.098, 1275: -0.854, 1300: -0.61, 1325: -0.366, 1350: -0.122, 1375: 0.122, 1400: 0.366, 1425: 0.61,
    #              1433: 0.688, 1450: 0.854, 1475: 1.098, 1500: 1.341, 1525: 1.585, 1550: 1.829, 1575: 2.073, 1600: 2.317,
    #              1625: 2.561, 1650: 2.805, 1675: 3.049, 1700: 3.293, 1725: 3.537, 1750: 3.78, 1775: 4.024, 1800: 4.268,
    #              1825: 4.512, 1850: 4.756, 1875: 5.0}
    # if test_clf_result.shape[1] == 1:
    predict_donors_df = pd.DataFrame(columns=["pseudotime"])
    for fold in range(len(test_donor_list)):
        predict_donors_df = pd.concat([predict_donors_df, predict_donors_dic[test_donor_list[fold]]])
    predict_donors_df['predicted_time'] = predict_donors_df['pseudotime'].apply(denormalize, args=(min(label_dic.keys()) / 100, max(label_dic.keys()) / 100,
                                                                                                   min(label_dic.values()), max(label_dic.values())))
    cell_time_stereo = pd.concat([cell_time.copy(), predict_donors_df], axis=1, join="inner")
    save_path = _logger.root.handlers[0].baseFilename.replace('.log', '')
    cell_time_stereo.to_csv(f"{save_path}/k_fold_test_result.csv")

    _logger.info("plot predicted time of each test donor is continuous ; and time style is {}.".format(time_standard_type))
    color_dic = plot_on_each_test_donor_violin_fromDF(cell_time_stereo.copy(), save_path, physical_str="predicted_time", x_str="time")

    plot_on_each_test_donor_continueTime_windowTime(test_donor_list, predict_donors_dic,
                                                    latent_dim,
                                                    time_standard_type, label_orginalAsKey_transAsValue_dic=label_dic,
                                                    model_name=config['model_params']['name'], cell_time=cell_time,
                                                    special_path_str=special_path_str,
                                                    plot_subtype_str="celltype_update", special_str="_celltype_update",
                                                    plot_on_each_cell_type=False)
    plot_on_each_test_donor_continueTime(test_donor_list, predict_donors_dic, latent_dim,
                                         time_standard_type, label_orginalAsKey_transAsValue_dic=label_dic,
                                         model_name=config['model_params']['name'], cell_time=cell_time,
                                         special_path_str=special_path_str,
                                         plot_subtype_str="celltype_update", special_str="_celltype_update",
                                         plot_on_each_cell_type=False)


    #  ---------------------------------------------- TASK: use all data to train a model  ----------------------------------------------
    # plot latent space
    sc_expression_train, y_time_nor_train, donor_index_train, runner, experiment, _m, train_clf_result, label_dic, total_result = onlyTrain_model(
        sc_expression_df, donor_dic,
        special_path_str,
        cell_time,
        time_standard_type, config, args, batch_size=args.batch_size,
        plot_latentSpaceUmap=False, time_saved_asFloat=True,
        adversarial_bool=args.adversarial_train_bool, batch_dic=batch_dic, checkpoint_file=checkpoint_file)
    import anndata as ad
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

    latent_mu_result_adata.write_h5ad(f"{save_path}/latent_mu.h5ad")
    latent_mu_result_adata_stereo = latent_mu_result_adata[latent_mu_result_adata.obs["donor"].isin(test_donor_list)].copy()
    latent_mu_result_adata_atlas = latent_mu_result_adata[~latent_mu_result_adata.obs["donor"].isin(test_donor_list)].copy()

    # plot pc1 pc2
    _logger.info("plot pca, pc1 pc2")
    from draw_images.read_json_plotViolin_oneTimeMulitDonor import cal_pc1pc2
    cal_pc1pc2(latent_mu_result_adata_stereo, "day", save_path=save_path, ncol=2)
    cal_pc1pc2(latent_mu_result_adata_atlas, "day", save_path=save_path, ncol=2)
    cal_pc1pc2(latent_mu_result_adata, "atlas_stereo", save_path=save_path, ncol=2)

    # plot umap
    _logger.info("plot umap")
    # plt_umap_byScanpy(latent_mu_result_adata_stereo.copy(), ["time"], save_path=save_path, mode=None, figure_size=(7, 6))
    plt_umap_byScanpy(latent_mu_result_adata_stereo.copy(), ["time"],
                      save_path=save_path, mode=None, figure_size=(7, 6),color_map="viridis")



if __name__ == '__main__':
    main()
