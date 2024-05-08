# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：VAE_mouse_kFoldOnStereoAndAtlas_withOrWithoutAdversarial_lessCellForFastCheck.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/9/12 20:13 


2023-08-29 15:16:49
for mouse embryo stereo, use Brain organ data only
combine 2 dataset, do k-fold test, plot umap, compare with non-adversarial training

2023-09-12 20:14:27
use 1/10 cell to fast check and compare with baseline method.
"""

import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import sys

# sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/CNNC-master/utils")
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/PyTorch-VAE-master")
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
    parser.add_argument('--KNN_smooth_type', type=str,
                        default="mingze",
                        help="KNN smooth method")  # don't use 2023-06-26 14:04:25

    parser.add_argument('--train_epoch_num', type=int,
                        default="3",
                        help="Train epoch num")
    parser.add_argument('--time_standard_type', type=str,
                        default="embryoneg5to5",
                        help="y_time_nor_train standard type may cause different latent space: log2, 0to1, neg1to1, labeldic,sigmoid,logit")

    # supervise_vae supervise_vae_regressionclfdecoder   supervise_vae_regressionclfdecoder_adversarial_0816_012
    parser.add_argument('--vae_param_file', type=str,
                        default="supervise_vae_regressionclfdecoder",
                        help="vae model parameters file.")
    parser.add_argument('--adversarial_train_bool', type=str2bool,
                        default="False",
                        help="if or not use adversarial training.")
    # 2023-08-31 10:25:04 add batch effect source
    parser.add_argument('--batch_effect_source', type=str,
                        default="dataset",
                        help="can select donor or dataset.")
    parser.add_argument('--parallel_bool', type=str2bool,
                        default="True",
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
    logger_file = '{}/{}_dim{}_time{}_epoch{}.log'.format(_path, args.vae_param_file, latent_dim,
                                                          time_standard_type, args.train_epoch_num)
    LogHelper.setup(log_path=logger_file, level='INFO')
    _logger = logging.getLogger(__name__)
    # 2023-09-02 16:29:00
    import multiprocessing
    from multiprocessing import Queue
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

    # 2023-09-12 19:39:08 to fast check performance only use 1/10 cells!!!!!
    sc_expression_df, cell_time = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path, sc_data_file_csv, cell_info_file_csv,
                                                                                donor_attr="donor", min_gene_num=50,
                                                                                external_file_name=external_sc_data_file_csv,
                                                                                external_cell_info_file=external_cell_info_file_csv,
                                                                                random_drop_cell_bool=True)

    special_path_str = "_mouseEmbryonicDevelopment" + "_" + args.time_standard_type  # LH7_1
    # ---------------------------------------- set donor list and dictionary -----------------------------------------------------
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

    # # # ----------------------------------TASK 1: K-FOLD TEST--------------------------------------
    if args.parallel_bool:
        # 2023-08-31 17:21:16 parallel version
        _logger.info("use parallel version and with works {}".format(args.works))

        with multiprocessing.Pool(processes=args.works) as pool:
            processed_results = pool.starmap(process_fold, [
                (fold, donor_list, sc_expression_df, donor_dic, batch_dic, special_path_str, cell_time,
                 time_standard_type, config, args, 50000, args.adversarial_train_bool) for fold in range(len(donor_list))])
        try:
            predict_donors_dic = dict()
            for fold, result in enumerate(processed_results):
                _predict_donors_dic, test_clf_result, label_dic = result
                predict_donors_dic.update(_predict_donors_dic)

        except:
            # time.sleep(random.randint(50, 200))
            while True:
                _logger.info("some embryo don't be calculated. recover from temp file.")
                temp_save_file_name = "{}{}/{}_plot_on_all_test_donor_time{}_temp.json".format(
                    _logger.root.handlers[0].baseFilename.replace(".log", ""),
                    special_path_str,
                    config['model_params']['name'],
                    time_standard_type)
                _logger.info("read saved temp json from".format(temp_save_file_name))
                if os.path.exists(temp_save_file_name):
                    predict_donors_dic = read_saved_temp_result_json(temp_save_file_name)
                    recalculate_donor_list = list(set(donor_list) - set(predict_donors_dic.keys()))
                else:
                    recalculate_donor_list = donor_list

                _logger.info("need to recalculate {}".format(recalculate_donor_list))
                if len(recalculate_donor_list) == 0:
                    _logger.info("All donor has been calculated.")
                    break
                # 创建一个空列表来存储每个元素在第二个列表中的索引
                re_fold_list = [donor_list.index(_e) for _e in recalculate_donor_list if _e in donor_list]
                with multiprocessing.Pool(processes=args.works) as pool:
                    processed_results = pool.starmap(process_fold, [
                        (fold, donor_list, sc_expression_df, donor_dic, batch_dic, special_path_str, cell_time,
                         time_standard_type, config, args, 50000, args.adversarial_train_bool) for fold in re_fold_list])
                for _in, _re in enumerate(processed_results):
                    if _re is not None:
                        _p, test_clf_result, label_dic = _re



    else:
        # no parallel version
        predict_donors_dic = dict()
        for fold in range(len(donor_list)):
            check_memory()
            _logger.info("the {}/{} fold train, use donor-{} as test set".format(fold + 1, len(donor_list), donor_list[fold]))
            device = auto_select_gpu_and_cpu()
            _logger.info("Auto select run on {}".format(device))
            if args.adversarial_train_bool:
                predict_donor_dic, test_clf_result, label_dic = one_fold_test_adversarialTrain(fold, donor_list, sc_expression_df,
                                                                                               donor_dic, batch_dic,
                                                                                               special_path_str, cell_time,
                                                                                               time_standard_type,
                                                                                               config, args.train_epoch_num,
                                                                                               plot_trainingLossLine=False,
                                                                                               plot_latentSpaceUmap=False,
                                                                                               time_saved_asFloat=True,
                                                                                               batch_size=100000)

            else:
                predict_donor_dic, test_clf_result, label_dic = one_fold_test(fold, donor_list, sc_expression_df, donor_dic,
                                                                              batch_dic,
                                                                              special_path_str, cell_time, time_standard_type,
                                                                              config, args.train_epoch_num,
                                                                              device,
                                                                              plot_trainingLossLine=False,
                                                                              plot_latentSpaceUmap=False,
                                                                              time_saved_asFloat=True, batch_size=100000)
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
    if (time_standard_type != "labeldic") & (time_standard_type is not None):
        # if test_clf_result.shape[1] == 1:
        _logger.info("plot predicted time of each test donor is continuous ; and time style is {}.".format(time_standard_type))
        plot_on_each_test_donor_continueTime_windowTime(donor_list, predict_donors_dic, latent_dim,
                                                        time_standard_type, label_orginalAsKey_transAsValue_dic=label_dic,
                                                        model_name=config['model_params']['name'], cell_time=cell_time,
                                                        special_path_str=special_path_str,
                                                        plot_subtype_str="celltype_update", special_str="_celltype_update",
                                                        plot_on_each_cell_type=False)
        plot_on_each_test_donor_continueTime(donor_list, predict_donors_dic, latent_dim,
                                             time_standard_type, label_orginalAsKey_transAsValue_dic=label_dic,
                                             model_name=config['model_params']['name'], cell_time=cell_time,
                                             special_path_str=special_path_str,
                                             plot_subtype_str="celltype_update", special_str="_celltype_update",
                                             plot_on_each_cell_type=False)
    elif time_standard_type == "labeldic":
        _logger.info(
            "plot predicted time of each test donor is discrete ; and label dict: {}.".format(label_dic))
        plot_on_each_test_donor_discreteTime(result_save_path, data_path, donor_list, predict_donors_dic, latent_dim,
                                             time_standard_type, label_orginalAsKey_transAsValue_dic=label_dic,
                                             model_name=config['model_params']['name'], cell_time=cell_time,
                                             special_path_str=special_path_str,
                                             plot_subtype_str="celltype_update", special_str="_celltype_update")
    else:
        _logger.info("Error in label")
        exit(1)
    # 2023-09-13 11:59:02 plot each
    _logger.info("Finish plot image and fold-test.")

    #
    # # # # ------------------------------TASK 2: TEST ON EXTERNAL DATASET--------------------------------------
    # # ------------------------------------- Test on YZ 2 dataset ---------------------------------------------
    # _logger.info("Start task: test on external dataset.")
    # external_data_list = ["test_onYZ/preprocess_Foxa2_tdTomato_mESC/",
    #                       "test_onYZ/preprocess_Noto_GFP_mESC/",
    #                       "test_onYZ_data_0809/preprocess_E8.5_CD_Mutant/",
    #                       "test_onYZ_data_0809/preprocess_E8.5_WT/",
    #                       "test_onYZ_data_0809/preprocess_E9.5_CD_Mutant/",
    #                       "test_onYZ_data_0809/preprocess_E9.5_WT/"]
    # predict_donors_dic = dict()
    #
    # for external_test_path in external_data_list:
    #     external_sc_data_file_csv = external_test_path + "/data_count_hvg.csv"
    #     external_cell_info_file_csv = external_test_path + "/cell_with_time.csv"
    #     cell_time_external = pd.read_csv("{}/{}".format(data_golbal_path, external_cell_info_file_csv), sep="\t", index_col=0)
    #
    #     _logger.info("external dataset is {}".format(external_test_path))
    #     _logger.info("Note: YZ's test data don't have donor info!")
    #     test_donor = np.unique(cell_time_external["donor"])[0]
    #     # ----------------------------Preprocess data, don't drop any donor.----------------------------------------
    #     sc_expression_df, cell_time = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path, sc_data_file_csv,
    #                                                                                 KNN_smooth_type, cell_info_file_csv,
    #                                                                                 attr="donor",
    #                                                                                 min_cell_num=50, min_gene_num=50,
    #                                                                                 external_file_name=external_sc_data_file_csv,
    #                                                                                 external_cell_info_file=external_cell_info_file_csv)
    #     _logger.info("For each donor (donor_id, cell_num):{} ".format(Counter(cell_time["donor"])))
    #     # ---------------------------------------- set donor list and dictionary -----------------------------------------------------
    #     sc_expression_train = sc_expression_df.loc[cell_time.index[cell_time["donor"] != test_donor]]
    #     sc_expression_test = sc_expression_df.loc[cell_time.index[cell_time["donor"] == test_donor]]
    #     train_donor_list = list(set(cell_time["donor"]))
    #     train_donor_list.remove(test_donor)
    #     train_donor_list = sorted(train_donor_list, key=Embryodonor_resort_key)
    #     train_donor_dic = dict()
    #     for i in range(len(train_donor_list)):
    #         train_donor_dic[train_donor_list[i]] = i
    #     _logger.info("Consider donor as batch effect, donor use label: {}".format(train_donor_dic))
    #     # # ---------------------------------------------- use all data to train a model and identify time-cor gene --------------------------------------------------
    #     sc_expression_train, y_time_nor_train, donor_index_train, runner, experiment, _m, train_clf_result, label_dic, _ = onlyTrain_model(
    #         sc_expression_train,
    #         train_donor_dic,
    #         result_save_path, data_path,
    #         latent_dim, special_path_str,
    #         cell_time,
    #         time_standard_type, config, args,
    #         device, time_saved_asFloat=True,
    #         plot_latentSpaceUmap=False, batch_size=50000)
    #     # -------------------------------------------------- test on new dataset --------------------------------------
    #     predict_donors_dic, test_clf_result, _ = test_on_newDonor(test_donor, sc_expression_test, runner, experiment,
    #                                                               predict_donors_dic)
    #     del runner
    #     del _m
    #     del experiment
    #     # 清除CUDA缓存
    #     torch.cuda.empty_cache()
    #
    # # --------------------------------------- save all results ----------------------------------------------------------
    # _save_path = "{}{}/".format(_logger.root.handlers[0].baseFilename.replace(".log", ""), special_path_str)
    # if not os.path.exists(_save_path):
    #     os.makedirs(_save_path)
    # _save_file_name = "{}/{}_testOnExternal_YZ_2data.json".format(_save_path, config['model_params']['name'])
    #
    # import json
    # save_dic = dict()
    # for _donor, val in predict_donors_dic.items():
    #     save_dic[_donor + "_pseudotime"] = list(val["pseudotime"].astype(float))
    #     save_dic[_donor + "_cellid"] = list(val.index)
    # with open(_save_file_name, 'w') as f:
    #     json.dump(save_dic, f)  # 2023-07-03 22:31:50
    # _logger.info("Finish save clf result at: {}".format(_save_file_name, ))
    # #  ---------------- plot total result directly from just saved json file --------------------------------------------------
    # from read_json_result_and_plot import plot_time_window_fromJSON
    # plot_time_window_fromJSON(json_file=_save_file_name, threshold_list=sorted(list(label_dic.values())))

    _logger.info("Finish all.")


if __name__ == '__main__':
    main()
