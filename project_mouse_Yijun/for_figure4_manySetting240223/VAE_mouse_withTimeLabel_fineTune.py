# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：VAE_mouse_withTimeLabel_fineTune.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024-02-25 14:25:46

2024-02-25 14:25:54
for setting:
use pre-trained model (by atlas data),
then use pre-trained model predicted time of stereo data as pseudo-time label,
then train source predictor (remove batch effect) ,
then fune-tuing on stereo, which use  sample with degree of confidence pseudotime as time label.

2024-02-23 11:18:38
No time label:
● direct prediction (baseline)
● adversarial (query: no time predictor, no decoder?)
● optional: fine-tuning (no adversarial;)
With time label:
● options 1, 2, 3 (presented in Fig. 4)
● option 4: similar to panel C, consider simpler fine-tuning.

"""

import os



os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
import sys

# sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/CNNC-master/utils")
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/model_master")
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan")
from utils.GPU_manager_pytorch import check_memory

import torch

torch.set_float32_matmul_precision('high')
import pyro

from utils.logging_system import LogHelper

smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
assert pyro.__version__.startswith('1.8.5')
pyro.set_rng_seed(1)
# from args_config import save_yaml_config
from utils.utils_DandanProject import *
import os
import argparse

# from models import *
from utils.utils_Dandan_plot import *
from multiprocessing import Queue
import yaml
import pandas as pd
def main():
    parser = argparse.ArgumentParser(description="CNN model for prediction of gene paris' regulatory relationship")
    parser.add_argument('--result_save_path', type=str,  # 2023-07-13 17:40:22
                        default="240225_Figure4_manySetting/noTimeLabel_adversarial_fineTune/",
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
    data_path = args.file_path + "/"
    yaml_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/vae_model_configs/"
    checkpoint_file = '/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/' \
                      '231020_plotLatentSpace_mouse_data_minGene50_hvg1000CalByEachOrgan_timeCorGene/' \
                      'mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/' \
                      'supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch200_minGeneNum100/' \
                      'wholeData/SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial/version_0/checkpoints/last.ckpt'
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
    if not os.path.exists(logger_file.replace(".log","")):
        os.makedirs(logger_file.replace(".log",""))
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
    # sc_expression_df, cell_time = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path, sc_data_file_csv, cell_info_file_csv,
    #                                                                             donor_attr="donor",
    #                                                                             min_cell_num=args.min_cell_num,
    #                                                                             min_gene_num=args.min_gene_num,
    #                                                                             external_file_name=external_sc_data_file_csv,
    #                                                                             external_cell_info_file=external_cell_info_file_csv,
    #                                                                             external_cellId_list=stereo_cellId)
    # sc_expression_df_stereo=sc_expression_df.loc[stereo_cellId]
    # cell_time_stereo=cell_time.loc[stereo_cellId]
    #
    gene_dic = geneId_geneName_dic()

    try:
        adata_stereo = anndata.read_csv(f"{data_golbal_path}/{external_sc_data_file_csv}", delimiter='\t')
    except:
        adata_stereo = anndata.read_csv(f"{data_golbal_path}/{external_sc_data_file_csv}", delimiter=',')
    cell_time_stereo = pd.read_csv(f"{data_golbal_path}/{external_cell_info_file_csv}", sep="\t", index_col=0)
    cell_time_stereo=cell_time_stereo.loc[stereo_cellId]
    adata_stereo=adata_stereo[:,stereo_cellId]
    adata_stereo = adata_stereo.copy().T
    adata_stereo.obs=cell_time_stereo
    mouse_atlas_file = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/" \
                       "mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/" \
                       "data_count_hvg.csv"

    trainData_renormalized_df, loss_gene_shortName_list, train_cell_info_df = predict_newData_preprocess_df(gene_dic, adata_stereo,
                                                                                                            min_gene_num=0,
                                                                                                            mouse_atlas_file=mouse_atlas_file,
                                                                                                            bool_change_geneID_to_geneShortName=False
                                                                                                            )


    special_path_str = ""
    # ---------------------------------------- set donor list and dictionary ----------------------------------
    donor_list=list(set(cell_time_stereo["donor"]))
    _logger.info(f"start task: k-fold test with {donor_list}.")
    donor_list = sorted(donor_list, key=Embryodonor_resort_key)
    donor_dic = dict()
    for i in range(len(donor_list)):
        donor_dic[donor_list[i]] = i
    batch_dic = donor_dic.copy()
    # ---------------------------------------- k-fold test ----------------------------------
    predict_donors_dic = dict()
    for fold in range(len(donor_list)):
        predict_donor_dic, test_clf_result, label_dic = one_fold_test(fold, donor_list, trainData_renormalized_df, donor_dic,
                                                                      batch_dic,
                                                                      special_path_str, train_cell_info_df, time_standard_type,
                                                                      config,5,
                                                                      plot_trainingLossLine=True,
                                                                      plot_latentSpaceUmap=False,
                                                                      time_saved_asFloat=True,
                                                                      checkpoint_file=checkpoint_file,min_max_val=(850,1875))
        predict_donors_dic.update(predict_donor_dic)
    predict_donors_df = pd.DataFrame(columns=["pseudotime"])
    for fold in range(len(donor_list)):
        predict_donors_df = pd.concat([predict_donors_df, predict_donors_dic[donor_list[fold]]])
    predict_donors_df['physical_pseudotime'] = predict_donors_df['pseudotime'].apply(denormalize, args=(8.5, 18.75, -5, 5))
    # cell_time = pd.concat([cell_time, predict_donors_df], axis=1)

    save_path=_logger.root.handlers[0].baseFilename.replace('.log', '')
    _logger.info(f"plot result save at {save_path}")
    cell_time_stereo=pd.concat([cell_time_stereo,predict_donors_df],axis=1)


    color_dic=plot_on_each_test_donor_violin_fromDF(cell_time_stereo,save_path,physical_str='physical_pseudotime')
    # plot_umap_240223(mu_predict_by_pretrained_model,cell_time_stereo, color_dic, save_path)

    _logger.info("Finish all.")



if __name__ == '__main__':
    main()
