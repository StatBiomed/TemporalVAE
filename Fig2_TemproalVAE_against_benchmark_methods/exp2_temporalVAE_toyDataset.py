# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：exp2_temporalVAE_toyDataset.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/9/21 12:18 

use dataset mentioned in psupertime manuscript

"""
import os
import sys

# necessary: change current path to TemporalVAE
# if os.getcwd().split("/")[-1] != "TemporalVAE":
#     os.chdir("..")
sys.path.append(os.getcwd())
from utils.GPU_manager_pytorch import auto_select_gpu_and_cpu, check_memory
import logging
from utils.logging_system import LogHelper
from utils.utils_project import trans_time, plot_training_loss_for_tags, denormalize
import anndata
import pandas as pd

import numpy as np
from model_master.experiment_temporalVAE import temporalVAEExperiment
from model_master.dataset import SupervisedVAEDataset, SupervisedVAEDataset_onlyPredict
from model_master import vae_models

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pathlib import Path

from plotFig2_check_corr import preprocess_parameters, corr


def main():
    dataset_list = ["embryoBeta","humanGermline", "acinarHVG",  ]
    for dataset in dataset_list:
        method_calculate(dataset)


def method_calculate(dataset):
    adata, data_x_df, data_y_df = preprocess_parameters(dataset)
    method = "temporalVAE"

    time_standard_type = "embryoneg5to5"  #
    print(f"time standard type: {time_standard_type}")
    if not os.path.exists(f'{os.getcwd()}/{method}_results'):
        os.makedirs(f'{os.getcwd()}/{method}_results')

    # ------- set config and logger ------

    train_epoch_num = 60  # 2024-04-16 18:19:48
    # train_epoch_num = 150
    import yaml
    with open("../vae_model_configs/supervise_vae_regressionclfdecoder_exp2_toyDataset.yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    logger_file = f'{os.getcwd()}/exp2_temporalVAE_toyDataset.log'
    LogHelper.setup(log_path=logger_file, level='INFO')
    _logger = logging.getLogger(__name__)
    print("Finished setting up the logger at: {}.".format(logger_file))
    print(f"Epoch number: {train_epoch_num}")

    donor_list = list(np.unique(data_y_df))
    kFold_test_result_df = pd.DataFrame(columns=['time', 'pseudotime', 'trans_label'])

    save_path = _logger.root.handlers[0].baseFilename.replace(".log", "")
    for fold in range(len(donor_list)):
        _result_df = process_fold_toyDataset(fold, donor_list, adata, time_standard_type, config, save_path, train_epoch_num)

        kFold_test_result_df = pd.concat([kFold_test_result_df, _result_df], axis=0)
    print("k-fold test final result:")
    print(kFold_test_result_df)
    corr(kFold_test_result_df["time"], kFold_test_result_df["pseudotime"])
    kFold_test_result_df.to_csv(f'{os.getcwd()}/{method}_results/{dataset}_{method}_result.csv', index=True)
    print(f"test result save at {os.getcwd()}/{method}_results/{dataset}_{method}_result.csv")

    # f = tp.plot_grid_search(title="Grid Search")
    # f.savefig(f"{os.getcwd()}/psupertime_results/gridSearch.png")
    # f = tp.plot_model_perf((adata.X, adata.obs.time), figsize=(6, 5))
    # f.savefig(f"{os.getcwd()}/psupertime_results/modelPred.png")
    # f = tp.plot_identified_gene_coefficients(adata, n_top=20)
    # f.savefig(f"{os.getcwd()}/psupertime_results/geneCoff.png")
    from exp2_psupertime_toyDataset import plot_psupertime_density
    save_path = f"{os.getcwd()}/{method}_results/"
    plot_psupertime_density(kFold_test_result_df, save_path=save_path, label_key="time", psupertime_key="pseudotime", method=f"{dataset}_")
    print(f"figure save at {save_path}/{dataset}_labelsOverPsupertime.png")


def corr(x1, x2):
    from scipy.stats import spearmanr, kendalltau
    sp_correlation, sp_p_value = spearmanr(x1, x2)
    ke_correlation, ke_p_value = kendalltau(x1, x2)
    print(f"spearman correlation score: {sp_correlation}, p-value: {sp_p_value}.")
    print(f"kendalltau correlation score: {ke_correlation},p-value: {ke_p_value}.")


def process_fold_toyDataset(fold, donor_list, adata, time_standard_type, config, save_path, train_epoch_num):
    # _logger = logging.getLogger(__name__)
    # time.sleep(random.randint(5, 20))
    print("the {}/{} fold train, use donor-{} as test set".format(fold + 1, len(donor_list), donor_list[fold]))

    train_adata = adata[adata.obs["time"] != donor_list[fold]].copy()
    test_adata = adata[adata.obs["time"] == donor_list[fold]].copy()
    # move one cell from test to train to occupy the test time
    one_test_cell = test_adata[:5].copy()
    test_adata = test_adata[5:].copy()
    train_adata = anndata.concat([train_adata.copy(), one_test_cell.copy()], axis=0)

    # ----------------------------------------split Train and Test dataset-----------------------------------------------------
    print("the {}/{} fold train, use donor-{} as test set".format(fold + 1, len(donor_list), donor_list[fold]))
    subFold_save_file_path = "{}/temporalVAE_results/{}/".format(save_path,
                                                         donor_list[fold])

    if not os.path.exists(subFold_save_file_path):
        os.makedirs(subFold_save_file_path)
    # we need to transpose data to correct its shape
    x_sc_train = torch.tensor(train_adata.X, dtype=torch.get_default_dtype()).t()
    x_sc_test = torch.tensor(test_adata.X, dtype=torch.get_default_dtype()).t()
    print("Set x_sc_train data with shape (gene, cells): {}".format(x_sc_train.shape))
    print("Set x_sc_test data with shape (gene, cells): {}".format(x_sc_test.shape))

    # trans y_time

    y_time_train = x_sc_train.new_tensor(np.array(train_adata.obs["time"]).astype(int))
    y_time_test = x_sc_test.new_tensor(np.array(test_adata.obs["time"]).astype(int))

    # for classification model with discrete time cannot use sigmoid and logit time type
    y_time_nor_train, label_dic = trans_time(y_time_train, time_standard_type, capture_time_other=y_time_test)
    y_time_nor_test, label_dic = trans_time(y_time_test, time_standard_type, label_dic_train=label_dic)
    print("label dictionary: {}".format(label_dic))
    print("Normalize train y_time_nor_train type: {}, with y_time_nor_train lable: {}, shape: {}, \ndetail: {}"
          .format(time_standard_type, np.unique(y_time_train), y_time_train.shape, np.unique(y_time_nor_train)))
    print("Normalize test y_time_nor_train type: {}, with y_time_nor_train lable: {}, shape: {}, \ndetail: {}"
          .format(time_standard_type, np.unique(y_time_test), y_time_test.shape, np.unique(y_time_nor_test)))

    # ------------------------------------------- Set up temporalVAE model and Start train process -------------------------------------------------
    print("Start training with epoch: {}. ".format(train_epoch_num))

    # if int(config['model_params']['in_channels']) == 0:
    config['model_params']['in_channels'] = x_sc_train.shape[0]
    tb_logger = TensorBoardLogger(save_dir=subFold_save_file_path,
                                  name=config['model_params']['name'], )

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    MyVAEModel = vae_models[config['model_params']['name']](**config['model_params'])

    ## print parameters of model
    # for name, param in MyVAEModel.named_parameters():
    #     print(name, param)
    train_data = [[x_sc_train[:, i], y_time_nor_train[i], y_time_train[i]] for i in range(x_sc_train.shape[1])]
    test_data = [[x_sc_test[:, i], y_time_nor_test[i], y_time_test[i]] for i in range(x_sc_test.shape[1])]
    print("don't set batch")
    data = SupervisedVAEDataset(train_data=train_data, val_data=test_data, test_data=test_data, predict_data=test_data,
                                train_batch_size=len(train_data), val_batch_size=len(test_data),
                                test_batch_size=len(test_data), predict_batch_size=len(test_data),
                                label_dic=label_dic)

    # data.setup("train")
    experiment = temporalVAEExperiment(MyVAEModel, config['exp_params'])

    # 创建一个 LearningRateMonitor 回调实例
    lr_monitor = LearningRateMonitor()
    # 2023-09-07 20:34:25 add check memory
    check_memory()
    device = auto_select_gpu_and_cpu()
    print("Auto select run on {}".format(device))

    runner = Trainer(logger=tb_logger, log_every_n_steps=1,
                     callbacks=[
                         lr_monitor,
                         ModelCheckpoint(save_top_k=2,
                                         dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                         monitor="val_loss",
                                         save_last=True),
                     ],
                     # check_val_every_n_epoch=1, val_check_interval=1,
                     devices=[int(device.split(":")[-1])],
                     accelerator="gpu", max_epochs=train_epoch_num
                     )

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, data)

    # test the model
    print("this epoch final, on test data:{}".format(runner.test(experiment, data)))
    # predict on train data
    data_predict = SupervisedVAEDataset_onlyPredict(predict_data=train_data, predict_batch_size=len(train_data))
    train_result = runner.predict(experiment, data_predict)
    train_clf_result, train_latent_mu_result, train_latent_log_var_result = train_result[0][0], train_result[0][1], \
        train_result[0][2]
    # predict on test data
    data_test = SupervisedVAEDataset_onlyPredict(predict_data=test_data, predict_batch_size=len(test_data))
    test_result = runner.predict(experiment, data_test)
    test_clf_result, test_latent_mu_result, test_latent_log_var_result = test_result[0][0], test_result[0][1], \
        test_result[0][2]
    if (np.isnan(np.squeeze(test_clf_result)).any()):
        print("The Array contain NaN values")
    else:
        print("The Array does not contain NaN values")
    print("predicted time of test donor is continuous.")

    _result_df = pd.DataFrame({'time': donor_list[fold],  # First column with a constant value of 1
                               'pseudotime': np.squeeze(test_clf_result, axis=1)})
    # _result_df['pseudotime'] = _result_df['pseudotime_normalized'].apply(denormalize, args=(min(label_dic.keys()), max(label_dic.keys()), min(label_dic.values()), max(label_dic.values())))
    _result_df["trans_label"] = _result_df["time"].map(label_dic)
    print("Plot training loss line for check.")

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    tags = EventAccumulator(tb_logger.log_dir).Reload().Tags()['scalars']
    print("All tags in logger: {}".format(tags))
    # Retrieve and print the metric results
    plot_tag_list = list(set(["train_clf_loss_epoch", "val_clf_loss", "test_clf_loss_epoch"]) & set(tags))
    print(f"plot tags {plot_tag_list}")
    plot_training_loss_for_tags(tb_logger, plot_tag_list, special_str=donor_list[fold], title=donor_list[fold])

    # ---------------------------------------------- save sub model parameters for check  --------------------------------------------------
    print("encoder and decoder structure: {}".format({"encoder": MyVAEModel.encoder, "decoder": MyVAEModel.decoder}))
    print("clf-decoder structure: {}".format({"encoder": MyVAEModel.clf_decoder}))
    torch.save(MyVAEModel, tb_logger.root_dir + "/version_" + str(tb_logger.version) + '/model.pth')
    print("detail information about structure save at： {}".format(
        tb_logger.root_dir + "/version_" + str(tb_logger.version) + '/model.pth'))

    del MyVAEModel
    del runner
    del experiment
    # 清除CUDA缓存
    torch.cuda.empty_cache()
    return _result_df


if __name__ == '__main__':
    main()
