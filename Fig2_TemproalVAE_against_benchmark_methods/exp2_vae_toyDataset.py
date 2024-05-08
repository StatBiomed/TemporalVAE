# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：exp2_vae_toyDataset.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/9/21 12:18 

use dataset mentioned in psupertime manuscript

"""
import sys

# sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/CNNC-master/utils")
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/PyTorch-VAE-master")
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan")
from utils.GPU_manager_pytorch import auto_select_gpu_and_cpu, check_memory
import logging
from utils.logging_system import LogHelper
from utils.utils_DandanProject import *
import anndata
import pandas as pd
import os
import numpy as np
from experiment import VAEXperiment
from dataset import SupervisedVAEDataset
from dataset import SupervisedVAEDataset_onlyPredict
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pathlib import Path

os.chdir('/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/Fig2_TemproalVAE_against_benchmark_methods')
import multiprocessing


def main():
    method = "vae"
    # ------------ for Mouse embryonic beta cells dataset:
    # result_file_name = "embryoBeta"
    # data_x_df = pd.read_csv(f'data_fromPsupertime/{result_file_name}_X.csv', index_col=0).T
    # hvg_gene_list = pd.read_csv(f'{os.getcwd()}/data_fromPsupertime/{result_file_name}_gene_list.csv', index_col=0)
    # data_x_df = data_x_df[hvg_gene_list["gene_name"]]
    # data_y_df = pd.read_csv(f'data_fromPsupertime/{result_file_name}_Y.csv', index_col=0)
    # data_y_df = data_y_df["time"]
    # preprocessing_params = {"select_genes": "all", "log": True}
    # time_standard_type = "embryoneg5to5"

    # for Human Germline dataset:
    result_file_name = "humanGermline"
    data_x_df = pd.read_csv('data_fromPsupertime/humanGermline_X.csv', index_col=0).T
    hvg_gene_list = pd.read_csv(f'{os.getcwd()}/data_fromPsupertime/{result_file_name}_gene_list.csv', index_col=0)
    data_x_df = data_x_df[hvg_gene_list["gene_name"]]
    data_y_df = pd.read_csv('data_fromPsupertime/humanGermline_Y.csv', index_col=0)
    data_y_df = data_y_df["time"]
    preprocessing_params = {"select_genes": "all", "log": True}
    time_standard_type = "embryoneg5to5"

    # for Acinar dataset, in acinar data set total 8 donors with 8 ages:
    # result_file_name = "acinarHVG"
    # data_x_df = pd.read_csv('data_fromPsupertime/acinar_hvg_sce_X.csv', index_col=0).T
    # data_y_df = pd.read_csv('data_fromPsupertime/acinar_hvg_sce_Y.csv')
    # data_y_df = np.array(data_y_df['x'])
    # preprocessing_params = {"select_genes": "all", "log": False}
    # time_standard_type = "embryoneg5to5"

    # START HERE
    adata_org = anndata.AnnData(data_x_df)
    adata_org.obs["time"] = data_y_df
    print(f"Input Data: n_genes={adata_org.n_vars}, n_cells={adata_org.n_obs}")

    # ------- set config and logger ------

    train_epoch_num = 60 # 2024-04-16 18:19:48
    # train_epoch_num = 150
    import yaml
    with open("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/vae_model_configs/"
              "supervise_vae_regressionclfdecoder_exp2_toyDataset.yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    logger_file = f'{os.getcwd()}/exp2_vae_toyDataset.log'
    LogHelper.setup(log_path=logger_file, level='INFO')
    _logger = logging.getLogger(__name__)
    print("Finished setting up the logger at: {}.".format(logger_file))
    print(f"Epoch number: {train_epoch_num}")
    # ------------------preprocess adata here ----------------
    from pypsupertime import Psupertime
    tp = Psupertime(n_jobs=5, n_folds=5,
                    preprocessing_params=preprocessing_params
                    )  # if for Acinar cell "select_genes": "all"

    adata = tp.preprocessing.fit_transform(adata_org.copy())
    del tp
    # ---------------------- PLOT adata umap----------------------
    # colors = colors_tuple()
    # plt.figure(figsize=(8, 6))
    #
    # import umap.umap_ as umap
    #
    # reducer = umap.UMAP(random_state=42)
    # embedding = reducer.fit_transform(adata.X)
    #
    # plt.gca().set_aspect('equal', 'datalim')
    #
    # i = 0
    # for label in np.unique(data_y_df):
    #     # print(label)
    #     indices = np.where(data_y_df == label)
    #     plt.scatter(embedding[indices, 0], embedding[indices, 1], label=f"{label} stage: {len(indices[0])} cells", s=7, alpha=0.9,
    #                 c=colors[i])
    #     i += 1
    #
    # plt.gca().set_aspect('equal', 'datalim')
    #
    # # plt.legend(bbox_to_anchor=(1.01, 0), loc="lower left", borderaxespad=0)
    # # 添加图例并设置样式
    # legend = plt.legend(bbox_to_anchor=(1.01, 0), loc="lower left", borderaxespad=0)
    # for handle in legend.legendHandles:
    #     handle.set_sizes([20])
    # # legend.legendHandles[0]._sizes = [10]  # 设置散点的大小
    # plt.setp(legend.get_title(), fontsize='small')  # 设置图例标题的字体大小
    #
    # plt.subplots_adjust(left=0.1, right=0.75)
    # plt.title('UMAP: ')
    #
    # plt.show()
    # plt.close()
    # --------------------------------------------
    #
    donor_list = list(np.unique(data_y_df))
    kFold_test_result_df = pd.DataFrame(columns=['time', 'pseudotime', 'trans_label'])

    save_path = _logger.root.handlers[0].baseFilename.replace(".log", "")
    for fold in range(len(donor_list)):

        _result_df = process_fold_toyDataset(fold, donor_list, adata, time_standard_type, config, save_path, train_epoch_num)
        # with multiprocessing.Pool(processes=len(donor_list)) as pool:
        #     processed_results = pool.starmap(process_fold_toyDataset, [
        #         (fold, donor_list, adata, time_standard_type, config, save_path, train_epoch_num) for fold in range(len(donor_list))])
        # use one donor as test set, other as train set
        # for fold, result_df in enumerate(processed_results):
        #     kFold_test_result_df = pd.concat([kFold_test_result_df, result_df], axis=0)
        kFold_test_result_df = pd.concat([kFold_test_result_df, _result_df], axis=0)
    print("k-fold test final result:")
    print(kFold_test_result_df)
    corr(kFold_test_result_df["time"], kFold_test_result_df["pseudotime"])
    kFold_test_result_df.to_csv(f'{os.getcwd()}/{method}_results/{result_file_name}_{method}_result.csv', index=True)
    print(f"test result save at {os.getcwd()}/{method}_results/{result_file_name}_{method}_result.csv")

    # f = tp.plot_grid_search(title="Grid Search")
    # f.savefig(f"{os.getcwd()}/psupertime_results/gridSearch.png")
    # f = tp.plot_model_perf((adata.X, adata.obs.time), figsize=(6, 5))
    # f.savefig(f"{os.getcwd()}/psupertime_results/modelPred.png")
    # f = tp.plot_identified_gene_coefficients(adata, n_top=20)
    # f.savefig(f"{os.getcwd()}/psupertime_results/geneCoff.png")
    from exp2_psupertime_toyDataset import plot_psupertime_density
    save_path = f"{os.getcwd()}/{method}_results/"
    plot_psupertime_density(kFold_test_result_df, save_path=save_path, label_key="time", psupertime_key="pseudotime", method=f"{result_file_name}_")
    print(f"figure save at {save_path}/{result_file_name}_labelsOverPsupertime.png")


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
    subFold_save_file_path = "{}/vae_results/{}/".format(save_path,
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

    # ------------------------------------------- Set up VAE model and Start train process -------------------------------------------------
    print("Start training with epoch: {}. ".format(train_epoch_num))

    # if int(config['model_params']['in_channels']) == 0:
    config['model_params']['in_channels'] = x_sc_train.shape[0]
    tb_logger = TensorBoardLogger(save_dir=subFold_save_file_path,
                                  name=config['model_params']['name'], )

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    MyVAEModel = vae_models[config['model_params']['name']](**config['model_params'])

    ## 打印模型的权重和偏置
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
    experiment = VAEXperiment(MyVAEModel, config['exp_params'])

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
