# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：TemporalVAE_science2022_ot_LR_PCA_RF_VAE_referenceAtlas_queryStereo.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/3/22 10:21 


"""

import os
import sys

if os.getcwd().split("/")[-1] != "TemporalVAE":
    os.chdir("..")
sys.path.append(os.getcwd())

import anndata as ad
import pandas as pd
from collections import Counter
from utils.utils_DandanProject import geneId_geneName_dic, predict_newData_preprocess_df, preprocessData_and_dropout_some_donor_or_gene
from utils.utils_Dandan_plot import calculate_real_predict_corrlation_score, plot_psupertime_density
import time
import logging
from utils.logging_system import LogHelper
import yaml
from utils.utils_Dandan_plot import plot_violin_240223
import gc


def main():
    min_gene_num = 100
    save_path = f"results/Figure3_LR_PCA_RF_directlyPredictOn_mouseStereo_minGeneNum{min_gene_num}/"
    plot_compare_corr_boxplot(save_path)


    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_golbal_path = "data/"
    baseline_data_path = "/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/"
    query_data_path = "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50/"
    checkpoint_file = "checkpoint_files/mouse_atlas.ckpt"
    config_file = "vae_model_configs/supervise_vae_regressionclfdecoder_mouse_stereo.yaml"
    # ----------------------set logger and parameters, creat result save path and folder------------------
    logger_file = f'{save_path}/run.log'
    LogHelper.setup(log_path=logger_file, level='INFO')
    _logger = logging.getLogger(__name__)
    _logger.info("Finished setting up the logger at: {}.".format(logger_file))
    _logger.info(f"baseline dataset: {data_golbal_path}/{baseline_data_path}, \n and query dataset: {data_golbal_path}/{query_data_path}")
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    # --------------------------------preprocess on query data-----------------------------------------
    gene_dic = geneId_geneName_dic()
    try:
        query_adata = ad.read_csv(f"{data_golbal_path}/{query_data_path}/data_count_hvg.csv", delimiter='\t')
    except:
        query_adata = ad.read_csv(f"{data_golbal_path}/{query_data_path}/data_count_hvg.csv", delimiter=',')
    cell_time_query_pd = pd.read_csv(f"{data_golbal_path}/{query_data_path}/cell_with_time.csv", sep="\t", index_col=0)
    query_adata = query_adata.copy().T
    query_adata.obs = cell_time_query_pd

    # 计算每个细胞类型的数量
    celltype_counts = query_adata.obs["celltype_update"].value_counts()
    print(f"plot for {len(query_adata)} cells")
    # if len(celltype_counts) > 10:
    #     # 选出数量前10的细胞类型
    #     top_10_attr = celltype_counts.head(10).index.tolist()
    #     # 根据选出的前10细胞类型筛选adata
    #     query_adata = query_adata[query_adata.obs["celltype_update"].isin(top_10_attr)].copy()
    #     print(f"plot for celltype_update, as the number is more than 10, select top 10 to plot umap: {top_10_attr}, the cell is filtered to {len(query_adata)}")
        # cell_time_query_pd = cell_time_query_pd[cell_time_query_pd["celltype_update"].isin(top_10_attr)]
        # cell_time_query_pd = cell_time_query_pd.loc[query_adata.obs.index]
    query_expression_df, _, query_cell_info_df = predict_newData_preprocess_df(gene_dic, query_adata,
                                                                               min_gene_num=min_gene_num,
                                                                               reference_file=f"{data_golbal_path}/{baseline_data_path}/data_count_hvg.csv",
                                                                               bool_change_geneID_to_geneShortName=False)
    query_adata = ad.AnnData(X=query_expression_df, obs=query_cell_info_df)
    print(f"get query adata {query_adata.shape}")


    # ------------------------------preprocess on baseline data----------------------------------------------------------
    reference_expression_df, reference_cell_info_df = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path,
                                                                                                    f"{baseline_data_path}/data_count_hvg.csv",
                                                                                                    f"{baseline_data_path}/cell_with_time.csv",
                                                                                                    min_cell_num=50,
                                                                                                    min_gene_num=100)
    # 2024-08-28 23:58:23 add science2022 and ot as benchmarking method.
    reference_adata = ad.AnnData(X=reference_expression_df, obs=reference_cell_info_df)
    train_science2022_model(reference_adata, query_adata, save_path)
    directly_predict_on_vae(query_adata, save_path, checkpoint_file, config)
    # train_ot_model(reference_adata, query_adata, save_path)
    plot_compare_corr_boxplot(save_path)

    train_LR_model(reference_expression_df, reference_cell_info_df, query_adata, save_path)
    train_PCA_model(reference_expression_df, reference_cell_info_df, query_adata, save_path)
    train_RF_model(reference_expression_df, reference_cell_info_df, query_adata, save_path)


def train_science2022_model(reference_adata, query_adata, save_path, ):
    method = "science2022"
    from benchmarking_methods.benchmarking_methods import science2022

    query_predictions = science2022(train_x=reference_adata.X, train_y=reference_adata.obs["time"], test_df=query_adata.X)

    query_result_df = pd.DataFrame(query_adata.obs["time"])
    query_result_df["pseudotime"] = query_predictions
    # query_result_df = pd.DataFrame({"time": query_adata.obs["time"], "pseudotime": query_predictions})

    print("Final corr:", calculate_real_predict_corrlation_score(query_result_df["time"], query_result_df["pseudotime"]))

    query_result_df.to_csv(f'{save_path}/{method}_result_df.csv', index=True)
    print(f"test result save at {save_path}/{method}_result_df.csv")

    plot_psupertime_density(query_result_df, save_path=save_path, label_key="time", psupertime_key="pseudotime", method=method)

    print(f"Finish {method} train on baseline data and predict on query data.")
    gc.collect()


def train_ot_model(reference_adata, query_adata, save_path, ):
    method = "ot"
    from benchmarking_methods.benchmarking_methods import ot_svm_classifier
    test_y_predicted = ot_svm_classifier(train_x=reference_adata.X, train_y=reference_adata.obs["time"],
                                         test_x=query_adata.X, test_y=query_adata.obs["time"])

    query_result_df = pd.DataFrame({"time": query_adata.obs["time"], "pseudotime": test_y_predicted})

    print("Final corr:", calculate_real_predict_corrlation_score(query_result_df["time"], query_result_df["pseudotime"]))

    query_result_df.to_csv(f'{save_path}/{method}_result_df.csv', index=True)
    print(f"test result save at {save_path}/{method}_result_df.csv")

    plot_psupertime_density(query_result_df, save_path=save_path, label_key="time", psupertime_key="pseudotime", method=method)

    print(f"Finish {method} train on baseline data and predict on query data.")
    gc.collect()


def plot_compare_corr_boxplot(save_path):
    # ---------- pretrain a model (TemporalVAE, LR, PCA, RF) on mouse atlas data, directly predict on mouse stereo data -----------------------
    from utils.utils_Dandan_plot import plot_boxplot_from_dic
    # ---------- pretrain a model (TemporalVAE, LR, PCA, RF) on mouse atlas data, directly predict on mouse stereo data -----------------------
    file_name = f"{save_path}/temporalVAE_result_df.csv"
    data_pd = pd.read_csv(file_name)
    VAE = calculate_real_predict_corrlation_score(data_pd["time"], data_pd["pseudotime"], only_str=False)

    file_name = f"{save_path}/linearRegression_result_df.csv"
    data_pd = pd.read_csv(file_name)
    LR = calculate_real_predict_corrlation_score(data_pd["time"], data_pd["pseudotime"], only_str=False)

    file_name = f"{save_path}/PCA_result_df.csv"
    data_pd = pd.read_csv(file_name)
    PCA = calculate_real_predict_corrlation_score(data_pd["time"], data_pd["pseudotime"], only_str=False)

    file_name = f"{save_path}/randomForest_result_df.csv"
    data_pd = pd.read_csv(file_name)
    RF = calculate_real_predict_corrlation_score(data_pd["time"], data_pd["pseudotime"], only_str=False)

    file_name = f"{save_path}/science2022_result_df.csv"
    data_pd = pd.read_csv(file_name)
    science2022 = calculate_real_predict_corrlation_score(data_pd["time"], data_pd["pseudotime"], only_str=False)

    # 构建数据，确保按照VAE、LR、PCA的顺序
    data = {
        'Method': ['TemporalVAE', 'TemporalVAE',
                   'LR', 'LR',
                   'PCA', 'PCA',
                   'RF', 'RF',
                   'Science2022','Science2022'],
        'Correlation Type': ['Spearman', 'Pearson',
                             'Spearman', 'Pearson',
                             'Spearman', 'Pearson',
                             'Spearman', 'Pearson',
                             'Spearman', 'Pearson'],
        'Value': [VAE[1]['spearman'].correlation, VAE[1]['pearson'].correlation,
                  LR[1]['spearman'].correlation, LR[1]['pearson'].correlation,
                  PCA[1]['spearman'].correlation, PCA[1]['pearson'].correlation,
                  RF[1]['spearman'].correlation, RF[1]['pearson'].correlation,
                  science2022[1]['spearman'].correlation, science2022[1]['pearson'].correlation]
    }
    data["Value"] = [abs(_i) if _i < 0 else _i for _i in data["Value"]]
    plot_boxplot_from_dic(data, legend_loc="upper right")


def directly_predict_on_vae(query_adata, save_path, checkpoint_file, config):
    method = "temporalVAE"
    import torch
    torch.set_float32_matmul_precision('high')

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
    config['model_params']['in_channels'] = query_adata.X.shape[1]  # the number of features

    from model_master import vae_models
    MyVAEModel = vae_models["SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial"](**config['model_params'])
    MyVAEModel.load_state_dict(new_state_dict)
    MyVAEModel.eval()

    from utils.GPU_manager_pytorch import check_memory, auto_select_gpu_and_cpu
    check_memory()
    # device = auto_select_gpu_and_cpu()
    device = auto_select_gpu_and_cpu(free_thre=5, max_attempts=100000000)  # device: e.g. "cuda:0"
    from pytorch_lightning import Trainer, seed_everything
    runner = Trainer(devices=[int(device.split(":")[-1])])
    seed_everything(config['exp_params']['manual_seed'], True)
    #
    x_sc = torch.tensor(query_adata.X, dtype=torch.get_default_dtype()).t()
    data_x = [[x_sc[:, i], 0, 0] for i in range(x_sc.shape[1])]

    # predict batch size will not influence the training
    from model_master.dataset import SupervisedVAEDataset_onlyPredict
    from model_master.experiment import VAEXperiment
    data_predict = SupervisedVAEDataset_onlyPredict(predict_data=data_x, predict_batch_size=len(data_x))

    experiment = VAEXperiment(MyVAEModel, config['exp_params'])
    # z=experiment.predict_step(data_predict,1)
    train_result = runner.predict(experiment, data_predict)
    pseudoTime_directly_predict_by_pretrained_model = train_result[0][0]
    pseudoTime_directly_predict_by_pretrained_model_df = pd.DataFrame(pseudoTime_directly_predict_by_pretrained_model, columns=["pseudotime_by_preTrained_mouseAtlas_model"])
    pseudoTime_directly_predict_by_pretrained_model_df.index = query_adata.obs_names
    from utils.utils_DandanProject import denormalize
    pseudoTime_directly_predict_by_pretrained_model_df["physical_pseudotime_by_preTrained_mouseAtlas_model"] = pseudoTime_directly_predict_by_pretrained_model_df[
        "pseudotime_by_preTrained_mouseAtlas_model"].apply(denormalize, args=(8.5, 18.75, -5, 5))
    mu_predict_by_pretrained_model = train_result[0][1].cpu().numpy()

    cell_time_stereo = pd.concat([query_adata.obs, pseudoTime_directly_predict_by_pretrained_model_df], axis=1)

    color_dic = plot_violin_240223(cell_time_stereo, save_path, real_attr="time")
    # print(color_dic)
    from utils.utils_Dandan_plot import plot_umap_240223
    plot_umap_240223(mu_predict_by_pretrained_model, cell_time_stereo, color_dic, save_path, attr_str="time")

    color_dic = {'Brain': "#FA8072", 'Connective tissue': "#32CD32",
                 'Muscle': "#4169E1", 'Cavity': "#FFA500",
                 'Liver': "#20B2AA", 'Spinal cord': "#FF4500",
                 'Meninges': "#EE82EE", 'Jaw and tooth': "#8B4513",
                 'Cartilage primordium': "#8E44AD", 'Epidermis': "#AD9BA5"}
    plot_umap_240223(mu_predict_by_pretrained_model, cell_time_stereo, color_dic, save_path, attr_str="celltype_update")
    cell_time_stereo["pseudotime"] = cell_time_stereo["physical_pseudotime_by_preTrained_mouseAtlas_model"]
    print("Final corr:", calculate_real_predict_corrlation_score(cell_time_stereo["time"], cell_time_stereo["pseudotime"]))

    cell_time_stereo.to_csv(f'{save_path}/{method}_result_df.csv', index=True)
    print(f"test result save at {save_path}/{method}_result_df.csv")

    plot_psupertime_density(cell_time_stereo, save_path=save_path, label_key="time", psupertime_key="pseudotime", method=method)
    print(f"Finish {method} train on baseline data and predict on query data.")


# ---------------------------------------- train LR model -----------------------------------------------------
def train_LR_model(atlas_sc_expression_df, cell_time_atlas, query_adata, save_path, ):
    method = "linearRegression"
    from sklearn.linear_model import LinearRegression
    # use one donor as test set, other as train set
    adata_atlas = ad.AnnData(X=atlas_sc_expression_df, obs=cell_time_atlas)
    train_adata = adata_atlas.copy()
    model = LinearRegression()
    model.fit(train_adata.X, train_adata.obs["time"])
    query_predictions = model.predict(query_adata.X)
    query_result_df = pd.DataFrame({"time": query_adata.obs["time"], "pseudotime": query_predictions})

    print("Final corr:", calculate_real_predict_corrlation_score(query_result_df["time"], query_result_df["pseudotime"]))

    query_result_df.to_csv(f'{save_path}/{method}_result_df.csv', index=True)
    print(f"test result save at {save_path}/{method}_result_df.csv")

    plot_psupertime_density(query_result_df, save_path=save_path, label_key="time", psupertime_key="pseudotime", method=method)

    print(f"Finish {method} train on baseline data and predict on query data.")
    gc.collect()


def train_RF_model(atlas_sc_expression_df, cell_time_atlas, query_adata, save_path, ):
    method = "randomForest"
    from sklearn.linear_model import LinearRegression
    # use one donor as test set, other as train set
    adata_atlas = ad.AnnData(X=atlas_sc_expression_df, obs=cell_time_atlas)
    train_adata = adata_atlas.copy()

    RF_model = random_forest_regressor(train_x=train_adata.X, train_y=train_adata.obs["time"])
    # RF_model = random_forest_classifier(train_x=train_adata.X, train_y=train_adata.obs["time"])
    test_y_predicted = RF_model.predict(query_adata.X)

    query_result_df = pd.DataFrame({"time": query_adata.obs["time"], "pseudotime": test_y_predicted})

    print("Final corr:", calculate_real_predict_corrlation_score(query_result_df["time"], query_result_df["pseudotime"]))

    query_result_df.to_csv(f'{save_path}/{method}_result_df.csv', index=True)
    print(f"test result save at {save_path}/{method}_result_df.csv")

    plot_psupertime_density(query_result_df, save_path=save_path, label_key="time", psupertime_key="pseudotime", method=method)

    print(f"Finish {method} train on baseline data and predict on query data.")


def train_PCA_model(atlas_sc_expression_df, cell_time_atlas, query_adata, save_path, ):
    method = "PCA"
    from sklearn.decomposition import PCA
    # use one donor as test set, other as train set
    adata_atlas = ad.AnnData(X=atlas_sc_expression_df, obs=cell_time_atlas)
    train_adata = adata_atlas.copy()

    pca = PCA(n_components=2)
    # classifier = DecisionTreeClassifier()
    # transform / fit
    train_lowDim = pca.fit_transform(train_adata.X)
    # classifier.fit(train_lowDim, train_adata.obs["time"])
    # predict "new" data
    test_lowDim = pca.transform(query_adata.X)

    query_result_df = pd.DataFrame({"time": query_adata.obs["time"], "pseudotime": test_lowDim[:, 0]})

    print("Final corr:", calculate_real_predict_corrlation_score(query_result_df["time"], query_result_df["pseudotime"]))

    query_result_df.to_csv(f'{save_path}/{method}_result_df.csv', index=True)
    print(f"test result save at {save_path}/{method}result_df.csv")

    plot_psupertime_density(query_result_df, save_path=save_path, label_key="time", psupertime_key="pseudotime", method=method)

    print(f"Finish {method} train on baseline data and predict on query data.")


#
# def corr(x1, x2, special_str=""):
#     from scipy.stats import spearmanr, kendalltau
#     sp_correlation, sp_p_value = spearmanr(x1, x2)
#     ke_correlation, ke_p_value = kendalltau(x1, x2)
#
#     sp = f"{special_str} spearman correlation score: {sp_correlation}, p-value: {sp_p_value}."
#     print(sp)
#     ke = f"{special_str} kendalltau correlation score: {ke_correlation},p-value: {ke_p_value}."
#     print(ke)
#
#     return sp, ke

def random_forest_regressor(train_x, train_y):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(max_depth=2, random_state=0)
    model.fit(train_x, train_y)
    return model


if __name__ == '__main__':
    main()
