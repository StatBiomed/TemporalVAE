# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE
@File    ：a-undetermined-TemporalVAE_science2022_LR_PCA_RF_referenceAtlas_queryStereo.py
@IDE     ：PyCharm
@Author  ：awa121
@Date    ：2024/9/8 22:34
"""
# -*-coding:utf-8 -*-
import os
if os.getcwd().split("/")[-1] != "TemporalVAE":
    os.chdir("../..")
import sys
sys.path.append(os.getcwd())

import anndata as ad
import pandas as pd
from TemporalVAE.utils import geneId_geneName_dic, predict_newData_preprocess_df
from TemporalVAE.utils import calculate_real_predict_corrlation_score, plot_psupertime_density
import logging
from TemporalVAE.utils import LogHelper
import yaml
from TemporalVAE.utils import plot_violin_240223


def main():
    min_gene_num = 50
    # save_path = f"results/test/"
    save_path = f"results/Fig3_TemporalVAE_referenceAtlas_queryStereo_240901/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_golbal_path = "data/"
    baseline_data_path = "/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/"
    query_data_path = "/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50/"
    checkpoint_file = "checkpoint_files/mouse_atlas.ckpt"
    config_file = "vae_model_configs/supervise_vae_regressionclfdecoder_mouse_stereo.yaml"

    # -------set logger and parameters, creat result save path and folder----------------------------------------------
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
        query_adata_raw = ad.read_csv(f"{data_golbal_path}/{query_data_path}/data_count_hvg.csv",
                                      delimiter='\t')
    except:
        query_adata_raw = ad.read_csv(f"{data_golbal_path}/{query_data_path}/data_count_hvg.csv",
                                      delimiter=',')
    cell_time_query_pd = pd.read_csv(f"{data_golbal_path}/{query_data_path}/cell_with_time.csv",
                                     sep="\t", index_col=0)
    query_adata_raw = query_adata_raw.copy().T
    query_adata_raw.obs = cell_time_query_pd

    # 计算每个细胞类型的数量
    celltype_counts = query_adata_raw.obs["celltype_update"].value_counts()
    print(f"plot for {len(query_adata_raw)} cells")
    if len(celltype_counts) > 10:
        # 选出数量前10的细胞类型
        top_10_attr = celltype_counts.head(10).index.tolist()
        # 根据选出的前10细胞类型筛选adata
        query_adata_raw = query_adata_raw[query_adata_raw.obs["celltype_update"].isin(top_10_attr)].copy()
        print(f"plot for celltype_update, as the number is more than 10, select top 10 to plot umap: {top_10_attr}, the cell is filtered to {len(query_adata_raw)}")
        # cell_time_query_pd = cell_time_query_pd[cell_time_query_pd["celltype_update"].isin(top_10_attr)]
        # cell_time_query_pd = cell_time_query_pd.loc[query_adata_raw.obs.index]
    trainData_renormalized_df, loss_gene_shortName_list, train_cell_info_df = predict_newData_preprocess_df(
        gene_dic, query_adata_raw,
        min_gene_num=min_gene_num,
        reference_file=f"{data_golbal_path}/{baseline_data_path}/data_count_hvg.csv",
        bool_change_geneID_to_geneShortName=False
    )
    query_adata = ad.AnnData(X=trainData_renormalized_df, obs=train_cell_info_df)
    print(f"get query adata {query_adata}")

    # ------------------------------preprocess on baseline data----------------------------------------------------------
    # reference_expression_df, reference_cell_info_df = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path,
    #                                                                                         f"{baseline_data_path}/data_count_hvg.csv",
    #                                                                                         f"{baseline_data_path}/cell_with_time.csv",
    #                                                                                         min_cell_num=50,
    #                                                                                         min_gene_num=100)
    # reference_adata = ad.AnnData(X=reference_expression_df, obs=reference_cell_info_df)
    # ------methods
    directly_predict_on_vae(query_adata, save_path, checkpoint_file, config)



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

    from TemporalVAE.model_master import vae_models
    MyVAEModel = vae_models["SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial"](**config['model_params'])
    MyVAEModel.load_state_dict(new_state_dict)
    MyVAEModel.eval()

    from TemporalVAE.utils import check_memory, auto_select_gpu_and_cpu
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
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyPredict
    from TemporalVAE.model_master.experiment_temporalVAE import temporalVAEExperiment
    data_predict = SupervisedVAEDataset_onlyPredict(predict_data=data_x, predict_batch_size=len(data_x))

    experiment = temporalVAEExperiment(MyVAEModel, config['exp_params'])
    # z=experiment.predict_step(data_predict,1)
    train_result = runner.predict(experiment, data_predict)
    pseudoTime_directly_predict_by_pretrained_model = train_result[0][0]
    pseudoTime_directly_predict_by_pretrained_model_df = pd.DataFrame(pseudoTime_directly_predict_by_pretrained_model, columns=["pseudotime_by_preTrained_mouseAtlas_model"])
    pseudoTime_directly_predict_by_pretrained_model_df.index = query_adata.obs_names
    from TemporalVAE.utils import denormalize
    pseudoTime_directly_predict_by_pretrained_model_df["physical_pseudotime_by_preTrained_mouseAtlas_model"] = pseudoTime_directly_predict_by_pretrained_model_df[
        "pseudotime_by_preTrained_mouseAtlas_model"].apply(denormalize, args=(8.5, 18.75, -5, 5))
    mu_predict_by_pretrained_model = train_result[0][1].cpu().numpy()

    cell_time_stereo = pd.concat([query_adata.obs, pseudoTime_directly_predict_by_pretrained_model_df], axis=1)

    color_dic = plot_violin_240223(cell_time_stereo, save_path, x_attr="time",special_legend_str="5")
    # print(color_dic)
    from TemporalVAE.utils import plot_umap_240223
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


if __name__ == '__main__':
    main()
