# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：test_fangXinData.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/10/23 10:07

return fangxin latent space data of http://127.0.0.1:18888/tree/public/for_yijun_fangxin

use check point of /mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/231020_plotLatentSpace_mouse_data_minGene50_hvg1000CalByEachOrgan_timeCorGene/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch200_minGeneNum100/wholeData/SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial/version_0/checkpoints
gene list with ENS name /mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/231020_plotLatentSpace_mouse_data_minGene50_hvg1000CalByEachOrgan_timeCorGene/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch200_minGeneNum100/preprocessed_gene_info.csv


"""

import sys

sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan")

from utils.utils_DandanProject import *

from utils.utils_Dandan_plot import *
import pandas as pd


def main(sc_file_name):
    print(f"for sc file {sc_file_name}")
    # ----------------- imput parameters  -----------------
    mouse_atlas_file = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/" \
                       "mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/" \
                       "data_count_hvg.csv"
    config_file = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/vae_model_configs/supervise_vae_regressionclfdecoder_mouse_stereo.yaml"
    checkpoint_file = '/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/' \
                      '231020_plotLatentSpace_mouse_data_minGene50_hvg1000CalByEachOrgan_timeCorGene/' \
                      'mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/' \
                      'supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch200_minGeneNum100/' \
                      'wholeData/SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial/version_0/checkpoints/last.ckpt'
    # spliced_min_gene_num = 0
    # unspliced_min_gene_num = 0
    _temp = sc_file_name.split("/")[-1].replace(".h5ad", "")
    save_result_path = f"/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/test/fangxinData/{_temp}/"
    if not os.path.exists(save_result_path):
        os.makedirs(save_result_path)
    print(f"make result save at {save_result_path}")
    # save parameters used
    _local_variables = locals().copy()
    _df = get_parameters_df(_local_variables)
    _df.to_csv(f"{save_result_path}/parameters_use_in_this_results.csv")

    # _temp_dic={sc_file_name:sc_file_name}
    # ----------------- read test adata -----------------
    adata = sc.read_h5ad(filename=sc_file_name)
    print("Import data, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    print("Annotation information of data includes: {}".format(adata.obs_keys()))  # 胞注釋信息的keys
    print("Cell id first 5: {}".format(adata.obs_names[:5]))  # 返回胞ID 数据类型是object
    print("Gene id first 5: {}".format(adata.var_names[:5]))  # 返回基因数据类型是list adata.obs.head()# 査看前5行的数据
    from utils.utils_DandanProject import geneId_geneName_dic
    gene_dic = geneId_geneName_dic()

    # adata_spliced = adata.copy()
    # adata_spliced.X = adata.layers["spliced"].toarray()
    # adata_unspliced = adata.copy()
    # adata_unspliced.X = adata.layers["unspliced"].toarray()
    adata_mrna = adata.copy()
    adata_mrna.X = adata.layers["spliced"].toarray()+adata.layers["unspliced"].toarray()

    # ----------------- get renormalized test data -----------------
    # adata_unspliced_renormalized_df, loss_gene_unspliced_shortName_list = predict_newData_preprocess_df(gene_dic, adata_unspliced,
    #                                                                                                     min_gene_num=0,
    #                                                                                                     mouse_atlas_file=mouse_atlas_file)
    # print("unsplice cell num:",len(adata_unspliced_renormalized_df))
    # adata_spliced_renormalized_df, loss_gene_spliced_shortName_list = predict_newData_preprocess_df(gene_dic, adata_spliced,
    #                                                                                                 min_gene_num=0,
    #                                                                                                 mouse_atlas_file=mouse_atlas_file)
    # print("splice cell num:",len(adata_spliced_renormalized_df))
    adata_mrna_renormalized_df, loss_gene_mrna_shortName_list,_ = predict_newData_preprocess_df(gene_dic, adata_mrna,
                                                                                                min_gene_num=0,
                                                                                                mouse_atlas_file=mouse_atlas_file)
    print("mrna cell num:",len(adata_mrna_renormalized_df))
    # adata_unspliced_renormalized_df.to_csv(f"{save_result_path}/preprocessed_unspliced_data.csv")
    # adata_spliced_renormalized_df.to_csv(f"{save_result_path}/preprocessed_spliced_data.csv")
    adata_mrna_renormalized_df.to_csv(f"{save_result_path}/preprocessed_mrna_data.csv")

    # ------------------ predict latent from a trained model  -------------------------------------------------
    # print("spliced")
    # spliced_result = read_model_parameters_fromCkpt(adata_spliced_renormalized_df, config_file, checkpoint_file)
    # spliced_clf_result, spliced_latent_mu_result = spliced_result[0][0], spliced_result[0][1]
    ## save
    # spliced_clf_result_pd = pd.DataFrame(data=np.squeeze(spliced_clf_result), index=adata_spliced_renormalized_df.index,
    #                                      columns=["predict_time"])
    # spliced_clf_result_pd.to_csv(f"{save_result_path}/spliced_predictedTime.csv")
    # spliced_latent_mu_result_pd = pd.DataFrame(data=spliced_latent_mu_result, index=adata_spliced_renormalized_df.index)
    # spliced_latent_mu_result_pd.to_csv(f"{save_result_path}/spliced_latentMu_dim50.csv")

    # print("unspliced")
    # unspliced_result = read_model_parameters_fromCkpt(adata_unspliced_renormalized_df, config_file, checkpoint_file)
    # unspliced_clf_result, unspliced_latent_mu_result = unspliced_result[0][0], unspliced_result[0][1]
    # # save
    # unspliced_clf_result_pd = pd.DataFrame(data=np.squeeze(unspliced_clf_result), index=adata_unspliced_renormalized_df.index,
    #                                        columns=["predict_time"])
    # unspliced_clf_result_pd.to_csv(f"{save_result_path}/unspliced_predictedTime.csv")
    # unspliced_latent_mu_result_pd = pd.DataFrame(data=unspliced_latent_mu_result, index=adata_unspliced_renormalized_df.index)
    # unspliced_latent_mu_result_pd.to_csv(f"{save_result_path}/unspliced_latentMu_dim50.csv")

    print("mrna")
    mrna_result = read_model_parameters_fromCkpt(adata_mrna_renormalized_df, config_file, checkpoint_file)
    mrna_clf_result, mrna_latent_mu_result = mrna_result[0][0], mrna_result[0][1]
    # save
    mrna_clf_result_pd = pd.DataFrame(data=np.squeeze(mrna_clf_result), index=adata_mrna_renormalized_df.index,
                                           columns=["predict_time"])
    mrna_clf_result_pd.to_csv(f"{save_result_path}/mrna_predictedTime.csv")
    mrna_latent_mu_result_pd = pd.DataFrame(data=mrna_latent_mu_result, index=adata_mrna_renormalized_df.index)
    mrna_latent_mu_result_pd.to_csv(f"{save_result_path}/mrna_latentMu_dim50.csv")

    # save gene list
    used_gene_list = pd.DataFrame(data=adata_mrna_renormalized_df.columns, columns=["used_gene_shortName"])
    used_gene_list.to_csv(f"{save_result_path}/used_geneShortName.csv")
    return


if __name__ == '__main__':
    main(sc_file_name="/mnt/yijun/public/for_yijun_fangxin/MouseTracheaE16_muhe/E16_Dec7v3_merged.h5ad")
    main(sc_file_name="/mnt/yijun/public/for_yijun_fangxin/mouse_erythoid/erythroid_lineage.h5ad")
    print("Finish all.")
