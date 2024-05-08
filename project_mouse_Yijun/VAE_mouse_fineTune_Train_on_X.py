# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：VAE_mouse_fineTune_Train_on_X.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/11/30 10:14 

2023-12-01 16:34:15
cd /mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/
source ~/.bashrc
nohup python -u project_mouse_Yijun/VAE_mouse_fineTune_Train_on_X.py > logs/VAE_mouse_fineTune.log 2>&1 &

2023-11-30 10:15:58
make a fine tune VAE version

2023/10/23 10:07
return fangxin latent space data of http://127.0.0.1:18888/tree/public/for_yijun_fangxin

use check point of /mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/231020_plotLatentSpace_mouse_data_minGene50_hvg1000CalByEachOrgan_timeCorGene/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch200_minGeneNum100/wholeData/SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial/version_0/checkpoints
gene list with ENS name /mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/231020_plotLatentSpace_mouse_data_minGene50_hvg1000CalByEachOrgan_timeCorGene/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch200_minGeneNum100/preprocessed_gene_info.csv


"""

import sys

sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan")

from utils.utils_DandanProject import *

from utils.utils_Dandan_plot import *
import pandas as pd

import argparse


def main():
    print("train (fine tune) on mrna, which is u plus s; and test on u and s, respectively.")
    parser = argparse.ArgumentParser(description="fine tune model")
    parser.add_argument('--special_path_str', type=str,  # 2023-07-13 17:40:22
                        default="test",
                        help="results all save here")
    parser.add_argument('--fine_tune_mode', type=str,  # 2023-07-13 17:40:22
                        # default="withMoreFeature",
                        # default="withCellType",
                        default="withoutCellType",
                        help="fine_tune_mode")
    parser.add_argument('--sc_file_name', type=str,
                        default="MouseTracheaE16_muhe/E16_Dec7v3_merged",
                        help="sc file folder path.")
    parser.add_argument('--clf_weight', type=float,
                        default=0.01,
                        help="clf_weight.")
    args = parser.parse_args()
    sc_file_name = f"/mnt/yijun/public/for_yijun_fangxin/{args.sc_file_name}.h5ad"
    fine_tune_mode = args.fine_tune_mode
    clf_weight = args.clf_weight
    special_path_str = args.special_path_str

    print(f"for sc file {sc_file_name} with fine tune mode {fine_tune_mode} with clf weight {clf_weight}.")
    # ----------------- imput parameters  -----------------
    mouse_atlas_file = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/" \
                       "mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/" \
                       "data_count_hvg.csv"
    config_file = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/vae_model_configs/supervise_vae_regressionclfdecoder_mouse_atlas_finetune.yaml"
    checkpoint_file = '/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/' \
                      '231020_plotLatentSpace_mouse_data_minGene50_hvg1000CalByEachOrgan_timeCorGene/' \
                      'mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/' \
                      'supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch200_minGeneNum100/' \
                      'wholeData/SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial/version_0/checkpoints/last.ckpt'
    _temp = sc_file_name.split("/")[-1].replace(".h5ad", "")
    save_result_path = f"/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/{special_path_str}fangxinData/{_temp}/{fine_tune_mode}_clfWeight{clf_weight}/"

    # save parameters used now 2023-12-01 16:30:17 useless
    # _local_variables = locals().copy()
    # _df = get_parameters_df(_local_variables)
    # _df.to_csv(f"{save_result_path}/parameters_use_in_this_results.csv")

    # ----------------- read test adata -----------------
    adata = sc.read_h5ad(filename=sc_file_name)
    try:
        adata = adata[adata.obs["celltype"] != "Doublet", :].copy()
    except:
        adata = adata[adata.obs["cell_type"] != "Doublet", :].copy()
    print("Import data, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    print("Annotation information of data includes: {}".format(adata.obs_keys()))  # 胞注釋信息的keys
    print("Cell id first 5: {}".format(adata.obs_names[:5]))  # 返回胞ID 数据类型是object
    print("Gene id first 5: {}".format(adata.var_names[:5]))  # 返回基因数据类型是list adata.obs.head()# 査看前5行的数据

    import scvelo as scv
    print(f"moment this new data.")
    scv.settings.verbosity = 3  # show errors(0), warnings(1), info(2), hints(3)
    scv.settings.presenter_view = True  # set max width size for presenter view
    scv.set_figure_params('scvelo')  # for beautified visualization
    scv.pl.proportions(adata)
    scv.pp.filter_and_normalize(adata)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    adata.X = adata.X.toarray()
    try:
        adata.obs.rename(columns={'celltype': 'cell_type'}, inplace=True)
    except:
        pass
    hvg_dic = dict()
    try:
        hvg_cellRanger_list = calHVG_adata(adata.copy(), gene_num=1000, method="cell_ranger")
        hvg_dic["cell ranger"] = hvg_cellRanger_list
    except:
        pass
    hvg_seurat_list = calHVG_adata(adata.copy(), gene_num=1000, method="seurat")
    hvg_seurat_v3_list = calHVG_adata(adata.copy(), gene_num=1000, method="seurat_v3")
    hvg_dic["seurat"] = hvg_seurat_list
    hvg_dic["seurat v3"] = hvg_seurat_v3_list
    draw_venn(hvg_dic)

    gene_dic = geneId_geneName_dic()
    adata.obs['cell_type_encoded'] = adata.obs['cell_type'].astype('category').cat.codes
    # -- unsplice data is more sparse than splice data
    adata_unspliced = adata.copy()
    adata_unspliced.X = adata.layers["Mu"]
    # adata_unspliced.X = adata.layers["unspliced"].toarray()
    adata_unspliced.obs["cell_id"] = adata_unspliced.obs_names
    adata_unspliced.obs_names = "unspliced_" + adata_unspliced.obs_names
    adata_unspliced.obs["s_or_mrna"] = "unspliced"

    adata_spliced = adata.copy()
    adata_spliced.X = adata.layers["Ms"]
    # adata_spliced.X = adata.layers["spliced"].toarray()
    adata_spliced.obs["cell_id"] = adata_spliced.obs_names
    adata_spliced.obs_names = "spliced_" + adata_spliced.obs_names
    adata_spliced.obs["s_or_mrna"] = "spliced"

    # adata_mrna = adata.copy()
    # adata_mrna.X = adata.layers["spliced"].toarray() + adata.layers["unspliced"].toarray()
    # adata_mrna.obs["cell_id"] = adata_mrna.obs_names
    # adata_mrna.obs_names = "mrna_" + adata_mrna.obs_names
    # adata_mrna.obs["s_or_mrna"] = "mrna"

    # adata.X[0, :].toarray()
    # adata.layers["spliced"][0, :].toarray()
    # adata.layers["unspliced"][0, :].toarray()
    # adata_mrna.X[0, :]
    # adata_spliced.X[0, :]
    # adata_spliced.layers["unspliced"][0, :].toarray()
    # adata_unspliced.X[0, :]
    # adata_unspliced.layers["spliced"][0, :].toarray()


    # adata_new_copy=adata_new.copy()

    # cal_new_adata(adata_spliced, gene_dic, mouse_atlas_file, f"{save_result_path}/spliced/", config_file, checkpoint_file, fine_tune_mode=fine_tune_mode,
    #               clf_weight=clf_weight, hvg_dic=hvg_dic)
    # adata_all = ad.concat([adata_spliced, adata_mrna], axis=0)
    # cal_new_adata(adata_all, gene_dic, mouse_atlas_file, f"{save_result_path}/all/", config_file, checkpoint_file, fine_tune_mode=fine_tune_mode,
    #               clf_weight=clf_weight, hvg_dic=hvg_dic)
    cal_new_adata(adata, gene_dic, mouse_atlas_file, f"{save_result_path}/all/", config_file, checkpoint_file, fine_tune_mode=fine_tune_mode,
                  clf_weight=clf_weight, hvg_dic=hvg_dic, adata_test={"spliced": adata_spliced, "unspliced": adata_unspliced})

    return


def cal_new_adata(adata_train, gene_dic, mouse_atlas_file, save_result_path, config_file, checkpoint_file, fine_tune_mode, clf_weight, hvg_dic=None,
                  adata_test=None):
    if not os.path.exists(f"{save_result_path}"):
        os.makedirs(save_result_path)
    print(f"make result save at {save_result_path}")
    # ----------------- get renormalized test data -----------------
    # adata_all = ad.concat([adata_train, adata_test], axis=0)
    trainData_renormalized_df, loss_gene_shortName_list, train_cell_info_df, trainData_renormalized_hvg_df = predict_newData_preprocess_df(gene_dic, adata_train,
                                                                                                                                           min_gene_num=0,
                                                                                                                                           mouse_atlas_file=mouse_atlas_file,
                                                                                                                                           hvg_dic=hvg_dic,
                                                                                                                                           )
    remain_hvg_list = list(trainData_renormalized_hvg_df.columns)
    if adata_test is not None:
        testData_spliced_renormalized_df, _, test_spliced_cell_info_df, testData_spliced_renormalized_hvg_df = predict_newData_preprocess_df(gene_dic,
                                                                                                                                             adata_test["spliced"],
                                                                                                                                             min_gene_num=0,
                                                                                                                                             mouse_atlas_file=mouse_atlas_file,
                                                                                                                                             hvg_dic=remain_hvg_list,
                                                                                                                                             )
        test_spliced = {"df": testData_spliced_renormalized_df,
                        "cell_info": test_spliced_cell_info_df,
                        "hvg_df": testData_spliced_renormalized_hvg_df}
        testData_unspliced_renormalized_df, _, test_unspliced_cell_info_df, testData_unspliced_renormalized_hvg_df = predict_newData_preprocess_df(gene_dic,
                                                                                                                                                   adata_test["unspliced"],
                                                                                                                                                   min_gene_num=0,
                                                                                                                                                   mouse_atlas_file=mouse_atlas_file,
                                                                                                                                                   hvg_dic=remain_hvg_list,
                                                                                                                                                   )
        test_unspliced = {"df": testData_unspliced_renormalized_df,
                          "cell_info": test_unspliced_cell_info_df,
                          "hvg_df": testData_unspliced_renormalized_hvg_df}
        testData_dic = {"spliced": test_spliced, "unspliced": test_unspliced}
    print("mrna cell num:", len(trainData_renormalized_df))
    trainData_renormalized_df.to_csv(f"{save_result_path}/preprocessed_data.csv")
    trainData_renormalized_hvg_df.to_csv(f"{save_result_path}/preprocessed_data_hvg.csv")

    # ------------------ predict latent from a trained model  -------------------------------------------------
    plt_attr = list(set(adata_train.obs.columns) & {"ClusterName", "specific_type", "cell_type", "celltype", "theiler", "stage", "sample", "sequencing.batch",
                                                    "s_or_mrna"})
    # 2023-12-13 12:34:12 finetuning on mrna data, and test on spliced data
    original_result, fine_tune_result_data, fine_tune_test_result_dic = read_model_parameters_fromCkpt(trainData_renormalized_df, config_file, checkpoint_file,
                                                                                                       y_label=16,
                                                                                                       save_result_path=save_result_path,
                                                                                                       cell_time_info=train_cell_info_df,
                                                                                                       fine_tune_mode=fine_tune_mode,
                                                                                                       clf_weight=clf_weight,
                                                                                                       sc_expression_df_add=trainData_renormalized_hvg_df,
                                                                                                       plt_attr=plt_attr,
                                                                                                       testData_dic=testData_dic)
    fine_tune_result_data.layers["original_latent_mu_result"] = original_result[0][1].cpu().numpy()
    fine_tune_result_data.layers["fine_tune_latent_mu_result"] = fine_tune_result_data.X
    fine_tune_result_data.write_h5ad(f"{save_result_path}/fine_tune_result.h5ad")

    spliced_fine_tune_result_data = fine_tune_test_result_dic["spliced"].copy()
    spliced_fine_tune_result_data.write_h5ad(f"{save_result_path}/spliced_latent_mu.h5ad")
    unspliced_fine_tune_result_data = fine_tune_test_result_dic["unspliced"].copy()
    unspliced_fine_tune_result_data.write_h5ad(f"{save_result_path}/unspliced_latent_mu.h5ad")

    # import numpy as np
    # from scipy.integrate import solve_ivp
    # from scipy.optimize import minimize
    # def ode_system(t, y, alph, beta, gmma):
    #     u, s = y[0], y[1]
    #     dyu_dt = alph - beta * u
    #     dys_dt = beta * u - gmma * s
    #     return [dyu_dt, dys_dt]
    #
    # #
    # def loss(params, t, u, s):
    #     a, b, c = params
    #     solution = solve_ivp(ode_system, [t[0], t[-1]], [u[0], s[0]], args=(a, b, c), t_eval=t)
    #     u_pred, s_pred = solution.y
    #     return np.mean((u - u_pred) ** 2 + (s - s_pred) ** 2)
    # #
    # y_s = spliced_fine_tune_result_data.X[:, 0]
    # y_u = mrna_fine_tune_result_data.X[:, 0] - spliced_fine_tune_result_data.X[:, 0]
    # t = np.linspace(0, 10, len(y_u)) + 16
    # #
    # initial_params = [0.1, 0.1, 0.1]
    # result = minimize(loss, initial_params, args=(t, y_u, y_s))
    #
    # #
    # optimized_params = result.x
    # print("Optimized parameters:", optimized_params)
    # dyu_dt, dys_dt = ode_system(16, [y_u, y_s], optimized_params[0], optimized_params[1], optimized_params[2])
    plt_latentDim(spliced_fine_tune_result_data, unspliced_fine_tune_result_data, save_result_path)

    # save gene list
    used_gene_pd = pd.DataFrame(data=trainData_renormalized_df.columns, columns=["used_gene_shortName"])
    used_gene_pd.to_csv(f"{save_result_path}/used_geneShortName.csv")
    loss_gene_mrna_shortName_pd = pd.DataFrame(data=loss_gene_shortName_list, columns=["miss_gene_shortName"])
    loss_gene_mrna_shortName_pd.to_csv(f"{save_result_path}/miss_gene_shortName.csv")


def plt_latentDim(spliced_fine_tune_result_data, unspliced_fine_tune_result_data, save_result_path):
    import matplotlib.pyplot as plt
    import numpy as np

    # 获取列数
    num_cols = spliced_fine_tune_result_data.X.shape[1]

    # 每行显示的子图数
    cols_per_row = 5

    # 计算需要的行数
    num_rows = (num_cols + cols_per_row - 1) // cols_per_row

    # 创建子图
    fig, axs = plt.subplots(num_rows, cols_per_row, figsize=(20, num_rows * 4))
    spliced_matrix = spliced_fine_tune_result_data.X
    unspliced_matrix = unspliced_fine_tune_result_data.X
    cell_type_list = spliced_fine_tune_result_data.obs["cell_type"]

    unique_labels = np.unique(cell_type_list)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, colors))

    for i in range(num_cols):
        _data = {"s": spliced_matrix[:, i], "u": unspliced_matrix[:, i], "cell_type": cell_type_list}
        _data = pd.DataFrame(_data)
        row = i // cols_per_row
        col = i % cols_per_row
        for _label in unique_labels:
            _data_subtype = _data[_data["cell_type"] == _label]
            axs[row, col].scatter(_data_subtype["s"], _data_subtype["u"], c=label_to_color[_label], label=_label, alpha=0.3, s=3)
            axs[row, col].set_title(f'Column {i + 1}')

    # 隐藏多余的子图
    for i in range(num_cols, num_rows * cols_per_row):
        row = i // cols_per_row
        col = i % cols_per_row
        axs[row, col].axis('off')

    plt.legend(markerscale=5)
    plt.tight_layout()
    plt.savefig(f"{save_result_path}/latent_dim.png", dpi=200)
    plt.savefig(f"{save_result_path}/latent_dim.pdf", format='pdf')
    plt.show()
    plt.close()


if __name__ == '__main__':
    main()

    print("Finish all.")
