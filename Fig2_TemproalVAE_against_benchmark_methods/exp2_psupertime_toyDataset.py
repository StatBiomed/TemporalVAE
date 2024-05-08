# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE
@File    ：exp2_psupertime_toyDataset.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/9/19 18:20

use dataset mentioned in psupertime manuscript

"""
import sys
import os

print(os.getcwd())
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan")
from utils.utils_Dandan_plot import plot_psupertime_density
from pypsupertime import Psupertime
import anndata
import pandas as pd
import os
import numpy as np
import seaborn as sns
import anndata as ad
import pandas as pd
from matplotlib import pyplot as plt
from collections import Counter

os.chdir('/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/Fig2_TemproalVAE_against_benchmark_methods')


def main():
    method="psupertime"
    # ------------ for Mouse embryonic beta cells dataset:
    result_file_name = "embryoBeta"
    data_x_df = pd.read_csv(f'data_fromPsupertime/{result_file_name}_X.csv', index_col=0).T
    hvg_gene_list = pd.read_csv(f'{os.getcwd()}/data_fromPsupertime/{result_file_name}_gene_list.csv', index_col=0)
    data_x_df = data_x_df[hvg_gene_list["gene_name"]]
    data_y_df = pd.read_csv(f'data_fromPsupertime/{result_file_name}_Y.csv', index_col=0)
    data_y_df = data_y_df["time"]
    preprocessing_params = {"select_genes": "all", "log": True}

    # ------------ for Human Germline dataset:
    # result_file_name="humanGermline"
    # data_x_df = pd.read_csv('data_fromPsupertime/humanGermline_X.csv', index_col=0).T
    # hvg_gene_list=pd.read_csv(f'{os.getcwd()}/data_fromPsupertime/{result_file_name}_gene_list.csv',index_col=0)
    # data_x_df=data_x_df[hvg_gene_list["gene_name"]]
    # data_y_df = pd.read_csv('data_fromPsupertime/humanGermline_Y.csv',index_col=0)
    # data_y_df=data_y_df["time"]
    # preprocessing_params = {"select_genes": "all", "log": True}
    #
    # ------------ for Acinar dataset, in acinar data set total 8 donors with 8 ages:
    # result_file_name = "acinarHVG"
    # data_x_df = pd.read_csv('data_fromPsupertime/acinar_hvg_sce_X.csv', index_col=0).T
    # data_y_df = pd.read_csv('data_fromPsupertime/acinar_hvg_sce_Y.csv')
    # data_y_df = np.array(data_y_df['x'])
    # preprocessing_params = {"select_genes": "all", "log": False}

    # START HERE
    adata_org = anndata.AnnData(data_x_df)
    adata_org.obs["time"] = data_y_df
    print(f"Input Data: n_genes={adata_org.n_vars}, n_cells={adata_org.n_obs}")

    # ------------------preprocess adata here
    tp = Psupertime(n_jobs=5, n_folds=5,
                    preprocessing_params=preprocessing_params
                    )  # if for Acinar cell "select_genes": "all"

    adata = tp.preprocessing.fit_transform(adata_org.copy())
    del tp
    # --------------------------------------------
    donor_list = list(np.unique(data_y_df))
    kFold_test_result_df = pd.DataFrame(columns=['time', 'pseudotime', 'predicted_label'])

    # use one donor as test set, other as train set
    for donor in donor_list:
        train_adata = adata[adata.obs["time"] != donor].copy()
        test_adata = adata[adata.obs["time"] == donor].copy()
        # move one cell from test to train to occupy the test time
        one_test_cell = test_adata[:5].copy()
        test_adata = test_adata[5:].copy()
        train_adata = anndata.concat([train_adata.copy(), one_test_cell.copy()], axis=0)
        _preprocessing_params = {"select_genes": "all", "log": False, "scale": False,
                                 "min_gene_mean": 0, "smooth": False, "normalize": False}
        # fit psupertime model
        tp = Psupertime(n_jobs=5, n_folds=5, preprocessing_params=_preprocessing_params)

        train_result_adata = tp.run(train_adata, "time")

        test_adata = test_adata[:, train_result_adata.var_names].copy()
        time_ordinalLabel_dic = train_result_adata.obs.set_index('time')['ordinal_label'].to_dict()
        test_adata.obs['ordinal_label'] = test_adata.obs['time'].map(time_ordinalLabel_dic)
        corr(train_result_adata.obs["time"], train_result_adata.obs["pseudotime"], special_str="For train set")
        try:
            test_result_df = tp.model.predict_psuper(test_adata, inplace=False)
        except:
            print("error here?")
        kFold_test_result_df = pd.concat([kFold_test_result_df, test_result_df], axis=0)
        del tp
    print("k-fold test final result:")
    corr(kFold_test_result_df["time"], kFold_test_result_df["pseudotime"])
    kFold_test_result_df.to_csv(f'{os.getcwd()}/{method}_results/{result_file_name}_{method}_result.csv', index=True)
    print(f"test result save at {os.getcwd()}/{method}_results/{result_file_name}_{method}_result.csv")
    # f = tp.plot_grid_search(title="Grid Search")
    # f.savefig(f"{os.getcwd()}/psupertime_results/gridSearch.png")
    # f = tp.plot_model_perf((adata.X, adata.obs.time), figsize=(6, 5))
    # f.savefig(f"{os.getcwd()}/psupertime_results/modelPred.png")
    # f = tp.plot_identified_gene_coefficients(adata, n_top=20)
    # f.savefig(f"{os.getcwd()}/psupertime_results/geneCoff.png")
    f = plot_psupertime_density(kFold_test_result_df, label_key="time", psupertime_key="pseudotime")
    f.savefig(f"{os.getcwd()}/{method}_results/{result_file_name}_labelsOverPsupertime.png")
    print(f"figure save at {os.getcwd()}/{method}_results/{result_file_name}_labelsOverPsupertime.png")




def corr(x1, x2, special_str=""):
    from scipy.stats import spearmanr, kendalltau
    sp_correlation, sp_p_value = spearmanr(x1, x2)
    ke_correlation, ke_p_value = kendalltau(x1, x2)

    sp = f"{special_str} spearman correlation score: {sp_correlation}, p-value: {sp_p_value}."
    print(sp)
    ke = f"{special_str} kendalltau correlation score: {ke_correlation},p-value: {ke_p_value}."
    print(ke)

    return sp, ke


if __name__ == '__main__':
    main()
