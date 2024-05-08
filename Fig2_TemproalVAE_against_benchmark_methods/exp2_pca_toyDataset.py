# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：exp2_pca_toyDataset.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/9/26 17:16
For principle component analysis (PCA),
we calculated the first principal component of the log counts and used this as the pseudotime.
Calculation of Monocle2 uses the following default settings: genes with mean expression <0.1
or expressed in <10 cells filtered out;
negbinomial expression family used;
dimensionality reduction method DDRTree;
root state selected as the state with the highest number of cells from the first label;
function orderCells used to extract the ordering.
对于主成分分析（PCA），
我们计算了日志计数的第一个主分量，并将其用作伪时间。
Monocle2的计算使用以下默认设置：平均表达＜0.1的基因
或在滤出的＜10个细胞中表达；
使用负二项表达式族；
降维方法DDRTree；
根状态，所述根状态被选择为具有来自所述第一标签的最高数目的细胞的状态；
function order用于提取排序的单元格。
"""
import sys
import os

print(os.getcwd())
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan")
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
from sklearn.decomposition import PCA
from scipy.stats import nbinom
import scanpy as sc
from sklearn.tree import DecisionTreeClassifier
from utils.utils_Dandan_plot import plot_psupertime_density
os.chdir('/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/Fig2_TemproalVAE_against_benchmark_methods')


def main():
    method="pca"
    # ------------ for Mouse embryonic beta cells dataset:
    result_file_name = "embryoBeta"
    data_x_df = pd.read_csv(f'data_fromPsupertime/{result_file_name}_X.csv', index_col=0).T
    hvg_gene_list = pd.read_csv(f'{os.getcwd()}/data_fromPsupertime/{result_file_name}_gene_list.csv', index_col=0)
    data_x_df = data_x_df[hvg_gene_list["gene_name"]]
    data_y_df = pd.read_csv(f'data_fromPsupertime/{result_file_name}_Y.csv', index_col=0)
    #### data_y_df["time"]=data_y_df["time"].astype(int)
    data_y_df = data_y_df["time"]
    preprocessing_params = {"select_genes": "all", "log": True}

    # ------------ for Human Germline dataset:
    # result_file_name = "humanGermline"
    # data_x_df = pd.read_csv('data_fromPsupertime/humanGermline_X.csv', index_col=0).T
    # hvg_gene_list = pd.read_csv(f'{os.getcwd()}/data_fromPsupertime/{result_file_name}_gene_list.csv', index_col=0)
    # data_x_df = data_x_df[hvg_gene_list["gene_name"]]
    # data_y_df = pd.read_csv('data_fromPsupertime/humanGermline_Y.csv', index_col=0)
    # data_y_df = data_y_df["time"]
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
    kFold_test_result_df = pd.DataFrame(columns=['time', 'pseudotime'])

    # use one donor as test set, other as train set
    for donor in donor_list:
        train_adata = adata[adata.obs["time"] != donor].copy()
        test_adata = adata[adata.obs["time"] == donor].copy()
        # move one cell from test to train to occupy the test time
        one_test_cell = test_adata[:5].copy()
        test_adata = test_adata[5:].copy()
        train_adata = anndata.concat([train_adata.copy(), one_test_cell.copy()], axis=0)

        # initiate PCA and classifier
        pca = PCA(n_components=2)
        # classifier = DecisionTreeClassifier()
        # transform / fit
        train_lowDim = pca.fit_transform(train_adata.X)
        # classifier.fit(train_lowDim, train_adata.obs["time"])
        # predict "new" data
        test_lowDim = pca.transform(test_adata.X)
        # predict labels using the trained classifier
        test_result_df = pd.DataFrame(test_adata.obs["time"])
        test_result_df["pseudotime"] = test_lowDim[:,0]
        # test_result_df["pseudotime"] = classifier.predict(test_lowDim)

        kFold_test_result_df = pd.concat([kFold_test_result_df, test_result_df], axis=0)

    # fit model by train, predict on test.
    # n_components = 2  # 选择要保留的主成分数量，根据实际情况进行调整
    # pca = PCA(n_components=n_components)
    # principal_components = pca.fit_transform(adata.X)
    # adata.obs["pseudotime"]=principal_components[:,0]
    #  计算DDRTree伪时间
    # sc.pp.neighbors( adata)
    # sc.tl.diffmap( adata)  # 计算Diffusion Map，类似于PCA
    #  adata.uns['iroot'] = 5
    # sc.tl.dpt( adata)  # 计算Diffusion Pseudotime

    # pseudotime_values =  adata.obs['dpt_pseudotime']
    print("k-fold test final result:")
    corr(kFold_test_result_df["time"], kFold_test_result_df["pseudotime"])
    kFold_test_result_df.to_csv(f'{os.getcwd()}/{method}_results/{result_file_name}_{method}_result.csv', index=True)
    print(f"test result save at {os.getcwd()}/{method}_results/{result_file_name}_{method}_result.csv")
    # print("test final result:")
    # corr(adata.obs["time"], adata.obs["pseudotime"])
    # result_df=pd.DataFrame()
    # result_df["time"]=adata.obs["time"]
    # result_df["pseudotime"]=adata.obs["pseudotime"]
    # result_df.to_csv(f'{os.getcwd()}/pca_results/{result_file_name}_{method}_result.csv', index=True)
    # print(f"test result save at {os.getcwd()}/pca_results/{result_file_name}_{method}_result.csv")
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

