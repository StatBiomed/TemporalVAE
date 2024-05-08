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

"""
import sys
import os

print(os.getcwd())
import anndata
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from utils.utils_Dandan_plot import plot_psupertime_density

from kFold_check_corr_addLR import preprocess_parameters,corr
def main():
    dataset_list = [ "acinarHVG", "embryoBeta", "humanGermline"]
    for dataset in dataset_list:
        method_calculate(dataset)
def method_calculate(dataset):
    adata, data_x_df, data_y_df = preprocess_parameters(dataset)
    method = "pca"
    if not os.path.exists(f'{os.getcwd()}/{method}_results'):
        os.makedirs(f'{os.getcwd()}/{method}_results')
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
        test_result_df["pseudotime"] = test_lowDim[:, 0]
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

    kFold_test_result_df.to_csv(f'{os.getcwd()}/{method}_results/{dataset}_{method}_result.csv', index=True)
    print(f"test result save at {os.getcwd()}/{method}_results/{dataset}_{method}_result.csv")
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
    save_path = f"{os.getcwd()}/{method}_results/"
    plot_psupertime_density(kFold_test_result_df, save_path=save_path, label_key="time", psupertime_key="pseudotime")
    print(f"figure save at {os.getcwd()}/{method}_results/{dataset}_labelsOverPsupertime.png")





if __name__ == '__main__':
    main()
