# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE
@File    ：exp2_psupertime_toyDataset.py
@IDE     ：PyCharm
@Author  ：awa121
@Date    ：2023/9/19 18:20

use dataset mentioned in psupertime manuscript

"""
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"project_root: {project_root}")
sys.path.append(project_root)
from TemporalVAE.utils.utils_plot import plot_psupertime_density
from pypsupertime import Psupertime
import anndata
import os
import numpy as np
import pandas as pd
from plotFig2_check_corr import preprocess_parameters,corr
def main():
    dataset_list = [ "acinarHVG", "embryoBeta", "humanGermline"]
    for dataset in dataset_list:
        method_calculate(dataset)
def method_calculate(dataset):
    adata, data_x_df, data_y_df = preprocess_parameters(dataset)
    method="psupertime"
    if not os.path.exists(f'demo/Fig2_TemproalVAE_against_benchmark_methods/{method}_results'):
        os.makedirs(f'demo/Fig2_TemproalVAE_against_benchmark_methods/{method}_results')
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
        corr(train_result_adata.obs["time"], train_result_adata.obs["psupertime"], special_str="For train set")
        try:
            test_result_df = tp.model.predict_psuper(test_adata, inplace=False)
        except:
            print("error here?")
        kFold_test_result_df = pd.concat([kFold_test_result_df, test_result_df], axis=0)
        del tp
    print("k-fold test final result:")
    kFold_test_result_df["pseudotime"]=kFold_test_result_df["psupertime"]
    corr(kFold_test_result_df["time"], kFold_test_result_df["pseudotime"])
    kFold_test_result_df.to_csv(f'demo/Fig2_TemproalVAE_against_benchmark_methods/{method}_results/{dataset}_{method}_result.csv', index=True)
    print(f"test result save at demo/Fig2_TemproalVAE_against_benchmark_methods/{method}_results/{dataset}_{method}_result.csv")
    # f = tp.plot_grid_search(title="Grid Search")
    # f.savefig(f"demo/Fig2_TemproalVAE_against_benchmark_methods/psupertime_results/gridSearch.png")
    # f = tp.plot_model_perf((adata.X, adata.obs.time), figsize=(6, 5))
    # f.savefig(f"demo/Fig2_TemproalVAE_against_benchmark_methods/psupertime_results/modelPred.png")
    # f = tp.plot_identified_gene_coefficients(adata, n_top=20)
    # f.savefig(f"demo/Fig2_TemproalVAE_against_benchmark_methods/psupertime_results/geneCoff.png")
    save_path = f"demo/Fig2_TemproalVAE_against_benchmark_methods/{method}_results/"
    plot_psupertime_density(kFold_test_result_df, save_path, label_key="time", psupertime_key="pseudotime", method=dataset)
    print(f"figure save at demo/Fig2_TemproalVAE_against_benchmark_methods/{method}_results/{dataset}_labelsOverPsupertime.png")







if __name__ == '__main__':
    main()
