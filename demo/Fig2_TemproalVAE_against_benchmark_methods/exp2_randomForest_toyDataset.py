# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE
@File    ：exp2_randomForest_toyDataset.py
@IDE     ：PyCharm
@Author  ：awa121
@Date    ：2023/9/26 14:29


use dataset mentioned in psupertime manuscript

"""
import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"project_root: {project_root}")
sys.path.append(project_root)
from TemporalVAE.utils.utils_plot import plot_psupertime_density
import anndata
import os
import numpy as np
import pandas as pd

from plotFig2_check_corr import preprocess_parameters,corr
def main():
    dataset_list = ["acinarHVG", "embryoBeta", "humanGermline"]
    for dataset in dataset_list:
        method_calculate(dataset)
def method_calculate(dataset):
    adata, data_x_df, data_y_df = preprocess_parameters(dataset)
    method = "randomForest"
    if not os.path.exists(f'demo/Fig2_TemproalVAE_against_benchmark_methods/{method}_results'):
        os.makedirs(f'demo/Fig2_TemproalVAE_against_benchmark_methods/{method}_results')
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
        RF_model = random_forest_regressor(train_x=train_adata.X, train_y=train_adata.obs["time"])
        # RF_model = random_forest_classifier(train_x=train_adata.X, train_y=train_adata.obs["time"])
        test_y_predicted = RF_model.predict(test_adata.X)

        test_result_df = pd.DataFrame(test_adata.obs["time"])
        test_result_df["pseudotime"] = test_y_predicted

        kFold_test_result_df = pd.concat([kFold_test_result_df, test_result_df], axis=0)
    print("k-fold test final result:")
    corr(kFold_test_result_df["time"], kFold_test_result_df["pseudotime"])
    kFold_test_result_df.to_csv(f'demo/Fig2_TemproalVAE_against_benchmark_methods/{method}_results/{dataset}_{method}_result.csv', index=True)
    print(f"test result save at demo/Fig2_TemproalVAE_against_benchmark_methods/{method}_results/{dataset}_{method}_result.csv")
    # f = tp.plot_grid_search(title="Grid Search")
    # f.savefig(f"demo/Fig2_TemproalVAE_against_benchmark_methods/psupertime_results/gridSearch.png")
    # f = tp.plot_model_perf((adata.X, adata.obs.time), figsize=(6, 5))
    # f.savefig(f"demo/Fig2_TemproalVAE_against_benchmark_methods/psupertime_results/modelPred.png")
    # f = tp.plot_identified_gene_coefficients(adata, n_top=20)
    # f.savefig(f"demo/Fig2_TemproalVAE_against_benchmark_methods/psupertime_results/geneCoff.png")
    save_path=f"demo/Fig2_TemproalVAE_against_benchmark_methods/{method}_results/"
    plot_psupertime_density(kFold_test_result_df, save_path,label_key="time", psupertime_key="pseudotime",method=dataset)
    # f.savefig(f"demo/Fig2_TemproalVAE_against_benchmark_methods/{method}_results/{result_file_name}_labelsOverPsupertime.png")
    print(f"figure save at demo/Fig2_TemproalVAE_against_benchmark_methods/{method}_results/{dataset}_labelsOverPsupertime.png")



def random_forest_regressor(train_x, train_y):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(max_depth=2, random_state=0)
    model.fit(train_x, train_y)
    return model


if __name__ == '__main__':
    main()
