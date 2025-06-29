# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE
@File    ：exp2_science2022_toyDataset.py
@IDE     ：PyCharm
@Author  ：awa121
@Date    ：2024-07-21 13:44:06

240714NCB revision: add science 2022 method as benchmarking method.
use dataset mentioned in psupertime manuscript

https://shendure-web.gs.washington.edu/content/members/DEAP_website/public/scripts/RNA/time_inference/

"""

# import sys
# import os
# if os.getcwd().split("/")[-1] != "TemporalVAE":
#     os.chdir("..")
# sys.path.append("..")

import os
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"project_root: {project_root}")
sys.path.append(project_root)
from TemporalVAE.utils.utils_plot import plot_psupertime_density
import anndata
# import os
import numpy as np
import pandas as pd
from plotFig2_check_corr import preprocess_parameters,corr
from benchmarking_methods.benchmarking_methods import science2022
def main():
    dataset_list = [ "acinarHVG", "embryoBeta", "humanGermline"]
    for dataset in dataset_list:
        method_calculate(dataset)

    print("Finish all.")
def method_calculate(dataset):
    adata, data_x_df, data_y_df = preprocess_parameters(dataset)
    method = "science2022"
    # os.makedirs(f'{os.getcwd()}/preprocessedData_ByPypsupertime_forSeurat')
    # data_x_df.to_csv(f'{os.getcwd()}/preprocessedData_ByPypsupertime_forSeurat/{dataset}_X.csv')
    # adata.obs["time"].to_csv(f'{os.getcwd()}/preprocessedData_ByPypsupertime_forSeurat/{dataset}_Y.csv')

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

        test_y_predicted = science2022(train_x=train_adata.X, train_y=train_adata.obs["time"],test_df=test_adata.X)
        # s = science_model.evaluate(test_adata.X, np.array(test_adata.obs["time"].tolist()).astype(np.float32), verbose=0)

        test_result_df = pd.DataFrame(test_adata.obs["time"])
        test_result_df["pseudotime"] = test_y_predicted

        kFold_test_result_df = pd.concat([kFold_test_result_df, test_result_df], axis=0)
    print("k-fold test final result:")
    corr(kFold_test_result_df["time"], kFold_test_result_df["pseudotime"])

    # emd, mmd, r_squared=distribution_metric(np.array(kFold_test_result_df["time"]),np.array(kFold_test_result_df["pseudotime"]))

    kFold_test_result_df.to_csv(f'{os.getcwd()}/{method}_results/{dataset}_{method}_result.csv', index=True)
    print(f"test result save at {os.getcwd()}/{method}_results/{dataset}_{method}_result.csv")

    save_path = f"{os.getcwd()}/{method}_results/"
    plot_psupertime_density(kFold_test_result_df, save_path=save_path, label_key="time", psupertime_key="pseudotime", method=method)
    print(f"figure save at {os.getcwd()}/{method}_results/{dataset}_labelsOverPsupertime.png")






if __name__ == '__main__':
    main()
