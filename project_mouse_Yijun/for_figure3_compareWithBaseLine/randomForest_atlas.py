# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：randomForest_atlas.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/3/22 10:20 


cd /mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/
source ~/.bashrc
nohup python -u project_mouse_Yijun/for_figure3_compareWithBaseLine/randomForest_atlas.py >> logs/for_figure3_compareWithBaseLine_randomForest_atlas.log 2>&1 &

"""

import os
import sys

# sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/CNNC-master/utils")
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/PyTorch-VAE-master")
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan")
import anndata as ad
import pandas as pd
from collections import Counter
from utils.utils_DandanProject import *
from utils.utils_Dandan_plot import *
import time
import logging
from utils.logging_system import LogHelper


def main():
    method = "randomForest"
    save_path=f"/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/240328_kFold_mouse_atlas_data_onlyTestTime/{method}_atlas/"
    # save_path=f"/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/240322_forFig3_compareWithBaseLine/{method}_atlas/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    data_golbal_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/"
    data_path = "/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/"
    sc_data_file_csv = f"{data_path}/data_count_hvg.csv"
    cell_info_file_csv = f"{data_path}/cell_with_time.csv"

    # ---------------------------------------set logger and parameters, creat result save path and folder----------------------------------------------
    logger_file = f'{save_path}/{method}_run.log'
    LogHelper.setup(log_path=logger_file, level='INFO')
    _logger = logging.getLogger(__name__)
    _logger.info("Finished setting up the logger at: {}.".format(logger_file))
    _logger.info("Train on dataset: {}.".format(data_golbal_path + data_path))

    sc_expression_df, cell_time = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path, sc_data_file_csv, cell_info_file_csv,
                                                                                min_cell_num=50,
                                                                                min_gene_num=100)
    # ---------------------------------------- set donor list and dictionary -----------------------------------------------------
    donor_list = np.unique(cell_time["donor"])
    donor_list = sorted(donor_list, key=Embryodonor_resort_key)
    donor_dic = dict()
    for i in range(len(donor_list)):
        donor_dic[donor_list[i]] = i
    # batch_dic = donor_dic.copy()
    print("Consider donor as batch effect, donor use label: {}".format(donor_dic))
    print("For each donor (donor_id, cell_num):{} ".format(Counter(cell_time["donor"])))

    kFold_test_result_df = pd.DataFrame(columns=['time', 'pseudotime'])

    # use one donor as test set, other as train set
    adata = ad.AnnData(X=sc_expression_df,obs=cell_time)
    print(len(donor_list))
    for donor in donor_list:
        # donor = "embryo_1"
        # start_time = time.time()
        train_adata = adata[adata.obs["donor"] != donor].copy()
        test_adata = adata[adata.obs["donor"] == donor].copy()
        # move one cell from test to train to occupy the test time
        RF_model = random_forest_regressor(train_x=train_adata.X, train_y=train_adata.obs["time"])
        # RF_model = random_forest_classifier(train_x=train_adata.X, train_y=train_adata.obs["time"])
        test_y_predicted = RF_model.predict(test_adata.X)

        test_result_df = pd.DataFrame(test_adata.obs["time"],index=test_adata.obs.index)
        test_result_df["pseudotime"] = test_y_predicted

        kFold_test_result_df = pd.concat([kFold_test_result_df, test_result_df], axis=0)
        # end_time = time.time()
        # elapsed_time = end_time - start_time
        # _logger.info(f"**** Total execution time: {elapsed_time} seconds **** k-fold on {donor}")
        # exit(1)
    print("k-fold test final result:")
    corr(kFold_test_result_df["time"], kFold_test_result_df["pseudotime"])
    kFold_test_result_df.to_csv(f'{save_path}/result_df.csv', index=True)
    print(f"test result save at {save_path}/result_df.csv")

    f = plot_psupertime_density(kFold_test_result_df, label_key="time", psupertime_key="pseudotime")
    f.savefig(f"{save_path}/labelsOverPsupertime.png")
    print(f"figure save at {save_path}/labelsOverPsupertime.png")




def corr(x1, x2, special_str=""):
    from scipy.stats import spearmanr, kendalltau
    sp_correlation, sp_p_value = spearmanr(x1, x2)
    ke_correlation, ke_p_value = kendalltau(x1, x2)

    sp = f"{special_str} spearman correlation score: {sp_correlation}, p-value: {sp_p_value}."
    print(sp)
    ke = f"{special_str} kendalltau correlation score: {ke_correlation},p-value: {ke_p_value}."
    print(ke)

    return sp, ke


def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=50, random_state=123)
    model.fit(train_x, train_y)
    return model


def random_forest_regressor(train_x, train_y):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(max_depth=2, random_state=0)
    model.fit(train_x, train_y)
    return model


if __name__ == '__main__':
    main()