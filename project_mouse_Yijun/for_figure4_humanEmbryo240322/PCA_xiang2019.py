# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：PCA_xiang2019.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/3/22 10:21 


cd /mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/
source ~/.bashrc
nohup python -u project_mouse_Yijun/for_figure4_humanEmbryo240322/PCA_xiang2019.py >> logs/for_figure4_compareWithBaseLine_PCA_xiang2019.log 2>&1 &

"""

import sys

sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan")
import anndata as ad
from collections import Counter
from utils.utils_DandanProject import *
from utils.utils_Dandan_plot import *
import time
import logging
from utils.logging_system import LogHelper
from sklearn.decomposition import PCA
from utils.utils_Dandan_plot import plot_psupertime_density


def main():
    method = "PCA"
    data_path = "/240322Human_embryo/xiang2019/hvg500/"
    save_path = f"/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/240403_forFig4_compareWithBaseLine/{data_path}/{method}/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    sc_data_file_csv = f"{data_path}/data_count_hvg.csv"
    cell_info_file_csv = f"{data_path}/cell_with_time.csv"

    data_golbal_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/"

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
    donor_list = np.unique(cell_time["day"])
    donor_list = sorted(donor_list, key=Embryodonor_resort_key)
    donor_dic = dict()
    for i in range(len(donor_list)):
        donor_dic[donor_list[i]] = i
    batch_dic = donor_dic.copy()
    print("Consider donor as batch effect, donor use label: {}".format(donor_dic))
    print("For each donor (donor_id, cell_num):{} ".format(Counter(cell_time["day"])))

    kFold_test_result_df = pd.DataFrame(columns=['time', 'pseudotime'])

    # use one donor as test set, other as train set
    adata = ad.AnnData(X=sc_expression_df, obs=cell_time)
    print(len(donor_list))
    start_time = time.time()
    for donor in donor_list:
        train_adata = adata[adata.obs["day"] != donor].copy()
        test_adata = adata[adata.obs["day"] == donor].copy()
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
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"The program took {elapsed_time} seconds to run.")
    print("k-fold test final result:")
    corr(kFold_test_result_df["time"], kFold_test_result_df["pseudotime"])
    kFold_test_result_df.to_csv(f'{save_path}/result_df.csv', index=True)
    print(f"test result save at {save_path}/result_df.csv")
    kFold_test_result_df['predicted_time'] = kFold_test_result_df['pseudotime']
    color_dic = plot_violin_240223(kFold_test_result_df, save_path, real_attr="time", pseudo_attr="predicted_time", special_file_name=method)

    plot_psupertime_density(kFold_test_result_df,save_path, label_key="time", psupertime_key="predicted_time")


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