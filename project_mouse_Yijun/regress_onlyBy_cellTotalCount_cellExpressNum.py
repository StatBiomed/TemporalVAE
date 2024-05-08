# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：regress_onlyBy_cellTotalCount_cellExpressNum.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/10/8 16:09 

"""

import os
import sys

# sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/CNNC-master/utils")
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/PyTorch-VAE-master")

import torch

torch.set_float32_matmul_precision('high')
import pyro

smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
assert pyro.__version__.startswith('1.8.5')
pyro.set_rng_seed(1)
from utils.utils_DandanProject import *
from models import *
from utils.utils_Dandan_plot import *

from utils.utils_Dandan_plot import plot_boxPlot_nonExpGene_percentage_whilePreprocess
from utils.utils_Dandan_plot import plot_boxPlot_total_count_per_cell_whilePreprocess


def main(organ):
    # organ = "Liver"
    # organ = "Heart"
    # organ = "Brain"
    data_golbal_path = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/"
    data_path = f"{data_golbal_path}/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_of{organ}/"
    # ---------------------------set logger and parameters, creat result save path and folder--------------------------
    sc_data_file_csv = data_path + "/data_count_hvg.csv"
    cell_info_file_csv = data_path + "/cell_with_time.csv"
    min_cell_num = 50
    min_gene_num = 50
    # ------------ Preprocess data, with hvg gene from preprocess_data_mouse_embryonic_development.py------------------------
    try:
        adata = anndata.read_csv(sc_data_file_csv, delimiter='\t')
    except:
        adata = anndata.read_csv(sc_data_file_csv, delimiter=',')
    adata = adata.T  # 基因和cell转置矩阵
    cell_time = pd.read_csv(cell_info_file_csv, sep="\t", index_col=0)

    plot_boxPlot_nonExpGene_percentage_whilePreprocess(adata, cell_time, "donor",
                                                       "", "",
                                                       special_file_str=f"{organ}:0RawAdata", save_images=False)
    plot_boxPlot_total_count_per_cell_whilePreprocess(adata, cell_time, "donor",
                                                      "", "",
                                                      special_file_str=f"{organ}:0RawAdata", save_images=False)
    # 数据数目统计
    _shape = adata.shape
    _new_shape = (0, 0)
    while _new_shape != _shape:  # make sure drop samples and genes
        _shape = adata.shape
        sc.pp.filter_cells(adata, min_genes=min_gene_num)  # drop samples with less than 20 gene expression
        sc.pp.filter_genes(adata, min_cells=min_cell_num)  # drop genes which none expression in 3 samples
        _new_shape = adata.shape
    plot_boxPlot_nonExpGene_percentage_whilePreprocess(adata, cell_time, "donor",
                                                       "", "",
                                                       special_file_str=f"{organ}:1FilterByminGene&CellSetting",
                                                       save_images=False)
    plot_boxPlot_total_count_per_cell_whilePreprocess(adata, cell_time, "donor",
                                                      "", "",
                                                      special_file_str=f"{organ}:1FilterByminGene&CellSetting", save_images=False)
    # sc.pp.normalize_total(adata, target_sum=1e6)
    sc.pp.log1p(adata)
    plot_boxPlot_nonExpGene_percentage_whilePreprocess(adata, cell_time, "donor",
                                                       "", "",
                                                       special_file_str=f"{organ}:2AfterLogRowCount", save_images=False)
    plot_boxPlot_total_count_per_cell_whilePreprocess(adata, cell_time, "donor",
                                                      "", "",
                                                      special_file_str=f"{organ}:2AfterLogRowCount", save_images=False)
    sc_expression_df = pd.DataFrame(data=adata.X, columns=adata.var.index, index=adata.obs.index)
    sc_reguress_df = pd.DataFrame(index=adata.obs.index)
    sc_reguress_df["express_gene_num"] = sc_expression_df.apply(calculate_gene_num, axis=1).copy()
    sc_reguress_df["express_total_log_count"] = sc_expression_df.apply(calculate_total_log_count, axis=1).copy()

    cell_time2 = cell_time.loc[sc_reguress_df.index].copy()
    # do k-fold test
    time_list = np.unique(cell_time2["time"])
    # use one donor as test set, other as train set
    kFold_test_result_df = pd.DataFrame(columns=['time', 'pseudotime'])
    for time in time_list:
        train_df = sc_reguress_df.loc[cell_time2[cell_time2['time'] != time].index].copy()
        train_y = cell_time2.loc[train_df.index].copy()
        test_df = sc_reguress_df.loc[cell_time2[cell_time2['time'] == time].index].copy()
        test_y = cell_time2.loc[test_df.index].copy()
        print(f"{time}: train {len(train_df)} cells, test {len(test_df)} cells")
        # move one cell from test to train to occupy the test time
        RF_model = random_forest_regressor(train_x=train_df, train_y=train_y["time"])
        # RF_model = random_forest_classifier(train_x=train_adata.X, train_y=train_adata.obs["time"])
        test_y_predicted = RF_model.predict(test_df)

        test_result_df = pd.DataFrame(test_y["time"])
        test_result_df["pseudotime"] = test_y_predicted

        kFold_test_result_df = pd.concat([kFold_test_result_df, test_result_df], axis=0)
    # RF_model = random_forest_regressor(train_x=sc_reguress_df, train_y=cell_time2["time"])
    # RF_model = random_forest_classifier(train_x=train_adata.X, train_y=train_adata.obs["time"])
    # y_predicted = RF_model.predict(sc_reguress_df)
    # print_corr(kFold_test_result_df["pseudotime"],kFold_test_result_df["time"],organ)
    from check_highGeneExpressNumCell_corr import plot_violin
    plot_violin(kFold_test_result_df,special_str=organ+' ')


def print_corr(x1, x2, organ):
    from scipy.stats import spearmanr, kendalltau, pearsonr
    sp_correlation, sp_p_value = spearmanr(x1, x2)
    pear_correlation, pear_p_value = pearsonr(x1, x2)

    ke_correlation, ke_p_value = kendalltau(x1, x2)
    print(f"Final {organ}: {len(x1)} cells.")
    print(f"-- spearman correlation score: {np.round(sp_correlation, 5)}, p-value: {sp_p_value}.")
    print(f"-- pearson correlation score: {np.round(pear_correlation, 5)}, p-value: {pear_p_value}.")
    print(f"-- kendalltau correlation score: {np.round(ke_correlation, 5)},p-value: {ke_p_value}.")


# 定义函数来计算 "express_gene_num"
def calculate_gene_num(row):
    return (row != 0).sum()


# 定义函数来计算 "express_total_log_count"
def calculate_total_log_count(row):
    return row.sum()


def random_forest_regressor(train_x, train_y):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(max_depth=2, random_state=0)
    model.fit(train_x, train_y)
    return model


if __name__ == '__main__':
    main("Brain")
    main("Heart")
    main("Liver")
