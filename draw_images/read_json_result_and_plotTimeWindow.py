# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：read_json_result_and_plotTimeWindow.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/8/7 11:27

2023-08-07 11:37:41
read test result (.json) of yz test data,
plot image.
"""
import json
from utils.utils_Dandan_plot import *
import pandas as pd


def plot_time_window_fromJSON(json_file="", threshold_list=[-1, -0.5, 0, 0.5, 1]):
    with open(json_file, "r") as f:
        data = json.load(f)
    # ---------- split save image for subplot.---------
    print("plot time window for total cells.")
    print(data.keys())
    donor_list = [key.replace("_pseudotime", "") for key, _ in data.items() if "pseudotime" in key]

    donors_pseudotime_df = pd.DataFrame(columns=["pseudotime", "donor"])
    for _donor in donor_list:
        _df = pd.DataFrame(data=data[_donor + "_pseudotime"], columns=["pseudotime"], index=data[_donor + "_cellid"])
        _df["donor"] = _donor
        donors_pseudotime_df = pd.concat([donors_pseudotime_df, _df])

    save_file_name = json_file.replace(".json", ".png")


    donor_color_dic = dict()
    donor_real_time_dic = dict()
    for _donor in donor_list:
        donor_color_dic[_donor] = (0.6, 0.6, 0.6)
        donor_real_time_dic[_donor] = 0
    categorical_kde_plot(donors_pseudotime_df, threshold_list=threshold_list, variable="pseudotime", category="donor",
                         category_order=donor_list, horizontal=True, save_png_name=save_file_name,
                         donor_color_dic=donor_color_dic,
                         donor_real_time_dic=donor_real_time_dic)
    return


if __name__ == '__main__':
    # json_file = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/results/230815_newAnno0717_Gene0720_testOnE35/preprocess_02_major_Anno0717_GeneVP0721/vae_prediction_result_nofilterTest_dim50_neg1to1/SuperviseVanillaVAE_regressionClfDecoder_testOnExternal_test_external0802_preprocess_E35_.json"
    # json_file = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/results/230815_newAnno0717_Gene0720_testOnE35/preprocess_02_major_Anno0717_Gene0720/vae_prediction_result_nofilterTest_dim50_neg1to1/SuperviseVanillaVAE_regressionClfDecoder_testOnExternal_test_external0802_preprocess_E35_.json"
    # json_file = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/230823_newAnno0717_testOnSpatialData/preprocess_02_major_Anno0717_Gene0720/supervise_vae_regressionclfdecoder_dim50_timeneg1to1_epoch100_dropDonorno_neg1to1/SuperviseVanillaVAE_regressionClfDecoder_testOnExternal_test_external0823_preprocess_A13_A30_scRNA.json"
    json_file = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/230809_trainOn_mouse_embryonic_development_testOnYZdata0809/mouse_embryonic_development/preprocess_adata_JAX_dataset_1/vae_prediction_result_nofilterTest_dim50_mouseEmbryonicDevelopment_embryoneg1to1/SuperviseVanillaVAE_regressionClfDecoder_testOnExternal_YZ_2data.json"
    threshold_list=[-0.818, -0.636, -0.455, -0.273, -0.091, 0.091, 0.273, 0.455, 0.636, 0.818, 1]
    plot_time_window_fromJSON(json_file=json_file,threshold_list=threshold_list)
