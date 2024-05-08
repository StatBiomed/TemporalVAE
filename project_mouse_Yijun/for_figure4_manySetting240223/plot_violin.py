# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：plot_violin.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/3/21 11:24 

"""
from utils.utils_DandanProject import denormalize
from utils.utils_Dandan_plot import plot_on_each_test_donor_violin_fromDF
import pandas as pd
import json



def main(json_file_name, cell_time_info_file, save_path):
    with open(json_file_name, "r") as json_file:
        data_list = []
        for line in json_file:
            json_obj = json.loads(line)
            data_list.append(json_obj)
    # get donor list from json file
    data_dic = data_list[0]
    donor_list = list(data_dic.keys())
    donor_list = list(set([i.replace("_pseudotime", "").replace("_cellid", "") for i in donor_list]))

    # only plot atlas donor
    donor_stereo_list = [i for i in donor_list if "stereo" in i]

    cell_time_stereo_pd = pd.read_csv(cell_time_info_file, sep=",", index_col=0)
    # color_dic = plot_on_each_test_donor_violin_fromDF(cell_time_stereo.copy(), save_path, physical_str="predicted_time", x_str="time")
    donor_stereo_df = pd.DataFrame(columns=["pseudotime"])
    # donor_stereo_df = pd.DataFrame(columns=["pseudotime", "time", "donor"])
    for _donor in donor_stereo_list:
        # _time = cell_time_stereo_pd[cell_time_stereo_pd['donor'] == _donor]['time'][0]
        # _temp_df = pd.DataFrame(dict(pseudotime=data_dic[_donor + "_pseudotime"], time=_time, donor=_donor))
        _temp_df = pd.DataFrame(index=data_dic[_donor + "_cellid"], data=data_dic[_donor + "_pseudotime"], columns=["pseudotime"])
        donor_stereo_df = pd.concat([donor_stereo_df, _temp_df], axis=0)

    donor_stereo_df["physical_pseudotime"] = donor_stereo_df["pseudotime"].apply(denormalize, args=(8.5, 18.75, -5, 5))
    cell_time_stereo_pd2 = pd.concat([cell_time_stereo_pd, donor_stereo_df], axis=1)
    color_dic = plot_on_each_test_donor_violin_fromDF(cell_time_stereo_pd2.copy(), save_path, physical_str="physical_pseudotime", x_str="time")

    return
if __name__ == '__main__':
    parameter_combinations = [
        {  # figure1
            "json_file_name": "/mnt/yijun/"
                         "nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/231004_trainOn_mouse_embryo_stereo_organs_kFold_minGene50_75_100/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofBrain/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch100_minGeneNum100_mouseEmbryonicDevelopment_embryoneg5to5/SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial_plot_on_all_test_donor_timeembryoneg5to5_celltype_update_testCLF.json",
            "cell_time_info_file": "/mnt/yijun/"
                                   "nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/231004_trainOn_mouse_embryo_stereo_organs_kFold_minGene50_75_100/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofBrain/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch100_minGeneNum100_mouseEmbryonicDevelopment_embryoneg5to5/preprocessed_cell_info.csv",
            "save_path": "/mnt/yijun/"
                         "nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/231004_trainOn_mouse_embryo_stereo_organs_kFold_minGene50_75_100/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofBrain/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch100_minGeneNum100_mouseEmbryonicDevelopment_embryoneg5to5/"
        },

        {  # figure2
            "json_file_name": "/mnt/yijun/"
                         "nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/231006_kFoldOnStereo_trainOnAtlasSubStereo_minGeneNum100_withoutAdversarial/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch300_externalDataofBrain/SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial_plot_on_all_test_donor_timeembryoneg5to5_celltype_update_testCLF.json",
            "cell_time_info_file": "/mnt/yijun/"
                                   "nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/231004_trainOn_mouse_embryo_stereo_organs_kFold_minGene50_75_100/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofBrain/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch100_minGeneNum100_mouseEmbryonicDevelopment_embryoneg5to5/preprocessed_cell_info.csv",
            "save_path": "/mnt/yijun/"
                         "nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/231006_kFoldOnStereo_trainOnAtlasSubStereo_minGeneNum100_withoutAdversarial/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch300_externalDataofBrain/"

        },
        {  # figure3
            "json_file_name": "/mnt/yijun/"
                         "nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/231006_kFoldOnStereo_trainOnAtlasSubStereo_minGeneNum100_withAdversarial/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/supervise_vae_regressionclfdecoder_adversarial0121212_231001_06_dim50_timeembryoneg5to5_epoch300_externalDataofBrain/SuperviseVanillaVAE_regressionClfDecoder_adversarial_plot_on_all_test_donor_timeembryoneg5to5_celltype_update_testCLF.json",
            "cell_time_info_file": "/mnt/yijun/"
                                   "nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/231004_trainOn_mouse_embryo_stereo_organs_kFold_minGene50_75_100/mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofBrain/supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch100_minGeneNum100_mouseEmbryonicDevelopment_embryoneg5to5/preprocessed_cell_info.csv",
            "save_path": "/mnt/yijun/"
                         "nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/231006_kFoldOnStereo_trainOnAtlasSubStereo_minGeneNum100_withAdversarial/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/supervise_vae_regressionclfdecoder_adversarial0121212_231001_06_dim50_timeembryoneg5to5_epoch300_externalDataofBrain/"
        },
    ]
    # 遍历参数组合列表，依次调用 main 函数
    for params in parameter_combinations:
        main(**params)
