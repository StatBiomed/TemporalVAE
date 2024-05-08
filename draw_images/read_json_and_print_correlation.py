# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：read_json_and_print_correlation.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/9/30 16:06 
"""
import pandas as pd
import json
from utils.utils_Dandan_plot import calculate_real_predict_corrlation_score


def main():
    file_name = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/" \
                "230931_trainOn_mouse_embryo_stereo_brain_kFold/mouse_embryo_stereo/" \
                "preprocess_Mouse_embryo_all_stage_minGene50_ofBrain/" \
                "supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch100_mouseEmbryonicDevelopment_embryoneg5to5/" \
                "SuperviseVanillaVAE_regressionClfDecoder_mouse_toyDataset_plot_on_all_test_donor_timeembryoneg5to5_celltype_update_testCLF.json"
    cell_time_stereo_file = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/" \
                            "mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofBrain/cell_with_time.csv"
    cell_time_atlas_file = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/" \
                           "mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/cell_with_time.csv"

    with open(file_name, "r") as json_file:
        data_list = []
        for line in json_file:
            json_obj = json.loads(line)
            data_list.append(json_obj)
    # get donor list from json file
    data_dic = data_list[0]
    donor_list = list(data_dic.keys())
    donor_list = list(set([i.replace("_pseudotime", "").replace("_cellid", "") for i in donor_list]))
    # split donor as atlas and stereo
    donor_atlas_list = [i for i in donor_list if "stereo" not in i]
    donor_stereo_list = [i for i in donor_list if "stereo" in i]

    # read atlas from file, generate cell real time and pseudotime df.
    cell_time_atlas_pd = pd.read_csv(cell_time_atlas_file, sep="\t", index_col=0)
    donor_atlas_df = pd.DataFrame(columns=["pseudotime", "time"])
    for _donor in donor_atlas_list:
        _time = cell_time_atlas_pd[cell_time_atlas_pd['donor'] == _donor]['time'][0]
        _temp_df = pd.DataFrame(dict(pseudotime=data_dic[_donor + "_pseudotime"], time=_time))
        donor_atlas_df = pd.concat([donor_atlas_df, _temp_df], axis=0)

    # read stereo from file, generate cell real time and pseudotime df.
    cell_time_stereo_pd = pd.read_csv(cell_time_stereo_file, sep="\t", index_col=0)
    donor_stereo_df = pd.DataFrame(columns=["pseudotime", "time"])
    for _donor in donor_stereo_list:
        _time = cell_time_stereo_pd[cell_time_stereo_pd['donor'] == _donor]['time'][0]
        _temp_df = pd.DataFrame(dict(pseudotime=data_dic[_donor + "_pseudotime"], time=_time))
        donor_stereo_df = pd.concat([donor_stereo_df, _temp_df], axis=0)
    # calculate correlation scores
    if len(donor_atlas_df) > 0:
        atlas_stats = calculate_real_predict_corrlation_score(donor_atlas_df["pseudotime"], donor_atlas_df["time"])
        print(f"=== atlas data correlation: \n{atlas_stats}")
    else:
        print("No atlas data.")
    if len(donor_stereo_df) > 0:
        stereo_stats = calculate_real_predict_corrlation_score(donor_stereo_df["pseudotime"], donor_stereo_df["time"])
        print(f"=== stereo data correlation: \n{stereo_stats}")
    else:
        print("No stereo data.")
    return





if __name__ == '__main__':
    main()
