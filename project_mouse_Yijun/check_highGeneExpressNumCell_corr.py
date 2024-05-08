# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：check_highGeneExpressNumCell_corr.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/10/6 17:06

check correlation between time and pseudotime of cells with more than 100 gene express

"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def main(organ):
    # organ = "Liver"
    kFold_result_ofminGeneNum50 = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/" \
                                  "231004_trainOn_mouse_embryo_stereo_organs_kFold_minGene50_75_100/mouse_embryo_stereo/" \
                                  f"preprocess_Mouse_embryo_all_stage_minGene50_of{organ}/" \
                                  "supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch100_minGeneNum50_mouseEmbryonicDevelopment_embryoneg5to5/" \
                                  "SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial_plot_on_all_test_donor_timeembryoneg5to5_celltype_update_testCLF.json"
    stereo_cell_df = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/results/" \
                     "231004_trainOn_mouse_embryo_stereo_organs_kFold_minGene50_75_100/mouse_embryo_stereo/" \
                     f"preprocess_Mouse_embryo_all_stage_minGene50_of{organ}/" \
                     "supervise_vae_regressionclfdecoder_mouse_stereo_dim50_timeembryoneg5to5_epoch100_minGeneNum100_mouseEmbryonicDevelopment_embryoneg5to5/" \
                     "preprocessed_cell_info.csv"
    with open(kFold_result_ofminGeneNum50, "r") as json_file:
        kFold_result_ofminGeneNum50_list = []
        for line in json_file:
            json_obj = json.loads(line)
            kFold_result_ofminGeneNum50_list.append(json_obj)
    kFold_result_ofminGeneNum50_dic = kFold_result_ofminGeneNum50_list[0]

    embryo_list = list(kFold_result_ofminGeneNum50_dic.keys())
    embryo_list = list(set([i.replace("_pseudotime", "").replace("_cellid", "") for i in embryo_list]))
    cell_time_stereo_file = "/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/GPLVM_dandan/data/" \
                            "mouse_embryo_stereo/preprocess_Mouse_embryo_all_stage_minGene50_ofBrain/cell_with_time.csv"
    # read stereo from file, generate cell real time and pseudotime df.
    cell_time_stereo_pd = pd.read_csv(cell_time_stereo_file, sep="\t", index_col=0)
    kFold_result_ofminGeneNum50_df = pd.DataFrame(columns=["pseudotime", "time"])
    for _donor in embryo_list:
        _time = cell_time_stereo_pd[cell_time_stereo_pd['donor'] == _donor]['time'][0]
        _temp_df = pd.DataFrame(dict(pseudotime=kFold_result_ofminGeneNum50_dic[_donor + "_pseudotime"], time=_time),
                                index=kFold_result_ofminGeneNum50_dic[_donor + "_cellid"])

        kFold_result_ofminGeneNum50_df = pd.concat([kFold_result_ofminGeneNum50_df, _temp_df], axis=0)

    stereo_cellId = list(pd.read_csv(stereo_cell_df, index_col=0).index)

    min50_100_df = kFold_result_ofminGeneNum50_df.drop(stereo_cellId).copy()
    min100_df = kFold_result_ofminGeneNum50_df.loc[stereo_cellId].copy()

    plot_violin(min100_df,special_str=organ+' ')
    plot_violin(min50_100_df,special_str=organ+' ')
    # plot_violin(kFold_result_ofminGeneNum50_df,special_str=organ+' ')
    return


def plot_violin(boxPlot_df, special_str=""):
    from utils.utils_Dandan_plot import colors_tuple
    print(f"total {len(boxPlot_df)} cells.")
    from scipy.stats import spearmanr, kendalltau, pearsonr
    sp_correlation, sp_p_value = spearmanr(list(boxPlot_df["pseudotime"]), list(boxPlot_df["time"]))
    pear_correlation, pear_p_value = pearsonr(list(boxPlot_df["pseudotime"]), list(boxPlot_df["time"]))

    ke_correlation, ke_p_value = kendalltau(list(boxPlot_df["pseudotime"]), list(boxPlot_df["time"]))
    print(f"Final {special_str}: {len(boxPlot_df)} cells.")
    print(f"{special_str}spearman correlation score: {np.round(sp_correlation, 5)}, p-value: {sp_p_value}.")
    print(f"{special_str}pearson correlation score: {np.round(pear_correlation, 5)}, p-value: {pear_p_value}.")
    print(f"{special_str}kendalltau correlation score: {np.round(ke_correlation, 5)},p-value: {ke_p_value}.")

    # 计算每个 "time" 值对应的 "pseudotime" 计数
    time_counts = boxPlot_df.groupby("time")["pseudotime"].count().reset_index()
    # 获取所有唯一的 "time" 值
    unique_times = list(time_counts["time"])
    # 使用颜色映射创建一个颜色字典
    colors = colors_tuple()
    cmap = colors[:len(unique_times)]
    # cmap = plt.get_cmap('RdYlBu')(np.linspace(0, 1, len(unique_times)))
    color_dict = dict(zip(unique_times, cmap))
    # 根据 "time" 值从颜色字典中获取颜色
    boxPlot_df["color"] = boxPlot_df["time"].map(color_dict)
    # 设置 Seaborn 主题和风格
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 7))  # 设置宽度为10，高度为6

    # 创建小提琴图，并指定 x 和 y 轴，设置颜色
    sns.violinplot(x="time", y="pseudotime", data=boxPlot_df, palette=color_dict, bw=0.2, scale='width')

    # 添加散点图，设置颜色
    # sns.stripplot(x="time", y="pseudotime", data=boxPlot_df, jitter=True, palette=palette, size=1, alpha=0.7)

    # 添加标题和标签
    plt.title(f"{special_str}Violin Plot of k-fold test.")
    plt.xlabel("Time")
    plt.ylabel("Pseudotime")
    # 添加颜色的图例
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  label=f"{time_counts[time_counts['time'] == time]['pseudotime'].values[0]}",
                                  markerfacecolor=color_dict[time], markersize=10) for time in unique_times]
    plt.legend(handles=legend_elements, title="Cell Num", loc='best')

    # 自定义 x 轴标签的显示顺序
    plt.xticks()

    # 显示图形

    # save_file_name = "{}{}/{}_plot_on_all_test_donor_time{}{}_violin".format(
    #     _logger.root.handlers[0].baseFilename.replace(".log", ""),
    #     special_path_str,
    #     model_name, time_standard_type, special_str)
    # plt.savefig(save_file_name + ".png", format='png')
    # plt.savefig(save_file_name + ".pdf", format='pdf')
    # _logger.info("Finish save images at: {}".format(save_file_name + ".png"))
    plt.show()
    plt.close()


if __name__ == '__main__':
    main("Brain")
    main("Heart")
    main("Liver")
