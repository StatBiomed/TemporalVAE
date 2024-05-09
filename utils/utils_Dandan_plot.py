# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：utils_Dandan_plot.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/7/27 16:37 
"""
import sys

sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/CNNC-master/utils/")
sys.path.append("/mnt/yijun/nfs_share/awa_project/pairsRegulatePrediction/model_master/")

import logging

_logger = logging.getLogger(__name__)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re


def LHdonor_resort_key(item):
    match = re.match(r'LH(\d+)_([0-9PGT]+)', item)
    if match:
        num1 = int(match.group(1))
        num2 = match.group(2)
        if num2 == 'PGT':
            num2 = 'ZZZ'  # 将'LH7_PGT'替换为'LH7_ZZZ'，确保'LH7_PGT'在'LH7'后面
        return num1, num2
    return item


def RIFdonor_resort_key(item):
    match = re.match(r'RIF_(\d+)', item)
    if match:
        num1 = int(match.group(1))
        return num1
    return item


def Embryodonor_resort_key(item):
    match = re.match(r'embryo(_stereo)?_(\d+)', item)
    if match:
        num1 = int(match.group(2))
        has_external = 1 if match.group(1) else 0
        return (has_external, num1)
    return (0, 0)


def plot_training_loss(latent_dim, losses, golbal_path, file_path, args=None):
    """For GPLVM model, plot training loss"""
    plt.plot(losses)
    plt.savefig("{}/{}/trainingLoss{}_time{}.png".format(golbal_path, file_path, latent_dim, args.time_standard_type),
                format='png')
    plt.show()
    plt.close()


def plot_latent_dim_image(latent_dim, labels, X, sc_expression_df_copy, golbal_path, file_path, attr,
                          reorder_labels=True, args=None, special_str=""):
    """For GPLVM model, plot latent dim X by color the attr"""
    colors = colors_tuple()

    # Define the number of plots you want
    num_plots = len(labels) + 1
    if reorder_labels:
        # reorder the labels
        def sort_key(item):
            lh_number = int(item.split("LH")[1].split("_")[0])

            if "_" in item:
                suffix = item.split("_")[1]
                return lh_number, suffix
            return lh_number, ""

        labels = sorted(labels, key=sort_key)
        # labels_temp = [int(i.split("_")[0].replace("LH", "")) for i in labels]
        # idx = np.argsort(labels_temp)
        # labels = labels[idx]
    _logger.info("Label list: {}".format(labels))

    if latent_dim > 2:
        elevs = [30, 0, 0, 90]  # 仰角
        azims = [60, 0, 90, 0]  # 方位角
        fig = plt.figure(figsize=(20, 40))
        # Iterate over the number of plots and create subplots
        for row_index in range(num_plots):
            for _subindex in range(4):
                ax = fig.add_subplot(num_plots, 4, _subindex + 1 + row_index * 4, projection='3d')
                # Your existing code for plotting the 3D graph
                if row_index == 0:
                    for i, label in enumerate(labels):
                        _logger.info("i:{}, label {}".format(i, label))
                        X_i = X[sc_expression_df_copy.index == label]
                        ax.scatter(X_i[:, 0], X_i[:, 1], X_i[:, 2], c=[colors[i + 1]], label=label, s=5)
                else:
                    i = row_index - 1
                    label = labels[i]
                    _logger.info("{}-th image for cells label {} ".format(i, label))
                    X_i = X[sc_expression_df_copy.index == label]
                    X_noti = X[sc_expression_df_copy.index != label]
                    ax.scatter(X_i[:, 0], X_i[:, 1], X_i[:, 2], c=[colors[i + 1]], label=label, s=5)
                    ax.scatter(X_noti[:, 0], X_noti[:, 1], X_noti[:, 2], c=[colors[0]], label="Not-" + label, s=5,
                               alpha=0.1)

                ax.set_xlabel('pseudotime', fontsize=14)
                ax.set_ylabel('Y Label', fontsize=14)
                ax.set_zlabel('Z Label', fontsize=14)

                ax.view_init(elev=elevs[_subindex], azim=azims[_subindex])

                plt.legend()
                if row_index == 0:
                    plt.title("GPLVM on " + file_path.split("_")[-1] + " cell data", fontsize=16)
                else:
                    plt.title("GPLVM on " + file_path.split("_")[-1] + " cell data with time " + labels[i],
                              fontsize=16)
    elif latent_dim == 2:
        import math
        num_cols = math.ceil(num_plots / 2)
        fig = plt.figure(figsize=(30, 20))

        # Iterate over the number of plots and create subplots
        for plot_index in range(1, num_plots + 1):
            ax = fig.add_subplot(2, num_cols, plot_index)
            # Your existing code for plotting the 3D graph
            if plot_index == 1:
                for i, label in enumerate(labels):
                    _logger.info("i:{}, label {}".format(i, label))
                    X_i = X[sc_expression_df_copy.index == label]
                    ax.scatter(X_i[:, 0], X_i[:, 1], c=[colors[i + 1]], label=label, s=5)
            else:
                i = plot_index - 2
                label = labels[i]
                _logger.info("{}-th image for cells label {} ".format(i, label))
                X_i = X[sc_expression_df_copy.index == label]
                X_noti = X[sc_expression_df_copy.index != label]

                ax.scatter(X_i[:, 0], X_i[:, 1], c=[colors[i + 1]], label=label, s=5)
                ax.scatter(X_noti[:, 0], X_noti[:, 1], c=[colors[0]], label="Not-" + label, s=5, alpha=0.1)

                ax.set_xlabel('pseudotime', fontsize=14)
                ax.set_ylabel('Y Label', fontsize=14)

            plt.legend()
            if plot_index == 1:
                plt.title("GPLVM on " + file_path.split("/")[0].split("_")[-1] + " cell data", fontsize=16)
            else:
                plt.title("GPLVM on " + file_path.split("/")[0].split("_")[-1] + " cell data with time " + labels[i],
                          fontsize=16)
    # Adjust the spacing between subplots
    fig.tight_layout()

    # Save the figure
    plt.savefig("{}/{}/latent_dim{}_{}_time{}{}.png".format(golbal_path, file_path, latent_dim, attr.replace("_", ""),
                                                            args.time_standard_type, special_str),
                format='png')
    _logger.info("Images saved at: {}".format(golbal_path + file_path))
    plt.show()
    plt.close()


def plot_on_each_test_donor_violin(predict_donors_dic, cell_time, special_path_str,
                                   model_name, time_standard_type,
                                   donor_str="donor", time_str="time",
                                   special_str="",
                                   save_file_bool=True):
    boxPlot_df = pd.DataFrame(columns=["pseudotime", time_str])
    for donor, donor_result_df in predict_donors_dic.items():
        donor_result_df = donor_result_df.copy()
        donor_result_df[time_str] = cell_time[cell_time[donor_str] == donor][time_str][0]
        boxPlot_df = pd.concat([boxPlot_df, donor_result_df], axis=0)
    # 计算每个 "time" 值对应的 "pseudotime" 计数
    time_counts = boxPlot_df.groupby(time_str)["pseudotime"].count().reset_index()
    # 获取所有唯一的 "time" 值
    unique_times = list(time_counts[time_str])
    # 使用颜色映射创建一个颜色字典
    colors = colors_tuple()
    cmap = colors[:len(unique_times)]
    # cmap = plt.get_cmap('RdYlBu')(np.linspace(0, 1, len(unique_times)))
    color_dict = dict(zip(unique_times, cmap))
    # 根据 "time" 值从颜色字典中获取颜色
    boxPlot_df["color"] = boxPlot_df[time_str].map(color_dict)
    # 设置 Seaborn 主题和风格
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 7))  # 设置宽度为10，高度为6

    # 创建小提琴图，并指定 x 和 y 轴，设置颜色
    sns.violinplot(x=time_str, y="pseudotime", data=boxPlot_df, palette=color_dict, bw=0.2, scale='width')

    # 添加散点图，设置颜色
    # sns.stripplot(x="time", y="pseudotime", data=boxPlot_df, jitter=True, palette=palette, size=1, alpha=0.7)

    # 添加标题和标签
    corr_str = calculate_real_predict_corrlation_score(list(boxPlot_df["pseudotime"]), list(boxPlot_df["time"]), only_str=True)
    plt.title(f"Violin Plot of k-fold test:{corr_str}.")
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
    if save_file_bool:
        save_file_name = "{}{}/{}_plot_on_all_test_donor_time{}{}_violin".format(
            _logger.root.handlers[0].baseFilename.replace(".log", ""),
            special_path_str,
            model_name, time_standard_type, special_str)
        plt.savefig(save_file_name + ".png", format='png')
        plt.savefig(save_file_name + ".pdf", format='pdf')
        _logger.info("Finish save images at: {}".format(save_file_name + ".png"))
    plt.show()
    plt.close()


def plot_on_each_test_donor_violin_fromDF(cell_time_df, save_path, physical_str, x_str="time", special_file_name_str="", cmap_color="viridis"):
    # from utils_Dandan_plot import calculate_real_predict_corrlation_score
    import matplotlib.pyplot as plt
    import seaborn as sns

    corr_stats = calculate_real_predict_corrlation_score(cell_time_df[physical_str], cell_time_df[x_str])
    print(f"=== data correlation: \n{corr_stats}")
    boxPlot_df = cell_time_df.copy()
    time_counts = boxPlot_df.groupby(x_str)[physical_str].count().reset_index()

    # plot violin
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))
    cmap = plt.get_cmap(cmap_color)(np.linspace(0, 1, len(time_counts[x_str])))
    color_dict = dict(zip(time_counts[x_str], cmap))
    sns.violinplot(x=x_str, y=physical_str, data=boxPlot_df, palette=cmap_color, bw=0.2, scale='width')

    corr_str = calculate_real_predict_corrlation_score(list(boxPlot_df[physical_str]), list(boxPlot_df[x_str]), only_str=True)
    plt.title(f"Violin Plot of k-fold test:{corr_str}.")
    plt.xlabel("Time")
    plt.ylabel("Pseudotime")
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  label=f"{time_counts[time_counts['time'] == time][physical_str].values[0]}",
                                  markerfacecolor=color_dict[time], markersize=10) for time in time_counts[x_str]]
    plt.legend(handles=legend_elements, title="Cell Num", loc='best')

    plt.xticks()

    plt.savefig(f"{save_path}/{special_file_name_str}violine.png", dpi=200)

    plt.show()
    plt.close()
    print(f"Save at {save_path}/{special_file_name_str}violine.png")
    return color_dict


def plot_on_each_test_donor_confusionMatrix_Joy2ClassMesen(predict_donors_dic, cell_time, special_path_str,
                                                           model_name, time_standard_type,
                                                           donor_str="donor", time_str="time",
                                                           special_str="",
                                                           save_file_bool=True):
    boxPlot_df = pd.DataFrame(columns=["pseudotime", time_str])
    for donor, donor_result_df in predict_donors_dic.items():
        donor_result_df = donor_result_df.copy()
        donor_result_df[time_str] = cell_time[cell_time[donor_str] == donor][time_str][0]
        boxPlot_df = pd.concat([boxPlot_df, donor_result_df], axis=0)
    from sklearn.metrics import confusion_matrix
    # 计算混淆矩阵
    confusion = confusion_matrix(boxPlot_df['mesen'].astype(int), boxPlot_df['pseudotime'].astype(int))
    # 设置 Seaborn 主题和风格
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 7))  # 设置宽度为10，高度为6

    sns.set(font_scale=1.2)  # 设置字体大小

    # 绘制混淆矩阵图
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=[0, 1],
                yticklabels=[0, 1])

    # 设置图形标题和坐标轴标签
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # 显示图形
    if save_file_bool:
        save_file_name = "{}{}/{}_plot_on_all_test_donor_time{}{}_confusionMatrix".format(
            _logger.root.handlers[0].baseFilename.replace(".log", ""),
            special_path_str,
            model_name, time_standard_type, special_str)
        plt.savefig(save_file_name + ".png", format='png')
        plt.savefig(save_file_name + ".pdf", format='pdf')
        _logger.info("Finish save images at: {}".format(save_file_name + ".png"))
    plt.show()
    plt.close()


def plot_on_each_test_donor_continueTime(donor_list, original_predict_donors_dic, latent_dim,
                                         time_standard_type, special_str="", model_name="",
                                         label_orginalAsKey_transAsValue_dic="", cell_time=None, special_path_str="",
                                         plot_subtype_str="type", plot_on_each_cell_type=True):
    """
    2023-06-30 15:42:07
    For GPLVM and VAE model, test on one donor and train on other donors,
    plot all test results in one image,
    time is label-type (continue)
    :param golbal_path:
    :param file_path:
    :param donor_list:
    :param original_predict_donors_dic:
    :param latent_dim:
    :param time_standard_type:
    :param special_str:
    :param model_name:
    :param label_orginalAsKey_transAsValue_dic:
    :param cell_time:
    :param special_path_str:
    :return:
    """
    _logger.info("plot sub image by {}".format(plot_subtype_str))
    donors_pseudotime_dic, subtype_dic, plot_subtype_str = change_cellId_to_cellType_for_dfIndex(original_predict_donors_dic,
                                                                                                 cell_time,
                                                                                                 str=plot_subtype_str)
    save_dic = dict()
    num_plots = len(subtype_dic) + 1 if plot_on_each_cell_type else 1

    colors = colors_tuple()

    test_on_LH_bool = True if "LH" in donor_list[0] else False
    if test_on_LH_bool:
        donor_list = sorted(donor_list, key=LHdonor_resort_key)
    elif "embryo" in donor_list[0]:
        donor_list = sorted(donor_list, key=Embryodonor_resort_key)
    else:
        donor_list = sorted(donor_list, key=RIFdonor_resort_key)
    donor_color_dic = dict()
    for _i in range(len(donor_list)):
        donor_color_dic[donor_list[_i]] = colors[_i]
    fig = plt.figure(figsize=(20, max(6 * num_plots, 20)))
    if 6 * num_plots > 500:  # 2023-08-06 15:25:36 avoid ValueError: Image size of pixels is too large. It must be less than 2^16 in each direction.
        fig = plt.figure(figsize=(20, 500))

    # -------------------------- set the save path and file name ----------------------------------
    # Save the figure
    if model_name == "GPLVM":
        save_file_name = "{}/GPLVM_plot_on_all_test_donor_time{}{}".format(_logger.root.handlers[0].baseFilename.replace(".log", ""),
                                                                           time_standard_type,
                                                                           special_str)
    else:
        save_file_name = "{}{}/{}_plot_on_all_test_donor_time{}{}".format(
            _logger.root.handlers[0].baseFilename.replace(".log", ""),
            special_path_str,
            model_name, time_standard_type, special_str)

    # Iterate over the number of plots and create subplots
    # ------------------------- first plot total cells-------------------------
    sub_real_time = []
    sub_pesudo_time = []
    ax1 = fig.add_subplot(num_plots, 1, 1)
    # 绘制散点图
    for _donor in donor_list:
        values = donors_pseudotime_dic[_donor]
        if isinstance(values, pd.DataFrame):
            pesudo_time = values["pseudotime"]
        # elif isinstance(values, np.ndarray):
        #     pesudo_time = values
        else:
            exit(1)
        save_dic[_donor + "_pseudotime"] = list(pesudo_time.astype(float))
        save_dic[_donor + "_cellid"] = list(values.index)
        real_time = [_donor] * len(values)
        if test_on_LH_bool:
            sub_real_time += [label_orginalAsKey_transAsValue_dic[int(_donor.split("_")[0].replace("LH", ""))]] * len(
                values)
        elif "time" in cell_time.columns:
            sub_real_time += list(cell_time.loc[values.index]["time"])
        else:
            _logger.info("no sub real time labels")

        sub_pesudo_time += list(pesudo_time.values)
        plt.scatter(pesudo_time, real_time, c=[donor_color_dic[_donor]], s=40, label=_donor, alpha=0.6,
                    edgecolors='white')
        # color_index += 1
    # plt.yticks(range(len(donor_list)), donor_list)
    if (test_on_LH_bool) or ("time" in cell_time.columns):
        corr_score = calculate_real_predict_corrlation_score(sub_real_time, sub_pesudo_time)
        plt.title('Test on each donor (time trans: {})-- {} cells\nCor_score: {}'.format(time_standard_type,
                                                                                         len(sub_pesudo_time),
                                                                                         corr_score))
    else:
        plt.title('Test on each donor (time trans: {})--{} cells'.format(time_standard_type, len(sub_pesudo_time)))

    # plt.xlabel('x')
    # plt.ylabel('y')
    # 获取当前图例的句柄和标签
    handles, labels = ax1.get_legend_handles_labels()

    # 反转图例项的顺序
    handles = handles[::-1]
    labels = labels[::-1]
    ax1.legend(handles, labels, bbox_to_anchor=(1.01, 0), loc="lower left", borderaxespad=0)
    plt.grid(True)
    # save result dic json
    import json
    with open(save_file_name + "_testCLF.json", 'w') as f:
        json.dump(save_dic, f)  # 2023-07-03 22:31:50
    _logger.info("save clf result at: {}".format(save_file_name + "_testCLF.json"))
    # ------------------------- second plot each sub cells type -------------------------
    for plot_index in range(2, num_plots + 1):
        subtype_list = sorted(list(subtype_dic))
        sub_real_time = []
        sub_pesudo_time = []

        # 绘制散点图
        ax = fig.add_subplot(num_plots, 1, plot_index, sharex=ax1)
        subtype = subtype_list[plot_index - 2]
        for _donor in donor_list:
            values = donors_pseudotime_dic[_donor]
            values_sub = values[values[plot_subtype_str] == subtype]
            # values_sub = values[values.index == subtype]
            if isinstance(values, pd.DataFrame):
                pesudo_time = values_sub["pseudotime"]
            elif isinstance(values_sub, np.ndarray):
                pesudo_time = values_sub
            real_time = [_donor] * len(values_sub)
            if test_on_LH_bool:
                sub_real_time += [label_orginalAsKey_transAsValue_dic[int(_donor.split("_")[0].replace("LH", ""))]] * len(
                    values_sub)
            elif "time" in cell_time.columns:
                sub_real_time += list(cell_time.loc[values_sub.index]["time"])
            else:
                _logger.info("no sub real time labels")
            sub_pesudo_time += list(pesudo_time.values)
            plt.scatter(pesudo_time, real_time, c=[donor_color_dic[_donor]], s=40, label=_donor, alpha=0.6,
                        edgecolors='white')
            # color_index += 1
        # plt.yticks(range(len(donor_list)), donor_list)
        if (test_on_LH_bool) or ("time" in cell_time.columns):
            corr_score = calculate_real_predict_corrlation_score(sub_real_time, sub_pesudo_time)
            plt.title('Test on each donor (time trans: {}) -- {}--{} cells\nCor_score: {}'.format(time_standard_type, subtype,
                                                                                                  len(sub_pesudo_time),
                                                                                                  corr_score))
        else:
            plt.title('Test on each donor (time trans: {}) -- {}--{} cells'.format(time_standard_type, subtype,
                                                                                   len(sub_pesudo_time)))

        # plt.xlabel('x')
        # plt.ylabel('y')
        # 获取当前图例的句柄和标签
        handles, labels = ax.get_legend_handles_labels()

        # 反转图例项的顺序
        handles = handles[::-1]
        labels = labels[::-1]
        ax.legend(handles, labels, bbox_to_anchor=(1.01, 0), loc="lower left", borderaxespad=0)
        plt.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.45)

    # Save the figure
    plt.savefig(save_file_name + ".png", format='png')
    plt.savefig(save_file_name + ".pdf", format='pdf')
    _logger.info("Finish save images at: {}".format(save_file_name + ".png"))
    plt.show()
    plt.close()


def colors_tuple():
    """
    color style is RGB
    :return: tuple ( (0.6, 0.6, 0.6), (0.6509803921568628, 0.33725490196078434, 0.1568627450980392)...)
    """
    # 定义要使用的颜色映射名称列表
    cmap_names = ["Set1", "tab10", "Dark2", "Set2", "Set3", "tab20", "Pastel1", "tab20c", "Pastel2", "tab20b", "Accent"]
    # 使用列表推导式生成颜色列表
    colors = [color for cmap_name in cmap_names for color in plt.get_cmap(cmap_name).colors[::-1]]
    return tuple(colors)


def changePostion(data, i, j):
    data = list(data)
    data[i], data[j] = data[j], data[i]
    return tuple(data)


def colors_tuple_hexadecimalColorCode():
    """
    color style is RGB
    :return: tuple ( '#FF0000', '#00FF00', '#0000FF'...)
    """
    # 使用列表推导式生成颜色列表
    colors = ["#00cdcd", "#FF4F00", "#FFB800", "#00D2FF", "#7F00FF",
              "#008c8c", "#c4b4f3", "#ff00FF", "#d280ff", "#80edff",
              "#80ff92", "#edff80"]
    more = colors_tuple()
    more = list("#{:02X}{:02X}{:02X}".format(int(R * 255), int(G * 255), int(B * 255)) for (R, G, B) in more)
    colors = colors + more
    colors = changePostion(colors, 0, 15)
    colors = changePostion(colors, 8, 143)
    colors = changePostion(colors, 9, 12)
    return tuple(colors)


def plot_on_each_test_donor_continueTime_windowTime(donor_list, original_predict_donors_dic, latent_dim,
                                                    time_standard_type, special_str="", model_name="",
                                                    label_orginalAsKey_transAsValue_dic="", cell_time=None, special_path_str="",
                                                    plot_subtype_str="type", plot_on_each_cell_type=True,
                                                    donor_real_time_dic=None):
    """
    plot window time
    For GPLVM and VAE model, test on one donor and train on other donors,
    plot all test results in one image,
    time is label-type (continue)
    :param golbal_path:
    :param file_path:
    :param donor_list:
    :param original_predict_donors_dic:
    :param latent_dim:
    :param time_standard_type:
    :param special_str:
    :param model_name:
    :param label_orginalAsKey_transAsValue_dic:
    :param cell_time:
    :param special_path_str:
    :return:
    """
    _logger.info("plot sub image by {}".format(plot_subtype_str))
    donors_pseudotime_dic, subtype_dic, plot_subtype_str = change_cellId_to_cellType_for_dfIndex(original_predict_donors_dic,
                                                                                                 cell_time,
                                                                                                 str=plot_subtype_str)

    colors = colors_tuple()
    # 创建横轴标签的字典映射, 使用字典推导式将键和值互换
    test_on_LH_bool = True if "LH" in donor_list[0] else False
    if test_on_LH_bool:
        donor_list = sorted(donor_list, key=LHdonor_resort_key)
    elif "embryo" in donor_list[0]:
        donor_list = sorted(donor_list, key=Embryodonor_resort_key)
        if "time" in cell_time.columns:
            donor_real_time_dic = {_d: int(cell_time[cell_time["donor"] == _d]["time"][0] * 100) for _d in donor_list}
    else:
        donor_list = sorted(donor_list, key=RIFdonor_resort_key)
    donor_list.reverse()
    donor_color_dic = dict()
    for _i in range(len(donor_list)):
        donor_color_dic[donor_list[_i]] = colors[_i]

    threshold_list = sorted([val for _, val in label_orginalAsKey_transAsValue_dic.items()])

    # ---------- split save image for subplot.---------
    _logger.info("plot time window for total cells.")
    donors_pseudotime_df = pd.DataFrame(columns=["pseudotime", "donor"])
    for _donor in donor_list:
        _df = pd.DataFrame(columns=["pseudotime", "donor"])
        _df["pseudotime"] = donors_pseudotime_dic[_donor]["pseudotime"]
        _df["donor"] = _donor
        donors_pseudotime_df = pd.concat([donors_pseudotime_df, _df], ignore_index=True)
    save_file_name = "{}{}/{}_plot_on_all_test_donor_time{}{}_timeWindow_allCell.png".format(
        _logger.root.handlers[0].baseFilename.replace(".log", ""), special_path_str, model_name, time_standard_type, special_str)
    categorical_kde_plot(donors_pseudotime_df, threshold_list=threshold_list, variable="pseudotime", category="donor",
                         category_order=donor_list, horizontal=True, save_png_name=save_file_name,
                         label_trans_dic=label_orginalAsKey_transAsValue_dic, donor_color_dic=donor_color_dic,
                         donor_real_time_dic=donor_real_time_dic)
    # ---------- split save image for each cell type.---------
    if plot_on_each_cell_type is True:
        subtype_list = sorted(list(subtype_dic))
        _logger.info("start plot image for each cell type.")
        for subtype in subtype_list:
            _logger.info("plot time window for total cells.")
            donors_pseudotime_df = pd.DataFrame(columns=["pseudotime", "donor"])
            for _donor in donor_list:
                _df = pd.DataFrame(columns=["pseudotime", "donor"])
                values = donors_pseudotime_dic[_donor]
                values_sub = values[values[plot_subtype_str] == subtype]
                _df["pseudotime"] = values_sub["pseudotime"]
                _df["donor"] = _donor
                donors_pseudotime_df = pd.concat([donors_pseudotime_df, _df], ignore_index=True)
            save_file_name = "{}{}/{}_plot_on_all_test_donor_time{}{}_timeWindow_subtype_{}.png".format(
                _logger.root.handlers[0].baseFilename.replace(".log", ""), special_path_str, model_name, time_standard_type,
                special_str,
                subtype.replace("/", "").replace(" ", "_"))
            categorical_kde_plot(donors_pseudotime_df, threshold_list=threshold_list, variable="pseudotime", category="donor",
                                 category_order=donor_list, horizontal=True, save_png_name=save_file_name, subtype=subtype,
                                 label_trans_dic=label_orginalAsKey_transAsValue_dic)


def plot_on_each_test_donor_discreteTime(golbal_path, file_path, donor_list, original_predict_donors_dic, latent_dim,
                                         time_standard_type, special_str="", model_name="",
                                         label_orginalAsKey_transAsValue_dic="", cell_time=None, special_path_str="",
                                         plot_subtype_str="type"):
    """
    2023-06-30 15:42:07
    For VAE model, test on one donor and train on other donors,
    plot all test results in one image,
    time is label-type(discrete)

    :param golbal_path:
    :param file_path:
    :param donor_list:
    :param predict_donors_dic:
    :param latent_dim:
    :param time_standard_type:
    :param special_str:
    :param model_name:
    :param label_orginalAsKey_transAsValue_dic:
    :param cell_time:
    :param special_path_str:
    :return:
    """

    _logger.info("plot sub image by {}".format(plot_subtype_str))
    donors_pseudotime_dic, subtype_dic, plot_subtype_str = change_cellId_to_cellType_for_dfIndex(original_predict_donors_dic,
                                                                                                 cell_time,
                                                                                                 str=plot_subtype_str)
    subtype_list = sorted(list(subtype_dic))
    # get true time and pesudo-time
    donors_list = list(donors_pseudotime_dic.keys())
    test_on_LH_bool = True if "LH" in donors_list[0] else False
    if test_on_LH_bool:
        _logger.info("Test on normal donors.")
        donors_list = sorted(donors_list, key=LHdonor_resort_key)
    elif "embryo" in donors_list[0]:
        _logger.info("Test on embryo donors.")
        donors_list = sorted(donors_list, key=Embryodonor_resort_key)
    else:
        _logger.info("Test on RIF donors")
        donors_list = sorted(donors_list, key=RIFdonor_resort_key)
    donors_list.reverse()
    # 创建横轴标签的字典映射, 使用字典推导式将键和值互换
    label_transAsKey_orginalAsValue_dic = {value: str(key) + "-LH" for key, value in
                                           label_orginalAsKey_transAsValue_dic.items()}
    predicted_labels_list = sorted(list(
        label_transAsKey_orginalAsValue_dic.keys()))  # predicted_labels_list = sorted(list(set([val[0] for sublist in donors_pseudotime_dic.values() for val in sublist.values])))
    _logger.info("trans label dic as: {}".format(label_transAsKey_orginalAsValue_dic))
    num_plots = len(subtype_list) + 1
    _logger.info("There are {} subplot in the final plot.".format(num_plots))
    fig = plt.figure(figsize=(7, 6 * num_plots))
    # Iterate over the number of plots and create subplots
    for plot_index in range(1, num_plots + 1):
        sub_real_time_list = []
        sub_pesudo_time_list = []
        if plot_index == 1:  # plot the total graph
            ax = fig.add_subplot(num_plots, 1, plot_index)
            # 创建热力图数据
            heatmap_data = np.zeros((len(donors_list), len(predicted_labels_list)))
            for _index, _donor_name in enumerate(donors_list):
                for _pseudotime in donors_pseudotime_dic[_donor_name]["pseudotime"]:
                    j = predicted_labels_list.index(_pseudotime)
                    heatmap_data[_index, j] += 1
                if test_on_LH_bool:
                    sub_real_time_list += [int(_donor_name.split("_")[0].replace("LH", ""))] * len(
                        donors_pseudotime_dic[_donor_name]["pseudotime"].values)
                else:
                    _logger.info("no sub real time labels")
                sub_pesudo_time_list += list(donors_pseudotime_dic[_donor_name]["pseudotime"].values)
            # 计算每个类别的预测数量所占比例
            percentage_data = heatmap_data / np.sum(heatmap_data, axis=1)[:, None]

            # 设置热力图的参数
            # fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(percentage_data, annot=heatmap_data, cmap='Blues', fmt='.0f', cbar=True,
                        cbar_kws={'label': 'Percentage of Total'},
                        xticklabels=[label_transAsKey_orginalAsValue_dic[label] for label in predicted_labels_list],
                        yticklabels=donors_list,
                        ax=ax)

            # 设置图形属性
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('Donor Id')
            if test_on_LH_bool:
                corr_score = calculate_real_predict_corrlation_score(sub_real_time_list, sub_pesudo_time_list)
                ax.set_title('Confusion Matrix about test on all donor\nCor_score: {}'.format(corr_score))
            else:
                ax.set_title('Confusion Matrix about test on all donor')

        else:  # plot on subtype
            ax = fig.add_subplot(num_plots, 1, plot_index)
            subtype = subtype_list[plot_index - 2]

            # 创建热力图数据
            heatmap_data = np.zeros((len(donors_list), len(predicted_labels_list)))
            for _index, _donor_name in enumerate(donors_list):
                values = donors_pseudotime_dic[_donor_name]
                values_sub = values[values[plot_subtype_str] == subtype]
                # values_sub = values[values.index == subtype]
                for _pseudotimes in values_sub["pseudotime"]:
                    j = predicted_labels_list.index(_pseudotimes)
                    heatmap_data[_index, j] += 1
                if test_on_LH_bool:
                    sub_real_time_list += [int(_donor_name.split("_")[0].replace("LH", ""))] * len(
                        list(values_sub["pseudotime"].values))
                else:
                    _logger.info("no sub real time labels")
                sub_pesudo_time_list += list(values_sub["pseudotime"].values)
            # 计算每个类别的预测数量所占比例
            percentage_data = heatmap_data / np.sum(heatmap_data, axis=1)[:, None]

            # 设置热力图的参数
            # fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(percentage_data, annot=heatmap_data, cmap='Blues', fmt='.0f', cbar=True,
                        cbar_kws={'label': 'Percentage of Total'},
                        xticklabels=[label_transAsKey_orginalAsValue_dic[label] for label in predicted_labels_list],
                        yticklabels=donors_list,
                        ax=ax)

            # 设置图形属性
            ax.set_xlabel('Predicted Labels')
            ax.set_ylabel('Donor Id')
            if test_on_LH_bool:
                corr_score = calculate_real_predict_corrlation_score(sub_real_time_list, sub_pesudo_time_list)
                ax.set_title(
                    'Confusion Matrix about test on all donor -- {}\nCorr_score: {}'.format(subtype, corr_score))
            else:
                ax.set_title('Confusion Matrix about test on all donor -- {}'.format(subtype))

        # 调整子图布局，将图像整体向右移动
        plt.subplots_adjust(left=0.2, right=0.9)
    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.45)
    # 展示热力图
    # Save the figure
    if model_name == "GPLVM":
        save_file_name = "{}/{}/GPLVM_plot_on_all_test_donor_latent_dim{}_time{}{}.png".format(golbal_path, file_path,
                                                                                               latent_dim,
                                                                                               time_standard_type,
                                                                                               special_str)
    else:
        save_file_name = "{}{}/{}_plot_on_all_test_donor_time{}{}.png".format(
            _logger.root.handlers[0].baseFilename.replace(".log", ""),
            special_path_str,
            model_name, time_standard_type, special_str)
    plt.savefig(save_file_name, format='png')
    import json
    save_dic = dict()
    for _donor, val in original_predict_donors_dic.items():
        save_dic[_donor + "_pseudotime"] = list(val.astype(float))
        save_dic[_donor + "_cellid"] = list(val.index)
    with open(save_file_name.replace(".png", "_testCLF.json"), 'w') as f:
        json.dump(save_dic, f)  # 2023-07-03 22:31:50
    _logger.info("Finish save images and clf result at: {} and {}".format(save_file_name, save_file_name.replace(".png",
                                                                                                                 "_testCLF.json")))
    plt.show()
    plt.close()


def plot_training_loss_for_tags(tb_logger, plot_tag_list, special_str="", title="for check", interval=5):
    """
    for VAE model, plot the training loss change, for check
    :param tb_logger:
    :param plot_tag_list:
    :param special_str:
    :return:
    """
    _logger.info("plot training process.")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_acc = EventAccumulator(tb_logger.log_dir).Reload()
    # Retrieve and print the metric results
    plt.figure(figsize=(15, 7))
    max_epoch = 0  # To keep track of the maximum epoch encountered

    for _tag in plot_tag_list:
        tag_values = event_acc.Scalars(_tag)
        try:
            tag_values_array = np.array([[_event.step, _event.predicted_df] for _event in tag_values])
        except:
            tag_values_array = np.array([[_event.step, _event.value] for _event in tag_values])
        # 绘制折线图
        plt.plot(tag_values_array[:, 0], tag_values_array[:, 1], marker='.', label=_tag)
        max_epoch = max(max_epoch, max(tag_values_array[:, 0]))
    # Add vertical lines every 'interval' epochs
    for epoch in range(0, int(max_epoch) + 1, interval):
        plt.axvline(x=epoch, color='r', linestyle='--', linewidth=1, label=f'Epoch {epoch}' if epoch == 0 else "")

    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.title(title)
    save_file_name = "{}/trainingLoss_{}_{}.png".format(tb_logger.log_dir, '_'.join(plot_tag_list), special_str)
    plt.savefig(save_file_name, format='png')
    _logger.info("Finish save images at: {}".format(save_file_name))
    plt.show()
    plt.close()


def umap_vae_latent_space(data, data_label, label_dic, special_path_str, config, special_str="", drop_batch_dim=0):
    """
    for VAE model latent space Visualization, use 2D-UMAP

    :param data:
    :param data_label:
    :param label_dic:
    :param tb_logger:
    :param config:
    :param special_str:
    :param drop_batch_dim:
    :return:
    """
    colors = colors_tuple_hexadecimalColorCode()
    # colors = colors_tuple()
    plt.figure(figsize=(12, 7))
    if drop_batch_dim > 0:
        _logger.info("{} dims use to drop batch effect".format(drop_batch_dim))
        data = data[:, drop_batch_dim:]
        special_str = special_str + "_dropBatchEffect" + str(drop_batch_dim)
    import umap.umap_ as umap
    if config['model_params']['name'] == 'SuperviseVanillaVAE_regressionClfDecoder_of_subLatentSpace':
        data = data[:, :10]
    if "time" in special_str.lower():
        label_mapping = {str(float(value)): str(value) + ": " + str(key) + "-th day" for key, value in
                         label_dic.items()}
    if "donor" in special_str.lower():
        label_mapping = {str(float(value)): str(key) for key, value in label_dic.items()}
    reducer = umap.UMAP(random_state=42)
    try:
        embedding = reducer.fit_transform(data.cpu().numpy())
    except:
        try:
            embedding = reducer.fit_transform(np.asanyarray(data.cpu().numpy()))
        except:
            try:
                embedding = reducer.fit_transform(np.asanyarray(data))
            except:
                print("Can't generate umap for latent space.")
                return
    plt.gca().set_aspect('equal', 'datalim')

    i = 0
    for label in np.unique(data_label):
        # print(label)
        indices = np.where(data_label == label)
        try:
            plt.scatter(embedding[indices, 0], embedding[indices, 1],
                        label=label_mapping[str(float(label))] + f"; {str(len(indices[0]))} cells", s=2,
                        alpha=0.7, c=colors[i])
        except:
            plt.scatter(embedding[indices, 0], embedding[indices, 1], label=label_mapping[str(label)], s=2, alpha=0.7,
                        c=colors[i])
        i += 1

    plt.gca().set_aspect('equal', 'datalim')

    plt.legend(bbox_to_anchor=(1.01, 0), loc="lower left", borderaxespad=0, scatterpoints=1, markerscale=3)
    # 添加图例并设置样式

    plt.subplots_adjust(left=0.1, right=0.75)
    plt.title('UMAP: ' + special_str)
    save_file_name = "{}{}/latentSpace_{}".format(_logger.root.handlers[0].baseFilename.replace(".log", ""),
                                                  special_path_str,
                                                  special_str)
    plt.savefig(save_file_name + ".png")
    plt.savefig(save_file_name + ".pdf")
    _logger.info("Finish plot latent space umap, save images at: {}".format(save_file_name))

    plt.show()
    plt.close()
    # save umap embedding
    temp_dic = {int(value): str(key) for key, value in label_dic.items()}
    temp_label = [temp_dic[int(i)] for i in data_label]
    _df = pd.DataFrame(data=embedding, index=temp_label, columns=["umap1", "umap2"])
    _df.to_csv(f"{save_file_name}_umapEmbedding.csv", sep="\t")
    # save pca embedding
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    X_standardized = StandardScaler().fit_transform(data.cpu().numpy())
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_standardized)
    _df = pd.DataFrame(data=X_pca, index=temp_label, columns=[f"pc1_{pca.explained_variance_ratio_[0]}", f"pc2_{pca.explained_variance_ratio_[1]}"])
    _df.to_csv(f"{save_file_name}_pcaEmbedding.csv", sep="\t")


def calculate_averages(input_list):
    output_list = []
    for i in range(len(input_list) - 1):
        avg = (input_list[i] + input_list[i + 1]) / 2
        output_list.append(avg)
    return output_list


def categorical_kde_plot(df, variable, category, threshold_list, save_png_name, category_order=None, horizontal=False, rug=True,
                         figsize=None, subtype="whole cell", label_trans_dic=None, donor_color_dic=None,
                         donor_real_time_dic=None):
    """Draw a categorical KDE plot

    Parameters
    ----------
    df: pd.DataFrame
        The data to plot
    variable: str
        The column in the `df` to plot (continuous variable)
    category: str
        The column in the `df` to use for grouping (categorical variable)
    horizontal: bool
        If True, draw density plots horizontally. Otherwise, draw them
        vertically.
    rug: bool
        If True, add also a sns.rugplot.
    figsize: tuple or None
        If None, use default figsize of (7, 1*len(categories))
        If tuple, use that figsize. Given to plt.subplots as an argument.
    """
    colors = colors_tuple()
    if category_order is None:
        categories = list(df[category].unique())
    else:
        categories = category_order[:]

    figsize = (10, 3 * len(categories))  # 2023-08-07 13:17:53 length * height
    # figsize = (9, 1.2 * len(categories))  # length * height

    fig, axes = plt.subplots(
        nrows=len(categories) if horizontal else 1,
        ncols=1 if horizontal else len(categories),
        figsize=figsize[::-1] if not horizontal else figsize,
        sharex=horizontal,
        sharey=not horizontal,
    )
    from collections.abc import Iterable
    if not isinstance(axes, Iterable):
        axes = [axes]

    sub_pesudo_time = list(df["pseudotime"])

    try:
        sub_real_time = list(df["donor"])
        sub_real_time = [label_trans_dic[int(_.split("_")[0].replace("LH", ""))] for _ in sub_real_time]

        corr_score = calculate_real_predict_corrlation_score(sub_real_time, sub_pesudo_time)
        fig.suptitle('Test on each donor -- {}--{} cells\nCor_score: {}'.format(subtype, len(sub_pesudo_time), corr_score))
    except:
        # plt.title('Test on each donor -- {}--{} cells'.format(subtype, len(sub_pesudo_time)))
        fig.suptitle('Test on each donor -- {}--{} cells'.format(subtype, len(sub_pesudo_time)), y=0.99)

    # threshold_list=[-1.0, -0.778, -0.556, -0.333, -0.111, 0.111, 0.333, 0.556, 0.778, 1.0]
    # Define the bins based on label_dic
    threshold_list = calculate_averages(threshold_list)
    # threshold_list = [-0.75, -0.25, 0.25, 0.75]
    bins = [-np.inf] + threshold_list + [np.inf]
    # pd.cut([val for _,val in label_trans_dic.items()],bins)

    df['set'] = pd.cut(df[variable], bins=bins)
    set_list = sorted(np.unique(df["set"]))

    if label_trans_dic is not None:
        temp_label = pd.DataFrame(data=[[_k, _v] for _k, _v in label_trans_dic.items()], columns=["real", "transed"])
        temp_label.index = pd.cut(temp_label["transed"], bins=bins)
        set_list_time_dic = {set_list[_i]: temp_label.loc[set_list[_i]]["real"] for _i in range(len(set_list))}
    set_list_color_dic = {set_list[_i]: colors[_i] for _i in range(len(set_list))}
    for i, (cat, ax) in enumerate(zip(categories, axes)):
        _data = df[df[category] == cat]

        sns.kdeplot(data=_data,
                    x=variable if horizontal else None,
                    y=None if horizontal else variable,
                    # kde kwargs
                    bw_adjust=0.5,
                    clip_on=False,
                    fill=True,
                    alpha=0.2,
                    linewidth=1.5,
                    ax=ax,
                    color=colors[-1],
                    )

        keep_variable_axis = (i == len(fig.axes) - 1) if horizontal else (i == 0)

        if rug:
            _data_copy = _data.copy(deep=True)
            # _data_copy['set'] = pd.cut(_data[variable], bins=bins)
            _set_list = np.unique(_data_copy["set"])
            count_str = ""
            for _set_index in range(len(_set_list)):
                _data_subThre = _data_copy[_data_copy["set"] == _set_list[_set_index]]
                if len(_data_subThre) == 0:
                    continue
                sns.rugplot(data=_data_subThre,
                            x=variable if horizontal else None,
                            y=None if horizontal else variable,
                            ax=ax,
                            color=set_list_color_dic[_set_list[_set_index]],
                            height=0.1 if keep_variable_axis else 0.1,
                            )
                # Calculate the count of data points in the current category
                count_str += "{}-{}: {}/{} cell, {}%\n".format(
                    _set_list[_set_index],
                    set_list_time_dic[
                        _set_list[_set_index]] if label_trans_dic is not None else "",
                    len(_data_subThre), len(_data),
                    round(len(_data_subThre) / len(_data), 3) * 100)

            # Add the count as text at the top of the rug plot
            # x_pos = ax.get_xlim()[1] if horizontal else (ax.get_xlim()[1] + ax.get_xlim()[0]) / 2
            # y_pos = ax.get_ylim()[1] if not horizontal else (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2
            x_pos = ax.get_xlim()[1] if horizontal else (ax.get_xlim()[1] + ax.get_xlim()[0]) / 2
            y_pos = ax.get_ylim()[1] if not horizontal else (ax.get_ylim()[1] + ax.get_ylim()[0]) / 2
            # print(x_pos)
            # print(y_pos)
            ax.text(x_pos,
                    y_pos,
                    f"Window count: {count_str}",
                    # f"Count: x_pos{x_pos},x_pos1{ax.get_xlim()[1]},x_pos0{ax.get_xlim()[0]}",
                    # f"Count: x_pos{x_pos},y_pos{y_pos}",
                    ha='right' if horizontal else 'center',
                    va='top' if not horizontal else 'center',
                    fontsize=5,
                    )
            # 在图的右侧（或其他位置）添加文本
            # ax.annotate(
            #     "hello!!",
            #     xy=(x_pos+0.5,y_pos),  # 文本的坐标，可以根据需要进行调整
            #     xycoords='axes fraction',  # 使用坐标系的分数表示
            #     fontsize=12,
            #     ha='left',  # 水平对齐方式
            # )
        _format_axis(
            ax,
            cat + ":{}".format(donor_real_time_dic[cat] / 100 if donor_real_time_dic is not None else ""),
            horizontal,
            keep_variable_axis=keep_variable_axis,
        )
    plt.tight_layout()

    plt.savefig(save_png_name, format='png')
    plt.savefig(save_png_name.replace(".png", ".pdf"), format='pdf')
    plt.show()
    plt.close()
    _logger.info("Window time image save at {}".format(save_png_name))


def _format_axis(ax, category, horizontal=False, keep_variable_axis=True):
    # Remove the axis lines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if horizontal:
        ax.set_ylabel(None)
        lim = ax.get_ylim()
        ax.set_yticks([(lim[0] + lim[1]) / 2])
        ax.set_yticklabels([category])
        if not keep_variable_axis:
            ax.get_xaxis().set_visible(False)
            ax.spines["bottom"].set_visible(False)
    else:
        ax.set_xlabel(None)
        lim = ax.get_xlim()
        ax.set_xticks([(lim[0] + lim[1]) / 2])
        ax.set_xticklabels([category])
        if not keep_variable_axis:
            ax.get_yaxis().set_visible(False)
            ax.spines["left"].set_visible(False)


def get_items_with_subtype(lst, str=""):
    return [item for item in lst if str in item]


def change_cellId_to_cellType_for_dfIndex(input_test_time_pesudo, cell_time, str):
    str = get_items_with_subtype(cell_time, str=str)[0]
    result_test_time_pesudo = dict()
    cell_id_to_sample_rds = dict(zip(cell_time.index, cell_time[str]))

    for donor, val_df in input_test_time_pesudo.items():
        # print(donor)
        val_df_copy = val_df.copy(deep=True)
        val_df_copy[str] = val_df_copy.index.map(cell_id_to_sample_rds)
        # for _cell_name in val_df_copy.index.values:
        #     val_df_copy.rename(index={_cell_name: cell_time.loc[_cell_name][column_name]}, inplace=True)
        result_test_time_pesudo[donor] = val_df_copy
    subtype_list = []
    for _, val_df in result_test_time_pesudo.items():
        subtype_list = subtype_list + list(val_df[str])
    subtype_dic = set(subtype_list)
    return result_test_time_pesudo, subtype_dic, str


def calculate_real_predict_corrlation_score(real_time, pseudo_time, only_str=True):
    """Input is dic """
    real_time = list(real_time)
    pseudo_time = list(pseudo_time)
    from scipy import stats
    from sklearn.metrics import r2_score
    stats_result = dict()
    try:
        stats_result["spearman"] = stats.spearmanr(real_time, pseudo_time)
        stats_result["pearson"] = stats.pearsonr(real_time, pseudo_time)
        stats_result["kendalltau"] = stats.kendalltau(real_time, pseudo_time)
        stats_result["r2"] = r2_score(real_time, pseudo_time)
    except:
        _logger.info("Error in calculate corr.")
        return "cannot calculate due to leak of cell"
    # 格式化输出字符串
    output_string = "Spearman correlation={:.5f}, p-value={:.5f}; \nPearson correlation={:.5f}, p-value={:.5f}; \nKendall correlation={:.5f}, p-value={:.5f}; \nR-squared={:.5f}."

    # 将科学计数法转换为小数形式的字符串
    # pvalue_decimal = "{:.4f}".format(stats_result['pearson']['pvalue'])

    # 根据字典变量进行格式化
    output = output_string.format(stats_result['spearman'].statistic,
                                  stats_result['spearman'].pvalue,
                                  stats_result['pearson'].statistic,
                                  # float(pvalue_decimal),
                                  stats_result['pearson'].pvalue,
                                  stats_result['kendalltau'].statistic,
                                  stats_result['kendalltau'].pvalue,
                                  stats_result['r2'])
    if only_str:
        return output
    else:
        return output, stats_result


def plot_boxPlot_nonExpGene_percentage_whilePreprocess(adata, cell_time, donor_attr, special_path_str, test_donor,
                                                       special_file_str="",
                                                       save_images=True):
    # output the distribution of the percentage of non-zero count HVGs in each cell for each sample
    _logger.info("output the distribution of the percentage of non-zero count HVGs in each cell for each sample")
    temp_cell_time = cell_time.loc[adata.obs_names]

    temp_cell_time_donor_list = list(set(temp_cell_time[donor_attr]))
    temp_cell_time_donor_list = sorted(temp_cell_time_donor_list)
    boxPlot_data_list = []

    for i in range(len(temp_cell_time_donor_list)):
        _cell = list(
            set(temp_cell_time.loc[temp_cell_time[donor_attr] == temp_cell_time_donor_list[i]].index) & set(adata.obs.index))
        _temp = adata[_cell].copy()

        _dataframe = pd.DataFrame(data=_temp.X, columns=_temp.var.index, index=_temp.obs.index)
        boxPlot_data_list.append(_dataframe)

    # 计算每个变量每个细胞的非零基因百分比
    non_zero_percentages_list = []
    for _a in boxPlot_data_list:
        non_zero_percentages_list.append(_a.apply(lambda row: (row != 0).sum() / len(row) * 100, axis=1))

    # 创建一个 DataFrame，方便绘图
    data = pd.DataFrame(non_zero_percentages_list).T
    data.columns = temp_cell_time_donor_list

    # 设置样式
    sns.set(style="whitegrid")

    # 绘制箱线图和数据点
    plt.figure(figsize=(2 * len(temp_cell_time_donor_list), 6))
    ax = sns.boxplot(data=data, width=0.5)

    sns.stripplot(data=data, jitter=True, color=".3", size=2, ax=ax, alpha=0.55)
    plt.title(f'{special_file_str + ": "}Non-Zero Gene Percentage by Donor')
    plt.ylabel('Non-Zero Gene Percentage of cells')
    plt.xlabel('Donor ID')

    if save_images:
        _save_path = "{}{}/".format(_logger.root.handlers[0].baseFilename.replace(".log", ""), special_path_str)
        _save_file = "{}/distribution_nonZeroCountGenes_of_TrainDonors_Test_{}_{}.png".format(_save_path, test_donor,
                                                                                              special_file_str)
        import os
        if not os.path.exists(_save_path):
            os.makedirs(_save_path)

        plt.savefig(_save_file, format='png')
        _logger.info(
            "the distribution of the percentage of non-zero count HVGs in each cell for each sample image save at {}".format(
                _save_file))
    plt.show()
    plt.close()


def plot_boxPlot_total_count_per_cell_whilePreprocess(adata, cell_time, donor_attr, special_path_str, test_donor,
                                                      special_file_str="",
                                                      save_images=True):
    # output the distribution of the percentage of non-zero count HVGs in each cell for each sample
    _logger.info("output the total counts per cell for each donor.")
    temp_cell_time = cell_time.loc[adata.obs_names]
    temp_cell_time_donor_list = list(set(temp_cell_time[donor_attr]))
    temp_cell_time_donor_list = sorted(temp_cell_time_donor_list)
    boxPlot_data_list = []

    for i in range(len(temp_cell_time_donor_list)):
        _cell = list(
            set(temp_cell_time.loc[temp_cell_time[donor_attr] == temp_cell_time_donor_list[i]].index) & set(adata.obs.index))
        _temp = adata[_cell].copy()

        _dataframe = pd.DataFrame(data=_temp.X, columns=_temp.var.index, index=_temp.obs.index)
        boxPlot_data_list.append(_dataframe)

    # the total counts per cell
    total_count_per_cell = []
    for _a in boxPlot_data_list:
        total_count_per_cell.append(_a.apply(lambda row: row.sum(), axis=1))

    # 创建一个 DataFrame，方便绘图
    data = pd.DataFrame(total_count_per_cell).T
    data.columns = temp_cell_time_donor_list

    # 设置样式
    sns.set(style="whitegrid")

    # 绘制箱线图和数据点
    plt.figure(figsize=(2 * len(temp_cell_time_donor_list), 6))
    ax = sns.boxplot(data=data, width=0.5)

    sns.stripplot(data=data, jitter=True, color=".3", size=2, ax=ax, alpha=0.55)
    plt.title(f'{special_file_str + ": "}total counts per cell by Donor')
    plt.ylabel('the total counts per cell')
    plt.xlabel('Donor ID')

    if save_images:
        _save_path = "{}{}/".format(_logger.root.handlers[0].baseFilename.replace(".log", ""), special_path_str)
        _save_file = "{}/totalCountsOfPerCell_of_TrainDonors_Test_{}_{}.png".format(_save_path, test_donor,
                                                                                    special_file_str)
        import os
        if not os.path.exists(_save_path):
            os.makedirs(_save_path)

        plt.savefig(_save_file, format='png')
        _logger.info("the total counts per cell for each donor image save at {}".format(_save_file))
    plt.show()
    plt.close()


def plot_time_change_gene_geneZero(gene, result_df, x_str, y_str, label_str, save_images=True, special_path_str="", title_str=""):
    plt.figure(figsize=(14, 24))
    plt.subplot(2, 1, 1)  # 2行1列，选择第1个子图

    sns.set(style="whitegrid")
    custom_palette = colors_tuple_hexadecimalColorCode()
    sns.scatterplot(data=result_df, x=x_str, y=y_str, hue=label_str, s=4,
                    palette=list(custom_palette))
    plt.plot(result_df[x_str], result_df[x_str], color='black', linestyle='--')
    # 添加标签和标题
    plt.xlabel("predicted time", fontsize=10)
    plt.ylabel(f"predicted time after knockout", fontsize=10)
    plt.title(f"{gene}: {title_str}", fontsize=13)

    # 显示图例
    plt.legend(title="real time", title_fontsize=10, bbox_to_anchor=(1.01, 0), loc="lower left", borderaxespad=0,
               prop={'size': 8})

    plt.subplot(2, 1, 2)  # 2行1列，选择第1个子图
    sns.set(style="whitegrid")
    sns.scatterplot(data=result_df, x=x_str, y="expression", hue=label_str, s=4,
                    palette=list(custom_palette))
    # plt.plot(result_df[x_str], result_df[x_str], color='red', linestyle='--')
    # 添加标签和标题
    plt.xlabel("predicted time", fontsize=10)
    plt.ylabel(f"gene expression", fontsize=10)
    plt.title(f"{gene}: expression value", fontsize=13)

    # 显示图例
    plt.legend(title="real time", title_fontsize=10, bbox_to_anchor=(1.01, 0), loc="lower left", borderaxespad=0,
               prop={'size': 8})
    if save_images:
        _save_path = f"{_logger.root.handlers[0].baseFilename.replace('.log', '')}{special_path_str}/predictedTimeChange_withPerturb/"
        import os
        if not os.path.exists(_save_path):
            os.makedirs(_save_path)

        _save_file = f"{_save_path}/{gene}_timeChanges.png"

        plt.savefig(_save_file)
        _logger.info("Time changes under perturb image save at {}".format(_save_file))
    # 显示散点图
    plt.show()
    plt.close()


def plot_detTandExp(gene_result_pd, special_path_str):
    # import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 10))
    # 使用 matplotlib 画散点图
    plt.scatter(gene_result_pd['total_raw_count'], gene_result_pd['mean'], s=15)
    # 为每个点添加标签
    for i, _g in enumerate(gene_result_pd['gene_short_name']):
        plt.annotate(_g, (gene_result_pd.iloc[i]['total_raw_count'], gene_result_pd.iloc[i]['mean']), fontsize=8)
    # 添加标签和标题
    plt.xlabel('Total raw expression of genes.')
    plt.ylabel('△t: pseudo-time after perturb and without perturb.')
    plt.title('Scatter Plot of each gene.')
    # 添加水平线 y=0
    plt.axhline(y=0, color='red', linestyle='--', label='y=0')

    # 设置 x 轴范围
    plt.xlim(gene_result_pd['total_raw_count'].min() - 10, gene_result_pd['total_raw_count'].max() + 100)
    # 显示图形
    save_path = f"{_logger.root.handlers[0].baseFilename.replace('.log', '')}{special_path_str}/"
    save_file = f"{save_path}/gene_detTandExp.png"
    plt.savefig(save_file)
    _logger.info("Time changes under perturb image save at {}".format(save_file))

    plt.show()
    plt.close()


def plot_detTandT_line(top_gene_list, plot_pd, special_path_str, gene_dic):
    # import matplotlib.pyplot as plt
    # import seaborn as sns
    # plt.figure(figsize=(50, 20))
    top_gene_list = [gene_dic[i] for i in top_gene_list]
    sns.set(style="white", color_codes=True)
    colors = colors_tuple_hexadecimalColorCode()[:len(top_gene_list)]

    # 创建散点图
    # plt.scatter(temp["trained_time"],temp["perturb_time"] ,s=2,c=colors[i],alpha=0.4)
    sns.lmplot(x="trained_time", y="det_time", data=plot_pd, hue="gene", palette=colors,
               order=2, ci=None, scatter_kws={"s": 5, "alpha": 0.7}, height=10, aspect=1.5)  # lmplot默认order参数是一阶的
    plt.axhline(y=0, color='red', linestyle='--', label='y=0')
    # 添加标题和标签
    plt.title(f'top {len(top_gene_list)} abs(△t) of genes')
    plt.xlabel('Pseudo-time predicted.')
    plt.ylabel('△t: pseudo-time after perturb and without perturb.')
    # 显示散点图
    save_path = f"{_logger.root.handlers[0].baseFilename.replace('.log', '')}{special_path_str}/"
    save_file = f"{save_path}/gene_detTandT_Line.png"
    plt.savefig(save_file)
    _logger.info("Time changes under perturb image save at {}".format(save_file))
    plt.show()
    plt.close()


def umap_vae_latent_space_adata_version(adata, label_str, save_path=None):
    """
    :param adata:
    :return:
    """
    data = adata.X
    data_label = adata.obs[label_str].values
    colors = colors_tuple()
    plt.figure(figsize=(12, 7))

    import umap.umap_ as umap

    reducer = umap.UMAP(random_state=42)
    try:
        embedding = reducer.fit_transform(data)
    except:
        print("Can't generate umap for latent space.")
        return
    plt.gca().set_aspect('equal', 'datalim')

    i = 0
    for label in np.unique(data_label):
        indices = np.where(data_label == label)
        plt.scatter(embedding[indices, 0], embedding[indices, 1], label=label, s=2, alpha=0.7,
                    c=colors[i])
        i += 1

    plt.gca().set_aspect('equal', 'datalim')

    plt.legend(bbox_to_anchor=(1.01, 0), loc="lower left", borderaxespad=0)
    # 添加图例并设置样式

    plt.subplots_adjust(left=0.1, right=0.75)
    plt.title('UMAP: ')
    if save_path is not None:
        plt.savefig(f"{save_path}/latentSpace_umap_{str}.png", dpi=200)
    plt.show()
    plt.close()
    return embedding


def draw_venn(lists_dict):
    import matplotlib.pyplot as plt
    from matplotlib_venn import venn2, venn3
    if len(lists_dict) > 3:
        raise ValueError("维恩图最多支持三个集合")
    elif len(lists_dict) < 2:
        raise ValueError("需要至少两个列表来绘制维恩图")

    sets = [set(lst) for lst in lists_dict.values()]
    labels = list(lists_dict.keys())

    if len(sets) == 3:
        venn3(sets, labels)
    elif len(sets) == 2:
        venn2(sets, labels)

    plt.show()
    plt.close()


def plt_umap_byScanpy(adata, attr_list, save_path, mode=None, special_file_name_str="",
                      figure_size=(15, 6), show_in_row=True, color_map=None, palette_dic=None, n_neighbors=10, n_pcs=40):
    import scanpy as sc
    # attr = attr + list({"physical_time", "physical_pseudotime_by_preTrained_mouseAtlas_model", "physical_pseudotime_by_finetune_model"} & set(adata.obs.columns))
    # attr = ["day", "physical_pseudotime_by_preTrained_mouseAtlas_model", "physical_pseudotime_by_finetune_model", "cell_type", "s_or_mrna", "sample"]
    # attr.sort()
    if mode is None:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
    elif mode == "write":  # this mode only for mouse atlas data
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)
        adata.write_h5ad(f"{save_path}/{special_file_name_str}top10CellType_subcell_latentmu.h5ad")
    elif mode == "read":  # this mode only for mouse atlas data
        adata = sc.read_h5ad(f"{save_path}/{special_file_name_str}top10CellType_subcell_latentmu.h5ad")
    sc.settings.figdir = save_path
    sc.tl.umap(adata, min_dist=0.75)  # 2024-01-19 11:33:26 add min_dist=0.75
    # sc.tl.umap(adata, min_dist=0.75)  # 2024-01-19 11:33:26 add min_dist=0.75
    # sc.tl.umap(adata, min_dist=0.75)  # 2024-01-19 11:33:26 add min_dist=0.75
    # sc.tl.umap(adata)
    if show_in_row:
        with plt.rc_context({'figure.figsize': figure_size}):
            sc.pl.umap(adata, color=attr_list, show=False, legend_fontsize=5.5, s=20, color_map=color_map)
            # sc.pl.umap(adata, color=attr, show=True, legend_fontsize=5.5, s=20, save=f"{special_file_name_str}latentSpace_umap_byScanpy.png")
    else:  # show all sub figures in one column
        fig, axs = plt.subplots(len(attr_list), 1, figsize=(figure_size[0], figure_size[1] * len(attr_list)))
        for ax, attr in zip(axs, attr_list):
            if (palette_dic is not None) and (attr in palette_dic.keys()):
                sc.pl.umap(adata, color=attr, ax=ax, show=False, legend_fontsize=7.5, s=20, palette=palette_dic[attr])
            else:
                sc.pl.umap(adata, color=attr, ax=ax, show=False, legend_fontsize=7.5, s=20, color_map=color_map)
        plt.tight_layout()  # 调整布局
    adata.write_h5ad(f"{save_path}/{special_file_name_str}latent_mu.h5ad")

    plt.savefig(f"{save_path}/{special_file_name_str}latentSpace_umap_byScanpy.png", dpi=300, )
    plt.show()
    plt.close()
    print(f"figure save as {save_path}/{special_file_name_str}latentSpace_umap_byScanpy.png")
    return adata


def plot_psupertime_density(test_results, save_path, label_key="time", psupertime_key="psupertime", method="a"):
    # always recalculate psupertime
    # psupertime_key = "psupertime"
    import matplotlib.pyplot as plt
    import seaborn as sns
    obs_copy = test_results

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111)

    palette = 'RdBu'
    col_vals = sns.color_palette(palette, len(obs_copy[label_key].unique()))
    sns.kdeplot(data=obs_copy, x=psupertime_key, fill=label_key, hue=label_key, alpha=0.5,
                palette=col_vals, legend=True, ax=ax)
    ax.set_xlabel("Psupertime")
    ax.set_ylabel("Density")
    sns.despine()

    plt.title(f'{method}: {calculate_real_predict_corrlation_score(obs_copy[label_key], obs_copy[psupertime_key])}', fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{method}_labelsOverPsupertime.png")
    plt.show()
    plt.close()
    print(f"figure save at {save_path}/{method}_labelsOverPsupertime.png")


def plot_violin_240223(cell_info_df, save_path, real_attr="time", pseudo_attr="physical_pseudotime_by_preTrained_mouseAtlas_model", special_file_name="", color_map="viridis"):
    from utils.utils_Dandan_plot import calculate_real_predict_corrlation_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    corr_stats = calculate_real_predict_corrlation_score(cell_info_df[pseudo_attr], cell_info_df[real_attr])
    print(f"=== data correlation: \n{corr_stats}")
    boxPlot_df = cell_info_df.copy()
    time_counts = boxPlot_df.groupby(real_attr)[pseudo_attr].count().reset_index()

    # plot violin
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))
    cmap = plt.get_cmap(color_map)(np.linspace(0, 1, len(time_counts[real_attr])))
    color_dict = dict(zip(time_counts[real_attr], cmap))
    sns.violinplot(x=real_attr, y=pseudo_attr, data=boxPlot_df, palette=color_map, bw=0.2, scale='width')

    corr_str = calculate_real_predict_corrlation_score(list(boxPlot_df[pseudo_attr]), list(boxPlot_df[real_attr]), only_str=True)
    plt.title(f"{special_file_name + ': '}Violin Plot of k-fold test:{corr_str}.")
    plt.xlabel("Time")
    plt.ylabel("Pseudotime")
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  label=f"{time_counts[time_counts[real_attr] == time][pseudo_attr].values[0]}",
                                  markerfacecolor=color_dict[time], markersize=10) for time in time_counts[real_attr]]
    plt.legend(handles=legend_elements, title="Cell Num", loc='best')

    plt.xticks()
    plt.tight_layout()
    plt.savefig(f"{save_path}/{special_file_name + '_'}violine.png", dpi=200)

    plt.show()
    plt.close()
    return color_dict


def plot_umap_240223(mu_predict_by_pretrained_model, cell_time_stereo, color_dic=None, save_path="", attr_str="time", color_map=None):
    import anndata as ad
    import scanpy as sc
    sc.settings.figdir = save_path

    adata = ad.AnnData(mu_predict_by_pretrained_model, obs=cell_time_stereo)

    # adata.obs["batch"]=-1 # for figure1
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    # adata_stereo.write_h5ad(f"{save_path}/latent_mu.h5ad")
    sc.tl.umap(adata)
    # if color_dic is not None:
    #     color_dic = {f"E{str(key)}": value for key, value in color_dic.items()}
    with plt.rc_context({'figure.figsize': (5, 4)}):
        try:
            sc.pl.umap(adata, color=attr_str, palette=color_dic, show=False, legend_fontsize=5.5, s=10, legend_loc='right margin', color_map=color_map)
        except:
            sc.pl.umap(adata, color=attr_str, show=False, legend_fontsize=5.5, s=10, legend_loc='right margin')
    plt.savefig(f"{save_path}/latentSpace_umap_byScanpy_{attr_str}.png", dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"save at {save_path}")


def plot_boxplot_from_dic(data, legend_loc="lower right"):
    # ----------------------------------------------------------------------------------------------------------
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.DataFrame(data)

    # 设置绘图风格
    sns.set(style="whitegrid")

    # 创建条形图
    plt.figure(figsize=(8, 6))
    barplot = sns.barplot(x='Method', y='Value', hue='Correlation Type', data=df, palette=["#B0E0E6", "#D8BFD8"])

    # 添加标题和坐标轴标签
    plt.title('Method Performance: Spearman vs Pearson Correlation', fontsize=16)
    plt.ylabel('Correlation Value', fontsize=14)
    plt.xlabel('Method', fontsize=14)

    # 调整图例
    plt.legend(title='Correlation Type', title_fontsize='13', fontsize='12', loc=legend_loc)

    # 在每个条形上显示数值
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.3f'),
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='center',
                         xytext=(0, 10),
                         textcoords='offset points')

    # 设置Y轴范围以清晰展示负相关性，添加Y=0参考线强调正负相关性
    plt.ylim(0, 1)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')

    # 美化图表
    sns.despine(offset=10, trim=True)  # 减少边框
    plt.tight_layout()  # 自动调整子图参数,使之填充整个图像区域

    # 显示图表
    plt.show()
    plt.close()
    return
