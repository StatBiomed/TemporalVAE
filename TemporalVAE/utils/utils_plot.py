# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：utils_plot.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2023/7/27 16:37 
"""

import logging

_logger = logging.getLogger(__name__)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import gc


def plot_data_quality(adata):
    import scanpy as sc
    print("Original data quality check by plot images.")
    sc.settings.set_figure_params(dpi=200, facecolor="white")
    sc.pl.highest_expr_genes(adata, n_top=20)
    try:
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt", "ribo", "hb"], inplace=True, log1p=True)
    except:
        try:
            sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, log1p=True)
        except:
            pass
    try:
        sc.pl.violin(adata, ["n_genes_by_counts", "total_counts", "pct_counts_mt"], jitter=0.4, multi_panel=True, )
    except:
        pass
    try:
        sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts", color="pct_counts_mt")
    except:
        pass
    try:
        sc.pl.scatter(adata, x="total_counts", y="pct_counts_mt")
    except:
        pass
    try:
        sc.pl.scatter(adata, x="total_counts", y="n_genes_by_counts")
    except:
        pass


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


def plot_on_each_test_donor_violin_fromDF(cell_time_df, save_path, y_attr, x_attr="time", special_file_name_str="", cmap_color="viridis"):
    # from utils_Dandan_plot import calculate_real_predict_corrlation_score
    import matplotlib.pyplot as plt
    import seaborn as sns

    corr_stats = calculate_real_predict_corrlation_score(cell_time_df[y_attr], cell_time_df[x_attr])
    print(f"=== data correlation: \n{corr_stats}")
    boxPlot_df = cell_time_df.copy()
    time_counts = boxPlot_df.groupby(x_attr)[y_attr].count().reset_index()

    # plot violin
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))
    cmap = plt.get_cmap(cmap_color)(np.linspace(0, 1, len(time_counts[x_attr])))
    color_dict = dict(zip(time_counts[x_attr], cmap))
    sns.violinplot(x=x_attr, y=y_attr, data=boxPlot_df, palette=cmap_color, bw=0.2, scale='width')

    corr_str = calculate_real_predict_corrlation_score(list(boxPlot_df[y_attr]), list(boxPlot_df[x_attr]), only_str=True)
    plt.title(f"Violin Plot of k-fold test:{corr_str}.")
    plt.xlabel("Time")
    plt.ylabel("Pseudotime")
    # legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
    #                               label=f"{time_counts[time_counts['time'] == time][physical_str].values[0]}",
    #                               markerfacecolor=color_dict[time], markersize=10) for time in time_counts[x_str]]
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  label=f"{time_counts[time_counts['time'] == time][y_attr].values[0]}",
                                  markerfacecolor=color_dict[time], markersize=10) for time in time_counts[x_attr]]
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


def plot_detTandExp(gene_result_pd,
                    special_path_str="",
                    stage_str="allStage",
                    x_str="total_raw_count", x_legend_str='Total raw expression of genes.',
                    y_str="mean", y_legend_str='Mean △t: pseudo-time after perturb and without perturb.',
                    special_filename_str="detT",
                    plt_yOrxZero="y",
                    save_path=None, scatter_strategy="linear",
                    min_font_size=10, max_font_size=15, fontsize_threshold=10.2, ):
    from sklearn.neighbors import NearestNeighbors
    from adjustText import adjust_text
    Q1_x = gene_result_pd[x_str].quantile(0.25)
    Q3_x = gene_result_pd[x_str].quantile(0.75)
    Q1_y = gene_result_pd[y_str].quantile(0.25)
    Q3_y = gene_result_pd[y_str].quantile(0.75)
    # 创建颜色标签
    colors = []
    for index, row in gene_result_pd.iterrows():
        if Q1_x < row[x_str] < Q3_x or row[y_str] < Q3_y:
            colors.append('#B4B4B4')
        else:
            colors.append('#1FA2FC')

    plt.figure(figsize=(10, 10))
    plt.scatter(gene_result_pd[x_str], gene_result_pd[y_str], color=colors, s=15, alpha=0.7)

    if scatter_strategy == "linear":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(gene_result_pd[[x_str, y_str]])
        df_scaled = pd.DataFrame(scaled_data, columns=[x_str, y_str])
        # Distance and map to the font size range through linear transformations
        # calculate distence from node to nearest nodes
        nbrs = NearestNeighbors(n_neighbors=5).fit(df_scaled[[x_str, y_str]])
        distances, indices = nbrs.kneighbors(df_scaled[[x_str, y_str]])
        distance_sums = distances[:, 1:].sum(axis=1)
        scaled_font_sizes = min_font_size + (max_font_size - min_font_size) * (distance_sums - distance_sums.min()) / (distance_sums.max() - distance_sums.min())
    # elif scatter_strategy =="baseY":

    # elif scatter_strategy == "percentile":
    #     scaled_font_sizes = min_font_size + (max_font_size - min_font_size) * (distance_sums - np.percentile(distance_sums, 25)) / (
    #             np.percentile(distance_sums, 75) - np.percentile(distance_sums, 25))
    #     scaled_font_sizes = np.clip(scaled_font_sizes, a_min=min_font_size, a_max=max_font_size)
    #     fontsize_threshold = np.percentile(scaled_font_sizes, 75)
    # elif scatter_strategy == "log":
    #     log_distance_sums = np.log1p(distance_sums - distance_sums.min() + 1)
    #     scaled_font_sizes = min_font_size + (max_font_size - min_font_size) * (log_distance_sums - log_distance_sums.min()) / (log_distance_sums.max() - log_distance_sums.min())
    #     scaled_font_sizes = np.clip(scaled_font_sizes, a_min=min_font_size, a_max=max_font_size)
    #     fontsize_threshold = 8.15
    scaled_font_sizes = np.round(scaled_font_sizes, decimals=2)
    # 为每个点添加标签
    texts = []
    for i, _g in enumerate(gene_result_pd['gene_short_name']):
        if scaled_font_sizes[i] < fontsize_threshold:
            continue
        loc_x = gene_result_pd.loc[_g][x_str]
        loc_y = gene_result_pd.loc[_g][y_str]
        # if loc_y < Q3_y:
        #     continue
        # if Q1_x < loc_x < Q3_x:
        #     continue
        texts.append(plt.text(loc_x, loc_y, f"{_g}", fontsize=scaled_font_sizes[i]))
        # plt.annotate(_g, (gene_result_pd.loc[_g][x_str], gene_result_pd.loc[_g][y_str]), fontsize=scaled_font_sizes[i])
        # plt.annotate(_g, (gene_result_pd.iloc[i][x_str], gene_result_pd.iloc[i][y_str]), fontsize=14)
    # 添加标签和标题

    adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))
    plt.xlabel(x_legend_str, fontsize=16)
    plt.ylabel(y_legend_str, fontsize=16)
    if stage_str == "allStage":
        plt.title(f'Scatter Plot of each gene.', fontsize=16)
    else:
        plt.title(f'Scatter Plot of each gene for {stage_str} time cluster.', fontsize=16)

    # 添加水平线 x=0
    if plt_yOrxZero == "y":
        plt.axhline(y=0, color='#F63C4C', linestyle='--', label='y=0')
    elif plt_yOrxZero == "x":
        plt.axvline(x=0, color='#F63C4C', linestyle='--', label='x=0')

    # 设置 x 轴范围
    # plt.xlim(gene_result_pd[x_str].min() - 10, gene_result_pd[x_str].max() + 100)
    # 修改 x 轴刻度值的字体大小
    plt.tick_params(axis='x', labelsize=14, rotation=45)  # 仅修改 x 轴
    plt.tick_params(axis='y', labelsize=14)  # 仅修改 x 轴
    # 显示图形
    if save_path is None:
        save_path = f"{_logger.root.handlers[0].baseFilename.replace('.log', '')}{special_path_str}/"
    save_file = f"{save_path}/{stage_str}_allGene_x{x_str.replace('_', '').capitalize()}_and_y{y_str.replace('_', '').capitalize()}_{special_filename_str}_scatter{scatter_strategy.capitalize()}.png"
    plt.savefig(save_file, dpi=200)
    print(f"Time changes under perturb image save as {save_file}")

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
    print(f"Time changes under perturb image save at {save_path}")
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
    # import numba
    # numba.core.caching.clear_cache()  # Clear Numba's disk cache
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
    print(f"latent mu save as {save_path}/{special_file_name_str}latent_mu.h5ad")
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


def plot_violin_240223(cell_info_df, save_path, x_attr="time",
                       y_attr="physical_pseudotime_by_preTrained_mouseAtlas_model",
                       special_file_name="", color_map="viridis", special_legend_str=""):
    # from utils.utils_Dandan_plot import calculate_real_predict_corrlation_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    corr_stats, corr_dic = calculate_real_predict_corrlation_score(cell_info_df[y_attr],
                                                                   cell_info_df[x_attr],
                                                                   only_str=False)
    print(f"=== data correlation: \n{corr_stats}")
    boxPlot_df = cell_info_df.copy()
    time_counts = boxPlot_df.groupby(x_attr)[y_attr].count().reset_index()

    # plot violin
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 7))
    cmap = plt.get_cmap(color_map)(np.linspace(0, 1, len(time_counts[x_attr])))
    color_dict = dict(zip(time_counts[x_attr], cmap))
    sns.violinplot(x=x_attr, y=y_attr, data=boxPlot_df, palette=color_map, bw=0.2, scale='width')

    corr_str = calculate_real_predict_corrlation_score(list(boxPlot_df[y_attr]), list(boxPlot_df[x_attr]), only_str=True)

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w',
                                  label=f"{time_counts[time_counts[x_attr] == time][y_attr].values[0]}",
                                  markerfacecolor=color_dict[time], markersize=10) for time in time_counts[x_attr]]
    plt.legend(handles=legend_elements, title="Cell Num", loc='lower right')
    # plt.legend(handles=legend_elements, title="Cell Num", loc='best')
    plt.text(0.15, 0.9,
             f'{special_legend_str}Correlation between x and y\n'
             f'Spearman: {np.round(corr_dic["spearman"][0], 3)}\n'
             f'Pearson: {np.round(corr_dic["pearson"][0], 3)}\n'
             f'Kendall’s τ: {np.round(corr_dic["kendalltau"][0], 3)}',
             verticalalignment='center', horizontalalignment='center',
             transform=plt.gca().transAxes,  # This makes the coordinates relative to the axes
             color='black', fontsize=12,
             bbox=dict(facecolor='white', alpha=0.5, edgecolor='black', boxstyle='round,pad=0.5'))
    plt.title(f"{special_file_name + ': '}Violin Plot of k-fold test:{corr_str}.", fontsize=12)
    plt.xlabel("Time", fontsize=12)
    plt.ylabel("Pseudotime", fontsize=12)
    plt.xticks(size=12)
    plt.yticks(size=12)
    plt.tight_layout()
    plt.savefig(f"{save_path}/{special_file_name + '_'}violine.png", dpi=200)

    plt.show()
    plt.close()
    print(f"figure save as {save_path}/{special_file_name + '_'}violine.png")
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


def plt_latentDim(spliced_fine_tune_result_data, unspliced_fine_tune_result_data, save_result_path):
    print("print latent dimension of spliced and unspliced data.")
    import matplotlib.pyplot as plt
    import numpy as np

    # 获取列数
    num_cols = spliced_fine_tune_result_data.X.shape[1]

    # 每行显示的子图数
    cols_per_row = 5

    # 计算需要的行数
    num_rows = (num_cols + cols_per_row - 1) // cols_per_row

    # 创建子图
    fig, axs = plt.subplots(num_rows, cols_per_row, figsize=(20, num_rows * 4))
    spliced_matrix = spliced_fine_tune_result_data.X
    unspliced_matrix = unspliced_fine_tune_result_data.X
    cell_type_list = spliced_fine_tune_result_data.obs["cell_type"]

    unique_labels = np.unique(cell_type_list)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
    label_to_color = dict(zip(unique_labels, colors))

    for i in range(num_cols):
        _data = {"s": spliced_matrix[:, i], "u": unspliced_matrix[:, i], "cell_type": cell_type_list}
        _data = pd.DataFrame(_data)
        row = i // cols_per_row
        col = i % cols_per_row
        for _label in unique_labels:
            _data_subtype = _data[_data["cell_type"] == _label]
            axs[row, col].scatter(_data_subtype["s"], _data_subtype["u"], c=label_to_color[_label], label=_label, alpha=0.3, s=3)
            axs[row, col].set_title(f'Column {i + 1}')

    # 隐藏多余的子图
    for i in range(num_cols, num_rows * cols_per_row):
        row = i // cols_per_row
        col = i % cols_per_row
        axs[row, col].axis('off')

    plt.legend(markerscale=5)
    plt.tight_layout()
    plt.savefig(f"{save_result_path}/latent_dim.png", dpi=200)
    plt.savefig(f"{save_result_path}/latent_dim.pdf", format='pdf')
    plt.show()
    plt.close()


def plot_tyser_mapping_to_datasets_attrTimeGT(adata_all, save_path, plot_attr,
                                              query_timePoint='16.5',
                                              legend_title="Cell stage",
                                              mask_dataset_label="t",
                                              reference_dataset_str='',
                                              special_file_str='',
                                              mask_color_alpha=0.7):
    import scanpy as sc
    adata_all.obs["time"] = adata_all.obs["time"].astype("float")
    time_values = adata_all.obs[plot_attr].unique()
    unique_times = sorted(set(time_values))

    min_time = np.min(unique_times)
    max_time = np.max(unique_times)
    normalized_times = (unique_times - min_time) / (max_time - min_time)
    turbo_cmap = plt.get_cmap('turbo')
    colors = [turbo_cmap(t) for t in normalized_times]
    # palette = sns.color_palette('turbo', len(time_values))
    # 创建颜色字典，将每个类别映射到一个颜色
    # color_dic = {str(time): color for time, color in zip(unique_times, palette)}
    color_dic = {str(time): color for time, color in zip(unique_times, colors)}
    if mask_dataset_label in ["t", "T"]:
        color_dic[query_timePoint] = (0.9, 0.9, 0.9, mask_color_alpha)
    elif mask_dataset_label in ["l & m & p & z & xiao", "L & M & P & Z & X & C","L & M & P & Z & Xiao & C"]:
        for _t in color_dic.keys():
            if _t != query_timePoint:
                color_dic[_t] = (0.9, 0.9, 0.9, mask_color_alpha)

    adata_all.obs["time"] = adata_all.obs["time"].astype("str")
    sc.settings.set_figure_params(dpi=200, facecolor="white", figsize=(5, 5), fontsize=18)

    sc.pl.umap(adata_all, color=plot_attr, show=False,
               s=25, palette=color_dic)
    plt.gca().set_title('')
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#808b96')
        spine.set_linewidth(0.5)  #
    plt.xlabel(plt.gca().get_xlabel(), fontsize=11)
    plt.ylabel(plt.gca().get_ylabel(), fontsize=11)
    plt.legend(title=legend_title, fontsize=14, title_fontsize=14,
               loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gca().set_position([0, 0, 1, 1])
    save_file_name = f"{save_path}/tyser_mapping_to_{reference_dataset_str}_{plot_attr}{special_file_str}.png"
    plt.savefig(save_file_name, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"figure save at {save_file_name}")
#
# def plot_query_mapping_to_referenceUmapSpace_attrTimeGT(adata_all, save_path, plot_attr,
#                                               query_timePoint='16.5',
#                                               legend_title="Cell stage",
#                                               mask_dataset_label="t",
#                                               reference_dataset_str='',
#                                               special_file_str='',
#                                               mask_color_alpha=0.7):
#     import scanpy as sc
#     adata_all.obs["time"] = adata_all.obs["time"].astype("float")
#     time_values = adata_all.obs[plot_attr].unique()
#     unique_times = sorted(set(time_values))
#
#     min_time = np.min(unique_times)
#     max_time = np.max(unique_times)
#     normalized_times = (unique_times - min_time) / (max_time - min_time)
#     turbo_cmap = plt.get_cmap('turbo')
#     colors = [turbo_cmap(t) for t in normalized_times]
#     # palette = sns.color_palette('turbo', len(time_values))
#     # 创建颜色字典，将每个类别映射到一个颜色
#     # color_dic = {str(time): color for time, color in zip(unique_times, palette)}
#     color_dic = {str(time): color for time, color in zip(unique_times, colors)}
#     if '&' not in mask_dataset_label:
#         color_dic[query_timePoint] = (0.9, 0.9, 0.9, mask_color_alpha)
#     elif mask_dataset_label in ["l & m & p & z & xiao", "L & M & P & Z & X & C","L & M & P & Z & Xiao & C"]:
#         for _t in color_dic.keys():
#             if _t != query_timePoint:
#                 color_dic[_t] = (0.9, 0.9, 0.9, mask_color_alpha)
#
#     adata_all.obs["time"] = adata_all.obs["time"].astype("str")
#     sc.settings.set_figure_params(dpi=200, facecolor="white", figsize=(5, 5), fontsize=18)
#
#     sc.pl.umap(adata_all, color=plot_attr, show=False,
#                s=25, palette=color_dic)
#     plt.gca().set_title('')
#     for spine in plt.gca().spines.values():
#         spine.set_edgecolor('#808b96')
#         spine.set_linewidth(0.5)  #
#     plt.xlabel(plt.gca().get_xlabel(), fontsize=11)
#     plt.ylabel(plt.gca().get_ylabel(), fontsize=11)
#     plt.legend(title=legend_title, fontsize=14, title_fontsize=14,
#                loc='center left', bbox_to_anchor=(1, 0.5))
#     plt.gca().set_position([0, 0, 1, 1])
#     save_file_name = f"{save_path}/tyser_mapping_to_{reference_dataset_str}_{plot_attr}{special_file_str}.png"
#     plt.savefig(save_file_name, dpi=200, bbox_inches='tight')
#     plt.show()
#     plt.close()
#     print(f"figure save at {save_file_name}")
# def plot_query_mapping_to_referenceUmapSpace_attrTimeGT(adata_all, save_path, plot_attr,
#                                                         legend_title="Cell stage",
#                                                         mask_dataset_label="t",
#                                                         reference_dataset_str='',
#                                                         special_file_str='',
#                                                         mask_color_alpha=0.7):
#
#     if isinstance(mask_dataset_label, str):
#         mask_dataset_label_list=[mask_dataset_label]
#     elif isinstance(mask_dataset_label, list):
#         mask_dataset_label_list=[mask_dataset_label]
#
#     import scanpy as sc
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from matplotlib.colors import to_rgba
#     adata_all.obs["time"] = adata_all.obs["time"].astype("float")
#     time_values = adata_all.obs[plot_attr].unique()
#     unique_times = sorted(set(time_values))
#
#     min_time = np.min(unique_times)
#     max_time = np.max(unique_times)
#     normalized_times = (unique_times - min_time) / (max_time - min_time)
#     turbo_cmap = plt.get_cmap('turbo')
#     colors = [turbo_cmap(t) for t in normalized_times]
#
#     # Create base color dictionary
#     color_dic = {str(time): to_rgba(color) for time, color in zip(unique_times, colors)}
#
#     # Create a new column for coloring that combines time and mask status
#     adata_all.obs['plot_color'] = adata_all.obs[plot_attr].astype(str)
#
#     # Identify cells to mask
#     mask_cells = adata_all.obs['dataset_label'].isin(mask_dataset_label_list)
#
#     # Create a new color dictionary that includes masked entries
#     masked_color = to_rgba((0.9, 0.9, 0.9, mask_color_alpha))
#
#     # For masked cells, append "_masked" to their time label
#     adata_all.obs.loc[mask_cells, 'plot_color'] = adata_all.obs.loc[mask_cells, 'plot_color'] + '_masked'
#
#     # Extend the color dictionary with masked entries
#     for time in unique_times:
#         color_dic[str(time) + '_masked'] = masked_color
#
#     sc.settings.set_figure_params(dpi=200, facecolor="white", figsize=(5, 5), fontsize=18)
#
#     # Plot using the new color column
#     sc.pl.umap(adata_all, color='plot_color', show=False,
#                s=25, palette=color_dic)
#
#     plt.gca().set_title('')
#     for spine in plt.gca().spines.values():
#         spine.set_edgecolor('#808b96')
#         spine.set_linewidth(0.5)
#
#     # Customize legend to hide the "_masked" suffix
#     handles, labels = plt.gca().get_legend_handles_labels()
#     new_labels = [l.replace('_masked', '') for l in labels]
#     plt.gca().legend(handles, new_labels,
#                      title=legend_title,
#                      fontsize=14,
#                      title_fontsize=14,
#                      loc='center left',
#                      bbox_to_anchor=(1, 0.5))
#
#     plt.xlabel(plt.gca().get_xlabel(), fontsize=11)
#     plt.ylabel(plt.gca().get_ylabel(), fontsize=11)
#     plt.gca().set_position([0, 0, 1, 1])
#
#     save_file_name = f"{save_path}/tyser_mapping_to_{reference_dataset_str}_{plot_attr}{special_file_str}.png"
#     plt.savefig(save_file_name, dpi=200, bbox_inches='tight')
#     plt.show()
#     plt.close()
#     print(f"figure save at {save_file_name}")
def plot_query_mapping_to_referenceUmapSpace_attrTimeGT(adata_all, save_path, plot_attr,
                                                        legend_title="Cell stage",
                                                        mask_dataset_label="t",
                                                        reference_dataset_str='',
                                                        special_file_str='',
                                                        mask_color_alpha=0.7):
    if isinstance(mask_dataset_label, str):
        mask_dataset_label_list = [mask_dataset_label]
    elif isinstance(mask_dataset_label, list):
        mask_dataset_label_list = mask_dataset_label  # Fixed this line to use the list directly

    import scanpy as sc
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    adata_all.obs["time"] = adata_all.obs["time"].astype("float")
    time_values = adata_all.obs[plot_attr].unique()
    unique_times = sorted(set(time_values))

    min_time = np.min(unique_times)
    max_time = np.max(unique_times)
    normalized_times = (unique_times - min_time) / (max_time - min_time)
    turbo_cmap = plt.get_cmap('turbo')
    colors = [turbo_cmap(t) for t in normalized_times]

    # Create base color dictionary
    color_dic = {str(time): to_rgba(color) for time, color in zip(unique_times, colors)}

    # Create a new column for coloring that combines time and mask status
    adata_all.obs['plot_color'] = adata_all.obs[plot_attr].astype(str)

    # Identify cells to mask
    mask_cells = adata_all.obs['dataset_label'].isin(mask_dataset_label_list)

    # Create a new color dictionary that includes masked entries
    masked_color = to_rgba((0.9, 0.9, 0.9, mask_color_alpha))

    # For masked cells, append "_masked" to their time label
    adata_all.obs.loc[mask_cells, 'plot_color'] = adata_all.obs.loc[mask_cells, 'plot_color'] + '_masked'

    # Extend the color dictionary with masked entries
    for time in unique_times:
        color_dic[str(time) + '_masked'] = masked_color

    sc.settings.set_figure_params(dpi=200, facecolor="white", figsize=(5, 5), fontsize=18)

    # Plot using the new color column
    sc.pl.umap(adata_all, color='plot_color', show=False,
               s=25, palette=color_dic)

    plt.gca().set_title('')
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#808b96')
        spine.set_linewidth(0.5)

    # Customize legend to exclude masked entries
    handles, labels = plt.gca().get_legend_handles_labels()

    # Filter out masked entries from legend
    filtered_handles = []
    filtered_labels = []
    for handle, label in zip(handles, labels):
        if not label.endswith('_masked'):
            filtered_handles.append(handle)
            filtered_labels.append(label)

    # Only create legend if there are items to show
    if filtered_handles:
        plt.gca().legend(filtered_handles, filtered_labels,
                         title=legend_title,
                         fontsize=14,
                         title_fontsize=14,
                         loc='center left',
                         bbox_to_anchor=(1, 0.5))

    plt.xlabel(plt.gca().get_xlabel(), fontsize=11)
    plt.ylabel(plt.gca().get_ylabel(), fontsize=11)
    plt.gca().set_position([0, 0, 1, 1])

    save_file_name = f"{save_path}/tyser_mapping_to_{reference_dataset_str}_{plot_attr}{special_file_str}.png"
    plt.savefig(save_file_name, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"figure save at {save_file_name}")
def plot_tyser_mapping_to_4dataset_predictedTime(adata_all, save_path, label_dic,
                                                 mask_dataset_label="t",
                                                 plot_attr='predicted_time',
                                                 reference_dataset_str="",
                                                 mask_str="data_type",
                                                 special_file_str="",
                                                 mask_color_alpha=0.7,
                                                 use_category_legend=False):
    min_t = min(label_dic.keys()) / 100
    max_t = max(label_dic.keys()) / 100
    import matplotlib.cm as matcm
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib as mpl
    scaler = MinMaxScaler()

    # normalized_time = scaler.fit_transform(adata_all.obs[attr].values[:, np.newaxis]).ravel()
    predict_time_list_add_min_max = np.concatenate((np.array([min_t, max_t]), adata_all.obs[plot_attr].values))
    normalized_time = scaler.fit_transform(predict_time_list_add_min_max[:, np.newaxis]).ravel()

    colors = matcm.turbo(normalized_time)[2:, ]
    color_dic = dict(zip(adata_all.obs[plot_attr], colors))
    for _k, _v in color_dic.items():
        color_dic[_k] = tuple(_v)
    mask = adata_all.obs[mask_str] == mask_dataset_label
    colors[mask] = (0.9, 0.9, 0.9, mask_color_alpha)

    _mask = (colors[:, -1] != mask_color_alpha)
    colored_unique_time = adata_all.obs.loc[_mask, 'time'].unique()
    plt.figure(figsize=(6.6, 6.6))
    plt.scatter(
        adata_all.obsm['X_umap'][:, 0],
        adata_all.obsm['X_umap'][:, 1],
        c=colors,  # 使用自定义颜色
        s=5,  # 点的大小
        # alpha=0.8  # 点的透明度
    )
    if use_category_legend:
        for _key in sorted(colored_unique_time):
            # for _key in sorted(color_dic.keys()):
            _color = color_dic[_key]
            plt.scatter([], [], c=[_color], s=50, label=_key)
        plt.legend(title="Cell stage", fontsize=14, title_fontsize=14,
                   loc='center left', bbox_to_anchor=(1, 0.5))
        # color_dic=dict(zip(adata_all.obs[plot_attr],colors))
        # unique_labels, indices = np.unique(colors, axis=0, return_inverse=True)

        # for idx, unique_color in enumerate(unique_labels):
        #     plt.scatter([], [], c=[unique_color], s=50, label=f'Category {idx}')  # 创建一个虚拟点用于图例
        # 添加类别图例
    else:
        cmap = plt.get_cmap('turbo')
        norm = mpl.colors.Normalize(vmin=min_t, vmax=max_t)
        scalar_mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
        scalar_mappable.set_array([])
        cbar = plt.colorbar(scalar_mappable, pad=0.01, fraction=0.03, shrink=1, aspect=30, ax=plt.gca())

        cbar.ax.invert_yaxis()
        cbar.ax.tick_params(labelsize=14)
        # cbar = plt.colorbar(scattering)

        cbar.set_label(plot_attr.replace("_", " ").capitalize(), fontsize=16)
        cbar.set_alpha(1)
        cbar._draw_all()
    plt.gca().set_title('')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#808b96')
        spine.set_linewidth(0.5)
    plt.xlabel(plt.gca().get_xlabel(), fontsize=11)  # 设置X轴字体大小
    plt.ylabel(plt.gca().get_ylabel(), fontsize=11)
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])

    # 移除网格线
    ax = plt.gca()
    ax.grid(False)  # 禁用网格线
    save_file_name = f"{save_path}/tyser_mapping_to_{reference_dataset_str}_{plot_attr}{special_file_str}.png"
    plt.savefig(save_file_name, dpi=200, bbox_inches='tight')
    # save_file_name = f"{save_path}/tyser_mapping_to_{reference_dataset_str}_{attr}{special_file_str}.png"
    plt.show()
    plt.close()
    print(f"figure save at {save_file_name}")


def plot_tyser_mapping_to_datasets_attrCellType_maskTyser(adata_all, save_path, attr,
                                                          masked_str, color_palette="hsv",
                                                          legend_title="Cell type",
                                                          reference_dataset_str="",
                                                          special_file_str='_maskT',
                                                          query_donor=None,
                                                          top_vis_cellType_num=15,
                                                          special_cell_type_list=None):
    import scanpy as sc
    unique_categories = adata_all.obs[attr].unique()
    # too long change, add \n to the cell type name.
    if 'Haemato-endothelial Progenitor' in unique_categories:
        adata_all.obs[attr] = adata_all.obs[attr].replace('Haemato-endothelial Progenitor',
                                                          'Haemato-endothelial\nProgenitor')
        unique_categories = adata_all.obs[attr].unique()
    category_counts = adata_all.obs[attr].value_counts()
    if len(unique_categories) > top_vis_cellType_num:
        print(f"more than 15 cell type, so trans cell type with less cells to Other cell type in image.")
        top_categories = category_counts.nlargest(top_vis_cellType_num).index
        other_categories = category_counts.index.difference(top_categories)
        adata_all.obs[attr] = adata_all.obs[attr].replace(other_categories, "Other cell type")
    else:
        top_categories = unique_categories
        other_categories = []

    palette = sns.color_palette(color_palette, len(top_categories))  # 可以选择不同的调色板
    color_dic = {cat: color for cat, color in zip(top_categories, palette)}
    color_dic[masked_str] = (0.9, 0.9, 0.9, 0.7)

    if len(other_categories) > 0:
        color_dic["Other cell type"] = "#d4efdf"
        _color = color_dic.pop("Other cell type")
        color_dic["Other cell type"] = _color
    _color = color_dic.pop(masked_str)
    color_dic[masked_str] = _color

    sc.settings.set_figure_params(dpi=200, facecolor="white", figsize=(5, 5), fontsize=18)

    sc.pl.umap(adata_all, color=attr, show=False, s=25,
               palette=color_dic)

    plt.gca().set_title('')
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#808b96')
        spine.set_linewidth(0.5)  #
    plt.xlabel(plt.gca().get_xlabel(), fontsize=11)
    plt.ylabel(plt.gca().get_ylabel(), fontsize=11)

    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dic[cat], markersize=8, label=cat)
               for cat in color_dic]
    if len(adata_all.obs[attr].unique()) > 16:
        print("legend use two col")
        plt.legend(title="Cell type", fontsize=13, title_fontsize=14,
                   loc='center left', bbox_to_anchor=(1, 0.5), ncol=2)
    else:
        plt.legend(handles=handles, title=legend_title, fontsize=13, title_fontsize=14,
               loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.legend(title=legend_title, fontsize=13, title_fontsize=13,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gca().set_position([0, 0, 1, 1])

    save_file_name = f"{save_path}/tyser_mapping_to_{reference_dataset_str}_{attr}{special_file_str}.png"
    plt.savefig(save_file_name, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"figure save at {save_file_name}")

    if query_donor == "T":
        adata_query = adata_all[adata_all.obs["dataset_label"] == "T"]

        manual_clusters = plot_to_identify_cluster_of_Tyser(adata_query.copy())

        # highlight cluster2 cell type distribution in Tyser.
        adata_query.obs["manual_cluster"] = manual_clusters
        cluster_celltype = pd.crosstab(adata_query.obs['manual_cluster'],
                                       adata_query.obs['cell_type'],
                                       normalize='index')
        cluster_celltype.plot.bar(stacked=True, figsize=(10, 5))
        plt.ylabel('Proportion')
        plt.title('Cell Type Composition per Manual Cluster'),

        plt.legend(title="Cell type", fontsize=14, title_fontsize=14,
                   loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(f'{save_path}/umap_cellTypeDistribution_queryOn{query_donor}.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()
    # return color_dic


def plot_tyser_mapping_to_datasets_attrDataset(adata_all, save_path, attr,
                                               color_dic, legend_title,
                                               masked_str='t',
                                               reference_dataset_str="",
                                               special_file_str=""):
    import scanpy as sc
    _color = color_dic.pop(masked_str)
    color_dic[masked_str] = _color
    sc.settings.set_figure_params(dpi=200, facecolor="white", figsize=(5, 5), fontsize=18)
    sc.pl.umap(adata_all, color=attr, show=False, s=25, palette=color_dic)
    plt.gca().set_title('')
    for spine in plt.gca().spines.values():
        spine.set_edgecolor('#808b96')  # 将边框颜色设置为红色
        spine.set_linewidth(0.5)  # 设置边框线条宽度
    plt.xlabel(plt.gca().get_xlabel(), fontsize=11)  # 设置X轴字体大小
    plt.ylabel(plt.gca().get_ylabel(), fontsize=11)
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_dic[cat], markersize=8, label=cat)
               for cat in color_dic]
    plt.legend(handles=handles, title=legend_title, fontsize=14, title_fontsize=14, loc='upper left', bbox_to_anchor=(0, 1))
    # plt.legend(title=legend_title, fontsize=13, title_fontsize=13,loc='center left', bbox_to_anchor=(1, 0.5))
    plt.gca().set_position([0, 0, 1, 1])

    save_file_name = f"{save_path}/tyser_mapping_to_{reference_dataset_str}_{attr}{special_file_str}.png"
    plt.savefig(save_file_name, dpi=200, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"figure save at {save_file_name}")


def plt_enrichmentResult(species, gene_set, env_gene_list, stage, save_path, top_term=5):
    import gseapy
    # gene_set=gseapy.get_library_name(organism=species)
    env_gene_num = len(env_gene_list)
    enr = gseapy.enrichr(gene_list=list(env_gene_list),  # or "./tests/data/gene_list.txt",
                         gene_sets=gene_set,
                         organism=species,  # don't forget to set organism to the one you desired! e.g. Yeast
                         outdir=None,  # don't write to disk
                         )  # enr.results.head(5)
    ax = gseapy.dotplot(enr.results,
                        column="Adjusted P-value",
                        x='Gene_set',  # set group, so you could do a multi-sample/library comparsion
                        size=10,
                        top_term=top_term,
                        figsize=(3, 5),
                        title=f"{stage.capitalize()} time cluster.",
                        xticklabels_rot=45,  # rotate xtick labels
                        show_ring=True,  # set to False to revmove outer ring
                        marker='o',
                        ofname=f"{save_path}/{stage}_enrichment_{env_gene_num}Genes_{species}_Top{env_gene_num}Genes.png")
    # ofname=f"{file_path}/{_s}_enrichment_{env_gene_num}Genes_{species}.png")
    # plt.savefig(f"{save_path}/{stage}_enrichment_{env_gene_num}Genes_{species}.png", bbox_inches="tight", dpi=300)
    # plt.gcf().subplots_adjust(left=0.05,top=0.91,bottom=0.09)
    # plt.show()
    plt.close()
    print(f"{stage} top {env_gene_num} pertub gene: {env_gene_list}")
    return enr


def plt_violinAndDot_topGene_inWhole_stage(top_gene_dic, perturb_data_denor,
                                           cell_info, perturb_show_gene_num, species,
                                           save_path):
    from collections import defaultdict
    # Create a defaultdict to hold the combined keys
    merged_dict = defaultdict(list)
    # Populate the merged_dict
    for key, genes in top_gene_dic.items():
        for gene in genes:
            merged_dict[gene].append(key)

    # Create the final dictionary with concatenated keys
    final_dict = {gene: '/'.join(keys) for gene, keys in merged_dict.items()}
    print(final_dict)
    # Plotting
    time_point_num = len(np.unique(np.array(cell_info["time"])))
    fig, axs = plt.subplots(len(final_dict), 1, figsize=(int(time_point_num / 2) + 1, 3 * len(final_dict)), sharey='row')

    for ax, (gene, str) in zip(axs, final_dict.items()):
        temp = {"det_time": np.array(perturb_data_denor[gene]) - np.array(cell_info["predicted_time_denor"]),
                "real_time": np.array(cell_info["time"]),
                "gene": gene,
                }
        temp = pd.DataFrame(temp, index=cell_info.index)
        sns.violinplot(x="real_time", y="det_time", data=temp, palette="tab10", inner="box", linewidth=1, alpha=0.5, bw_method=0.2, scale='width', ax=ax)
        sns.stripplot(data=temp, x="real_time", y="det_time", size=0.6, ax=ax)
        ax.axhline(y=0, color='red', linestyle='--', label='△t=0')
        # ax.scatter(cell_info["time"],np.array(perturb_data_denor[gene]) - np.array(cell_info["predicted_time_denor"]),label=gene)
        ax.set_title(f"{gene} is Top-{perturb_show_gene_num} temporal-sensitive in {str} stage", fontsize=16)
        ax.set_xlabel('Time', fontsize=16)
        ax.set_ylabel('△t', fontsize=16)
        # Rotating x-axis labels
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=16)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
        ax.legend(fontsize=16)

        # ax.legend(loc='upper right', fontsize=16)
    sns.despine()
    plt.tight_layout()
    save_file_name = f"{save_path}/{species}_wholeStage_perturb_top{perturb_show_gene_num}Gene_xTime_yDetT.png"
    plt.savefig(save_file_name, dpi=200)
    print(f"figure save as {save_file_name}")
    plt.show()
    plt.close()


def plt_lineChart_stageGeneDic_inStages(top_gene_dic, perturb_data_denor,
                                        cell_info, perturb_show_gene_num, species, stage_timePoint_dic,
                                        save_path, cal_detT_str="mean",
                                        plt_stage="whole", plt_timePoint="whole",
                                        special_filename_head_str="", special_filename_tail_str="",
                                        figsize_hight_weight=3.5):
    print(top_gene_dic)
    # Plotting
    # time_point_list = np.unique(np.array(cell_info["time"]))
    # print(time_point_list)
    if plt_stage == "whole":
        figsize_length, figsize_hight = int(len(np.unique(np.array(cell_info["time"]))) / 3) + 8, figsize_hight_weight * len(top_gene_dic)
    else:  # plt_stage is early, middle, late
        top_gene_dic = {plt_stage: top_gene_dic[plt_stage]}
        if plt_timePoint == "whole":
            figsize_length, figsize_hight = int(len(np.unique(np.array(cell_info["time"]))) / 3) + 8, figsize_hight_weight
        else:
            cell_info = cell_info[cell_info["time"].isin(stage_timePoint_dic[plt_timePoint])]
            perturb_data_denor = perturb_data_denor.loc[cell_info.index]
            figsize_length, figsize_hight = 10, 6
    time_point_list = np.unique(np.array(cell_info["time"]))
    print(time_point_list)
    fig, axs = plt.subplots(len(top_gene_dic), 1,
                            figsize=(figsize_length, figsize_hight),
                            sharey='row', sharex=True,
                            )

    if len(top_gene_dic) == 1:
        axs = [axs]
    for ax, (stage, gene_list) in zip(axs, top_gene_dic.items()):
        # print(type(ax))
        for gene in gene_list:
            temp = {"det_time": np.array(perturb_data_denor[gene]) - np.array(cell_info["predicted_time_denor"]),
                    "real_time": np.array(cell_info["time"]),
                    "gene": gene, }
            temp = pd.DataFrame(temp, index=cell_info.index)
            if cal_detT_str == "mean":
                average_det_time = temp.groupby("real_time")["det_time"].mean()
            elif cal_detT_str == "median":
                average_det_time = temp.groupby("real_time")["det_time"].median()
            #  Series trans to DataFrame
            average_det_time_df = average_det_time.reset_index()
            average_det_time_df.columns = ['real_time', 'average_det_time']  #
            sns.lineplot(x='real_time', y='average_det_time', data=average_det_time_df, marker='o', ax=ax, label=gene)
        ax.axhline(y=0, color='red', linestyle='--', label='△t=0')
        ax.set_title(f"Temporal-sensitive genes in {stage} time cluster.", fontsize=16)

        ax.set_ylabel(f'{cal_detT_str.capitalize()} △t', fontsize=16)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=16)
        ax.legend(loc='upper right', fontsize=16)
    for i, ax in enumerate(axs):
        ax.grid(True, which='both', axis='x', linestyle='--', linewidth=0.5)  # 启用 X 轴的网格线
        if plt_timePoint == "whole":
            if species == "human":
                ax.axvline(x=7, color='#9370DB', linestyle='--', linewidth=2, label='x = 7')
                ax.axvline(x=14, color='#9370DB', linestyle='--', linewidth=2, label='x = 14')

            elif species == "mouse":
                ax.axvline(x=10.25, color='#9370DB', linestyle='--', linewidth=2, label='x = 10.25')
                ax.axvline(x=14, color='#9370DB', linestyle='--', linewidth=2, label='x = 14')
        if i < len(axs) - 1:  # 如果不是最后一个子图
            ax.tick_params(labelbottom=False)  # 隐藏除最下面子图外的X轴刻度标签
        else:
            ax.tick_params(labelbottom=True)  # 确保最后一个子图显示X轴刻度标签
            ax.set_xlabel('Biological time', fontsize=16)

            if plt_timePoint == "whole":
                ax.set_xticklabels(ax.get_xticks(), rotation=45, fontsize=16)
                highlight_ticks_dic = {"human": [7.0, 14.0], "mouse": [14.0, 10.25]}
                for tick in highlight_ticks_dic[species]:
                    ax.text(tick, ax.get_ylim()[0], f'x={tick}', color='#9370DB', verticalalignment='bottom', fontsize=16,
                            bbox=dict(boxstyle="square,pad=0.", ec="none", alpha=0.4, facecolor='#ADD8E6'))
            else:
                ax.set_xticks(time_point_list)
                ax.set_xticklabels(time_point_list, rotation=45, fontsize=16)  # 确保使用正确的刻度和格式
    # sns.despine()
    plt.tight_layout()
    save_file_name = f"{save_path}/{special_filename_head_str}{plt_stage}Stage_perturb_top{perturb_show_gene_num}Gene_xTime_y{cal_detT_str.capitalize()}DetT_line_{species}{special_filename_tail_str}.png"
    plt.savefig(save_file_name, dpi=200)
    print(f"figure save as {save_file_name}")
    plt.show()
    plt.close()


def plt_venn_fromDict(enr_top_gene_dic2, save_path, perturb_show_gene_num, species):
    from matplotlib_venn import venn3
    # 创建一个Venn图
    plt.figure(figsize=(6, 6))
    venn_diagram = venn3([set(enr_top_gene_dic2["early"]), set(enr_top_gene_dic2["middle"]), set(enr_top_gene_dic2["late"])],
                         ('Early', 'Middle', 'Late'),
                         set_colors=("#D6A2E8", "#3498DB", "#FFC93C"), alpha=0.9, )
    # 设置字体大小
    for text in venn_diagram.set_labels:
        if text:  # 防止在空标签上设置属性
            text.set_fontsize(16)  # 更改集合名字体大小

    for text in venn_diagram.subset_labels:
        if text:
            text.set_fontsize(14)  # 更改集合内元素计数的字体大小
    # venn3_circles([set(enr_top_gene_dic2["early"]), set(enr_top_gene_dic2["middle"]), set(enr_top_gene_dic2["late"])],
    #               linestyle="dashed", linewidth=2)
    # 查找所有三个组的交集
    intersection = set(enr_top_gene_dic2["early"]) & set(enr_top_gene_dic2["middle"]) & set(enr_top_gene_dic2["late"])
    print(f"Top50 in each time cluster intersection: {intersection}")
    # if intersection:
    #     # 在图的中心添加交集元素的名称
    #     plt.text(venn_diagram.get_label_by_id('111').get_position()[0],
    #              venn_diagram.get_label_by_id('111').get_position()[1],
    #              '\n'.join(intersection),
    #              ha='center', va='center', fontsize=8, color='black')

    # 显示图表
    plt.title("Venn Diagram of cell time clusters.", fontsize=16)
    plt.tight_layout()
    save_file_name = f"{save_path}/3Stage_perturb_top{perturb_show_gene_num}Gene_{species}_venn.png"
    plt.savefig(save_file_name, dpi=200)
    print(f"figure save as {save_file_name}")
    plt.show()
    plt.close()
    return intersection


def plt_perturb_xTime_yDetT(plot_pd, gene_list, save_path, stage):
    plt.figure(figsize=(len(plot_pd["real_time"].unique()) * 2 + 2, 8))
    sns.set(style="white", color_codes=True)
    # 创建散点图
    # plt.scatter(temp["trained_time"],temp["perturb_time"] ,s=2,c=colors[i],alpha=0.4)
    sns.violinplot(x="real_time", y="det_time", hue="gene", data=plot_pd, palette="tab10", inner="box", linewidth=1, alpha=0.5, bw_method=0.2, scale='width')
    # sns.boxenplot(x="real_time", y="det_time", hue="gene", data=plot_pd, palette="tab10", width=0.05)
    # sns.scatterplot(x="real_time", y="det_time", data=plot_pd, hue="gene", palette="tab10",s=5, alpha=0.7, )

    # sns.lmplot(x="real_time", y="det_time", data=plot_pd, hue="gene", palette="tab10",
    #            order=2, ci=None, scatter_kws={"s": 5, "alpha": 0.7}, height=10, aspect=1.5)  # lmplot默认order参数是一阶的
    plt.axhline(y=0, color='red', linestyle='--', label='△t=0')
    # 添加标题和标签
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title(f'In {stage} stage: top {len(gene_list)} abs(△t) of genes', fontsize=16)
    plt.xlabel('Biological time.', fontsize=16)
    plt.ylabel('△t: pseudo-time with perturbation - without perturbation.', fontsize=16)
    plt.legend(fontsize=16)
    #
    # _logger.info
    # handles, labels = plt.gca().get_legend_handles_labels()
    # plt.legend(handles, labels, title='Gene')
    plt.tight_layout()
    save_file_name = f"{save_path}/{stage}_perturb_top{len(gene_list)}Gene_xTime_yDetT.png"
    plt.savefig(save_file_name, dpi=200)
    print(f"Time changes under perturb image save as {save_file_name}")
    plt.show()
    plt.close()
    gc.collect()


def plt_allGene_dot_voteNum_meanDetT_Exp(cell_info, perturb_data_denor, pertrub_gene_df, file_path, top_gene_num=10, stage_str="allStage",
                                         x_str="mean",
                                         species="mouse"):  # x_str: "mean" or "median"
    cell_df = cell_info.loc[perturb_data_denor.index]

    # abs_mean_df = pert_data.apply(lambda col: abs(np.array(col) - np.array(cell_df["predicted_time_denor"])).mean())
    from TemporalVAE.utils.utils_project import voteScore_genePerturbation
    column_counts = voteScore_genePerturbation(cell_df, perturb_data_denor, top_gene_num, predictedTime_attr="predicted_time_denor")

    pertrub_gene_df[f"top{top_gene_num}VoteNum"] = pertrub_gene_df['gene_short_name'].map(column_counts).fillna(0).astype(int)
    pertrub_gene_df[f"top{top_gene_num}VoteNum_proportion"] = pertrub_gene_df[f"top{top_gene_num}VoteNum"] / (len(cell_df))
    pertrub_gene_df["mean"] = perturb_data_denor.apply(lambda col: np.mean((np.array(col) - np.array(cell_df["predicted_time_denor"]))))
    pertrub_gene_df["median"] = perturb_data_denor.apply(lambda col: np.median(np.array(col) - np.array(cell_df["predicted_time_denor"])))

    # plot_detTandExp(pertrub_gene_df.copy(),
    #                 y_str=f"top{top_gene_num}VoteNum", y_legend_str=f'Total Votes for Top {top_gene_num} Genes per Sample',
    #                 x_str=x_str, x_legend_str=f'{x_str.capitalize()} △t: with - without perturbation.',
    #                 special_filename_str=f"voteTop{top_gene_num}", save_path=file_path, stage_str=stage_str,
    #                 plt_yOrxZero="x",
    #                 scatter_strategy="linear",
    #                 )
    plot_detTandExp(pertrub_gene_df.copy(),
                    y_str=f"top{top_gene_num}VoteNum_proportion", y_legend_str=f'Vote proportion',
                    x_str=x_str, x_legend_str=f'{x_str.capitalize()} △t: with - without perturbation.',
                    special_filename_str=f"voteTop{top_gene_num}", save_path=file_path, stage_str=stage_str,
                    plt_yOrxZero="x",
                    scatter_strategy="linear",
                    )

    # plot_detTandExp(pertrub_cor_data.copy(),
    #                 x_str="total_raw_count", x_legend_str='Total raw expression of genes.',
    #                 y_str=f"top{top_gene_num}VoteNum", y_legend_str=f'Total Votes for Top {top_gene_num} Genes per Sample',
    #                 special_filename_str=f"voteTop{top_gene_num}", save_path=file_path,stage_str=stage_str,
    #                 )
    # plot_detTandExp(pertrub_cor_data.copy(),
    #                 x_str="total_raw_count", x_legend_str='Total raw expression of genes.',
    #                 y_str="median", y_legend_str=f'Median △t: pseudo-time after perturb and without perturb.',
    #                 special_filename_str=f"medianDetT", save_path=file_path,stage_str=stage_str,
    #                 )
    # plot_detTandExp(pertrub_cor_data.copy(),
    #                 x_str="total_raw_count", x_legend_str='Total raw expression of genes.',
    #                 y_str="mean", y_legend_str=f'Mean △t: pseudo-time after perturb and without perturb.',
    #                 special_filename_str=f"meanDetT", save_path=file_path,stage_str=stage_str,
    #                 )


def plt_muiltViolin_forGenes_xRawCount(adata_df, intersection, cell_info,
                                       save_path, perturb_show_gene_num,
                                       species, special_filename_str=""):
    # 从 anndata 中提取基因表达数据
    expr_matrix = adata_df[list(intersection)]
    # 确保 cell_anno 的索引与 expr_matrix 对应
    expr_matrix['cell_id'] = expr_matrix.index
    cell_info["cell_id"] = cell_info.index
    cell_info2 = cell_info[cell_info['cell_id'].isin(expr_matrix['cell_id'])]

    # 合并表达数据和细胞注释
    full_data = pd.merge(expr_matrix, cell_info2, on='cell_id')

    # 选择一个颜色调色板
    palette = sns.color_palette("hsv", len(full_data['time'].unique()))

    # 绘制 violin plot
    unique_time_points = full_data['time'].nunique()
    fig, axes = plt.subplots(nrows=len(intersection), figsize=(unique_time_points / 2, len(intersection) / 1.5),
                             # constrained_layout=True
                             )
    print(unique_time_points / 2, len(intersection) / 1.5)
    for i, gene in enumerate(intersection):
        vplot = sns.violinplot(x='time', y=gene, data=full_data, palette=palette, inner=None, ax=axes[i])
        axes[i].set_ylabel(gene, rotation=0, horizontalalignment='right', labelpad=2)
        # axes[i].set_xlabel('Time')
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        if i < len(intersection) - 1:
            axes[i].tick_params(labelbottom=False)  # 隐藏非最后一个图的 x 轴标签
            axes[i].set_xlabel('')
        else:

            axes[i].tick_params(labelbottom=True)
            axes[i].set_xlabel('Biogical time')
            axes[i].set_xticklabels(vplot.get_xticklabels(), rotation=45)
    plt.tight_layout()
    save_file_name = f"{save_path}/geneExpression_top{perturb_show_gene_num}Gene_{species}_violin{special_filename_str}.png"
    plt.savefig(save_file_name, dpi=200)
    print(f"figure save as {save_file_name}")
    plt.show()
    plt.close()
def plot_to_identify_cluster_of_Tyser(adata):
    import matplotlib.pyplot as plt
    # 1. plot location (x,y) of cells in umap to identify cell clusters.
    umap_coords = adata.obsm['X_umap']
    x, y = umap_coords[:, 0], umap_coords[:, 1]

    plt.figure(figsize=(10, 8))
    plt.scatter(x, y, s=5, alpha=0.5, c='gray', label='Unassigned')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('UMAP with Manual Cluster Boundaries')
    plt.show()
    plt.close()
    # 2. manual set 3 cluster of cells
    cluster_ranges = {
        'Cluster 0': {'x_range': (-10, -5), 'y_range': (0, 10)},  # 左下
        'Cluster 1': {'x_range': (0, 5), 'y_range': (5, 10)},  # 中部
        'Cluster 2': {'x_range': (0, 5), 'y_range': (10, 15)}  # 右上
    }
    manual_clusters = np.full(len(adata), '-1', dtype=object)  # -1 表示未分配

    for i, (xi, yi) in enumerate(zip(x, y)):
        for cluster, ranges in cluster_ranges.items():
            x_min, x_max = ranges['x_range']
            y_min, y_max = ranges['y_range']
            if x_min <= xi <= x_max and y_min <= yi <= y_max:
                manual_clusters[i] = cluster
                break
    plt.figure(figsize=(10, 8))
    colors = {'Cluster 0': 'red', 'Cluster 1': 'green', 'Cluster 2': 'blue', '-1': 'gray'}
    plt.scatter(x, y, s=5, c=[colors[c] for c in manual_clusters], alpha=0.7)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('Cells Colored by Manual Cluster')

    for cluster, ranges in cluster_ranges.items():
        x_min, x_max = ranges['x_range']
        y_min, y_max = ranges['y_range']
        plt.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min],
                 'k--', linewidth=1, alpha=0.5)

    plt.show()
    plt.close()
    return manual_clusters