# -*-coding:utf-8 -*-
"""
@Project ：pairsRegulatePrediction
@File    ：vae_humanEmbryo_Melania.py
@IDE     ：PyCharm
@Author  ：awa121
@Date    ：2024/3/3 21:02


"""

import os
import sys

if os.getcwd().split("/")[-1] != "TemporalVAE":
    os.chdir("..")
sys.path.append(os.getcwd())

import torch

torch.set_float32_matmul_precision('high')
import pyro

from utils.logging_system import LogHelper

smoke_test = ('CI' in os.environ)  # ignore; used to check code integrity in the Pyro repo
assert pyro.__version__.startswith('1.8.5')
pyro.set_rng_seed(1)
from utils.utils_DandanProject import *
from collections import Counter
import os
import yaml
import argparse
from utils.utils_Dandan_plot import *
import anndata as ad
from draw_images.read_json_plotViolin_oneTimeMulitDonor import plt_umap_byScanpy
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="CNN model for prediction of gene paris' regulatory relationship")
    parser.add_argument('--result_save_path', type=str,  # 2023-07-13 17:40:22
                        default="240505_preimplantation_Melania",
                        help="results all save here")
    parser.add_argument('--file_path', type=str,
                        default="/240405_preimplantation_Melania/",
                        help="sc file folder path.")
    # ------------------ preprocess sc data setting ------------------
    parser.add_argument('--min_gene_num', type=int,
                        default="50",
                        help="filter cell with min gene num, default 50")
    parser.add_argument('--min_cell_num', type=int,
                        default="50",
                        help="filter gene with min cell num, default 50")
    # ------------------ model training setting ------------------
    parser.add_argument('--train_epoch_num', type=int,
                        default="100",
                        help="Train epoch num")
    parser.add_argument('--batch_size', type=int,
                        default=100000,
                        help="batch size")
    parser.add_argument('--time_standard_type', type=str,
                        default="embryoneg5to5",
                        help="y_time_nor_train standard type may cause different latent space: log2, 0to1, neg1to1, labeldic,sigmoid,logit")
    # supervise_vae            supervise_vae_regressionclfdecoder
    parser.add_argument('--vae_param_file', type=str,
                        # default="supervise_vae_regressionclfdecoder_mouse_stereo_humanEmbryo240401",
                        default="supervise_vae_regressionclfdecoder_mouse_stereo",
                        # default="supervise_vae_regressionclfdecoder",
                        help="vae model parameters file.")
    # ------------------ task setting ------------------
    parser.add_argument('--kfold_test', action="store_true", help="(Optional) make the task k fold test on dataset.", default=True)
    parser.add_argument('--train_whole_model', action="store_true", help="(Optional) use all data to train a model.", default=True)
    parser.add_argument('--identify_time_cor_gene', action="store_true", help="(Optional) identify time-cor gene by model trained by all.", default=False)

    # Todo, useless, wait to delete "KNN_smooth_type"
    parser.add_argument('--KNN_smooth_type', type=str,
                        default="mingze",
                        help="KNN smooth method")  # don't use 2023-06-26 14:04:25

    args = parser.parse_args()

    data_golbal_path = "data/"
    result_save_path = "results/" + args.result_save_path + "/"
    data_path = args.file_path + "/"
    yaml_path = "vae_model_configs/"
    # --------------------------------------- import vae model parameters from yaml file----------------------------------------------
    with open(yaml_path + "/" + args.vae_param_file + ".yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # ---------------------------------------set logger and parameters, creat result save path and folder----------------------------------------------
    latent_dim = config['model_params']['latent_dim']
    # KNN_smooth_type = args.KNN_smooth_type

    time_standard_type = args.time_standard_type
    sc_data_file_csv = data_path + "/data_count_hvg.csv"
    cell_info_file_csv = data_path + "/cell_with_time.csv"

    _path = '{}/{}/'.format(result_save_path, data_path)
    if not os.path.exists(_path):
        os.makedirs(_path)

    logger_file = '{}/{}_dim{}_time{}_epoch{}_minGeneNum{}.log'.format(_path, args.vae_param_file, latent_dim,
                                                                       time_standard_type, args.train_epoch_num,
                                                                       args.min_gene_num)
    LogHelper.setup(log_path=logger_file, level='INFO')
    _logger = logging.getLogger(__name__)
    _logger.info("Finished setting up the logger at: {}.".format(logger_file))
    _logger.info("Train on dataset: {}.".format(data_golbal_path + data_path))
    device = auto_select_gpu_and_cpu()
    _logger.info("Auto select run on {}".format(device))
    _logger.info("load vae model parameters from file: {}".format(yaml_path + args.vae_param_file + ".yaml"))
    # ------------ Preprocess data, with hvg gene from preprocess_data_mouse_embryonic_development.py------------------------
    if "Melania" in sc_data_file_csv:
        data_raw_count_bool = False
    else:
        data_raw_count_bool = True

    sc_expression_df, cell_time = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path, sc_data_file_csv, cell_info_file_csv,
                                                                                # donor_attr=donor_attr, drop_out_donor=drop_out_donor,
                                                                                min_cell_num=args.min_cell_num,
                                                                                min_gene_num=args.min_gene_num,
                                                                                data_raw_count_bool=data_raw_count_bool)  # 2024-04-20 15:38:58

    special_path_str = ""
    # ---------------------------------------- set donor list and dictionary -----------------------------------------------------
    donor_list = np.unique(cell_time["day"])
    donor_list = sorted(donor_list, key=Embryodonor_resort_key)
    donor_dic = dict()
    for i in range(len(donor_list)):
        donor_dic[donor_list[i]] = i
    batch_dic = donor_dic.copy()
    _logger.info("Consider donor as batch effect, donor use label: {}".format(donor_dic))
    _logger.info("For each donor (donor_id, cell_num):{} ".format(Counter(cell_time["day"])))
    save_file_name = f"{_logger.root.handlers[0].baseFilename.replace('.log', '')}/"
    #  ---------------------------------------------- TASK: use all data to train a model  ----------------------------------------------
    if args.train_whole_model:
        drop_out_donor = "t"
        print(f"drop the donor: {drop_out_donor}")
        cell_drop_index_list = cell_time.loc[cell_time["dataset_label"] == drop_out_donor].index
        sc_expression_df_filter = sc_expression_df.drop(cell_drop_index_list, axis=0)
        cell_time_filter = cell_time.drop(cell_drop_index_list, axis=0)
        cell_time_filter = cell_time_filter.loc[sc_expression_df_filter.index]
        sc_expression_train, y_time_nor_train, donor_index_train, runner, experiment, _m, train_clf_result, label_dic, total_result = onlyTrain_model(
            sc_expression_df_filter, donor_dic,
            special_path_str,
            cell_time_filter,
            time_standard_type, config, args,
            device=device, plot_latentSpaceUmap=False, plot_trainingLossLine=True, time_saved_asFloat=True, batch_dic=batch_dic, donor_str="day",
            batch_size=int(args.batch_size))  # 2023-10-24 17:44:31 batch as 10,000 due to overfit, batch size as 100,000 may be have different result
        predict_donors_df = pd.DataFrame(train_clf_result, columns=["pseudotime"], index=cell_time_filter.index)
        predict_donors_df['predicted_time'] = predict_donors_df['pseudotime'].apply(denormalize, args=(min(label_dic.keys()) / 100, max(label_dic.keys()) / 100,
                                                                                                       min(label_dic.values()), max(label_dic.values())))
        cell_time_filter = pd.concat([cell_time_filter, predict_donors_df], axis=1)

        plt_image_adata = ad.AnnData(X=total_result["mu"].cpu().numpy())
        plt_image_adata.obs = cell_time_filter[["time", "predicted_time", "dataset_label", "cell_type", "day"]]

        plt_umap_byScanpy(plt_image_adata.copy(), ["time", "predicted_time", "dataset_label", "cell_type"], save_path=save_file_name, mode=None, figure_size=(5, 4),
                          color_map="turbo",
                          n_neighbors=50, n_pcs=20, special_file_name_str="n50_")  # color_map="viridis"

    ## # # ----------------------------------TASK 1: K-FOLD TEST--------------------------------------
    if args.kfold_test:
        test_donor_list = ["D_14_21_t"]
        predict_donors_dic, label_dic, mu_result = task_kFoldTest(test_donor_list, sc_expression_df, donor_dic, batch_dic, special_path_str, cell_time, time_standard_type,
                                                                  config, args.train_epoch_num, _logger, donor_str="day", batch_size=args.batch_size, recall_predicted_mu=True)
        train_mu_result, test_mu_result = mu_result
        cell_time_tyser = cell_time.loc[predict_donors_dic["D_14_21_t"].index]
        cell_time_tyser["predicted_time"] = predict_donors_dic["D_14_21_t"]['pseudotime'].apply(denormalize, args=(min(label_dic.keys()) / 100, max(label_dic.keys()) / 100,
                                                                                                                   min(label_dic.values()), max(label_dic.values())))
        cell_time_tyser = cell_time_tyser[["time", "predicted_time", "dataset_label", "day", "cell_type"]]
        adata_mu_tyser = ad.AnnData(X=test_mu_result.cpu().numpy(), obs=cell_time_tyser)
        adata_mu_tyser.obs['data_type'] = 't'

        adata_mu_4dataset = ad.read_h5ad(f"{save_file_name}/n50_latent_mu.h5ad")
        adata_mu_4dataset.obs['data_type'] = 'l & m & p & z'

        adata_all = anndata.concat([adata_mu_4dataset.copy(), adata_mu_tyser.copy()], axis=0)
        adata_all.obs["cell_typeMask4dataset"] = adata_all.obs.apply(lambda row: 'l & m & p & z' if row['dataset_label'] != 't' else row['cell_type'], axis=1)
        adata_all.obs["cell_typeMaskTyser"] = adata_all.obs.apply(lambda row: 't' if row['dataset_label'] == 't' else row['cell_type'], axis=1)

        # ---- 1 method: mapping tyser data to other 4 dataset's umap, just use different umap model
        # sc.pp.neighbors(adata_mu_tyser, n_neighbors=50, n_pcs=20)
        # sc.tl.umap(adata_mu_tyser, min_dist=0.75)
        # ----

        # ----2 method mapping tyser data to other 4 dataset's umap, use same umap model by 4 dataset,
        # Create a UMAP model instance
        import umap
        reducer = umap.UMAP(n_neighbors=50, min_dist=0.75, n_components=2, random_state=0)
        embedding_4dataset = reducer.fit_transform(adata_mu_4dataset.X)
        embedding_tyser = reducer.transform(adata_mu_tyser.X)
        adata_mu_4dataset.obsm['X_umap'] = embedding_4dataset
        adata_mu_tyser.obsm['X_umap'] = embedding_tyser
        # ----

        # combin two AnnData's UMAP loc
        combined_umap = np.vstack([adata_mu_4dataset.obsm['X_umap'], adata_mu_tyser.obsm['X_umap']])
        adata_all.obsm["X_umap"] = combined_umap
        adata_all.write_h5ad(f"{save_file_name}/5dataset_mu.h5ad")
        # ----
        with plt.rc_context({'figure.figsize': (6, 5)}):
            sc.pl.umap(adata_all, color="dataset_label", show=False, legend_fontsize=5.5, s=25,
                       palette={'l': "#B292CA", 'm': '#7ED957', 'p': '#FFC947', 'z': '#00CED1', 't': (0.9, 0.9, 0.9, 0)})
        plt.savefig(f"{save_file_name}/tyser_mapping_to_4dataset_datatype_maskt.png", dpi=300, )
        plt.show()
        plt.close()
        with plt.rc_context({'figure.figsize': (6, 5)}):
            sc.pl.umap(adata_all, color="data_type", show=False, legend_fontsize=5.5, s=25, palette={'l & m & p & z': (0.9, 0.9, 0.9, 0.7), 't': "#E06D83"})
        plt.tight_layout()
        plt.savefig(f"{save_file_name}/tyser_mapping_to_4dataset_datatype_maskl & m & p & z.png", dpi=300)
        plt.show()
        plt.close()

        plot_tyser_mapping_to_4dataset_cellType_maskTyser(adata_all.copy(), save_file_name, mask_dataset_label='t', attr="cell_typeMaskTyser")
        plot_tyser_mapping_to_4dataset_cellType_mask4dataset(adata_all.copy(), save_file_name, mask_dataset_label='l & m & p & z', attr="cell_typeMask4dataset")
        plot_tyser_mapping_to_4dataset_time_catgorial(adata_all.copy(), save_file_name, mask_dataset_label="t", attr="time")
        plot_tyser_mapping_to_4dataset_time_catgorial(adata_all.copy(), save_file_name, mask_dataset_label="l & m & p & z", attr="time")
        plot_tyser_mapping_to_4dataset_predictedTime(adata_all.copy(), save_file_name, mask_dataset_label='t', attr='predicted_time')
        plot_tyser_mapping_to_4dataset_predictedTime(adata_all.copy(), save_file_name, mask_dataset_label='l & m & p & z', attr='predicted_time')

        with plt.rc_context({'figure.figsize': (6, 5)}):
            sc.pl.umap(adata_all, color="data_type", show=False, legend_fontsize=5.5, s=25, palette={'l & m & p & z': (0.9, 0.9, 0.9, 0.7), 't': "#E06D83"})
        plt.tight_layout()
        plt.savefig(f"{save_file_name}/tyser_mapping_to_4dataset_datatype_maskl & m & p & z.png", dpi=300)
        plt.show()
        plt.close()

        adata_all.write_h5ad(f"{save_file_name}/mappint_tyser_to4dataset_latent_mu.h5ad")

        # ---- combine and plot umap
        # adata_all.obs["time"] = adata_all.obs["time"].astype("str")
        # with plt.rc_context({'figure.figsize': (6, 5)}):
        #     sc.pl.umap(adata_all, color="time", show=False, legend_fontsize=5.5, s=25, palette="turbo")
        # plt.tight_layout()
        # plt.savefig(f"{save_file_name}/combine_tyser_to4dataset_predictedTime.png", dpi=300)
        # plt.show()
        # plt.close()

        plt_umap_byScanpy(adata_all.copy(), ["time", "predicted_time", "dataset_label", "day", "cell_type", "cell_typeMask4dataset"],
                          save_path=save_file_name, mode=None, figure_size=(13, 4),
                          color_map="turbo", show_in_row=False,
                          n_neighbors=50, n_pcs=20, special_file_name_str="combine_tyser_mapping_to_n50_4dataset")

        _logger.info("Finish fold-test.")

    _logger.info("Finish all.")


def plot_tyser_mapping_to_4dataset_time_catgorial(adata_all, save_file_name, mask_dataset_label="t", attr="time"):
    # adata_all.obs["time"] = adata_all.obs["time"].astype("float")
    import matplotlib.pyplot as plt
    time_values = adata_all.obs[attr].unique()

    # 提取唯一值并排序
    unique_times = sorted(set(time_values))

    # 获取turbo调色板
    turbo_cmap = plt.get_cmap('turbo')
    # 计算最小和最大时间值，用于归一化
    min_time = np.min(unique_times)
    max_time = np.max(unique_times)

    # 归一化时间值
    normalized_times = (unique_times - min_time) / (max_time - min_time)
    # 分配颜色，为每个类别均匀选择颜色
    # colors = [turbo_cmap(i / (len(unique_times) - 1)) for i in range(len(unique_times))]
    colors = [turbo_cmap(t) for t in normalized_times]
    # 创建颜色字典，将每个类别映射到一个颜色
    color_dict = {str(time): color for time, color in zip(unique_times, colors)}
    if mask_dataset_label == "t":
        color_dict['17.5'] = (0.9, 0.9, 0.9, 0)
    elif mask_dataset_label == "l & m & p & z":
        for _t in color_dict.keys():
            if _t != '17.5':
                color_dict[_t] = (0.9, 0.9, 0.9, 0.7)
    # mask = adata_all.obs['data_type'] == mask_dataset_label
    # 应用颜色映射
    # colors[mask] = [0.9, 0.9, 0.9, 1]
    # color_map = {cat: color for cat, color in zip(unique_categories, palette)}
    # color_map["NA"] = [0.9, 0.9, 0.9,1]
    adata_all.obs["time"] = adata_all.obs["time"].astype("str")
    with plt.rc_context({'figure.figsize': (6, 5)}):
        sc.pl.umap(adata_all, color=attr, show=False, legend_fontsize=5.5, s=25, palette=color_dict)
    plt.tight_layout()
    plt.savefig(f"{save_file_name}/tyser_mapping_to_4dataset_{attr}_mask{mask_dataset_label}.png", dpi=300)
    plt.show()
    plt.close()


def plot_tyser_mapping_to_4dataset_predictedTime(adata_all, save_file_name, mask_dataset_label="t", attr='predicted_time'):
    import matplotlib.pyplot as plt
    import matplotlib.cm as matcm
    from sklearn.preprocessing import MinMaxScaler
    import matplotlib as mpl
    # 为所有点计算归一化的时间值
    scaler = MinMaxScaler()
    # normalized_time = scaler.fit_transform(adata_all.obs[attr].values[:, np.newaxis]).ravel()
    predict_time_list_add_min_max = np.concatenate((np.array([3.0, 17.5]), adata_all.obs[attr].values))
    normalized_time = scaler.fit_transform(predict_time_list_add_min_max[:, np.newaxis]).ravel()

    # 初始化一个 RGBA 颜色数组，默认所有颜色为 'lightgray'
    colors = matcm.turbo(normalized_time)[2:, ]
    # 找出是 'l & m & p & z' 的点 and mask
    mask = adata_all.obs['data_type'] == mask_dataset_label
    # 应用颜色映射
    if mask_dataset_label == "t":
        colors[mask] = (0.9, 0.9, 0.9, 0)
    else:
        colors[mask] = (0.9, 0.9, 0.9, 0.7)
    plt.figure(figsize=(13, 10))
    plt.scatter(
        adata_all.obsm['X_umap'][:, 0],
        adata_all.obsm['X_umap'][:, 1],
        c=colors,  # 使用自定义颜色
        s=20,  # 点的大小
        # alpha=0.8  # 点的透明度
    )
    plt.title(f'UMAP colored by {attr}')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')

    # 创建一个颜色条
    # 定义颜色映射和数值范围
    cmap = plt.get_cmap('turbo')
    norm = mpl.colors.Normalize(vmin=3, vmax=17.5)
    scalar_mappable = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_mappable.set_array([])
    cbar = plt.colorbar(scalar_mappable, pad=0.01, fraction=0.03)
    # cbar = plt.colorbar(scattering)
    cbar.set_label('Predicted Time')
    cbar.set_alpha(1)
    cbar.draw_all()
    # 移除网格线
    ax = plt.gca()
    ax.grid(False)  # 禁用网格线
    plt.savefig(f"{save_file_name}/tyser_mapping_to_4dataset_{attr}_mask{mask_dataset_label}.png", dpi=300, )
    plt.show()
    plt.close()


def plot_tyser_mapping_to_4dataset_cellType_mask4dataset(adata_all, save_file_name, mask_dataset_label='l & m & p & z', attr="cell_typeMask4dataset"):
    import matplotlib.pyplot as plt
    # dataset_celltype_dic = adata_all.obs.groupby('data_type')[attr].apply(set).apply(list).to_dict()

    unique_categories = adata_all.obs[attr].unique()
    palette = sns.color_palette("tab20", len(unique_categories))  # 可以选择不同的调色板
    # mask = adata_all.obs['data_type'] == mask_dataset_label
    # 应用颜色映射
    # colors[mask] = [0.9, 0.9, 0.9, 1]
    color_map = {cat: color for cat, color in zip(unique_categories, palette)}
    color_map['l & m & p & z'] = (0.9, 0.9, 0.9, 0.7)
    # for _c in dataset_celltype_dic[mask_dataset_label]:
    #     color_map[_c] = (0.9, 0.9, 0.9)
    with plt.rc_context({'figure.figsize': (8, 5)}):
        sc.pl.umap(adata_all, color=attr, show=False, legend_fontsize=5.5, s=25, palette=color_map)
    plt.tight_layout()
    plt.savefig(f"{save_file_name}/tyser_mapping_to_4dataset_{attr}_mask{mask_dataset_label}.png", dpi=300)
    plt.show()
    plt.close()


def plot_tyser_mapping_to_4dataset_cellType_maskTyser(adata_all, save_file_name, mask_dataset_label='t', attr="cell_typeMaskTyser"):
    import matplotlib.pyplot as plt
    # dataset_celltype_dic = adata_all.obs.groupby('data_type')[attr].apply(set).apply(list).to_dict()

    unique_categories = adata_all.obs[attr].unique()
    palette = sns.color_palette("hsv", len(unique_categories))  # 可以选择不同的调色板
    # palette = plt.get_cmap('tab20b',len(unique_categories))
    # 应用颜色映射
    # colors[mask] = [0.9, 0.9, 0.9, 1]
    color_map = {cat: color for cat, color in zip(unique_categories, palette)}
    color_map['t'] = (0.9, 0.9, 0.9, 0)
    # for _c in dataset_celltype_dic[mask_dataset_label]:
    #     color_map[_c] = (0.9, 0.9, 0.9)
    with plt.rc_context({'figure.figsize': (8, 5)}):
        sc.pl.umap(adata_all, color=attr, show=False, legend_fontsize=5.5, s=25, palette=color_map)
    plt.tight_layout()
    plt.savefig(f"{save_file_name}/tyser_mapping_to_4dataset_{attr}_mask{mask_dataset_label}.png", dpi=300)
    plt.show()
    plt.close()


# def task_kFoldTest(donor_list, sc_expression_df, donor_dic, batch_dic,
#                    special_path_str, cell_time, time_standard_type,
#                    config, args, _logger):
#     save_path = _logger.root.handlers[0].baseFilename.replace('.log', '')
#     _logger.info(f"start task: k-fold test with {donor_list}.")
#     predict_donors_dic = dict()
#
#     for fold in range(len(donor_list)):
#         # fold=3
#         predict_donor_dic, test_clf_result, label_dic = one_fold_test(fold, donor_list, sc_expression_df, donor_dic,
#                                                                       batch_dic,
#                                                                       special_path_str, cell_time, time_standard_type,
#                                                                       config, args.train_epoch_num,
#                                                                       plot_trainingLossLine=True,
#                                                                       plot_latentSpaceUmap=False,
#                                                                       time_saved_asFloat=True, batch_size=int(args.batch_size),donor_str="day")
#         predict_donors_dic.update(predict_donor_dic)
#     predict_donors_df = pd.DataFrame(columns=["pseudotime"])
#     for fold in range(len(donor_list)):
#         predict_donors_df = pd.concat([predict_donors_df, predict_donors_dic[donor_list[fold]]])
#     predict_donors_df['predicted_time'] = predict_donors_df['pseudotime'].apply(denormalize, args=(min(label_dic.keys()) / 100, max(label_dic.keys()) / 100,
#                                                                                                    min(label_dic.values()), max(label_dic.values())))
#     cell_time = pd.concat([cell_time, predict_donors_df], axis=1)
#     cell_time.to_csv(f"{save_path}/k_fold_test_result.csv")
#
#     color_dic=plot_on_each_test_donor_violin_fromDF(cell_time.copy(), save_path, physical_str="predicted_time", x_str="time")
#
#
#     _logger.info("Finish plot image and fold-test.")
#     return predict_donors_dic, label_dic


if __name__ == '__main__':
    main()
