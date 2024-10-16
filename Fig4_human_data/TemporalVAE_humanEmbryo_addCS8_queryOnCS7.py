# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：TemporalVAE_humanEmbryo_addCS8_queryOnCS7.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/8/14 09:23 
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
import anndata as ad
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="TemporalVAE")
    parser.add_argument('--result_save_path', type=str,  # 2023-07-13 17:40:22
                        default="/Fig4_TemporalVAE_human_addCS8as18.5_queryOnCS7_240901/",
                        help="results all save here")
    parser.add_argument('--file_path', type=str,
                        default="/240405_preimplantation_Melania/",
                        help="sc file folder path.")
    parser.add_argument('--cs8_file_path', type=str,
                        default="/240322Human_embryo/xiaoCellCS8/",
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
                        default="supervise_vae_regressionclfdecoder_mouse_stereo",
                        help="vae model parameters file.")
    # ------------------ task setting ------------------
    parser.add_argument('--kfold_test', action="store_true", help="(Optional) make the task k fold test on dataset.", default=True)
    parser.add_argument('--train_whole_model', action="store_true", help="(Optional) use all data to train a model.", default=True)  # necessary here
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
    # ---------------------------- import vae model parameters from yaml file----------------------------------------------
    with open(yaml_path + "/" + args.vae_param_file + ".yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # ----------set logger and parameters, creat result save path and folder -----------------------------
    latent_dim = config['model_params']['latent_dim']
    # KNN_smooth_type = args.KNN_smooth_type

    time_standard_type = args.time_standard_type
    sc_data_file_csv = data_path + "/data_count_hvg_raw.csv"
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
    # trainData_renormalized_df, loss_gene_shortName_list, train_cell_info_df = predict_newData_preprocess_df(gene_dic, adata_query,
    #                                                                                                         min_gene_num=0,
    #                                                                                                         reference_file=f"{data_golbal_path}/{baseline_data_path}/data_count_hvg.csv",
    #                                                                                                         bool_change_geneID_to_geneShortName=False)
    # if "Melania" in sc_data_file_csv:
    #     data_raw_count_bool = False
    # else:
    #     data_raw_count_bool = True
    query_data_file_csv = "/240322Human_embryo/xiaoCellCS8/data_count_hvg.csv"
    query_cell_info_file_csv = "/240322Human_embryo/xiaoCellCS8/cell_with_time.csv"
    sc_expression_df, cell_time = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path,
                                                                                sc_data_file_csv,
                                                                                cell_info_file_csv,
                                                                                # donor_attr=donor_attr,
                                                                                # drop_out_donor=drop_out_donor,
                                                                                external_file_name=query_data_file_csv,
                                                                                external_cell_info_file=query_cell_info_file_csv,
                                                                                min_cell_num=args.min_cell_num,
                                                                                min_gene_num=args.min_gene_num,
                                                                                data_raw_count_bool=True)  # 2024-04-20 15:38:58

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
    # """2024-08-23 15:57:41 not use here, remove
    #  ---------------------- TASK: use reference data to train a model  ------------------------
    if args.train_whole_model:
        drop_out_donor = "t"  #
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
            batch_size=int(args.batch_size))  # 2023-10-24 17:44:31 batch as 10,000 due to overfit, batch size as 100,000 may have different result
        predict_donors_df = pd.DataFrame(train_clf_result, columns=["pseudotime"], index=cell_time_filter.index)
        predict_donors_df['predicted_time'] = predict_donors_df['pseudotime'].apply(denormalize, args=(min(label_dic.keys()) / 100, max(label_dic.keys()) / 100,
                                                                                                       min(label_dic.values()), max(label_dic.values())))
        cell_time_filter = pd.concat([cell_time_filter, predict_donors_df], axis=1)

        plt_image_adata = ad.AnnData(X=total_result["mu"].cpu().numpy())
        plt_image_adata.obs = cell_time_filter[["time", "predicted_time", "dataset_label", "cell_type", "day"]]

        plt_umap_byScanpy(plt_image_adata.copy(), ["time", "predicted_time", "dataset_label", "cell_type"], save_path=save_file_name, mode=None, figure_size=(5, 4),
                          color_map="turbo",
                          n_neighbors=50, n_pcs=20, special_file_name_str="n50_")  # color_map="viridis"
    # """
    ### ------------TASK : K-FOLD TEST--------------------------------------
    if args.kfold_test:
        test_donor_list = ["D_14_21_t"]
        predict_donors_dic, label_dic, kFold_result_recall_dic = task_kFoldTest(test_donor_list, sc_expression_df, donor_dic, batch_dic,
                                                                  special_path_str, cell_time, time_standard_type,
                                                                  config, args.train_epoch_num, _logger,
                                                                  donor_str="day",
                                                                  batch_size=args.batch_size, recall_predicted_mu=True)
        mu_result=kFold_result_recall_dic["D_14_21_t"][-1]
        train_mu_result, test_mu_result = mu_result
        cell_time_tyser = cell_time.loc[predict_donors_dic["D_14_21_t"].index]
        cell_time_tyser["predicted_time"] = predict_donors_dic["D_14_21_t"]['pseudotime'].apply(denormalize, args=(min(label_dic.keys()) / 100, max(label_dic.keys()) / 100,
                                                                                                                   min(label_dic.values()), max(label_dic.values())))
        cell_time_tyser = cell_time_tyser[["time", "predicted_time", "dataset_label", "day", "cell_type"]]
        adata_mu_tyser = ad.AnnData(X=test_mu_result.cpu().numpy(), obs=cell_time_tyser)
        adata_mu_tyser.obs['data_type'] = 't'

        # cell_time_filter = cell_time.drop(cell_drop_index_list, axis=0)
        # cell_time_filter = cell_time_filter.loc[sc_expression_df_filter.index]
        # cell_time_referenceDataset=
        # adata_mu_4dataset=ad.AnnData(X=train_mu_result.cpu().numpy(),obs=cell_time)
        try:
            adata_mu_4dataset = ad.read_h5ad(f"{save_file_name}/n50_latent_mu.h5ad")
        except:
            print("error on predict on query dataset. \n"
                  "Note: *TASK: use reference data to train a model* is necessary, "
                  "because it generate train's n50_latent_mu.h5ad file. \n"
                  "TASK : K-FOLD TEST is based on Function task_kFoldTest, "
                  "and it's difficult to return lantent_mu of each fold.")

        adata_mu_4dataset.obs['data_type'] = 'l & m & p & z & xiao'

        adata_all = anndata.concat([adata_mu_4dataset.copy(), adata_mu_tyser.copy()], axis=0)
        adata_all.obs["cell_typeMask4dataset"] = adata_all.obs.apply(lambda row: 'l & m & p & z & xiao' if row['dataset_label'] != 't' else row['cell_type'], axis=1)
        adata_all.obs["cell_typeMaskTyser"] = adata_all.obs.apply(lambda row: 't' if row['dataset_label'] == 't' else row['cell_type'], axis=1)

        # ---- 1 method: mapping tyser data to other 4 dataset's umap, just use different umap model
        # sc.pp.neighbors(adata_mu_tyser, n_neighbors=50, n_pcs=20)
        # sc.tl.umap(adata_mu_tyser, min_dist=0.75)
        # ----

        # ----2 method mapping tyser data to other 4 dataset's umap, use same umap model by 4 dataset,
        # Create a UMAP model instance
        import umap
        # reducer = umap.UMAP(n_neighbors=50, min_dist=0.75, n_components=2, random_state=101)
        reducer = umap.UMAP(n_neighbors=50, min_dist=0.75, n_components=2, random_state=0)
        embedding_4dataset = reducer.fit_transform(adata_mu_4dataset.X)
        embedding_tyser = reducer.transform(adata_mu_tyser.X)
        adata_mu_4dataset.obsm['X_umap'] = embedding_4dataset
        adata_mu_tyser.obsm['X_umap'] = embedding_tyser

        ### ---------------- Plot images ---------------
        reference_dataset_str = '&'.join(adata_all.obs['dataset_label'].unique().astype('str'))
        # combin two AnnData's UMAP loc
        adata_all.obsm["X_umap"] = np.vstack([adata_mu_4dataset.obsm['X_umap'], adata_mu_tyser.obsm['X_umap']])
        adata_all.write_h5ad(f"{save_file_name}/{reference_dataset_str}_mu.h5ad")
        # --- plot on Predict Time
        plot_tyser_mapping_to_4dataset_predictedTime(adata_all.copy(), save_file_name, label_dic,
                                                     mask_dataset_label='t', plot_attr='predicted_time',
                                                     reference_dataset_str=reference_dataset_str,
                                                     special_file_str="_maskT"
                                                     )
        plot_tyser_mapping_to_4dataset_predictedTime(adata_all.copy(), save_file_name, label_dic,
                                                     mask_dataset_label='l & m & p & z & xiao',
                                                     plot_attr='predicted_time',
                                                     reference_dataset_str=reference_dataset_str,
                                                     special_file_str="_maskl&m&p&z&xiao")
        # --- plot on cell type
        plot_tyser_mapping_to_datasets_attrCellType_maskTyser(adata_all.copy(), save_file_name, attr="cell_typeMask4dataset",
                                                              masked_str='l & m & p & z & xiao', color_palette="tab20",
                                                              legend_title="Cell type",
                                                              reference_dataset_str=reference_dataset_str,
                                                              special_file_str='_maskl&m&p&z&xiao')
        plot_tyser_mapping_to_datasets_attrCellType_maskTyser(adata_all.copy(), save_file_name, attr="cell_typeMaskTyser",
                                                              masked_str="t", color_palette="hsv",
                                                              legend_title="Cell type",
                                                              reference_dataset_str=reference_dataset_str,
                                                              special_file_str='_maskT')
        # --- plot on dataset
        plot_tyser_mapping_to_datasets_attrDataset(adata_all.copy(), save_file_name,
                                                   attr="dataset_label", masked_str='t',
                                                   color_dic={'l': "#B292CA",
                                                              'm': '#7ED957',
                                                              'p': '#FFC947',
                                                              'z': '#00CED1',
                                                              'Xiao': '#E06377',
                                                              't': (0.9, 0.9, 0.9, 0.7)},
                                                   legend_title="Dataset",
                                                   reference_dataset_str=reference_dataset_str,
                                                   special_file_str="_maskT")
        plot_tyser_mapping_to_datasets_attrDataset(adata_all.copy(), save_file_name,
                                                   attr="data_type", masked_str='l & m & p & z & xiao',
                                                   color_dic={'l & m & p & z & xiao': (0.9, 0.9, 0.9, 0.7),
                                                              't': "#E06D83"},
                                                   reference_dataset_str=reference_dataset_str,
                                                   legend_title="Dataset", special_file_str="_maskl&m&p&z&xiao")

        # --- plot on time categorical
        plot_tyser_mapping_to_datasets_attrTimeGT(adata_all.copy(), save_file_name, attr='time',
                                                  query_timePoint='17.5',
                                                  legend_title="Cell stage",
                                                  mask_dataset_label="t",
                                                  reference_dataset_str=reference_dataset_str,
                                                  special_file_str='_maskT')
        plot_tyser_mapping_to_datasets_attrTimeGT(adata_all.copy(), save_file_name, attr='time',
                                                  query_timePoint='17.5',
                                                  legend_title="Cell stage",
                                                  mask_dataset_label="l & m & p & z & xiao",
                                                  reference_dataset_str=reference_dataset_str,
                                                  special_file_str='_maskl&m&p&z&xiao')

        # with plt.rc_context({'figure.figsize': (6, 5)}):
        #     sc.pl.umap(adata_all, color="data_type", show=False, legend_fontsize=5.5, s=25, palette={'l & m & p & z & xiao': (0.9, 0.9, 0.9, 0.7), 't': "#E06D83"})
        # plt.tight_layout()
        # plt.savefig(f"{save_file_name}/tyser_mapping_to_4dataset_datatype_maskl & m & p & z & xiao.png", dpi=300)
        # plt.show()
        # plt.close()

        # ---- combine and plot umap
        # adata_all.obs["time"] = adata_all.obs["time"].astype("str")
        # with plt.rc_context({'figure.figsize': (6, 5)}):
        #     sc.pl.umap(adata_all, color="time", show=False, legend_fontsize=5.5, s=25, palette="turbo")
        # plt.tight_layout()
        # plt.savefig(f"{save_file_name}/combine_tyser_to4dataset_predictedTime.png", dpi=300)
        # plt.show()
        # plt.close()

        # plt_umap_byScanpy(adata_all.copy(), ["time", "predicted_time", "dataset_label", "day", "cell_type", "cell_typeMask4dataset"],
        #                   save_path=save_file_name, mode=None, figure_size=(13, 4),
        #                   color_map="turbo", show_in_row=False,
        #                   n_neighbors=50, n_pcs=20, special_file_name_str="combine_tyser_mapping_to_n50_4dataset")
        import gc
        gc.collect()
        _logger.info("Finish fold-test.")

    _logger.info("Finish all.")




if __name__ == '__main__':
    main()
