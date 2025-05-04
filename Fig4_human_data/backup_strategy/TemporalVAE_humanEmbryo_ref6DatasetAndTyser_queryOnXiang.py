# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：TemporalVAE_humanEmbryo_ref6dataset_queryOnTyser.py
@Author  ：awa121
@Date    ：2025/3/21 8:05 

Description: 
"""

import os
import sys

if os.getcwd().split("/")[-1] != "TemporalVAE":
    os.chdir("../..")
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
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="TemporalVAE")
    parser.add_argument('--result_save_path', type=str,  # 2023-07-13 17:40:22
                        default="/Fig4_TemporalVAE_humanEmbryo_ref6DatasetAndTyser_queryOnXiang_epoch95/",
                        help="results all save here")
    parser.add_argument('--file_path', type=str,
                        default="/human_embryo_preimplantation/integration_8dataset/",
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
                        default="95",  # original 100
                        help="Train epoch num")
    parser.add_argument('--batch_size', type=int,
                        default=100000,  # original 100000
                        help="batch size")
    parser.add_argument('--time_standard_type', type=str,
                        default="embryoneg1to1",
                        # default="embryoneg5to5", #  2025-03-27 15:27:56 original
                        help="y_time_nor_train standard type may cause different latent space: "
                             "log2, 0to1, neg1to1, labeldic,sigmoid,logit")
    # supervise_vae            supervise_vae_regressionclfdecoder
    parser.add_argument('--vae_param_file', type=str,
                        default="supervise_vae_regressionclfdecoder_mouse_stereo",
                        help="vae model parameters file.")
    # ------------------ task setting ------------------
    # parser.add_argument('--kfold_test', action="store_true", help="(Optional) make the task k fold test on dataset.", default=True)
    # parser.add_argument('--train_whole_model', action="store_true", help="(Optional) use all data to train a model.", default=True)  # necessary here
    # parser.add_argument('--identify_time_cor_gene', action="store_true", help="(Optional) identify time-cor gene by model trained by all.", default=False)

    args = parser.parse_args()

    data_golbal_path = "data/"  # dataset all save at folder "data"
    result_save_path = "results/" + args.result_save_path + "/"
    yaml_path = "vae_model_configs/"
    # ---------------------------- import vae model parameters from yaml file----------------------------------------------
    with open(yaml_path + "/" + args.vae_param_file + ".yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    # ----------set logger and parameters, creat result save path and folder -----------------------------
    latent_dim = config['model_params']['latent_dim']

    time_standard_type = args.time_standard_type
    sc_data_file_csv = args.file_path + 'rawCount_Z&C&Xiao&M&P&Liu&Tyser&Xiang.h5ad'
    _path = '{}/{}/'.format(result_save_path, args.file_path)
    if not os.path.exists(_path):
        os.makedirs(_path)

    logger_file = f'{_path}/{args.vae_param_file}_dim{latent_dim}_time{time_standard_type}_epoch{args.train_epoch_num}_batchSize{args.batch_size}_minGeneNum{args.min_gene_num}.log'
    LogHelper.setup(log_path=logger_file, level='INFO')
    _logger = logging.getLogger(__name__)
    _logger.info("Finished setting up the logger at: {}.".format(logger_file))
    _logger.info("Train on dataset: {}.".format(data_golbal_path + args.file_path))
    _logger.info("load vae model parameters from file: {}".format(yaml_path + args.vae_param_file + ".yaml"))

    # ------------ Preprocess data, with hvg gene from preprocess_data_mouse_embryonic_development.py------------------------
    # query_data_file_csv = "/human_embryo_preimplantation/XiaoCS8/data_count_hvg.csv"
    # query_cell_info_file_csv = "/human_embryo_preimplantation/XiaoCS8/cell_with_time.csv"
    sc_expression_df, cell_time = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path,
                                                                                sc_data_file_csv,
                                                                                cell_info_file=None,
                                                                                # donor_attr=donor_attr,
                                                                                # drop_out_donor=drop_out_donor,
                                                                                # external_file_name=query_data_file_csv,
                                                                                # external_cell_info_file=query_cell_info_file_csv,
                                                                                min_cell_num=args.min_cell_num,
                                                                                min_gene_num=args.min_gene_num,
                                                                                data_raw_count_bool=True)  # 2024-04-20 15:38:58

    special_path_str = ""
    # ---------------------------------------- set donor list and dictionary -----------------------------------------------------
    donor_list = np.unique(cell_time["dataset_label"])
    donor_list = sorted(donor_list, key=Embryodonor_resort_key)
    donor_dic = dict()
    for i in range(len(donor_list)):
        donor_dic[donor_list[i]] = i
    batch_dic = donor_dic.copy()
    _logger.info("Consider donor as batch effect, donor use label: {}".format(donor_dic))
    _logger.info("For each donor (donor_id, cell_num):{} ".format(Counter(cell_time["day"])))
    save_path = f"{_logger.root.handlers[0].baseFilename.replace('.log', '')}/"
    # """2024-08-23 15:57:41 not use here, remove

    #  ---------------------- TASK: use reference data to train a model  ------------------------
    drop_out_donor = ["Xiang"]  #
    print(f"drop the donor: {drop_out_donor}")
    cell_drop_index_list = cell_time.loc[cell_time["dataset_label"].isin(drop_out_donor)].index
    sc_expression_df_filter = sc_expression_df.drop(cell_drop_index_list, axis=0)
    cell_time_filter = cell_time.drop(cell_drop_index_list, axis=0)
    cell_time_filter = cell_time_filter.loc[sc_expression_df_filter.index]
    sc_expression_train, y_time_nor_train, donor_index_train, runner, experiment, _m, train_clf_result, label_dic, total_result = onlyTrain_model(
        sc_expression_df_filter, donor_dic,
        special_path_str,
        cell_time_filter,
        time_standard_type, config, args,
        plot_latentSpaceUmap=False, plot_trainingLossLine=True, time_saved_asFloat=True, batch_dic=batch_dic, donor_str="dataset_label",
        batch_size=int(args.batch_size))  # 2023-10-24 17:44:31 batch as 10,000 due to overfit, batch size as 100,000 may have different result
    predict_donors_df = pd.DataFrame(train_clf_result, columns=["pseudotime"], index=cell_time_filter.index)
    predict_donors_df['predicted_time'] = predict_donors_df['pseudotime'].apply(denormalize, args=(min(label_dic.keys()) / 100, max(label_dic.keys()) / 100,
                                                                                                   min(label_dic.values()), max(label_dic.values())))
    cell_time_filter = pd.concat([cell_time_filter, predict_donors_df], axis=1)

    adata_mu_reference = ad.AnnData(X=total_result["mu"].cpu().numpy())
    adata_mu_reference.obs = cell_time_filter[["time", "predicted_time", "dataset_label", "cell_type", "day"]]

    plt_umap_byScanpy(adata_mu_reference.copy(), ["time", "predicted_time", "dataset_label", "cell_type"], save_path=save_path, mode=None, figure_size=(5, 4),
                      color_map="turbo",
                      n_neighbors=50, n_pcs=20, special_file_name_str="n50_")  # color_map="viridis"
    # add reference and query annotation
    adata_mu_reference.obs['data_type'] = 'L & M & P & Z & Xiao & C & T'
    print(f"{Counter(adata_mu_reference.obs['dataset_label'])}")

    # adata_subset=from_adata_randomSelect_cells_equityTimePoint(adata_mu_reference, random_select_n_timePoint=200, random_seed=0)
    # first use subset of reference data ( 2025-04-14 15:34:22 6 dataset) to train a umap space,
    # second map all reference data, Tyser, Xiang19 to this umap space, respectively.
    # queryOneDataset_referenceOn6Datasets_humanEmbryo("T",
    #                                                  cell_time, sc_expression_df,
    #                                                  time_standard_type, label_dic, batch_dic,
    #                                                  runner, experiment, adata_subset.copy(), _logger, save_path,
    #                                                  special_file_name=f"_subRefTime{random_select_n_timePoint}",umap_n_neighbors=20,
    #                                                  )
    # queryOneDataset_referenceOn6Datasets_humanEmbryo("Xiang",
    #                                                  cell_time, sc_expression_df,
    #                                                  time_standard_type, label_dic, batch_dic,
    #                                                  runner, experiment, adata_subset.copy(), _logger, save_path,
    #                                                  special_file_name=f"_subRefTime{random_select_n_timePoint}",umap_n_neighbors=100,
    #                                                  )
    umap_n_neighbors = 50
    import umap
    umap_reducer = umap.UMAP(n_neighbors=umap_n_neighbors, min_dist=0.75, n_components=2, random_state=0)
    print(f"{Counter(adata_mu_reference.obs['dataset_label'])}")
    print(f"{Counter(adata_mu_reference.obs['time'])}")

    adata_mu_reference.obsm['X_umap'] = umap_reducer.fit_transform(adata_mu_reference.X)
    adata_mu_reference.obsm['X_umap'] = umap_reducer.transform(adata_mu_reference.X)
    # queryOneDataset_referenceOn6Datasets_humanEmbryo("T",
    #                                                  cell_time, sc_expression_df,
    #                                                  time_standard_type, label_dic, batch_dic,
    #                                                  runner, experiment, adata_mu_reference.copy(), _logger, save_path,
    #                                                  umap_reducer,
    #                                                  special_file_name=f"_umapNei{umap_n_neighbors}"
    #                                                  )
    queryOneDataset_referenceOn6Datasets_humanEmbryo("Xiang",
                                                     cell_time, sc_expression_df,
                                                     time_standard_type, label_dic, batch_dic,
                                                     runner, experiment, adata_mu_reference.copy(), _logger, save_path,
                                                     umap_reducer,
                                                     special_file_name=f"_umapNei{umap_n_neighbors}"
                                                     )
    # less cell to train umap reducer.
    umap_n_neighbors = 20
    random_select_n_timePoint = 200
    random_seed = 0
    adata_subset = from_adata_randomSelect_cells_equityTimePoint(adata_mu_reference, random_select_n_timePoint=random_select_n_timePoint, random_seed=random_seed)
    print(f"umap_n_neighbors: {umap_n_neighbors}, random_select_n_timePoint: {random_select_n_timePoint},random seed: {random_seed}")
    umap_reducer = umap.UMAP(n_neighbors=umap_n_neighbors, min_dist=0.75, n_components=2, random_state=0)
    adata_subset.obsm['X_umap'] = umap_reducer.fit_transform(adata_subset.X)
    adata_mu_reference.obsm['X_umap'] = umap_reducer.transform(adata_mu_reference.X)
    # queryOneDataset_referenceOn6Datasets_humanEmbryo("T",
    #                                                  cell_time, sc_expression_df,
    #                                                  time_standard_type, label_dic, batch_dic,
    #                                                  runner, experiment, adata_mu_reference.copy(), _logger, save_path,
    #                                                  umap_reducer,
    #                                                  special_file_name=f"_subCell{random_select_n_timePoint}_umapNei{umap_n_neighbors}"
    #                                                  )
    queryOneDataset_referenceOn6Datasets_humanEmbryo("Xiang",
                                                     cell_time, sc_expression_df,
                                                     time_standard_type, label_dic, batch_dic,
                                                     runner, experiment, adata_mu_reference.copy(), _logger, save_path,
                                                     umap_reducer,
                                                     special_file_name=f"_subCell{random_select_n_timePoint}_umapNei{umap_n_neighbors}"
                                                     )

    # """
    ### ------------TASK : K-FOLD TEST--------------------------------------
    # if args.kfold_test:
    #     queryOneDataset_dropOutOneDataset(["T"],
    #                                       ["Xiang"],
    #                                       cell_time, sc_expression_df, donor_dic, batch_dic, special_path_str, time_standard_type,
    #                                       config, args, _logger, save_file_name
    #                                       )
    #     queryOneDataset_dropOutOneDataset(["Xiang"],
    #                                       ["T"],
    #                                       cell_time, sc_expression_df, donor_dic, batch_dic, special_path_str, time_standard_type,
    #                                       config, args, _logger, save_file_name
    #                                       )

    _logger.info("Finish all.")
    return


def from_adata_randomSelect_cells_equityTimePoint(adata_mu_reference, random_select_n_timePoint=200, random_seed=0):
    # Randomly select 200 variables (genes) from adata
    np.random.seed(random_seed)

    # Ensure 'time' is categorical
    time_categories = adata_mu_reference.obs['time'].astype('category').cat.categories

    # Initialize a list to store selected cell indices
    selected_cell_indices = []

    # Iterate over each time point
    for time_point in time_categories:
        # Get indices of cells for the current time point
        time_mask = adata_mu_reference.obs['time'] == time_point
        cell_indices = np.where(time_mask)[0]  # or use adata_mu_reference.obs.index[time_mask]

        # Determine how many cells to sample (min between 200 and available cells)
        n_cells = min(random_select_n_timePoint, len(cell_indices))

        # Randomly select cells without replacement
        selected_indices = np.random.choice(cell_indices, size=n_cells, replace=False)
        selected_cell_indices.extend(selected_indices)

    # Subset the AnnData object using the selected indices
    adata_subset = adata_mu_reference[selected_cell_indices, :].copy()  # .copy() to avoid warnings

    # Optional: Reset indices if needed
    adata_subset.obs.reset_index(drop=True, inplace=True)
    print(f"Before random select:\n\t"
          f"{Counter(adata_mu_reference.obs['dataset_label'])}\n\t "
          f"{Counter(adata_mu_reference.obs['time'])}\n\t"
          f"{Counter(adata_mu_reference.obs['cell_type'])}\n"
          f"After random select:"
          f"{Counter(adata_subset.obs['dataset_label'])}\n \t "
          f"{Counter(adata_subset.obs['time'])}\n\t"
          f"{Counter(adata_subset.obs['cell_type'])}")
    return adata_subset


def queryOneDataset_referenceOn6Datasets_humanEmbryo(test_donor,
                                                     cell_time, sc_expression_df,
                                                     time_standard_type, label_dic, batch_dic,
                                                     runner, experiment, adata_mu_reference, _logger,
                                                     save_path,
                                                     umap_reducer,
                                                     special_file_name='',

                                                     # umap_n_neighbors=50,
                                                     # umap_space_withSubsetRef=False,random_select_n_timePoint=200,
                                                     ):
    # ---- set query data and predict on query data
    cell_time_dic = dict(zip(cell_time.index, cell_time['time']))
    sc_expression_test = sc_expression_df.loc[cell_time.index[cell_time['dataset_label'] == test_donor]]
    x_sc_test = torch.tensor(sc_expression_test.values, dtype=torch.get_default_dtype()).t()
    y_time_test = x_sc_test.new_tensor(np.array(sc_expression_test.index.map(cell_time_dic) * 100).astype(int))
    try:
        y_time_nor_test, label_dic = trans_time(y_time_test, time_standard_type, label_dic_train=label_dic)
    except:
        print("error")
    x_sc_test = torch.tensor(sc_expression_test.values, dtype=torch.get_default_dtype()).t()
    donor_index_test = x_sc_test.new_tensor(
        [int(batch_dic[cell_time.loc[_cell_name]['dataset_label']]) for _cell_name in sc_expression_test.index.values])
    test_data = [[x_sc_test[:, i], y_time_nor_test[i], donor_index_test[i]] for i in range(x_sc_test.shape[1])]
    from model_master.dataset import SupervisedVAEDataset_onlyPredict
    data_test = SupervisedVAEDataset_onlyPredict(predict_data=test_data, predict_batch_size=len(test_data))
    test_result = runner.predict(experiment, data_test)
    test_clf_result, test_mu_result, test_latent_log_var_result = test_result[0][0], test_result[0][1], \
        test_result[0][2]
    if test_clf_result.shape[1] == 1:
        # time is continues, supervise_vae_regressionclfdecoder  supervise_vae_regressionclfdecoder_of_sublatentspace
        _logger.info("predicted time of test donor is continuous.")
        import pandas as pd
        test_clf_result = pd.DataFrame(data=np.squeeze(test_clf_result, axis=1), index=sc_expression_test.index, columns=["pseudotime"])

    cell_time_tyser = cell_time.loc[sc_expression_test.index]
    cell_time_tyser["predicted_time"] = test_clf_result.apply(denormalize, args=(min(label_dic.keys()) / 100, max(label_dic.keys()) / 100,
                                                                                 min(label_dic.values()), max(label_dic.values())))
    cell_time_tyser = cell_time_tyser[["time", "predicted_time", "dataset_label", "day", "cell_type"]]
    adata_mu_query = ad.AnnData(X=test_mu_result.cpu().numpy(), obs=cell_time_tyser)
    adata_mu_query.obs['data_type'] = test_donor
    # ----

    # ---- combine low-dim representation of reference and query data
    adata_combined = anndata.concat([adata_mu_reference.copy(), adata_mu_query.copy()], axis=0)

    adata_combined.obs["cell_typeMask4dataset"] = adata_combined.obs.apply(lambda row: 'L & M & P & Z & Xiao & C & T' if row['dataset_label'] != test_donor else row['cell_type'],
                                                                           axis=1)
    adata_combined.obs["cell_typeMaskTyser"] = adata_combined.obs.apply(lambda row: test_donor if row['dataset_label'] == test_donor else row['cell_type'], axis=1)
    # ----

    # ---- plot predicted violin images.
    plot_violin_240223(adata_combined.obs.copy(),
                       save_path,
                       x_attr="time",
                       y_attr="predicted_time",
                       special_file_name=f"queryOn{test_donor}_violinAll{special_file_name}")
    plot_violin_240223(adata_combined.obs.loc[adata_combined.obs.index[adata_combined.obs['dataset_label'] == test_donor]].copy(),
                       save_path,
                       x_attr="time",
                       y_attr="predicted_time",
                       special_file_name=f"queryOn{test_donor}_violoinTest{special_file_name}")
    plot_violin_240223(adata_combined.obs.loc[adata_combined.obs.index[adata_combined.obs['dataset_label'] != test_donor]].copy(),
                       save_path,
                       x_attr="time",
                       y_attr="predicted_time",
                       special_file_name=f"queryOn{test_donor}_violoinTrain{special_file_name}")

    # ---- 1 method: mapping tyser data to other 4 dataset's umap, just use different umap model
    # sc.pp.neighbors(adata_mu_query, n_neighbors=50, n_pcs=20)
    # sc.tl.umap(adata_mu_query, min_dist=0.75)
    # ----2 method mapping tyser data to other 4 dataset's umap, use same umap model by 4 dataset,
    # Create a UMAP model instance
    # import umap
    # reducer = umap.UMAP(n_neighbors=umap_n_neighbors, min_dist=0.75, n_components=2, random_state=0)
    # reducer = umap.UMAP(n_neighbors=50, min_dist=0.75, n_components=2, random_state=101)
    # reducer = umap.UMAP(n_neighbors=15, min_dist=0.75, n_components=2, random_state=10)
    print(f"{Counter(adata_mu_reference.obs['dataset_label'])}")
    print(f"{Counter(adata_mu_reference.obs['time'])}")

    # adata_mu_reference.obsm['X_umap'] = reducer.fit_transform(adata_mu_reference.X)
    adata_mu_query.obsm['X_umap'] = umap_reducer.transform(adata_mu_query.X)

    ### ---------------- Plot images ---------------
    reference_dataset_str = '&'.join(adata_combined.obs['dataset_label'].unique().astype('str'))
    # combin two AnnData's UMAP loc
    adata_combined.obsm["X_umap"] = np.vstack([adata_mu_reference.obsm['X_umap'], adata_mu_query.obsm['X_umap']])
    # clear unused categories
    adata_combined.obs["dataset_label"] = adata_combined.obs["dataset_label"].astype('category').cat.remove_unused_categories()
    # save reference and query low-dim .h5ad, includes .obsm["X_umap"]
    adata_combined.write_h5ad(f"{save_path}/{reference_dataset_str}_mu{special_file_name}.h5ad")
    print(f"Final plot dataset information: {Counter(adata_combined.obs['dataset_label'])}")

    # --- plot on cell type

    plot_tyser_mapping_to_datasets_attrCellType_maskTyser(adata_combined.copy(), save_path, attr="cell_typeMaskTyser",
                                                          masked_str=test_donor, color_palette="hsv",
                                                          legend_title="Cell type",
                                                          reference_dataset_str=reference_dataset_str,
                                                          special_file_str=f'_mask{test_donor}_query{test_donor}{special_file_name}', top_vis_cellType_num=15,
                                                          )
    plot_tyser_mapping_to_datasets_attrCellType_maskTyser(adata_combined.copy(), save_path, attr="cell_typeMask4dataset",
                                                          masked_str='L & M & P & Z & Xiao & C & T', color_palette="tab20",
                                                          legend_title="Cell type",
                                                          reference_dataset_str=reference_dataset_str,
                                                          special_file_str=f'_maskL&M&P&Z&Xiao&C_query{test_donor}{special_file_name}',
                                                          query_donor=test_donor, top_vis_cellType_num=15)
    # --- plot on dataset observed cell stage
    plot_query_mapping_to_referenceUmapSpace_attrTimeGT(adata_combined.copy(), save_path, plot_attr='time',
                                                        legend_title=f"Cell stage\nof Ref.",
                                                        mask_dataset_label=test_donor,
                                                        reference_dataset_str=reference_dataset_str,
                                                        special_file_str=f'_cellStageOnDataset_mask{test_donor}_query{test_donor}{special_file_name}')
    plot_query_mapping_to_referenceUmapSpace_attrTimeGT(adata_combined.copy(), save_path, plot_attr='time',
                                                        legend_title=f"Cell stage\nof {test_donor}",
                                                        mask_dataset_label=['L', 'M', 'P', 'Z', 'Xiao', 'C'],
                                                        reference_dataset_str=reference_dataset_str,
                                                        special_file_str=f'_cellStageOnDataset_maskL&M&P&Z&Xiao&C_query{test_donor}{special_file_name}')

    # --- plot on time categorical
    # plot_tyser_mapping_to_datasets_attrTimeGT(adata_combined.copy(), save_file_name, plot_attr='time',
    #                                           query_timePoint='17.5',
    #                                           legend_title="Cell stage",
    #                                           mask_dataset_label=test_donor,
    #                                           reference_dataset_str=reference_dataset_str,
    #                                           special_file_str=f'_mask{test_donor}_query{test_donor}')
    # plot_tyser_mapping_to_datasets_attrTimeGT(adata_combined.copy(), save_file_name, plot_attr='time',
    #                                           query_timePoint='17.5',
    #                                           legend_title="Cell stage",
    #                                           mask_dataset_label="Liu & Lv & M & P & Z & Xiao & C",
    #                                           reference_dataset_str=reference_dataset_str,
    #                                           special_file_str=f'_maskL&M&P&Z&Xiao&C_query{test_donor}')
    # --- plot on Predict Time
    plot_tyser_mapping_to_4dataset_predictedTime(adata_combined.copy(), save_path, label_dic,
                                                 mask_dataset_label=test_donor, plot_attr='predicted_time',
                                                 reference_dataset_str=reference_dataset_str,
                                                 special_file_str=f"_mask{test_donor}_query{test_donor}{special_file_name}"
                                                 )
    plot_tyser_mapping_to_4dataset_predictedTime(adata_combined.copy(), save_path, label_dic,
                                                 mask_dataset_label='L & M & P & Z & Xiao & C & T',
                                                 plot_attr='predicted_time',
                                                 reference_dataset_str=reference_dataset_str,
                                                 special_file_str=f"_maskL&M&P&Z&Xiao&C_query{test_donor}{special_file_name}")

    # --- plot on dataset

    plot_tyser_mapping_to_datasets_attrDataset(adata_combined.copy(), save_path,
                                               attr="dataset_label", masked_str=test_donor,
                                               color_dic={'L': '#E06377',
                                                          'M': '#7ED957',
                                                          'P': '#FFC947',
                                                          'Z': '#00CED1',
                                                          'Xiao': "#B292CA",
                                                          'C': '#c76f00',
                                                          'T': '#8f5239',
                                                          test_donor: (0.9, 0.9, 0.9, 0.7)},
                                               legend_title="Dataset",
                                               reference_dataset_str=reference_dataset_str,
                                               special_file_str=f"_mask{test_donor}_query{test_donor}{special_file_name}")
    plot_tyser_mapping_to_datasets_attrDataset(adata_combined.copy(), save_path,
                                               attr="data_type", masked_str='L & M & P & Z & Xiao & C & T',
                                               color_dic={'L & M & P & Z & Xiao & C & T': (0.9, 0.9, 0.9, 0.7),
                                                          test_donor: "#E06D83"},
                                               reference_dataset_str=reference_dataset_str,
                                               legend_title="Dataset", special_file_str=f"_maskL&M&P&Z&Xiao&C_query{test_donor}{special_file_name}")

    # if umap_space_withSubsetRef:
    #     adata_subset=from_adata_randomSelect_cells_equityTimePoint(adata_mu_reference, random_select_n_timePoint=random_select_n_timePoint, random_seed=0)
    #     reducer = umap.UMAP(n_neighbors=umap_n_neighbors, min_dist=0.75, n_components=2, random_state=0)
    #     adata_subset.obsm['X_umap'] = reducer.fit_transform(adata_subset.X)
    #     adata_mu_query.obsm['X_umap'] = reducer.transform(adata_mu_query.X)
    #     adata_combined_subMapping=adata_combined.copy()
    #     adata_combined_subMapping.obsm["X_umap"] =  reducer.transform(adata_combined_subMapping.X)
    #     plot_tyser_mapping_to_datasets_attrCellType_maskTyser(adata_combined_subMapping.copy(), save_path, attr="cell_typeMask4dataset",
    #                                                           masked_str='L & M & P & Z & Xiao & C & T', color_palette="tab20",
    #                                                           legend_title="Cell type",
    #                                                           reference_dataset_str=reference_dataset_str,
    #                                                           special_file_str=f'_maskL&M&P&Z&Xiao&C_query{test_donor}_subMapping_{special_file_name}',
    #                                                           query_donor=test_donor)
    #     plot_tyser_mapping_to_datasets_attrCellType_maskTyser(adata_combined_subMapping.copy(), save_path, attr="cell_typeMaskTyser",
    #                                                           masked_str=test_donor, color_palette="hsv",
    #                                                           legend_title="Cell type",
    #                                                           reference_dataset_str=reference_dataset_str,
    #                                                           special_file_str=f'_mask{test_donor}_subMapping_query{test_donor}{special_file_name}')
    import gc
    gc.collect()


def queryOneDataset_dropOutOneDataset(test_donor_list, drop_out_donor,
                                      cell_time, sc_expression_df, donor_dic, batch_dic, special_path_str, time_standard_type,
                                      config, args, _logger, save_file_name,
                                      k_fold_attr_str="dataset_label"
                                      ):
    test_donor = test_donor_list[0]
    print(f"query on {test_donor_list}, drop {drop_out_donor}")

    cell_drop_index_list = cell_time.loc[cell_time["dataset_label"].isin(drop_out_donor)].index
    sc_expression_df_filter = sc_expression_df.drop(cell_drop_index_list, axis=0)
    cell_time_filter = cell_time.drop(cell_drop_index_list, axis=0)
    cell_time_filter = cell_time_filter.loc[sc_expression_df_filter.index]
    predict_donors_dic, label_dic, kFold_result_recall_dic = task_kFoldTest(test_donor_list, sc_expression_df_filter, donor_dic, batch_dic,
                                                                            special_path_str, cell_time_filter, time_standard_type,
                                                                            config, args.train_epoch_num, _logger,
                                                                            donor_str=k_fold_attr_str,
                                                                            batch_size=args.batch_size, recall_predicted_mu=True)
    mu_result = kFold_result_recall_dic[test_donor][-1]
    train_mu_result, test_mu_result = mu_result
    cell_time_tyser = cell_time.loc[predict_donors_dic[test_donor].index]
    cell_time_tyser["predicted_time"] = predict_donors_dic[test_donor]['pseudotime'].apply(denormalize, args=(min(label_dic.keys()) / 100, max(label_dic.keys()) / 100,
                                                                                                              min(label_dic.values()), max(label_dic.values())))
    cell_time_tyser = cell_time_tyser[["time", "predicted_time", "dataset_label", "day", "cell_type"]]
    adata_mu_tyser = ad.AnnData(X=test_mu_result.cpu().numpy(), obs=cell_time_tyser)
    adata_mu_tyser.obs['data_type'] = test_donor

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

    adata_mu_4dataset.obs['data_type'] = 'L & M & P & Z & Xiao & C & T'
    print(f"{Counter(adata_mu_4dataset.obs['dataset_label'])}")
    # adata_mu_4dataset = downsampling_specificDataset(adata_mu_4dataset.copy(), "C")
    # adata_mu_4dataset = downsampling_specificDataset(adata_mu_4dataset.copy(), "X")

    adata_all = anndata.concat([adata_mu_4dataset.copy(), adata_mu_tyser.copy()], axis=0)

    adata_all.obs["cell_typeMask4dataset"] = adata_all.obs.apply(lambda row: 'L & M & P & Z & Xiao & C & T' if row['dataset_label'] != test_donor else row['cell_type'], axis=1)
    adata_all.obs["cell_typeMaskTyser"] = adata_all.obs.apply(lambda row: test_donor if row['dataset_label'] == test_donor else row['cell_type'], axis=1)

    # ---- 1 method: mapping tyser data to other 4 dataset's umap, just use different umap model
    # sc.pp.neighbors(adata_mu_tyser, n_neighbors=50, n_pcs=20)
    # sc.tl.umap(adata_mu_tyser, min_dist=0.75)
    # ----

    # ----2 method mapping tyser data to other 4 dataset's umap, use same umap model by 4 dataset,
    # Create a UMAP model instance
    import umap
    # reducer = umap.UMAP(n_neighbors=50, min_dist=0.75, n_components=2, random_state=101)
    # reducer = umap.UMAP(n_neighbors=15, min_dist=0.75, n_components=2, random_state=10)
    reducer = umap.UMAP(n_neighbors=50, min_dist=0.75, n_components=2, random_state=0)
    print(f"{Counter(adata_mu_4dataset.obs['dataset_label'])}")

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
                                                 mask_dataset_label=test_donor, plot_attr='predicted_time',
                                                 reference_dataset_str=reference_dataset_str,
                                                 special_file_str=f"_mask{test_donor}_query{test_donor}"
                                                 )
    plot_tyser_mapping_to_4dataset_predictedTime(adata_all.copy(), save_file_name, label_dic,
                                                 mask_dataset_label='L & M & P & Z & Xiao & C & T',
                                                 plot_attr='predicted_time',
                                                 reference_dataset_str=reference_dataset_str,
                                                 special_file_str=f"_maskL&M&P&Z&Xiao&C_query{test_donor}")
    # --- plot on cell type
    plot_tyser_mapping_to_datasets_attrCellType_maskTyser(adata_all.copy(), save_file_name, attr="cell_typeMask4dataset",
                                                          masked_str='L & M & P & Z & Xiao & C & T', color_palette="tab20",
                                                          legend_title="Cell type",
                                                          reference_dataset_str=reference_dataset_str,
                                                          special_file_str=f'_maskL&M&P&Z&Xiao&C_query{test_donor}')
    plot_tyser_mapping_to_datasets_attrCellType_maskTyser(adata_all.copy(), save_file_name, attr="cell_typeMaskTyser",
                                                          masked_str=test_donor, color_palette="hsv",
                                                          legend_title="Cell type",
                                                          reference_dataset_str=reference_dataset_str,
                                                          special_file_str=f'_mask{test_donor}_query{test_donor}')
    # --- plot on dataset
    plot_tyser_mapping_to_datasets_attrDataset(adata_all.copy(), save_file_name,
                                               attr="dataset_label", masked_str='T',
                                               color_dic={'L': "#B292CA",
                                                          'M': '#7ED957',
                                                          'P': '#FFC947',
                                                          'Z': '#00CED1',
                                                          'Xiao': '#E06377',
                                                          'C': '#c76f00',
                                                          test_donor: (0.9, 0.9, 0.9, 0.7)},
                                               legend_title="Dataset",
                                               reference_dataset_str=reference_dataset_str,
                                               special_file_str=f"_mask{test_donor}_query{test_donor}")
    plot_tyser_mapping_to_datasets_attrDataset(adata_all.copy(), save_file_name,
                                               attr="data_type", masked_str='L & M & P & Z & Xiao & C & T',
                                               color_dic={'L & M & P & Z & Xiao & C & T': (0.9, 0.9, 0.9, 0.7),
                                                          test_donor: "#E06D83"},
                                               reference_dataset_str=reference_dataset_str,
                                               legend_title="Dataset", special_file_str=f"_maskL&M&P&Z&Xiao&C_query{test_donor}")

    # --- plot on time categorical
    plot_tyser_mapping_to_datasets_attrTimeGT(adata_all.copy(), save_file_name, plot_attr='time',
                                              query_timePoint='17.5',
                                              legend_title="Cell stage",
                                              mask_dataset_label=test_donor,
                                              reference_dataset_str=reference_dataset_str,
                                              special_file_str=f'_mask{test_donor}_query{test_donor}')
    plot_tyser_mapping_to_datasets_attrTimeGT(adata_all.copy(), save_file_name, plot_attr='time',
                                              query_timePoint='17.5',
                                              legend_title="Cell stage",
                                              mask_dataset_label="Liu & Lv & M & P & Z & Xiao & C",
                                              reference_dataset_str=reference_dataset_str,
                                              special_file_str=f'_maskL&M&P&Z&Xiao&C_query{test_donor}')

    import gc
    gc.collect()
    _logger.info("Finish fold-test.")


def downsampling_specificDataset(adata, dataset_label):
    c_mask = adata.obs["dataset_label"] == dataset_label
    c_indices = np.where(c_mask)[0]
    total_c = len(c_indices)
    if total_c > 1000:
        # 随机选择要删除的索引（保留1000个）
        keep_indices = np.random.choice(c_indices, size=1000, replace=False)
        delete_mask = c_mask.copy()
        delete_mask[keep_indices] = False  # 将要保留的设为False

        # 3. 创建反向选择器（保留所有非"C"或选中的1000个"C"）
        adata = adata[~delete_mask].copy()
        print(f"{Counter(adata.obs['dataset_label'])}")
    else:
        print(f"数据集'C'只有{total_c}个样本，不足1000个，不做删除")
    return adata


if __name__ == '__main__':
    main()
