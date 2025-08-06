# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE
@File    ：utils_DandanProject.py
@IDE     ：PyCharm
@Author  ：awa121
@Date    ：2023/6/10 18:49
"""
import logging

_logger = logging.getLogger(__name__)

import os

from .GPU_manager_pytorch import auto_select_gpu_and_cpu, check_memory, check_gpu_memory
from TemporalVAE.model_master import *
import time
import random
import json
import anndata

import scanpy as sc

from .utils_plot import *


def read_saved_temp_result_json(file_name):
    with open(file_name, "r") as json_file:
        data_list = []
        for line in json_file:
            json_obj = json.loads(line)
            data_list.append(json_obj)

    # 还原 DataFrame 对象
    restored_dic = {}
    for em_info in data_list:
        for embryo_id, predicted_df in em_info.items():
            # 将 JSON 数据转换为 DataFrame
            value2 = predicted_df.strip().split('\n')
            # 解析每行的 JSON 数据并构建 DataFrame 列表
            dataframes = []
            for line in value2:
                json_obj = json.loads(line)
                df = pd.DataFrame([json_obj])
                dataframes.append(df)

            # 合并 DataFrame 列表为一个 DataFrame
            final_df = pd.concat(dataframes, ignore_index=True)
            final_df.set_index("cell_index", inplace=True)
            restored_dic[embryo_id] = final_df
        # 显示还原后的 DataFrame

    return restored_dic


def str2bool(str):
    return True if str.lower() == 'true' else False


def setup_model(y, time, latent_dim, device):
    from torch.nn import Parameter
    import pyro
    import pyro.contrib.gp as gp
    import pyro.distributions as dist
    """ pro.infer.elbo
    Users are therefore strongly encouraged to use this interface in
        conjunction with ``pyro.settings.set(module_local_params=True)`` which
        will override the default implicit sharing of parameters across
        :class:`~pyro.nn.PyroModule` instances.
    """
    pyro.settings.set(module_local_params=True)
    # we setup the mean of our prior over X
    X_prior_mean = torch.zeros(y.size(1), latent_dim)  # shape: 437 x 2
    X_prior_mean[:, 0] = time
    _logger.info("Set latent variable x with prior info:{}".format(X_prior_mean))

    kernel = gp.kernels.RBF(input_dim=latent_dim, lengthscale=torch.ones(latent_dim)).to(device)

    # we clone here so that we don't change our prior during the course of training
    X = Parameter(X_prior_mean.clone()).to(device)
    y = y.to(device)
    # we will use SparseGPRegression model with num_inducing=32;
    # initial values for Xu are sampled randomly from X_prior_mean
    import pyro.ops.stats as stats
    Xu = stats.resample(X_prior_mean.clone(), 32).to(device)
    # jitter: A small positive term which is added into the diagonal part of a covariance matrix to help stablize its Cholesky decomposition.
    gplvm = gp.models.SparseGPRegression(X, y, kernel, Xu, noise=torch.tensor(0.01).to(device), jitter=1e-4).to(device)
    # we use `.to_event()` to tell Pyro that the prior distribution for X has no batch_shape
    gplvm.X = pyro.nn.PyroSample(dist.Normal(X_prior_mean.to(device), 0.1).to_event())
    gplvm.autoguide("X", dist.Normal)
    return gplvm


def predict_on_one_donor(gplvm, cell_time, sc_expression_train, sc_expression_test, golbal_path, file_path, donor,
                         sample_num=10, args=None):
    # ---------------------------------------------- Freeze gplvm for prediction  --------------------------------------------------
    """
    After training GPLVM, we get good hyperparameters for the data.
    Then when new data is provided, we train GPLVM again with these fixed hyperparameters to learn X.
    """
    import pyro
    # Freeze gplvm
    # for param in gplvm.parameters():
    #     param.requires_grad_(False)
    pyro.settings.set(module_local_params=False)
    gplvm.mode = "guide"
    # sample 10 times
    sample_y_data = dict()
    labels = cell_time["time"].unique()
    sc_expression_df_copy = sc_expression_train.copy()
    for _cell_name in sc_expression_train.index.values:
        sc_expression_df_copy.rename(index={_cell_name: cell_time.loc[_cell_name]["time"]}, inplace=True)
    _logger.info("Sample from latent space {} times".format(sample_num))
    for i in range(sample_num):
        _X_one_sample = gplvm.X
        _Y_one_sample = gplvm.forward(_X_one_sample)
        # _Y_one_sample[0] is loc, _Y_one_sample[1] is var
        _Y_one_sample_df = pd.DataFrame(data=_Y_one_sample[0].cpu().detach().numpy().T,
                                        index=_X_one_sample[:, 0].cpu().detach().numpy(),
                                        columns=sc_expression_train.columns)
        for i, label in enumerate(labels):
            if label not in sample_y_data.keys():
                sample_y_data[label] = _Y_one_sample_df[sc_expression_df_copy.index == label]
            else:
                sample_y_data[label] = sample_y_data[label]._append(
                    _Y_one_sample_df[sc_expression_df_copy.index == label])

    each_type_num = np.inf
    for key in sample_y_data.keys():
        _logger.info("label {} have sampled {} cells".format(key, len(sample_y_data[key])))
        if len(sample_y_data[key]) < each_type_num:
            each_type_num = len(sample_y_data[key])
    for key in sample_y_data.keys():
        sample_y_data[key] = sample_y_data[key].sample(n=each_type_num)

    # # filter sc_expression_test; 2023-06-19 20:22:38 don't filter gene for test data, so delete code here
    # for _gene in sc_expression_test:
    #     if np.var(sc_expression_test[_gene].values) == 0:
    #         sc_expression_test = sc_expression_test.drop(_gene, axis=1)
    from utils_project import cosSim
    sample_y_data_df = pd.DataFrame(columns=sc_expression_train.columns)
    # sample_y_data_df_attr_time =pd.DataFrame(columns=sc_expression_train.columns)
    for key in sample_y_data.keys():
        sample_y_data_df = sample_y_data_df._append(sample_y_data[key])
        # temp=pd.DataFrame(data=sample_y_data[key].values,columns=sc_expression_train.columns,index=[key for i in range(len(sample_y_data[key]))])
        # sample_y_data_df_attr_time=sample_y_data_df_attr_time._append(temp)
    # plot the generated cells
    # plot_latent_dim_image(_X_one_sample.shape[1], labels, X, sample_y_data_df_attr_time, golbal_path, file_path, "time", reorder_labels=True, args=args,special_str="Sampled_")

    sample_y_data_df = sample_y_data_df[sc_expression_test.columns]
    result_test_pseudotime = pd.DataFrame(index=sc_expression_test.index, columns=["pseudotime"])
    for test_index, row_test in sc_expression_test.iterrows():
        _cell_samilars = []
        for time, row_sampled in sample_y_data_df.iterrows():
            _cell_samilars.append([time, cosSim(row_test, row_sampled)])
        _cell_samilars = np.array(_cell_samilars)
        _cell_samilars = _cell_samilars[_cell_samilars[:, 1].argsort()][::-1][:10, 0]
        _cell_samilars = np.mean(_cell_samilars)
        result_test_pseudotime.loc[test_index]["pseudotime"] = _cell_samilars
    result_test_pseudotime.to_csv('{}/{}/test_onOneDonor_{}.csv'.format(golbal_path, file_path, donor), sep="\t",
                                  index=True, header=True)
    _logger.info("Test donor: {}, pseudotime for each cell is {}".format(donor, result_test_pseudotime))
    return result_test_pseudotime


def preprocessData(golbal_path, file_name, KNN_smooth_type, cell_info_file, gene_list=None, min_cell_num=50,
                   min_gene_num=100):
    """
    2023-08-23 10:02:28 careful use, has update (add more function) to function preprocessData_and_dropout_some_donor_or_gene.
    :param golbal_path:
    :param file_name:
    :param KNN_smooth_type:
    :param cell_info_file:
    :param gene_list:
    :param min_cell_num:
    :param min_gene_num:
    :return:
    """
    adata = anndata.read_csv(golbal_path + file_name, delimiter='\t')
    if gene_list is not None:
        overlap_gene = list(set(adata.obs_names) & set(gene_list))
        adata = adata[overlap_gene].copy()

    adata = adata.T  # 基因和cell转置矩阵
    # 数据数目统计
    _logger.info("Import data, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    _shape = adata.shape
    _new_shape = (0, 0)
    while _new_shape != _shape:  # make sure drop samples and genes
        _shape = adata.shape
        sc.pp.filter_cells(adata, min_genes=min_gene_num)  # drop samples with less than 20 gene expression
        sc.pp.filter_genes(adata, min_cells=min_cell_num)  # drop genes which none expression in 3 samples
        _new_shape = adata.shape
    _logger.info("Drop cells with less than {} gene expression, drop genes which none expression in {} samples".format(
        min_gene_num, min_cell_num))
    _logger.info("After filter, get cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    _logger.info("Finish normalize per cell, so that every cell has the same total count after normalization.")

    denseM = adata.X
    # denseM = KNN_smoothing_start(denseM, type=KNN_smooth_type)
    # _logger.info("Finish smooth by {} method.".format(KNN_smooth_type))
    from sklearn.preprocessing import scale
    denseM = scale(denseM.astype(float), axis=0, with_mean=True, with_std=True)
    _logger.info("Finish normalize per gene as Gaussian-dist (0, 1).")

    sc_expression_df = pd.DataFrame(data=denseM, columns=adata.var.index, index=adata.obs.index)
    cell_time = pd.read_csv(golbal_path + cell_info_file, sep="\t", index_col=0)
    cell_time = cell_time.loc[sc_expression_df.index]
    _logger.info("Get expression dataframe with shape (cell, gene): {}, and cell time info with shape: {}.".format(
        sc_expression_df.shape, cell_time.shape))

    return sc_expression_df, cell_time


def downSample_matrix(matrix, target_location="row", reduce_multiple=10):
    matrix = matrix.astype("float64")
    import numpy.random as npr
    npr.seed(123)
    # Normalize matrix along rows
    if target_location == "row":
        row_sums = matrix.sum(axis=1)
        print(row_sums)
        normalized_matrix = matrix / row_sums[:, np.newaxis]
        print(normalized_matrix)
        # Initialize an empty array to store downsampled matrix
        downsampled_matrix = np.zeros(matrix.shape)

        # Generate multinomial samples row by row
        for i in range(len(matrix)):
            if row_sums[i] == 0:
                row_sample = matrix[i]
            else:
                row_sample = np.random.multinomial(row_sums[i] / reduce_multiple, normalized_matrix[i])
            downsampled_matrix[i] = row_sample

        return downsampled_matrix
    elif target_location == "col":
        col_sums = matrix.sum(axis=0)
        print(col_sums)
        normalized_matrix = matrix / col_sums[np.newaxis]
        print(normalized_matrix)
        # Initialize an empty array to store downsampled matrix
        downsampled_matrix = np.zeros(matrix.shape)

        # Generate multinomial samples row by row
        for i in range(len(matrix[0])):
            if col_sums[i] == 0:
                col_sample = matrix[:, i]
            else:
                col_sample = np.random.multinomial(col_sums[i] / reduce_multiple, normalized_matrix[:, i])
            downsampled_matrix[:, i] = col_sample

        return downsampled_matrix
    elif target_location == "line":
        # Flatten the matrix into a 1D array
        flattened_matrix = matrix.flatten().astype('float64')
        # Normalize the flattened array to create probabilities
        probabilities = flattened_matrix / np.sum(flattened_matrix)
        print(probabilities)
        # Generate multinomial samples
        print(flattened_matrix.sum())
        samples = np.random.multinomial(flattened_matrix.sum() / reduce_multiple, probabilities)

        # Reshape the samples back to the original matrix shape
        downsampled_matrix = samples.reshape(matrix.shape)

        return downsampled_matrix


def cosSim(x, y):
    '''
    余弦相似度
    '''
    tmp = np.sum(x * y)
    non = np.linalg.norm(x) * np.linalg.norm(y)
    return np.round(tmp / float(non), 9)


def eculidDisSim(x, y):
    '''
    欧几里得相似度
    '''
    return np.sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))


def manhattanDisSim(x, y):
    '''
    曼哈顿相似度
    '''
    return sum(abs(a - b) for a, b in zip(x, y))


def pearsonrSim(x, y):
    '''
    皮尔森相似度
    '''
    from scipy.stats import pearsonr
    return pearsonr(x, y)[0]


# def donor_resort_key(item):
#     import re
#     match = re.match(r'LH(\d+)_([0-9PGT]+)', item)
#     if match:
#         num1 = int(match.group(1))
#         num2 = match.group(2)
#         if num2 == 'PGT':
#             num2 = 'ZZZ'  # 将'LH7_PGT'替换为'LH7_ZZZ'，确保'LH7_PGT'在'LH7'后面
#         return num1, num2
#     return item
#
#
# def RIFdonor_resort_key(item):
#     import re
#     match = re.match(r'RIF_(\d+)', item)
#     if match:
#         num1 = int(match.group(1))
#         return num1
#     return item


def test_on_newDataset(sc_expression_train, data_golbal_path, result_save_path, KNN_smooth_type, runner, experiment,
                       config, latent_dim,
                       special_path_str, time_standard_type, test_data_path):
    """
    2023-07-13 14:39:38 dandan share a new dataset (download from public database, with epi and fibro, different platfrom: ct and 10X)
    use all dandan data as train data to train a model and test on the new dataset.
    :param sc_expression_train:
    :param data_golbal_path:
    :param KNN_smooth_type:
    :param runner:
    :param experiment:
    :param config:
    :param latent_dim:
    :param special_path_str:
    :param time_standard_type:
    :return:
    """
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyPredict
    _logger.info("Test on new dataset.")

    gene_list = sc_expression_train.columns

    _logger.info("Test on dataset: {}".format(test_data_path))
    gene_dic = dict()
    file_name_test = test_data_path + "/data_count.csv"
    cell_info_file_test = test_data_path + "/cell_info.csv"
    sc_expression_df_test, cell_time_test = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path, file_name_test, cell_info_file_test,
                                                                                          gene_list=gene_list, min_cell_num=0,
                                                                                          min_gene_num=10)

    loss_gene = list(set(gene_list) - set(sc_expression_df_test.columns))
    _logger.info("loss {} gene in test data, set them to 0".format(len(loss_gene)))
    _logger.info("test data don't have gene: {}".format(loss_gene))
    gene_dic["model_gene"] = list(gene_list)
    gene_dic["loss_gene"] = list(loss_gene)
    gene_dic["testdata_gene"] = list(sc_expression_df_test.columns)
    for _g in loss_gene:
        sc_expression_df_test[_g] = 0
    x_sc_test = torch.tensor(sc_expression_df_test.values, dtype=torch.get_default_dtype()).t()
    _logger.info("Set x_sc_test data with shape (gene, cells): {}".format(x_sc_test.shape))
    train_data = [[x_sc_test[:, i], torch.tensor(0), torch.tensor(0)] for i in range(x_sc_test.shape[1])]
    data_test = SupervisedVAEDataset_onlyPredict(predict_data=train_data, predict_batch_size=len(train_data))
    test_result = runner.predict(experiment, data_test)
    test_clf_result, test_latent_mu_result, test_latent_log_var_result = test_result[0][0], test_result[0][1], \
        test_result[0][2]
    if test_clf_result.shape[1] == 1:
        test_clf_result = test_clf_result.squeeze()
        test_clf_result_df = pd.DataFrame(data=test_clf_result, index=sc_expression_df_test.index,
                                          columns=["pseudotime"])
        _logger.info("Time type is continues.")
    else:
        test_clf_result = test_clf_result.squeeze()
        test_clf_result = np.argmax(test_clf_result, axis=1)
        test_clf_result_df = pd.DataFrame(data=test_clf_result, index=sc_expression_df_test.index,
                                          columns=["pseudotime"])
        _logger.info("Time type is discrete.")

    _save_path = "{}{}/".format(_logger.root.handlers[0].baseFilename.replace(".log", ""), special_path_str)
    if not os.path.exists(_save_path):
        os.makedirs(_save_path)
    _save_file_name = "{}/{}_testOnExternal_{}.csv".format(_save_path, config['model_params']['name'],
                                                           test_data_path.replace("/", "").replace(".rds", "").replace(" ", "_"))

    test_clf_result_df.to_csv(_save_file_name, sep="\t")
    import json
    with open(_save_file_name.replace(".csv", "geneUsed.json"), 'w') as f:
        json.dump(gene_dic, f)
    _logger.info("result save at: {}".format(_save_file_name))


def test_on_newDonor(test_donor_name, sc_expression_test, runner, experiment, predict_donors_dic):
    """
    2023-07-16 15:05:18 dandan share 10 RIF donor, test on these donor
    use all dandan data as train data to train a model and test on the new dataset.
    :param sc_expression_train:
    :param data_golbal_path:
    :param KNN_smooth_type:
    :param runner:
    :param experiment:
    :param config:
    :param latent_dim:
    :param special_path_str:
    :param time_standard_type:
    :return:
    """
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyPredict

    x_sc_test = torch.tensor(sc_expression_test.values, dtype=torch.get_default_dtype()).t()
    _logger.info("Set x_sc_test data with shape (gene, cells): {}".format(x_sc_test.shape))
    test_data = [[x_sc_test[:, i], torch.tensor(0), torch.tensor(0)] for i in range(x_sc_test.shape[1])]
    data_test = SupervisedVAEDataset_onlyPredict(predict_data=test_data, predict_batch_size=len(test_data))
    test_result = runner.predict(experiment, data_test)
    test_clf_result, test_latent_mu_result, test_latent_log_var_result = test_result[0][0], test_result[0][1], \
        test_result[0][2]
    # time is continues, supervise_vae_regressionclfdecoder  supervise_vae_regressionclfdecoder_of_sublatentspace
    if test_clf_result.shape[1] == 1:
        _logger.info("predicted time of test donor is continuous.")
        predict_donors_dic[test_donor_name] = pd.DataFrame(data=np.squeeze(test_clf_result),
                                                           index=sc_expression_test.index, columns=["pseudotime"])
    else:  # time is discrete and probability on each time point, supervise_vae supervise_vae_noclfdecoder
        _logger.info("predicted time of test donor is discrete.")
        labels_pred = torch.argmax(torch.tensor(test_clf_result), dim=1)
        predict_donors_dic[test_donor_name] = pd.DataFrame(data=labels_pred.cpu().numpy(),
                                                           index=sc_expression_test.index, columns=["pseudotime"])
    test_latent_info_dic = {"mu": test_latent_mu_result,
                            "log_var": test_latent_log_var_result,
                            "label_true": np.zeros(len(test_latent_mu_result)),
                            "label_dic": {"test": "test"},
                            "donor_index": np.zeros(len(test_latent_mu_result)) - 1,
                            "donor_dic": {test_donor_name: -1}}
    return predict_donors_dic, test_clf_result, test_latent_info_dic


def process_fold(fold, donor_list, sc_expression_df, donor_dic, batch_dic, special_path_str, cell_time,
                 time_standard_type, config, args, batch_size, adversarial_train_bool, checkpoint_file=None):
    # _logger = logging.getLogger(__name__)
    time.sleep(random.randint(10, 100))
    _logger.info("the {}/{} fold train, use donor-{} as test set".format(fold + 1, len(donor_list), donor_list[fold]))

    try:
        if adversarial_train_bool:
            predict_donor_dic, test_clf_result, label_dic = one_fold_test_adversarialTrain(fold, donor_list, sc_expression_df,
                                                                                           donor_dic, batch_dic,
                                                                                           special_path_str, cell_time,
                                                                                           time_standard_type, config, args.train_epoch_num,
                                                                                           plot_trainingLossLine=False,
                                                                                           plot_latentSpaceUmap=False,
                                                                                           time_saved_asFloat=True,
                                                                                           batch_size=batch_size,
                                                                                           max_attempts=5,
                                                                                           checkpoint_file=checkpoint_file)

        else:
            predict_donor_dic, test_clf_result, label_dic = one_fold_test(fold, donor_list, sc_expression_df, donor_dic,
                                                                          batch_dic,
                                                                          special_path_str,
                                                                          cell_time, time_standard_type, config, args.train_epoch_num,
                                                                          plot_trainingLossLine=False, plot_latentSpaceUmap=False,
                                                                          time_saved_asFloat=True,
                                                                          batch_size=batch_size,
                                                                          max_attempts=5,
                                                                          checkpoint_file=checkpoint_file)
        temp_save_dic(special_path_str, config, time_standard_type, predict_donor_dic.copy(), label_dic=label_dic)
        result = (predict_donor_dic, test_clf_result, label_dic)
        return result
    except Exception as e:
        # 如果有异常，记录错误日志，日志消息将通过队列传递给主进程
        _logger.error("Note!!! Error {} processing data at embryo {}".format(donor_list[fold], e))
        return None


def temp_save_dic(special_path_str, config, time_standard_type, predict_donor_dic, label_dic=None):
    import fcntl
    temp_save_file_name = "{}{}/{}_plot_on_all_test_donor_time{}_temp.json".format(
        _logger.root.handlers[0].baseFilename.replace(".log", ""),
        special_path_str,
        config['model_params']['name'],
        time_standard_type)
    for key, value in predict_donor_dic.items():
        value.index.name = "cell_index"
        _dic = value.reset_index().to_json(orient='records', lines=True)
        predict_donor_dic.update({key: _dic})
    # 检查 JSON 文件是否被占用，如果被占用则等待60秒

    while True:
        try:
            with open(temp_save_file_name, "a") as json_file:
                # 尝试获取文件锁
                fcntl.flock(json_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                # 文件没有被占用，写入结果并释放文件锁
                json.dump(predict_donor_dic, json_file)
                json_file.write("\n")  # 添加换行符以分隔每个结果
                fcntl.flock(json_file.fileno(), fcntl.LOCK_UN)
                break
        except BlockingIOError:
            # 文件被占用，等待60秒
            time.sleep(60)
    _logger.info("Temp predict_donor_dic save at {}".format(temp_save_file_name))
    if label_dic is not None:
        for key, value in label_dic.items():
            label_dic[key] = float(value)
        while True:
            try:
                with open(temp_save_file_name.replace("temp.json", "labelDic.json"), "w") as json_file:
                    # 尝试获取文件锁
                    fcntl.flock(json_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # 文件没有被占用，写入结果并释放文件锁
                    json.dump(label_dic, json_file)
                    json_file.write("\n")  # 添加换行符以分隔每个结果
                    fcntl.flock(json_file.fileno(), fcntl.LOCK_UN)
                    break
            except BlockingIOError:
                # 文件被占用，等待60秒
                time.sleep(60)
        _logger.info("Save label dic at {}".format(temp_save_file_name.replace("temp.json", "labelDic.json")))


def str2bool(str):
    return True if str.lower() == 'true' else False


def setup_model(y, time, latent_dim, device):
    """ pro.infer.elbo
    Users are therefore strongly encouraged to use this interface in
        conjunction with ``pyro.settings.set(module_local_params=True)`` which
        will override the default implicit sharing of parameters across
        :class:`~pyro.nn.PyroModule` instances.
    """
    from torch.nn import Parameter
    import pyro
    import pyro.contrib.gp as gp
    import pyro.distributions as dist
    pyro.settings.set(module_local_params=True)
    # we setup the mean of our prior over X
    X_prior_mean = torch.zeros(y.size(1), latent_dim)  # shape: 437 x 2
    X_prior_mean[:, 0] = time
    _logger.info("Set latent variable x with prior info:{}".format(X_prior_mean))

    kernel = gp.kernels.RBF(input_dim=latent_dim, lengthscale=torch.ones(latent_dim)).to(device)

    # we clone here so that we don't change our prior during the course of training
    X = Parameter(X_prior_mean.clone()).to(device)
    y = y.to(device)
    # we will use SparseGPRegression model with num_inducing=32;
    # initial values for Xu are sampled randomly from X_prior_mean
    import pyro.ops.stats as stats
    Xu = stats.resample(X_prior_mean.clone(), 32).to(device)
    # jitter: A small positive term which is added into the diagonal part of a covariance matrix to help stablize its Cholesky decomposition.
    gplvm = gp.models.SparseGPRegression(X, y, kernel, Xu, noise=torch.tensor(0.01).to(device), jitter=1e-4).to(device)
    # we use `.to_event()` to tell Pyro that the prior distribution for X has no batch_shape
    gplvm.X = pyro.nn.PyroSample(dist.Normal(X_prior_mean.to(device), 0.1).to_event())
    gplvm.autoguide("X", dist.Normal)
    return gplvm


def predict_on_one_donor(gplvm, cell_time, sc_expression_train, sc_expression_test, golbal_path, file_path, donor,
                         sample_num=10, args=None):
    # ---------------------------------------------- Freeze gplvm for prediction  --------------------------------------------------
    """
    After training GPLVM, we get good hyperparameters for the data.
    Then when new data is provided, we train GPLVM again with these fixed hyperparameters to learn X.
    """
    import pyro
    # Freeze gplvm
    # for param in gplvm.parameters():
    #     param.requires_grad_(False)
    pyro.settings.set(module_local_params=False)
    gplvm.mode = "guide"
    # sample 10 times
    sample_y_data = dict()
    labels = cell_time["time"].unique()
    sc_expression_df_copy = sc_expression_train.copy()
    for _cell_name in sc_expression_train.index.values:
        sc_expression_df_copy.rename(index={_cell_name: cell_time.loc[_cell_name]["time"]}, inplace=True)
    _logger.info("Sample from latent space {} times".format(sample_num))
    for i in range(sample_num):
        _X_one_sample = gplvm.X
        _Y_one_sample = gplvm.forward(_X_one_sample)
        # _Y_one_sample[0] is loc, _Y_one_sample[1] is var
        _Y_one_sample_df = pd.DataFrame(data=_Y_one_sample[0].cpu().detach().numpy().T,
                                        index=_X_one_sample[:, 0].cpu().detach().numpy(),
                                        columns=sc_expression_train.columns)
        for i, label in enumerate(labels):
            if label not in sample_y_data.keys():
                sample_y_data[label] = _Y_one_sample_df[sc_expression_df_copy.index == label]
            else:
                sample_y_data[label] = sample_y_data[label]._append(
                    _Y_one_sample_df[sc_expression_df_copy.index == label])

    each_type_num = np.inf
    for key in sample_y_data.keys():
        _logger.info("label {} have sampled {} cells".format(key, len(sample_y_data[key])))
        if len(sample_y_data[key]) < each_type_num:
            each_type_num = len(sample_y_data[key])
    for key in sample_y_data.keys():
        sample_y_data[key] = sample_y_data[key].sample(n=each_type_num)

    # # filter sc_expression_test; 2023-06-19 20:22:38 don't filter gene for test data, so delete code here
    # for _gene in sc_expression_test:
    #     if np.var(sc_expression_test[_gene].values) == 0:
    #         sc_expression_test = sc_expression_test.drop(_gene, axis=1)
    from utils_project import cosSim
    sample_y_data_df = pd.DataFrame(columns=sc_expression_train.columns)
    # sample_y_data_df_attr_time =pd.DataFrame(columns=sc_expression_train.columns)
    for key in sample_y_data.keys():
        sample_y_data_df = sample_y_data_df._append(sample_y_data[key])
        # temp=pd.DataFrame(data=sample_y_data[key].values,columns=sc_expression_train.columns,index=[key for i in range(len(sample_y_data[key]))])
        # sample_y_data_df_attr_time=sample_y_data_df_attr_time._append(temp)
    # plot the generated cells
    # plot_latent_dim_image(_X_one_sample.shape[1], labels, X, sample_y_data_df_attr_time, golbal_path, file_path, "time", reorder_labels=True, args=args,special_str="Sampled_")

    sample_y_data_df = sample_y_data_df[sc_expression_test.columns]
    result_test_pseudotime = pd.DataFrame(index=sc_expression_test.index, columns=["pseudotime"])
    for test_index, row_test in sc_expression_test.iterrows():
        _cell_samilars = []
        for time, row_sampled in sample_y_data_df.iterrows():
            _cell_samilars.append([time, cosSim(row_test, row_sampled)])
        _cell_samilars = np.array(_cell_samilars)
        _cell_samilars = _cell_samilars[_cell_samilars[:, 1].argsort()][::-1][:10, 0]
        _cell_samilars = np.mean(_cell_samilars)
        result_test_pseudotime.loc[test_index]["pseudotime"] = _cell_samilars
    result_test_pseudotime.to_csv('{}/{}/test_onOneDonor_{}.csv'.format(golbal_path, file_path, donor), sep="\t",
                                  index=True, header=True)
    _logger.info("Test donor: {}, pseudotime for each cell is {}".format(donor, result_test_pseudotime))
    return result_test_pseudotime


def preprocessData(golbal_path, file_name, KNN_smooth_type, cell_info_file, gene_list=None, min_cell_num=50,
                   min_gene_num=100):
    """
    2023-08-23 10:02:28 careful use, has update (add more function) to function preprocessData_and_dropout_some_donor_or_gene.
    :param golbal_path:
    :param file_name:
    :param KNN_smooth_type:
    :param cell_info_file:
    :param gene_list:
    :param min_cell_num:
    :param min_gene_num:
    :return:
    """
    adata = anndata.read_csv(golbal_path + file_name, delimiter='\t')
    if gene_list is not None:
        overlap_gene = list(set(adata.obs_names) & set(gene_list))
        adata = adata[overlap_gene].copy()

    adata = adata.T  # 基因和cell转置矩阵
    # 数据数目统计
    _logger.info("Import data, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    _shape = adata.shape
    _new_shape = (0, 0)
    while _new_shape != _shape:  # make sure drop samples and genes
        _shape = adata.shape
        sc.pp.filter_cells(adata, min_genes=min_gene_num)  # drop samples with less than 20 gene expression
        sc.pp.filter_genes(adata, min_cells=min_cell_num)  # drop genes which none expression in 3 samples
        _new_shape = adata.shape
    _logger.info("Drop cells with less than {} gene expression, drop genes which none expression in {} samples".format(
        min_gene_num, min_cell_num))
    _logger.info("After filter, get cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    _logger.info("Finish normalize per cell, so that every cell has the same total count after normalization.")

    denseM = adata.X
    # denseM = KNN_smoothing_start(denseM, type=KNN_smooth_type)
    # _logger.info("Finish smooth by {} method.".format(KNN_smooth_type))
    from sklearn.preprocessing import scale
    denseM = scale(denseM.astype(float), axis=0, with_mean=True, with_std=True)
    _logger.info("Finish normalize per gene as Gaussian-dist (0, 1).")

    sc_expression_df = pd.DataFrame(data=denseM, columns=adata.var.index, index=adata.obs.index)
    cell_time = pd.read_csv(golbal_path + cell_info_file, sep="\t", index_col=0)
    cell_time = cell_time.loc[sc_expression_df.index]
    _logger.info("Get expression dataframe with shape (cell, gene): {}, and cell time info with shape: {}.".format(
        sc_expression_df.shape, cell_time.shape))

    return sc_expression_df, cell_time


def downSample_matrix(matrix, target_location="row", reduce_multiple=10):
    matrix = matrix.astype("float64")
    import numpy.random as npr
    npr.seed(123)
    # Normalize matrix along rows
    if target_location == "row":
        row_sums = matrix.sum(axis=1)
        print(row_sums)
        normalized_matrix = matrix / row_sums[:, np.newaxis]
        print(normalized_matrix)
        # Initialize an empty array to store downsampled matrix
        downsampled_matrix = np.zeros(matrix.shape)

        # Generate multinomial samples row by row
        for i in range(len(matrix)):
            if row_sums[i] == 0:
                row_sample = matrix[i]
            else:
                row_sample = np.random.multinomial(row_sums[i] / reduce_multiple, normalized_matrix[i])
            downsampled_matrix[i] = row_sample

        return downsampled_matrix
    elif target_location == "col":
        col_sums = matrix.sum(axis=0)
        print(col_sums)
        normalized_matrix = matrix / col_sums[np.newaxis]
        print(normalized_matrix)
        # Initialize an empty array to store downsampled matrix
        downsampled_matrix = np.zeros(matrix.shape)

        # Generate multinomial samples row by row
        for i in range(len(matrix[0])):
            if col_sums[i] == 0:
                col_sample = matrix[:, i]
            else:
                col_sample = np.random.multinomial(col_sums[i] / reduce_multiple, normalized_matrix[:, i])
            downsampled_matrix[:, i] = col_sample

        return downsampled_matrix
    elif target_location == "line":
        # Flatten the matrix into a 1D array
        flattened_matrix = matrix.flatten().astype('float64')
        # Normalize the flattened array to create probabilities
        probabilities = flattened_matrix / np.sum(flattened_matrix)
        print(probabilities)
        # Generate multinomial samples
        print(flattened_matrix.sum())
        samples = np.random.multinomial(flattened_matrix.sum() / reduce_multiple, probabilities)

        # Reshape the samples back to the original matrix shape
        downsampled_matrix = samples.reshape(matrix.shape)

        return downsampled_matrix


# 2024-02-23 12:56:15 remove KNN_smooth_type
def preprocessData_and_dropout_some_donor_or_gene(global_data_path, file_name, cell_info_file,
                                                  drop_out_donor=None, donor_attr="donor", gene_list=None,
                                                  drop_out_cell_type=None,
                                                  min_cell_num=50, min_gene_num=100, keep_sub_type_with_cell_num=None,
                                                  external_file_name=None, external_cell_info_file=None,
                                                  # external_file_name2=None, external_cell_info_file2=None,
                                                  # external_cellId_list=None,
                                                  downSample_on_testData_bool=False, test_donor=None,
                                                  downSample_location_type=None,
                                                  augmentation_on_trainData_bool=False,
                                                  plot_boxPlot_bool=False,
                                                  special_path_str="",
                                                  random_drop_cell_bool=False,
                                                  normalized_cellTotalCount=1e6,
                                                  data_raw_count_bool=True,
                                                  return_normalized_raw_count=False, ):
    """
    :param global_data_path:
    :param file_name:
    :param cell_info_file:
    :param drop_out_donor:
    :param donor_attr:
    :param gene_list:
    :param drop_out_cell_type:
    :param min_cell_num:
    :param min_gene_num:
    :param keep_sub_type_with_cell_num:
    :param external_file_name:
    :param external_cell_info_file:
    :param external_cellId_list:
    :param downSample_on_testData_bool:
    :param test_donor:
    :param downSample_location_type:
    :param augmentation_on_trainData_bool:
    :param plot_boxPlot_bool:
    :param special_path_str:
    :param random_drop_cell_bool:
    :param normalized_cellTotalCount:
    :param data_raw_count_bool: default is True, and do normalized bu scanpy. Require the input adata is raw count, but if input is already normalized, don't do normalization.
    :return:
    """
    # drop should before perprocess sc data
    # from .utils_Dandan_plot import plot_boxPlot_nonExpGene_percentage_whilePreprocess
    _logger.info("the original sc expression anndata should be gene as row, cell as column")
    if file_name.split(".")[-1] in ["h5", "h5ad"]:
        _logger.info("input data is .h5 or .h5ad file, where cell as column and gene as row.")
        adata = anndata.read_h5ad(f"{global_data_path}/{file_name}")
        # cell_time = adata.obs
        _logger.info("Import data, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))

    elif file_name.split(".")[-1] == "csv":
        _logger.info("input data is .csv file, where gene as row, cell as column.")
        try:
            adata = anndata.read_csv("{}/{}".format(global_data_path, file_name), delimiter='\t')
        except:
            adata = anndata.read_csv("{}/{}".format(global_data_path, file_name), delimiter=',')
        _logger.info("read the original sc expression anndata with shape (gene, cell): {}".format(adata.shape))
        adata = adata.T

        adata.obs = pd.read_csv(global_data_path + cell_info_file, sep="\t", index_col=0)
    # read external file,
    if external_file_name is not None:
        external_file_name = [external_file_name] if isinstance(external_file_name, str) else external_file_name

        for i in range(len(external_file_name)):
            if external_file_name[i].split(".")[-1] in ["h5", "h5ad"]:
                _logger.info("External file name {} is .h5 or h5ad type".format(external_file_name[i]))
                adata_external = anndata.read_h5ad("{}/{}".format(global_data_path, external_file_name[i]))
            elif external_file_name[i].split(".")[-1] == "csv":
                _logger.info("External file name {} is .csv type".format(external_file_name[i]))
                try:
                    adata_external = anndata.read_csv("{}/{}".format(global_data_path, external_file_name[i]), delimiter='\t')
                except:
                    adata_external = anndata.read_csv("{}/{}".format(global_data_path, external_file_name[i]), delimiter=',')
                _logger.info("read the external test dataset sc expression anndata with shape (gene, cell): {}".format(adata_external.shape))
                _logger.info("here is important, we want to use same cell compare with no integration method.")
                adata_external = adata_external.T
                # read cell info file
                external_cell_time = pd.read_csv(global_data_path + external_cell_info_file[i], sep="\t", index_col=0)
                external_cell_time = external_cell_time.reindex(adata_external.obs_names)
                adata_external.obs = external_cell_time

            # if external_cellId_list is not None:
            #     _logger.info(f"external data select cells by external_cellId_list with {len(external_cellId_list)} cells")
            #     adata_external = adata_external[external_cellId_list]
            # identify and remove duplicate cell
            duplicate_columns = set(adata.obs_names) & set(adata_external.obs_names)
            adata_external = adata_external[~adata_external.obs_names.isin(duplicate_columns)]
            _logger.info("drop out {} duplicate cell (with the same cell name) from external data".format(len(duplicate_columns)))

            col_gene_list = [_gene for _gene in adata.var_names if _gene in adata_external.var_names]
            adata = adata[:, col_gene_list]
            adata_external = adata_external[:, col_gene_list]
            adata = anndata.concat([adata, adata_external], axis=0, join='outer')
            _logger.info("merged sc data and external test dataset with shape (cell, gene): {}".format(adata.shape))

    cell_time = adata.obs
    _logger.info(f"cell annotation includes {cell_time.columns}")
    # -------------
    if gene_list is not None:
        overlap_gene = list(set(adata.var_names) & set(gene_list))
        adata = adata[:, overlap_gene].copy()
        _logger.info("with gene list require, adata filted with {} genes.".format(len(overlap_gene)))
    _logger.info("Import data, cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    if drop_out_donor is not None:
        drop_out_cell = list(
            set(cell_time.loc[cell_time[donor_attr].isin(list(drop_out_donor))].index) & set(adata.obs.index))
        adata = adata[adata.obs_names.drop(drop_out_cell)].copy()
        _logger.info("After drop out {}, get expression dataframe with shape (cell, gene): {}.".format(drop_out_donor,
                                                                                                       adata.shape))
    if drop_out_cell_type is not None:
        _logger.info(f"drop out {len(drop_out_cell_type)} cell type: {drop_out_cell_type}")
        # 2023-11-07 11:05:26 for Joy project, she focus on epi cells, so remove fibro cells, and the attr name in cell_time is "broad_celltype"
        drop_out_cell = list(set(cell_time.loc[cell_time["broad_celltype"].isin(list(drop_out_cell_type))].index) & set(adata.obs.index))
        adata = adata[adata.obs_names.drop(drop_out_cell)].copy()
        _logger.info("After drop out {}, get expression dataframe with shape (cell, gene): {}.".format(drop_out_donor, adata.shape))
    if plot_boxPlot_bool:
        try:
            plot_boxPlot_nonExpGene_percentage_whilePreprocess(adata, cell_time, donor_attr,
                                                               special_path_str, test_donor,
                                                               special_file_str="1InitialImportData")
            plot_boxPlot_total_count_per_cell_whilePreprocess(adata, cell_time, donor_attr,
                                                              special_path_str, test_donor,
                                                              special_file_str="1InitialImportData")
        except:
            print("some error while plot before boxplot.")
    if random_drop_cell_bool:
        _logger.info("To fast check model performance, random dropout cells, only save few cells to train and test")
        random.seed(123)
        # 随机生成不重复的样本索引
        random_indices = random.sample(range(adata.shape[0]), int(adata.n_obs / 10), )

        # 从 anndata 对象中获取选定的样本数据
        adata = adata[random_indices, :].copy()
        _logger.info("After random select 1/10 samples from adata, remain adata shape: {}".format(adata.shape))
        if plot_boxPlot_bool:
            try:
                plot_boxPlot_nonExpGene_percentage_whilePreprocess(adata, cell_time, donor_attr,
                                                                   special_path_str, test_donor,
                                                                   special_file_str="2RandomDropPartCell")
                plot_boxPlot_total_count_per_cell_whilePreprocess(adata, cell_time, donor_attr,
                                                                  special_path_str, test_donor,
                                                                  special_file_str="2RandomDropPartCell")
            except:
                print("some error while plot before boxplot.")

    if downSample_on_testData_bool:
        _logger.info("Down sample on test donor: {}, downSample location type: {}".format(test_donor, downSample_location_type))
        test_cell = list(set(cell_time.loc[cell_time[donor_attr] == test_donor].index) & set(adata.obs.index))
        adata_test = adata[test_cell].copy()

        test_dataframe = pd.DataFrame(data=adata_test.X, columns=adata_test.var.index, index=adata_test.obs.index)
        temp = downSample_matrix(np.array(test_dataframe.values), target_location=downSample_location_type)
        downSample_test_dataframe = pd.DataFrame(data=temp, columns=test_dataframe.columns, index=test_dataframe.index)
        downSample_test_anndata = anndata.AnnData(downSample_test_dataframe)
        adata = adata[adata.obs_names.drop(test_cell)].copy()
        adata = anndata.concat([adata.copy(), downSample_test_anndata.copy()], axis=0)
        # downSample_test_dataframe.values.sum()

        if plot_boxPlot_bool:
            try:
                plot_boxPlot_nonExpGene_percentage_whilePreprocess(adata, cell_time, donor_attr,
                                                                   special_path_str, test_donor,
                                                                   special_file_str=f"3DownSampleOnTestBy{downSample_location_type}")
                plot_boxPlot_total_count_per_cell_whilePreprocess(adata, cell_time, donor_attr,
                                                                  special_path_str, test_donor,
                                                                  special_file_str=f"3DownSampleOnTestBy{downSample_location_type}")
            except:
                print("some error while plot before boxplot.")
    if augmentation_on_trainData_bool:
        _logger.info("Data augmentation on train set by down sample 1/10 and 1/3, downSample location type line")
        train_cell = list(set(cell_time.loc[cell_time[donor_attr] != test_donor].index) & set(adata.obs.index))
        _logger.info(f"the train cell number is {len(train_cell)}")
        adata_train = adata[train_cell].copy()

        train_dataframe = pd.DataFrame(data=adata_train.X, columns=adata_train.var.index, index=adata_train.obs.index)
        temp10 = downSample_matrix(np.array(train_dataframe.values), target_location="line", reduce_multiple=10)
        temp3 = downSample_matrix(np.array(train_dataframe.values), target_location="line", reduce_multiple=3)
        downSample10_train_df = pd.DataFrame(data=temp10, columns=train_dataframe.columns,
                                             index=train_dataframe.index + "_downSample10")
        downSample3_train_df = pd.DataFrame(data=temp3, columns=train_dataframe.columns,
                                            index=train_dataframe.index + "_downSample3")
        downSample10_train_anndata = anndata.AnnData(downSample10_train_df)
        downSample3_train_anndata = anndata.AnnData(downSample3_train_df)

        adata = anndata.concat([adata.copy(), downSample10_train_anndata.copy()], axis=0)
        adata = anndata.concat([adata.copy(), downSample3_train_anndata.copy()], axis=0)
        _logger.info(f"After concat downSample 1/10 and 1/3 train cell, the adata with {adata.n_obs} cell, {adata.n_vars} gene.")
        train_cell_time = cell_time.loc[train_cell]
        downSample10_train_cell_time = train_cell_time.copy()
        downSample10_train_cell_time.index = downSample10_train_cell_time.index + "_downSample10"
        downSample3_train_cell_time = train_cell_time.copy()
        downSample3_train_cell_time.index = downSample3_train_cell_time.index + "_downSample3"

        cell_time = pd.concat([cell_time, downSample10_train_cell_time])
        cell_time = pd.concat([cell_time, downSample3_train_cell_time])
        _logger.info(
            f"Also add the downSample cell info to cell time dataframe and shape to {cell_time.shape}, columns is {cell_time.columns}")
        if plot_boxPlot_bool:
            try:
                plot_boxPlot_nonExpGene_percentage_whilePreprocess(adata, cell_time, donor_attr,
                                                                   special_path_str, test_donor,
                                                                   special_file_str="4DataAugmentationOnTrainByline")
                plot_boxPlot_total_count_per_cell_whilePreprocess(adata, cell_time, donor_attr,
                                                                  special_path_str, test_donor,
                                                                  special_file_str="4DataAugmentationOnTrainByline")
            except:
                print("some error while plot before boxplot.")

    # 数据数目统计
    _shape = adata.shape
    _new_shape = (0, 0)
    while _new_shape != _shape:  # make sure drop samples and genes
        _shape = adata.shape
        sc.pp.filter_cells(adata, min_genes=min_gene_num)  # drop samples with less than 20 gene expression
        sc.pp.filter_genes(adata, min_cells=min_cell_num)  # drop genes which none expression in 3 samples
        _new_shape = adata.shape
    _logger.info("After drop gene threshold: {}, cell threshold: {}, remain adata shape: {}".format(min_cell_num,
                                                                                                    min_gene_num,
                                                                                                    adata.shape))

    if plot_boxPlot_bool:
        try:
            plot_boxPlot_nonExpGene_percentage_whilePreprocess(adata, cell_time, donor_attr,
                                                               special_path_str, test_donor,
                                                               special_file_str=f"5filterGene{min_cell_num}Cell{min_gene_num}")
            plot_boxPlot_total_count_per_cell_whilePreprocess(adata, cell_time, donor_attr,
                                                              special_path_str, test_donor,
                                                              special_file_str=f"5filterGene{min_cell_num}Cell{min_gene_num}")
        except:
            print("some error while plot before boxplot.")

    if keep_sub_type_with_cell_num is not None:
        drop_out_cell = []
        _logger.info(f"Random select {keep_sub_type_with_cell_num} cells in each sub celltype; "
                     f"if one sub type number less than the threshold, keep all.")
        _cell_time = cell_time.loc[adata.obs_names]
        from collections import Counter
        _num_dic = dict(Counter(_cell_time["major_cell_type"]))
        _logger.info("In this cell type (sub cell type, number of cells): {}".format(_num_dic))
        for _cell_type, _cell_num in _num_dic.items():
            if _cell_num > keep_sub_type_with_cell_num:
                _cell_type_df = _cell_time.loc[_cell_time["major_cell_type"] == _cell_type]
                _drop_cell = _cell_type_df.sample(n=len(_cell_type_df) - keep_sub_type_with_cell_num,
                                                  random_state=0).index
                drop_out_cell += list(_drop_cell)
                _logger.info("Drop out {} cells for {}, which has {} cells before".format(len(_drop_cell), _cell_type,
                                                                                          _cell_num))
        adata = adata[adata.obs_names.drop(drop_out_cell)].copy()
        _logger.info("After drop out {}, get expression dataframe with shape (cell, gene): {}.".format(drop_out_donor,
                                                                                                       adata.shape))

    _logger.info("Drop cells with less than {} gene expression, drop genes which none expression in {} samples".format(min_gene_num, min_cell_num))

    _logger.info("After filter, get cell number: {}, gene number: {}".format(adata.n_obs, adata.n_vars))
    gene_raw_total_count = pd.DataFrame(data=adata.X.sum(axis=0), index=adata.var_names, columns=["raw_total_count"])

    # ---
    # sc.pp.normalize_total(adata, target_sum=normalized_cellTotalCount)
    adata_normalized = sc.pp.normalize_total(adata, target_sum=normalized_cellTotalCount, copy=True)
    print(f"normalized sample to {normalized_cellTotalCount}")
    if data_raw_count_bool:
        sc.pp.log1p(adata_normalized)
        print("Input data is raw count, do the log1p().")
    else:
        print("Input data is log-ed, skip the log-ed.")
    _logger.info(f"Finish normalize per cell to {normalized_cellTotalCount}, "
                 f"so that every cell has the same total count after normalization.")
    if plot_boxPlot_bool:
        try:
            plot_boxPlot_nonExpGene_percentage_whilePreprocess(adata_normalized, cell_time, donor_attr,
                                                               special_path_str, test_donor,
                                                               special_file_str=f"6NormalizeTo1e6AndLog")
            plot_boxPlot_total_count_per_cell_whilePreprocess(adata_normalized, cell_time, donor_attr,
                                                              special_path_str, test_donor,
                                                              special_file_str=f"6NormalizeTo1e6AndLog")
        except:
            print("some error while plot before boxplot.")
    sc_expression_df = pd.DataFrame(data=adata_normalized.X, columns=adata_normalized.var.index, index=adata_normalized.obs.index)
    # 2023-08-05 15:12:28 for debug
    # sc_expression_df = sc_expression_df.sample(n=6000, random_state=0)

    denseM = sc_expression_df.values
    # denseM = KNN_smoothing_start(denseM, type=KNN_smooth_type) # 2024-02-23 12:54:26 remove this useless KNN_smooth_type
    # _logger.info("Finish smooth by {} method.".format(KNN_smooth_type))
    from sklearn.preprocessing import scale
    denseM = scale(denseM.astype(float), axis=0, with_mean=True, with_std=True)
    _logger.info("Finish normalize per gene as Gaussian-dist (0, 1).")

    sc_expression_df = pd.DataFrame(data=denseM, columns=sc_expression_df.columns, index=sc_expression_df.index)

    cell_time = cell_time.loc[sc_expression_df.index]
    _logger.info("Get expression dataframe with shape (cell, gene): {}, and cell time info with shape: {}.".format(
        sc_expression_df.shape, cell_time.shape))
    try:

        _save_path = "{}{}/".format(_logger.root.handlers[0].baseFilename.replace(".log", ""), special_path_str)
        if not os.path.exists(_save_path):
            os.makedirs(_save_path)
        cell_time.to_csv(f"{_save_path}/preprocessed_cell_info.csv")
        gene_raw_total_count.to_csv(f"{_save_path}/preprocessed_gene_info.csv")
        _logger.info(f"save preprocessed_cell_info.csv and preprocessed_gene_info.csv at {_save_path}")
    except:
        print("Not save preprocessed_cell_info.csv and preprocessed_gene_info.csv")
        pass
    if return_normalized_raw_count:
        import scipy.sparse as sp
        adata.X = sp.csr_matrix(adata.X)
        adata.layers['processed'] = sc_expression_df.values
        return sc_expression_df, cell_time, adata
    else:
        return sc_expression_df, cell_time


def cosSim(x, y):
    '''
    余弦相似度
    '''
    tmp = np.sum(x * y)
    non = np.linalg.norm(x) * np.linalg.norm(y)
    return np.round(tmp / float(non), 9)


def eculidDisSim(x, y):
    '''
    欧几里得相似度
    '''
    return np.sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))


def manhattanDisSim(x, y):
    '''
    曼哈顿相似度
    '''
    return sum(abs(a - b) for a, b in zip(x, y))


def pearsonrSim(x, y):
    '''
    皮尔森相似度
    '''
    from scipy.stats import pearsonr
    return pearsonr(x, y)[0]


def trans_time(capture_time, time_standard_type, capture_time_other=None, label_dic_train=None, min_max_val=None):
    if label_dic_train is not None:  # the label dic is given by label_dic_train
        label_dic = label_dic_train
    elif time_standard_type == "log2":
        label_dic = {3: 0, 5: 0.528, 7: 0.774, 9: 0.936, 11: 1.057}
        # time = (capture_time - 2).log2() / 3
    elif time_standard_type == "0to1":
        label_dic = {3: 0, 5: 0.25, 7: 0.5, 9: 0.75, 11: 1}
        # time = (capture_time - min(capture_time)) / (max(capture_time) - min(capture_time))
    elif time_standard_type == "neg1to1":
        label_dic = {3: -1, 5: -0.5, 7: 0, 9: 0.5, 11: 1}
        # time = (capture_time - np.unique(capture_time).mean()) / (max(capture_time) - min(capture_time))
    elif time_standard_type == "labeldic":
        label_dic = {3: 0, 5: 1, 7: 2, 9: 3, 11: 4}
        # time = torch.tensor([label_dic[int(_i)] for _i in capture_time])
    elif time_standard_type == "sigmoid":
        label_dic = {3: -5, 5: -3.5, 7: 0, 9: 3.5, 11: 5}
        # time = torch.tensor([label_dic[int(_i)] for _i in capture_time])
    elif time_standard_type == "logit":
        label_dic = {3: -5, 5: -1.5, 7: 0, 9: 1.5, 11: 5}
        # time = torch.tensor([label_dic[int(_i)] for _i in capture_time])
    # if data_set=="test":
    elif time_standard_type == "acinardic":
        label_dic = {1: 0, 5: 1, 6: 2, 21: 3, 22: 4, 38: 5, 44: 6, 54: 7}
    elif time_standard_type[:6] == "embryo":
        multiple = int(time_standard_type.split("to")[-1])
        new_max = multiple
        new_min = -1 * new_max
        unique_time = np.unique(capture_time)
        if capture_time_other is not None:
            unique_time = np.concatenate((unique_time, np.unique(capture_time_other)))  # all time point
        # 计算最小值和最大值
        if min_max_val is not None:
            min_val, max_val = min_max_val
        else:
            min_val = np.min(unique_time)
            max_val = np.max(unique_time)
        # full_range = np.arange(min_val, max_val + 50, 50)


        # unique_time = sorted(list(set(unique_time.tolist() + full_range.tolist())))
        # 最小-最大归一化
        normalized_data = (unique_time - min_val) * (new_max - new_min) / (max_val - min_val) + new_min
        # normalized_data = (unique_time - min_val) / (max_val - min_val) * 2 - 1


        label_dic = {int(key): round(value, 3) for key, value in zip(unique_time, normalized_data)}
    elif time_standard_type == "organdic":  # 2023-11-02 10:50:09 add for Joy organ project
        # label_dic = {100: 1, 0: 0} # 2023-11-07 12:18:52 for "mesen" attr, don't use.
        label_dic = {i: i for i in np.unique(capture_time)}
    time = torch.tensor([label_dic[int(_i)] for _i in capture_time])
    return time, label_dic


# def donor_resort_key(item):
#     import re
#     match = re.match(r'LH(\d+)_([0-9PGT]+)', item)
#     if match:
#         num1 = int(match.group(1))
#         num2 = match.group(2)
#         if num2 == 'PGT':
#             num2 = 'ZZZ'  # 将'LH7_PGT'替换为'LH7_ZZZ'，确保'LH7_PGT'在'LH7'后面
#         return num1, num2
#     return item
#
#
# def RIFdonor_resort_key(item):
#     import re
#     match = re.match(r'RIF_(\d+)', item)
#     if match:
#         num1 = int(match.group(1))
#         return num1
#     return item


def identify_timeCorGene(sc_expression_df, cell_info, y_time_nor_tensor, donor_index_tensor, runner, experiment, trained_total_dic,
                         special_path_str,
                         config, parallel_bool=False, top_gene_num=15):
    """
    input a trained vae model (train dataset is all donor data) and the predicted cell pseudotime
    1 add pertubation to each gene: set gene expression to the min expression
    2 forward new data
    3 compare the pseudotime change between gene-non-pertubation and gene-with-pertubation
    :param sc_expression_df:
    :param y_time_nor_tensor:
    :param donor_index_tensor:
    :param runner:
    :param experiment:
    :param trained_clf_ndarray:
    :param golbal_path:
    :param file_path:
    :param latent_dim:
    :param special_path_str:
    :param config:
    :return:
    """
    save_file_path = f"{_logger.root.handlers[0].baseFilename.replace('.log', '')}{special_path_str}/"

    trained_result = trained_total_dic["total"]
    if trained_total_dic["time_continues_bool"]:  # time is continues
        _logger.info("predicted time is continuous.")
        trained_clf = torch.tensor(trained_result[0][0]).squeeze()
    else:  # time is discrete
        _logger.info("predicted time is discrete.")
        trained_clf = torch.argmax(torch.tensor(trained_result[0][0]), dim=1)
    # trained_latent = trained_result[0][1]
    # trained_latent_logVar = trained_result[0][2]
    # trained_recons = trained_result[0][4]

    gene_raw_total_count_pd = pd.read_csv(f"{save_file_path}/preprocessed_gene_info.csv", header=0, index_col=0)

    gene_list = sc_expression_df.columns
    gene_result_pd = pd.DataFrame(columns=["gene", "gene_short_name",
                                           "spearman", "spearman_pval",
                                           "pearson", "pearson_pval",
                                           "kendalltau",
                                           "kendalltau_pval",
                                           "r2",
                                           "mean_abs", "mean", "median_abs", "median",
                                           "t_test", "total_raw_count"])
    if gene_list[0].startswith("EN") and gene_list[1].startswith("EN"):
        gene_dic = geneId_geneName_dic()
    else:
        gene_dic = {item: item for item in gene_list}
    # import concurrent.futures
    # if parallel_bool:
    #     _logger.info("use parallel_bool")
    #     with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    #         # 并行执行每个基因的计算任务
    #         futures = [executor.submit(pertub_one_gene_parallel, gene, sc_expression_df.copy(), y_time_nor_tensor,
    #                                    donor_index_tensor, runner, experiment, trained_total_dic, trained_clf, gene_dic,
    #                                    gene_raw_total_count_pd.loc[gene]["raw_total_count"], special_path_str) for gene in
    #                    gene_list]
    #
    #         # 收集计算结果
    #         gene_result_pd = pd.concat([future.result() for future in futures], axis=0)

    # with multiprocessing.Pool(processes=5) as pool:
    #     processed_results = pool.starmap(pertub_one_gene, [
    #         (gene, sc_expression_df.copy(), y_time_nor_tensor, donor_index_tensor,
    #          runner, experiment, trained_total_dic, trained_clf, gene_dic,
    #          gene_raw_total_count_pd.loc[gene]["raw_total_count"], special_path_str) for gene in gene_list])
    # for _i, _pd in enumerate(processed_results):
    #     if _pd is not None:
    #         gene_result_pd = pd.concat([gene_result_pd, _pd], axis=0)

    # else:
    perturb_gene_predictedTime_pd = pd.DataFrame(index=sc_expression_df.index)
    for gene in gene_list:
        _pd, _perturbTime = pertub_one_gene(gene, sc_expression_df, y_time_nor_tensor, donor_index_tensor,
                                            runner, experiment, trained_total_dic, trained_clf, gene_dic,
                                            gene_raw_total_count_pd.loc[gene]["raw_total_count"], special_path_str,
                                            plot_geneTimeDetTime_bool=False)
        gene_result_pd = pd.concat([gene_result_pd, _pd], axis=0)
        perturb_gene_predictedTime_pd[gene] = _perturbTime
    # save result
    gene_result_pd.to_csv(f"{save_file_path}/{config['model_params']['name']}_timeCorGene_predTimeChange.csv")
    perturb_gene_predictedTime_pd.to_csv(f"{save_file_path}/After_perturb_singleGene_eachSample_predcitedTime.csv")
    cell_info["predicted_time"] = trained_clf
    cell_info["normalized_time"] = y_time_nor_tensor
    cell_info.to_csv(f"{save_file_path}/eachSample_info.csv")
    _logger.info("Time cor gene save at: {}".format(save_file_path))

    # plot det t and total count on each gene
    # plot_detTandExp(gene_result_pd.copy(), special_path_str=special_path_str)

    _logger.info(f"plot expression and detT of {len(sc_expression_df.columns)} genes.")
    # plot top 20 line
    gene_result_pd.reset_index(inplace=True)
    gene_result_pd_sorted = gene_result_pd.loc[gene_result_pd['mean'].abs().sort_values(ascending=False).index]
    top_gene_list = list(gene_result_pd_sorted["gene"][:top_gene_num])
    _logger.info(f"plot t and detT of top mean change {top_gene_num} genes.")
    plot_pd = pd.DataFrame(columns=["perturb_time", "trained_time", "gene"])
    for i in range(len(top_gene_list)):
        temp = {"det_time": np.array(perturb_gene_predictedTime_pd[top_gene_list[i]]) - np.array(trained_clf),
                "trained_time": np.array(trained_clf),
                "gene": gene_dic[top_gene_list[i]]}
        temp = pd.DataFrame(temp)
        temp = temp.sample(n=int(len(temp) / 10), random_state=42)
        plot_pd = pd.concat([plot_pd, temp], axis=0)
    plot_detTandT_line(top_gene_list, plot_pd, special_path_str, gene_dic)
    return gene_result_pd


# def pertub_one_gene_parallel(gene, sc_expression_df, y_time_nor_tensor, donor_index_tensor,
#                              runner, experiment, trained_total_dic, trained_clf, gene_dic,
#                              raw_total_count, special_path_str):
#     # 这是 pertub_one_gene 函数的并行化版本
#     _pd, _ = pertub_one_gene(gene, sc_expression_df.copy(), y_time_nor_tensor, donor_index_tensor,
#                              runner, experiment, trained_total_dic, trained_clf, gene_dic,
#                              raw_total_count, special_path_str)
#     return _pd


def pertub_one_gene(gene, sc_expression_df, y_time_nor_tensor, donor_index_tensor,
                    runner, experiment, trained_total_dic, trained_clf, gene_dic,
                    gene_raw_total_count, special_path_str, plot_geneTimeDetTime_bool=True):
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyPredict
    from scipy import stats
    # gene = gene_list[index]
    if gene not in sc_expression_df.columns:
        _logger.info("{} not in the gene list, pass".format(gene))
        return
    _logger.info(f"calculate for gene {gene}/{gene_dic[gene]}.")
    # set express to min value of this gene
    sc_expression_geneZero = sc_expression_df.copy(deep=True)
    sc_expression_geneZero[gene] = sc_expression_geneZero[gene].min()  # set to
    # make data with gene express min
    x_sc_geneZero = torch.tensor(sc_expression_geneZero.values, dtype=torch.get_default_dtype()).t()
    data_geneZero = [[x_sc_geneZero[:, i], y_time_nor_tensor[i], donor_index_tensor[i]] for i in
                     range(x_sc_geneZero.shape[1])]
    # input to model and predict
    data_geneZero = SupervisedVAEDataset_onlyPredict(predict_data=data_geneZero,
                                                     predict_batch_size=len(data_geneZero))
    check_gpu_memory(runner.device_ids[0])
    result_geneZero = runner.predict(experiment, data_geneZero)
    # get clf, mu, var, recons result
    result_clf_geneZero = result_geneZero[0][0]
    # result_latent_mu_geneZero = result_geneZero[0][1]
    # result_latent_logVar_geneZero = result_geneZero[0][2]
    # result_recons_geneZero = result_geneZero[0][4]

    if trained_total_dic["time_continues_bool"]:  # time is continues
        _logger.info("predicted time is continuous.")
        result_clf_geneZero = result_clf_geneZero.squeeze()
    else:  # time is discrete
        _logger.info("predicted time is discrete.")
        result_clf_geneZero = torch.argmax(torch.tensor(result_clf_geneZero), dim=1)

    # calculate correlation change between orginal express and gene-changed
    _, stats_result = calculate_real_predict_corrlation_score(result_clf_geneZero, trained_clf, only_str=False)
    mean_abs = abs(np.array(result_clf_geneZero) - np.array(trained_clf)).mean()
    mean = np.mean(np.array(result_clf_geneZero) - np.array(trained_clf))
    median_abs = np.median(abs(np.array(result_clf_geneZero) - np.array(trained_clf)))
    median = np.median(np.array(result_clf_geneZero) - np.array(trained_clf))
    _gene_result_df = pd.DataFrame(index=sc_expression_df.index)
    _gene_result_df["trained_clf"] = trained_clf
    _gene_result_df["result_clf_geneZero"] = result_clf_geneZero
    _gene_result_df["real_time"] = y_time_nor_tensor
    _gene_result_df["expression"] = sc_expression_df[gene]

    # 获取唯一的标签
    unique_labels = _gene_result_df["real_time"].unique()
    # 执行独立样本 t 检验并检查显著性水平
    tTest_stat_str = ""
    for label in unique_labels:
        data_label = _gene_result_df[_gene_result_df["real_time"] == label]
        t_stat, p_value = stats.ttest_ind(data_label["trained_clf"], data_label["result_clf_geneZero"])
        tTest_stat_str = tTest_stat_str + f"{str(label)}:{np.round(p_value, 5)};"
    gene_result_pd = pd.DataFrame(
        columns=["gene", "gene_short_name", "spearman", "spearman_pval", "pearson", "pearson_pval", "kendalltau",
                 "kendalltau_pval", "r2",
                 "mean_abs", "mean",
                 "median_abs", "median", "t_test", "total_raw_count"])
    gene_result_pd.loc[0] = [gene,
                             gene_dic[gene],
                             stats_result['spearman'].statistic,
                             stats_result['spearman'].pvalue,
                             stats_result['pearson'].statistic,
                             stats_result['pearson'].pvalue,
                             stats_result['kendalltau'].statistic,
                             stats_result['kendalltau'].pvalue,
                             stats_result['r2'],
                             mean_abs,
                             mean,
                             median_abs,
                             median,
                             tTest_stat_str,
                             gene_raw_total_count]
    if plot_geneTimeDetTime_bool:
        plot_time_change_gene_geneZero(f"{gene_dic[gene]}({gene})", _gene_result_df, x_str='trained_clf',
                                       y_str='result_clf_geneZero',
                                       label_str='real_time', special_path_str=special_path_str,
                                       title_str=f"SpearmanCorr {round(stats_result['spearman'].statistic, 3)}")
    gc.collect()
    return gene_result_pd, result_clf_geneZero


def geneId_geneName_dic(return_total_gene_pd_bool=False):
    gene_data = pd.read_csv("data/mouse_embryonic_development//df_gene.csv", index_col=0)
    gene_dict = gene_data.set_index('gene_id')['gene_short_name'].to_dict()
    if return_total_gene_pd_bool:
        return gene_dict, gene_data
    else:
        return gene_dict


def test_on_newDataset(sc_expression_train, data_golbal_path, result_save_path, KNN_smooth_type, runner, experiment,
                       config, latent_dim,
                       special_path_str, time_standard_type, test_data_path):
    """
    2023-07-13 14:39:38 dandan share a new dataset (download from public database, with epi and fibro, different platfrom: ct and 10X)
    use all dandan data as train data to train a model and test on the new dataset.
    :param sc_expression_train:
    :param data_golbal_path:
    :param KNN_smooth_type:
    :param runner:
    :param experiment:
    :param config:
    :param latent_dim:
    :param special_path_str:
    :param time_standard_type:
    :return:
    """
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyPredict
    _logger.info("Test on new dataset.")

    gene_list = sc_expression_train.columns

    _logger.info("Test on dataset: {}".format(test_data_path))
    gene_dic = dict()
    file_name_test = test_data_path + "/data_count.csv"
    cell_info_file_test = test_data_path + "/cell_info.csv"
    sc_expression_df_test, cell_time_test = preprocessData_and_dropout_some_donor_or_gene(data_golbal_path, file_name_test, cell_info_file_test,
                                                                                          gene_list=gene_list, min_cell_num=0,
                                                                                          min_gene_num=10)

    loss_gene = list(set(gene_list) - set(sc_expression_df_test.columns))
    _logger.info("loss {} gene in test data, set them to 0".format(len(loss_gene)))
    _logger.info("test data don't have gene: {}".format(loss_gene))
    gene_dic["model_gene"] = list(gene_list)
    gene_dic["loss_gene"] = list(loss_gene)
    gene_dic["testdata_gene"] = list(sc_expression_df_test.columns)
    for _g in loss_gene:
        sc_expression_df_test[_g] = 0
    x_sc_test = torch.tensor(sc_expression_df_test.values, dtype=torch.get_default_dtype()).t()
    _logger.info("Set x_sc_test data with shape (gene, cells): {}".format(x_sc_test.shape))
    train_data = [[x_sc_test[:, i], torch.tensor(0), torch.tensor(0)] for i in range(x_sc_test.shape[1])]
    data_test = SupervisedVAEDataset_onlyPredict(predict_data=train_data, predict_batch_size=len(train_data))
    test_result = runner.predict(experiment, data_test)
    test_clf_result, test_latent_mu_result, test_latent_log_var_result = test_result[0][0], test_result[0][1], \
        test_result[0][2]
    if test_clf_result.shape[1] == 1:
        test_clf_result = test_clf_result.squeeze()
        test_clf_result_df = pd.DataFrame(data=test_clf_result, index=sc_expression_df_test.index,
                                          columns=["pseudotime"])
        _logger.info("Time type is continues.")
    else:
        test_clf_result = test_clf_result.squeeze()
        test_clf_result = np.argmax(test_clf_result, axis=1)
        test_clf_result_df = pd.DataFrame(data=test_clf_result, index=sc_expression_df_test.index,
                                          columns=["pseudotime"])
        _logger.info("Time type is discrete.")

    _save_path = "{}{}/".format(_logger.root.handlers[0].baseFilename.replace(".log", ""), special_path_str)
    if not os.path.exists(_save_path):
        os.makedirs(_save_path)
    _save_file_name = "{}/{}_testOnExternal_{}.csv".format(_save_path, config['model_params']['name'],
                                                           test_data_path.replace("/", "").replace(".rds", "").replace(" ", "_"))

    test_clf_result_df.to_csv(_save_file_name, sep="\t")
    import json
    with open(_save_file_name.replace(".csv", "geneUsed.json"), 'w') as f:
        json.dump(gene_dic, f)
    _logger.info("result save at: {}".format(_save_file_name))


def test_on_newDonor(test_donor_name, sc_expression_test, runner, experiment, predict_donors_dic):
    """
    2023-07-16 15:05:18 dandan share 10 RIF donor, test on these donor
    use all dandan data as train data to train a model and test on the new dataset.
    :param sc_expression_train:
    :param data_golbal_path:
    :param KNN_smooth_type:
    :param runner:
    :param experiment:
    :param config:
    :param latent_dim:
    :param special_path_str:
    :param time_standard_type:
    :return:
    """
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyPredict

    x_sc_test = torch.tensor(sc_expression_test.values, dtype=torch.get_default_dtype()).t()
    _logger.info("Set x_sc_test data with shape (gene, cells): {}".format(x_sc_test.shape))
    test_data = [[x_sc_test[:, i], torch.tensor(0), torch.tensor(0)] for i in range(x_sc_test.shape[1])]
    data_test = SupervisedVAEDataset_onlyPredict(predict_data=test_data, predict_batch_size=len(test_data))
    test_result = runner.predict(experiment, data_test)
    test_clf_result, test_latent_mu_result, test_latent_log_var_result = test_result[0][0], test_result[0][1], \
        test_result[0][2]
    # time is continues, supervise_vae_regressionclfdecoder  supervise_vae_regressionclfdecoder_of_sublatentspace
    if test_clf_result.shape[1] == 1:
        _logger.info("predicted time of test donor is continuous.")
        predict_donors_dic[test_donor_name] = pd.DataFrame(data=np.squeeze(test_clf_result),
                                                           index=sc_expression_test.index, columns=["pseudotime"])
    else:  # time is discrete and probability on each time point, supervise_vae supervise_vae_noclfdecoder
        _logger.info("predicted time of test donor is discrete.")
        labels_pred = torch.argmax(torch.tensor(test_clf_result), dim=1)
        predict_donors_dic[test_donor_name] = pd.DataFrame(data=labels_pred.cpu().numpy(),
                                                           index=sc_expression_test.index, columns=["pseudotime"])
    test_latent_info_dic = {"mu": test_latent_mu_result,
                            "log_var": test_latent_log_var_result,
                            "label_true": np.zeros(len(test_latent_mu_result)),
                            "label_dic": {"test": "test"},
                            "donor_index": np.zeros(len(test_latent_mu_result)) - 1,
                            "donor_dic": {test_donor_name: -1}}
    return predict_donors_dic, test_clf_result, test_latent_info_dic


def one_fold_test(fold, donor_list, sc_expression_df, donor_dic, batch_dic,
                  special_path_str,
                  cell_time, time_standard_type, config, train_epoch_num,
                  donor_str="donor", time_str="time",
                  device=None,
                  plot_trainingLossLine=True, plot_tags=["train_clf_loss_epoch", "val_clf_loss", "test_clf_loss_epoch"],
                  plot_latentSpaceUmap=True,
                  time_saved_asFloat=False, batch_size=None,
                  max_attempts=10000000,
                  checkpoint_file=None, min_max_val=None,
                  recall_used_device=False, recall_predicted_mu=False):
    """
    2024-03-17 20:05:28 in this file, have a new version for this function, kindly check.
    use donor_list[fold] as test data, and use other donors as train data,
    then train a vae model with {latent dim} in latent space
    :param fold:
    :param donor_list:
    :param sc_expression_df:
    :param donor_dic:
    :param golbal_path:
    :param file_path:
    :param latent_dim:
    :param special_path_str:
    :param cell_time:
    :param time_standard_type:
    :param config:
    :param args:
    :param predict_donors_dic:
    :param device:
    :param batch_dim:
    :param plot_trainingLossLine:
    :param plot_latentSpaceUmap:
    :return:
    """
    from TemporalVAE.model_master.experiment_temporalVAE import temporalVAEExperiment
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyPredict
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pathlib import Path
    predict_donors_dic = dict()
    # ----------------------------------------split Train and Test dataset-----------------------------------------------------
    _logger.info("the {}/{} fold train, use donor-{} as test set".format(fold + 1, len(donor_list), donor_list[fold]))
    subFold_save_file_path = "{}{}/{}/".format(_logger.root.handlers[0].baseFilename.replace(".log", ""), special_path_str,
                                               donor_list[fold])

    if not os.path.exists(subFold_save_file_path):
        os.makedirs(subFold_save_file_path)
    if checkpoint_file is not None:
        gene_list_file = "data/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/gene_info.csv"
        gene_list = list(pd.read_csv(gene_list_file, sep="\t")["gene_id"])
        _logger.info(f"checkpoint file is not none, make the train data have the same size with checkpoint file and "
                     f"make the gene with the same order, so use the gene list file: "
                     f"{gene_list_file}.")
        sc_expression_df = sc_expression_df.reindex(columns=gene_list).fillna(0)
        _logger.info(f"sc_expression_df re-size: {sc_expression_df.shape}")
    sc_expression_train = sc_expression_df.loc[cell_time.index[cell_time[donor_str] != donor_list[fold]]]
    sc_expression_test = sc_expression_df.loc[cell_time.index[cell_time[donor_str] == donor_list[fold]]]

    # we need to transpose data to correct its shape
    x_sc_train = torch.tensor(sc_expression_train.values, dtype=torch.get_default_dtype()).t()
    x_sc_test = torch.tensor(sc_expression_test.values, dtype=torch.get_default_dtype()).t()
    _logger.info("Set x_sc_train data with shape (gene, cells): {}".format(x_sc_train.shape))
    _logger.info("Set x_sc_test data with shape (gene, cells): {}".format(x_sc_test.shape))

    # trans y_time

    if time_saved_asFloat:  # 2023-08-06 13:25:48 trans 8.50000 to 850; 9.2500000 to 925; easy to  manipulate.
        cell_time_dic = dict(zip(cell_time.index, cell_time[time_str]))
        y_time_train = x_sc_train.new_tensor(np.array(sc_expression_train.index.map(cell_time_dic) * 100).astype(int))
        y_time_test = x_sc_test.new_tensor(np.array(sc_expression_test.index.map(cell_time_dic) * 100).astype(int))

    else:
        y_time_train = x_sc_train.new_tensor(
            [int(cell_time.loc[_cell_name][time_str].split("_")[0].replace("LH", "")) for _cell_name in
             sc_expression_train.index.values])
        y_time_test = x_sc_test.new_tensor(
            [int(cell_time.loc[_cell_name][time_str].split("_")[0].replace("LH", "")) for _cell_name in
             sc_expression_test.index.values])
    donor_index_train = x_sc_train.new_tensor(
        [int(batch_dic[cell_time.loc[_cell_name][donor_str]]) for _cell_name in sc_expression_train.index.values])
    donor_index_test = x_sc_test.new_tensor(
        [int(batch_dic[cell_time.loc[_cell_name][donor_str]]) for _cell_name in sc_expression_test.index.values])

    # for classification model with discrete time cannot use sigmoid and logit time type
    y_time_nor_train, label_dic = trans_time(y_time_train, time_standard_type, capture_time_other=y_time_test, min_max_val=min_max_val)
    y_time_nor_test, label_dic = trans_time(y_time_test, time_standard_type, label_dic_train=label_dic)
    _logger.info("label dictionary: {}".format(label_dic))
    _logger.info("Normalize train y_time_nor_train type: {}, with y_time_nor_train lable: {}, shape: {}, \ndetail: {}"
                 .format(time_standard_type, np.unique(y_time_train), y_time_train.shape, np.unique(y_time_nor_train)))
    _logger.info("Normalize test y_time_nor_train type: {}, with y_time_nor_train lable: {}, shape: {}, \ndetail: {}"
                 .format(time_standard_type, np.unique(y_time_test), y_time_test.shape, np.unique(y_time_nor_test)))

    # ------------------------------------------- Set up VAE model and Start train process -------------------------------------------------
    _logger.info("Start training with epoch: {}. ".format(train_epoch_num))

    # if int(config['model_params']['in_channels']) == 0:
    config['model_params']['in_channels'] = x_sc_train.shape[0]
    tb_logger = TensorBoardLogger(save_dir=subFold_save_file_path,
                                  name=config['model_params']['name'], )

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    MyVAEModel = vae_models[config['model_params']['name']](**config['model_params'])
    if checkpoint_file is not None:
        _logger.info(f"use checkpoint file: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        # 去掉每层名字前面的 "model."
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            # 去掉前缀 "model."
            if key.startswith('model.'):
                key = key[6:]
            new_state_dict[key] = value
        MyVAEModel.load_state_dict(new_state_dict)
    ## 打印模型的权重和偏置
    # for name, param in MyVAEModel.named_parameters():
    #     print(name, param)
    train_data = [[x_sc_train[:, i], y_time_nor_train[i], donor_index_train[i]] for i in range(x_sc_train.shape[1])]
    test_data = [[x_sc_test[:, i], y_time_nor_test[i], donor_index_test[i]] for i in range(x_sc_test.shape[1])]
    if batch_size is None:
        _logger.info("batch size is none, so don't set batch")
        data = SupervisedVAEDataset(train_data=train_data, val_data=test_data, test_data=test_data, predict_data=test_data,
                                    train_batch_size=len(train_data), val_batch_size=len(test_data),
                                    test_batch_size=len(test_data), predict_batch_size=len(test_data),
                                    label_dic=label_dic)
    else:
        _logger.info("batch size is {}".format(batch_size))
        data = SupervisedVAEDataset(train_data=train_data, val_data=test_data, test_data=test_data, predict_data=test_data,
                                    train_batch_size=batch_size, val_batch_size=batch_size,
                                    test_batch_size=batch_size, predict_batch_size=batch_size,
                                    label_dic=label_dic)
    # data.setup("train")
    experiment = temporalVAEExperiment(MyVAEModel, config['exp_params'])

    # 创建一个 LearningRateMonitor 回调实例
    lr_monitor = LearningRateMonitor()
    # 2023-09-07 20:34:25 add check memory
    check_memory(max_attempts=max_attempts)
    device = auto_select_gpu_and_cpu(max_attempts=max_attempts)
    _logger.info("Auto select run on {}".format(device))

    runner = Trainer(logger=tb_logger, log_every_n_steps=1,
                     callbacks=[
                         lr_monitor,
                         ModelCheckpoint(save_top_k=2,
                                         dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                         monitor="val_loss",
                                         save_last=True),
                     ],
                     # check_val_every_n_epoch=1, val_check_interval=1,
                     devices=[int(device.split(":")[-1])],
                     accelerator="gpu", max_epochs=train_epoch_num
                     )

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment, data)

    # test the model
    _logger.info("this epoch final, on test data:{}".format(runner.test(experiment, data)))
    # predict on train data
    data_predict = SupervisedVAEDataset_onlyPredict(predict_data=train_data, predict_batch_size=len(train_data))
    train_result = runner.predict(experiment, data_predict)
    train_clf_result, train_latent_mu_result, train_latent_log_var_result = train_result[0][0], train_result[0][1], \
        train_result[0][2]
    # predict on test data
    data_test = SupervisedVAEDataset_onlyPredict(predict_data=test_data, predict_batch_size=len(test_data))
    test_result = runner.predict(experiment, data_test)
    test_clf_result, test_latent_mu_result, test_latent_log_var_result = test_result[0][0], test_result[0][1], \
        test_result[0][2]
    if (np.isnan(np.squeeze(test_clf_result)).any()):
        _logger.info("The Array contain NaN values")
    else:
        _logger.info("The Array does not contain NaN values")
    if test_clf_result.shape[1] == 1:
        # time is continues, supervise_vae_regressionclfdecoder  supervise_vae_regressionclfdecoder_of_sublatentspace
        _logger.info("predicted time of test donor is continuous.")
        try:
            predict_donors_dic[donor_list[fold]] = pd.DataFrame(data=np.squeeze(test_clf_result, axis=1),
                                                                index=sc_expression_test.index, columns=["pseudotime"])
        except:
            print("error here")
    else:  # time is discrete and probability on each time point, supervise_vae supervise_vae_noclfdecoder
        _logger.info("predicted time of test donor is discrete.")
        labels_pred = torch.argmax(torch.tensor(test_clf_result), dim=1)
        predict_donors_dic[donor_list[fold]] = pd.DataFrame(data=labels_pred.cpu().numpy(),
                                                            index=sc_expression_test.index, columns=["pseudotime"])
    # acc = torch.tensor(torch.sum(labels_pred == y_time_nor_test).item() / (len(y_time_nor_test) * 1.0))
    # # ---------------------------------------------- plot sub result of training process for check  --------------------------------------------------
    if plot_trainingLossLine:
        _logger.info("Plot training loss line for check.")

        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        tags = EventAccumulator(tb_logger.log_dir).Reload().Tags()['scalars']
        _logger.info("All tags in logger: {}".format(tags))
        # Retrieve and print the metric results
        # plot_tag_list = ["train_loss_epoch", "val_loss", "test_loss_epoch"]
        # plot_training_loss_for_tags(tb_logger, plot_tag_list, special_str="")
        plot_tag_list = list(set(plot_tags) & set(tags))
        _logger.info(f"plot tags {plot_tag_list}")
        plot_training_loss_for_tags(tb_logger, plot_tag_list, special_str=donor_list[fold], title=donor_list[fold])
        # plot_tag_list = ["train_Reconstruction_loss_epoch", "val_Reconstruction_loss", "test_Reconstruction_loss_epoch"]
        # plot_training_loss_for_tags(tb_logger, plot_tag_list, special_str="")
        try:
            plot_tag_list = ["train_batchEffect_loss_epoch", "val_batchEffect_loss", "test_batchEffect_loss_epoch"]
            plot_training_loss_for_tags(tb_logger, plot_tag_list, special_str=donor_list[fold], title=donor_list[fold])
        except:
            _logger.info("No batchEffect decoder.")
    else:
        _logger.info("Don't plot training loss line for check.")

    # # # ---------------------------------------------- plot sub latent space of sub model for check  --------------------------------------------------
    if plot_latentSpaceUmap:
        _logger.info("Plot each fold's UMAP of latent space for check.")
        umap_vae_latent_space(train_latent_mu_result, y_time_nor_train, label_dic, special_path_str, config,
                              special_str="trainData_mu_time", drop_batch_dim=0)
        umap_vae_latent_space(train_latent_mu_result, donor_index_train, donor_dic, special_path_str, config,
                              special_str="trainData_mu_donor", drop_batch_dim=0)
        umap_vae_latent_space(train_latent_log_var_result, y_time_nor_train, label_dic, special_path_str, config,
                              special_str="trainData_logVar_time", drop_batch_dim=0)
        umap_vae_latent_space(train_latent_log_var_result, donor_index_train, donor_dic, special_path_str, config,
                              special_str="trainData_logVar_donor", drop_batch_dim=0)
        try:
            umap_vae_latent_space(test_latent_mu_result, y_time_nor_test, label_dic, special_path_str, config,
                                  special_str="testData_mu_time", drop_batch_dim=0)
            umap_vae_latent_space(test_latent_mu_result, donor_index_test, donor_dic, special_path_str, config,
                                  special_str="testData_mu_donor", drop_batch_dim=0)
            umap_vae_latent_space(test_latent_log_var_result, y_time_nor_test, label_dic, special_path_str, config,
                                  special_str="testData_logVar_time", drop_batch_dim=0)
            umap_vae_latent_space(test_latent_log_var_result, donor_index_test, donor_dic, special_path_str, config,
                                  special_str="testData_logVar_donor", drop_batch_dim=0)
        except:
            _logger.info("Too few test cells, can't plot umap of the latent space: {}.".format(len(y_time_nor_test)))
    else:
        _logger.info("Don't plot each fold's UMAP of latent space for check.")

    # ---------------------------------------------- save sub model parameters for check  --------------------------------------------------
    _logger.info(
        "encoder and decoder structure: {}".format({"encoder": MyVAEModel.encoder, "decoder": MyVAEModel.decoder}))
    _logger.info("clf-decoder: {}".format({"clf-decoder": MyVAEModel.clf_decoder}))
    torch.save(MyVAEModel, tb_logger.root_dir + "/version_" + str(tb_logger.version) + '/model.pth')
    # _logger.info("detail information about structure save at： {}".format(tb_logger.root_dir + "/version_" + str(tb_logger.version) + '/model.pth'))

    del MyVAEModel
    del runner
    del experiment
    # 清除CUDA缓存
    torch.cuda.empty_cache()
    if recall_used_device:
        return predict_donors_dic, test_clf_result, label_dic, device
    elif recall_predicted_mu:
        return predict_donors_dic, test_clf_result, label_dic, (train_latent_mu_result, test_latent_mu_result)
    else:
        return predict_donors_dic, test_clf_result, label_dic


def one_fold_test_adversarialTrain(fold, donor_list, sc_expression_df, donor_dic, batch_dic,
                                   special_path_str,
                                   cell_time, time_standard_type, config, train_epoch_num,
                                   donor_str="donor", time_str="time",
                                   plot_trainingLossLine=True,
                                   plot_tags=["train_clf_loss_epoch", "val_clf_loss", "test_clf_loss_epoch"],
                                   plot_latentSpaceUmap=True,
                                   time_saved_asFloat=False, batch_size=None,
                                   max_attempts=10000000,
                                   checkpoint_file=None, min_max_val=None):
    """
    use donor_list[fold] as test data, and use other donors as train data,
    then train a vae model with {latent dim} in latent space
    :param fold:
    :param donor_list:
    :param sc_expression_df:
    :param donor_dic:
    :param special_path_str:
    :param cell_time:
    :param time_standard_type:
    :param config:
    :param args:
    :param predict_donors_dic:
    :param device:
    :param plot_trainingLossLine:
    :param plot_latentSpaceUmap:
    :return:
    """
    from TemporalVAE.model_master.experiment_adversarial import VAEXperiment_adversarial
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyPredict
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pathlib import Path
    predict_donor_dic = dict()
    # ----------------------------------------split Train and Test dataset-----------------------------------------------------
    _logger.info("the {}/{} fold train, use donor-{} as test set".format(fold + 1, len(donor_list), donor_list[fold]))
    # subFold_save_file_path = "{}/{}/vae_prediction_result_nofilterTest_dim{}{}/{}/".format(golbal_path, file_path,
    #                                                                                        latent_dim, special_path_str,
    #                                                                                        donor_list[fold])
    # 2023-08-17 15:58:51 change save path
    try:
        subFold_save_file_path = "{}{}/{}/".format(_logger.root.handlers[0].baseFilename.replace(".log", ""), special_path_str,
                                                   donor_list[fold])
    except:
        print("error")
    if not os.path.exists(subFold_save_file_path):
        os.makedirs(subFold_save_file_path)
    if checkpoint_file is not None:
        gene_list_file = "data/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/gene_info.csv"
        gene_list = list(pd.read_csv(gene_list_file, sep="\t")["gene_id"])
        _logger.info(f"checkpoint file is not none, make the train data have the same size with checkpoint file and "
                     f"make the gene with the same order, so use the gene list file: "
                     f"{gene_list_file}.")
        sc_expression_df = sc_expression_df.reindex(columns=gene_list).fillna(0)
        _logger.info(f"sc_expression_df re-size: {sc_expression_df.shape}")
    sc_expression_train = sc_expression_df.loc[cell_time.index[cell_time[donor_str] != donor_list[fold]]]
    sc_expression_test = sc_expression_df.loc[cell_time.index[cell_time[donor_str] == donor_list[fold]]]

    # we need to transpose data to correct its shape
    x_sc_train = torch.tensor(sc_expression_train.values, dtype=torch.get_default_dtype()).t()
    x_sc_test = torch.tensor(sc_expression_test.values, dtype=torch.get_default_dtype()).t()
    _logger.info("Set x_sc_train data with shape (gene, cells): {}".format(x_sc_train.shape))
    _logger.info("Set x_sc_test data with shape (gene, cells): {}".format(x_sc_test.shape))

    # trans y_time

    if time_saved_asFloat:  # 2023-08-06 13:25:48 trans 8.50000 to 850; 9.2500000 to 925; easy to  manipulate.
        cell_time_dic = dict(zip(cell_time.index, cell_time[time_str]))
        y_time_train = x_sc_train.new_tensor(np.array(sc_expression_train.index.map(cell_time_dic) * 100).astype(int))
        y_time_test = x_sc_test.new_tensor(np.array(sc_expression_test.index.map(cell_time_dic) * 100).astype(int))
    else:
        y_time_train = x_sc_train.new_tensor(
            [int(cell_time.loc[_cell_name][time_str].split("_")[0].replace("LH", "")) for _cell_name in
             sc_expression_train.index.values])
        y_time_test = x_sc_test.new_tensor(
            [int(cell_time.loc[_cell_name][time_str].split("_")[0].replace("LH", "")) for _cell_name in
             sc_expression_test.index.values])

    # get each cell batch info
    cell_donor_dic = dict(zip(cell_time.index, cell_time[donor_str]))
    donor_index_train = sc_expression_train.index.map(cell_donor_dic)
    donor_index_train = x_sc_train.new_tensor([int(batch_dic[_key]) for _key in donor_index_train])
    donor_index_test = sc_expression_test.index.map(cell_donor_dic)
    donor_index_test = x_sc_test.new_tensor([int(batch_dic[_key]) for _key in donor_index_test])

    # for classification model with discrete time cannot use sigmoid and logit time type
    y_time_nor_train, label_dic = trans_time(y_time_train, time_standard_type, capture_time_other=y_time_test, min_max_val=min_max_val)
    y_time_nor_test, label_dic = trans_time(y_time_test, time_standard_type, label_dic_train=label_dic)
    _logger.info("label dictionary: {}".format(label_dic))
    _logger.info("Normalize train y_time_nor_train type: {}, with y_time_nor_train lable: {}, shape: {}, \ndetail: {}"
                 .format(time_standard_type, np.unique(y_time_train), y_time_train.shape, np.unique(y_time_nor_train)))
    _logger.info("Normalize test y_time_nor_train type: {}, with y_time_nor_train lable: {}, shape: {}, \ndetail: {}"
                 .format(time_standard_type, np.unique(y_time_test), y_time_test.shape, np.unique(y_time_nor_test)))

    # ------------------------------------------- Set up VAE model and Start train process -------------------------------------------------
    _logger.info("Start training with epoch: {}. ".format(train_epoch_num))

    # if int(config['model_params']['in_channels']) == 0:
    config['model_params']['in_channels'] = x_sc_train.shape[0]
    _logger.info("batch effect dic: {}".format(batch_dic))
    config['model_params']['batch_num'] = len(set(batch_dic.values()))

    tb_logger = TensorBoardLogger(save_dir=subFold_save_file_path,
                                  name=config['model_params']['name'], )

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    MyVAEModel = vae_models[config['model_params']['name']](**config['model_params'])

    if checkpoint_file is not None:  # 2024-03-17 20:15:39 add
        _logger.info(f"use checkpoint file: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        # 去掉每层名字前面的 "model."
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            # 去掉前缀 "model."
            if key.startswith('model.'):
                key = key[6:]
            new_state_dict[key] = value
        # MyVAEModel.load_state_dict(new_state_dict)
        MyVAEModel.load_state_dict(new_state_dict, strict=False)

    ## 打印模型的权重和偏置
    # for name, param in MyVAEModel.named_parameters():
    #     print(name, param)
    train_data = [[x_sc_train[:, i], y_time_nor_train[i], donor_index_train[i]] for i in range(x_sc_train.shape[1])]
    test_data = [[x_sc_test[:, i], y_time_nor_test[i], donor_index_test[i]] for i in range(x_sc_test.shape[1])]
    if batch_size is None:
        _logger.info("batch size is none, so don't set batch")
        data = SupervisedVAEDataset(train_data=train_data, val_data=test_data, test_data=test_data, predict_data=test_data,
                                    train_batch_size=len(train_data), val_batch_size=len(test_data),
                                    test_batch_size=len(test_data), predict_batch_size=len(test_data),
                                    label_dic=label_dic)
    else:
        _logger.info("batch size is {}".format(batch_size))
        data = SupervisedVAEDataset(train_data=train_data, val_data=test_data, test_data=test_data, predict_data=test_data,
                                    train_batch_size=batch_size, val_batch_size=batch_size,
                                    test_batch_size=batch_size, predict_batch_size=batch_size,
                                    label_dic=label_dic)
    experiment = VAEXperiment_adversarial(MyVAEModel, config['exp_params'])
    lr_monitor = LearningRateMonitor()
    # 2023-09-07 20:31:43 setting check of memory and cuda
    check_memory(max_attempts=max_attempts)
    device = auto_select_gpu_and_cpu(max_attempts=max_attempts)
    _logger.info("Auto select run on {}".format(device))

    runner = Trainer(logger=tb_logger, log_every_n_steps=1,
                     callbacks=[
                         lr_monitor,
                         ModelCheckpoint(save_top_k=2,
                                         dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                         monitor="val_loss",
                                         save_last=True),
                     ],
                     devices=[int(device.split(":")[-1])],
                     accelerator="gpu", max_epochs=train_epoch_num
                     )

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"======= Training {config['model_params']['name']} =======")
    # step 0 train with 2 decoder

    runner.fit(experiment, data)
    # step 1 fix encoder, train a classifier for batch effect
    # step 2 fix classifier, retrain model, add loss of classifier and hope the batch classification as bad as possible.

    # test the model
    _logger.info("this epoch final, on test data:{}".format(donor_list[fold]))
    # predict on train data
    data_predict = SupervisedVAEDataset_onlyPredict(predict_data=train_data, predict_batch_size=len(train_data))
    train_result = runner.predict(experiment, data_predict)
    train_clf_result, train_latent_mu_result, train_latent_log_var_result = train_result[0][0], train_result[0][1], \
        train_result[0][2]
    # predict on test data
    data_test = SupervisedVAEDataset_onlyPredict(predict_data=test_data, predict_batch_size=len(test_data))
    test_result = runner.predict(experiment, data_test)
    test_clf_result, test_latent_mu_result, test_latent_log_var_result = test_result[0][0], test_result[0][1], \
        test_result[0][2]

    if (np.isnan(np.squeeze(test_clf_result)).any()):
        _logger.info("The Array contain NaN values")
    else:
        _logger.info("The Array does not contain NaN values")
    if test_clf_result.shape[1] == 1:
        # time is continues, supervise_vae_regressionclfdecoder  supervise_vae_regressionclfdecoder_of_sublatentspace
        _logger.info("predicted time of test donor is continuous.")
        try:
            predict_donor_dic[donor_list[fold]] = pd.DataFrame(data=np.squeeze(test_clf_result, axis=1),
                                                               index=sc_expression_test.index, columns=["pseudotime"])
        except:
            print("error here")
    else:  # time is discrete and probability on each time point, supervise_vae supervise_vae_noclfdecoder
        _logger.info("predicted time of test donor is discrete.")
        labels_pred = torch.argmax(torch.tensor(test_clf_result), dim=1)
        predict_donor_dic[donor_list[fold]] = pd.DataFrame(data=labels_pred.cpu().numpy(),
                                                           index=sc_expression_test.index, columns=["pseudotime"])
    # ----------------------------------- plot sub result of training process for check  -------------------
    if plot_trainingLossLine:
        _logger.info("Plot training loss line for check.")

        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        tags = EventAccumulator(tb_logger.log_dir).Reload().Tags()['scalars']
        _logger.info("All tags in logger: {}".format(tags))
        # Retrieve and print the metric results
        plot_tag_list = list(set(plot_tags) & set(tags))
        _logger.info(f"plot tags {plot_tag_list}")
        plot_training_loss_for_tags(tb_logger, plot_tag_list, special_str=donor_list[fold], title=donor_list[fold])
        try:
            plot_tag_list = ["train_batchEffect_loss_epoch", "val_batchEffect_loss"]
            plot_training_loss_for_tags(tb_logger, plot_tag_list, special_str="")
        except:
            _logger.info("No batchEffect decoder.")
        try:
            plot_training_loss_for_tags(tb_logger, ["train_clf_loss_epoch", "val_clf_loss"], special_str="")
        except:
            pass
    else:
        _logger.info("Don't plot training loss line for check.")

    # # # ---------------------------------------------- plot sub latent space of sub model for check  --------------------------------------------------
    if plot_latentSpaceUmap:
        _logger.info("Plot each fold's UMAP of latent space for check.")

        umap_vae_latent_space(train_latent_mu_result, y_time_nor_train, batch_dic, special_path_str, config,
                              special_str="trainData_mu_batch", drop_batch_dim=0)
        umap_vae_latent_space(train_latent_mu_result, y_time_nor_train, label_dic, special_path_str, config,
                              special_str="trainData_mu_time", drop_batch_dim=0)
        umap_vae_latent_space(train_latent_mu_result, donor_index_train, donor_dic, special_path_str, config,
                              special_str="trainData_mu_donor", drop_batch_dim=0)
        umap_vae_latent_space(train_latent_log_var_result, y_time_nor_train, batch_dic, special_path_str, config,
                              special_str="trainData_logVar_batch", drop_batch_dim=0)
        umap_vae_latent_space(train_latent_log_var_result, y_time_nor_train, label_dic, special_path_str, config,
                              special_str="trainData_logVar_time", drop_batch_dim=0)
        umap_vae_latent_space(train_latent_log_var_result, donor_index_train, donor_dic, special_path_str, config,
                              special_str="trainData_logVar_donor", drop_batch_dim=0)

        try:
            umap_vae_latent_space(test_latent_mu_result, y_time_nor_test, batch_dic, special_path_str, config,
                                  special_str="testData_mu_batch", drop_batch_dim=0)
            umap_vae_latent_space(test_latent_mu_result, y_time_nor_test, label_dic, special_path_str, config,
                                  special_str="testData_mu_time", drop_batch_dim=0)
            umap_vae_latent_space(test_latent_mu_result, donor_index_test, donor_dic, special_path_str, config,
                                  special_str="testData_mu_donor", drop_batch_dim=0)
            umap_vae_latent_space(test_latent_log_var_result, y_time_nor_test, batch_dic, special_path_str, config,
                                  special_str="testData_logVar_batch", drop_batch_dim=0)
            umap_vae_latent_space(test_latent_log_var_result, y_time_nor_test, label_dic, special_path_str, config,
                                  special_str="testData_logVar_time", drop_batch_dim=0)
            umap_vae_latent_space(test_latent_log_var_result, donor_index_test, donor_dic, special_path_str, config,
                                  special_str="testData_logVar_donor", drop_batch_dim=0)
        except:
            _logger.info("Too few test cells, can't plot umap of the latent space: {}.".format(len(y_time_nor_test)))
    else:
        _logger.info("Don't plot each fold's UMAP of latent space for check.")

    # ---------------------------------------------- save sub model parameters for check  --------------------------------------------------
    _logger.info(
        "encoder and decoder structure: {}".format({"encoder": MyVAEModel.encoder, "decoder": MyVAEModel.decoder}))
    try:
        _logger.info("clf-decoder structure: {}".format({"encoder": MyVAEModel.clf_decoder}))
        _logger.info("batch-decoder structure: {}".format({"encoder": MyVAEModel.batchEffect_decoder}))
    except:
        _logger.info("error here?")
    torch.save(MyVAEModel, tb_logger.root_dir + "/version_" + str(tb_logger.version) + '/model.pth')
    # _logger.info("detail information about structure save at： {}".format(tb_logger.root_dir + "/version_" + str(tb_logger.version) + '/model.pth'))

    del MyVAEModel
    del runner
    del experiment
    # 清除CUDA缓存
    torch.cuda.empty_cache()
    return predict_donor_dic, test_clf_result, label_dic


def onlyTrain_model(sc_expression_df, donor_dic,
                    special_path_str,
                    cell_time,
                    time_standard_type, config, train_epoch_num, device=None, batch_dim=0, plot_latentSpaceUmap=True,
                    time_saved_asFloat=False,
                    batch_size=None, max_attempts=10000000, adversarial_bool=False, batch_dic=None,
                    donor_str="donor", time_str="time", checkpoint_file=None, min_max_val=None,
                    plot_trainingLossLine=False):
    """
    use all donors as train data,
    then train a vae model with {latent dim} in latent space
    :param sc_expression_df:
    :param donor_dic:
    :param golbal_path:
    :param file_path:
    :param latent_dim:
    :param special_path_str:
    :param cell_time:
    :param time_standard_type:
    :param config:
    :param train_epoch_num:
    :param device:
    :param batch_dim:
    :param plot_latentSpaceUmap:
    :return:
    """
    from TemporalVAE.model_master.experiment_adversarial import VAEXperiment_adversarial
    from TemporalVAE.model_master.experiment_temporalVAE import temporalVAEExperiment
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyPredict, SupervisedVAEDataset_onlyTrain
    from pytorch_lightning import Trainer
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning import seed_everything
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pathlib import Path
    # _logger.info("calculate time cor gene. Use whole data retrain a model, predict average and medium change of each gene between normal-express and non-express")
    subFold_save_file_path = "{}{}/wholeData/".format(_logger.root.handlers[0].baseFilename.replace(".log", ""),
                                                      special_path_str)
    if not os.path.exists(subFold_save_file_path):
        os.makedirs(subFold_save_file_path)
    if checkpoint_file is not None:
        gene_list_file = "data/mouse_embryonic_development/preprocess_adata_JAX_dataset_combine_minGene100_minCell50_hvg1000/gene_info.csv"
        gene_list = list(pd.read_csv(gene_list_file, sep="\t")["gene_id"])
        _logger.info(f"checkpoint file is not none, make the train data have the same size with checkpoint file and "
                     f"make the gene with the same order, so use the gene list file: "
                     f"{gene_list_file}.")
        sc_expression_df = sc_expression_df.reindex(columns=gene_list).fillna(0)
        _logger.info(f"sc_expression_df re-size: {sc_expression_df.shape}")
    sc_expression_train = sc_expression_df.copy(deep=True)
    # we need to transpose data to correct its shape
    x_sc_train = torch.tensor(sc_expression_train.values, dtype=torch.get_default_dtype()).t()
    _logger.info("Set x_sc_train data with shape (gene, cells): {}".format(x_sc_train.shape))

    if time_saved_asFloat:  # 2023-08-06 13:25:48 trans 8.50000 to 850; 9.2500000 to 925; easy to  manipulate.
        cell_time_dic = dict(zip(cell_time.index, cell_time[time_str]))
        y_time_train = x_sc_train.new_tensor(np.array(sc_expression_train.index.map(cell_time_dic) * 100).astype(int))
    elif time_standard_type == "organdic":
        cell_time_dic = dict(zip(cell_time.index, cell_time[time_str]))
        y_time_train = x_sc_train.new_tensor(np.array(sc_expression_train.index.map(cell_time_dic)).astype(int))
    else:
        y_time_train = x_sc_train.new_tensor(
            [int(cell_time.loc[_cell_name][time_str].split("_")[0].replace("LH", "")) for _cell_name in
             sc_expression_train.index.values])

    # get each cell batch info
    if batch_dic is None:  # if no batch dic, just use donor id as batch index
        _logger.info("No batch dic, just use donor id as batch index")
        donor_index_train = x_sc_train.new_tensor(
            [int(donor_dic[cell_time.loc[_cell_name][donor_str]]) for _cell_name in sc_expression_train.index.values])
        batch_dic = donor_dic.copy()
    else:
        cell_donor_dic = dict(zip(cell_time.index, cell_time[donor_str]))
        donor_index_train = sc_expression_train.index.map(cell_donor_dic)
        donor_index_train = x_sc_train.new_tensor([int(batch_dic[_key]) for _key in donor_index_train])

    # for classification model with discrete time cannot use sigmoid and logit time type
    y_time_nor_train, label_dic = trans_time(y_time_train, time_standard_type)
    _logger.info("label dictionary: {}".format(label_dic))
    _logger.info(f"Normalize train y_time_train type: {time_standard_type}, "
                 f"with y_time_train lable: {np.unique(y_time_train)}, shape: {y_time_train.shape}, "
                 f"\nAfter trans y_time_nor_train detail: {np.unique(y_time_nor_train)}")

    # ------------------------------------------- Set up VAE model and Start train process -------------------------------------------------
    _logger.info("Start training with epoch: {}. ".format(train_epoch_num))

    # if (int(config['model_params']['in_channels']) == 0) :
    config['model_params']['in_channels'] = x_sc_train.shape[0]
    _logger.info("batch effect dic: {}".format(batch_dic))
    config['model_params']['batch_num'] = len(set(batch_dic.values()))

    tb_logger = TensorBoardLogger(save_dir=subFold_save_file_path,
                                  name=config['model_params']['name'], )

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    MyVAEModel = vae_models[config['model_params']['name']](**config['model_params'])
    if checkpoint_file is not None:  # 2024-03-17 20:15:39 add
        _logger.info(f"use checkpoint file: {checkpoint_file}")
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        # remove "model." before layer name
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for key, value in state_dict.items():
            # remove "model."
            if key.startswith('model.'):
                key = key[6:]
            new_state_dict[key] = value
        # MyVAEModel.load_state_dict(new_state_dict)
        MyVAEModel.load_state_dict(new_state_dict, strict=False)

    train_data = [[x_sc_train[:, i], y_time_nor_train[i], donor_index_train[i]] for i in range(x_sc_train.shape[1])]
    if batch_size is None:
        _logger.info("batch size is none, so don't set batch")
        data = SupervisedVAEDataset_onlyTrain(train_data=train_data, train_batch_size=len(train_data), label_dic=label_dic)

    else:
        _logger.info("batch size is {}".format(batch_size))
        data = SupervisedVAEDataset_onlyTrain(train_data=train_data, train_batch_size=batch_size, label_dic=label_dic)

    if adversarial_bool:
        experiment = VAEXperiment_adversarial(MyVAEModel, config['exp_params'])
    else:
        experiment = temporalVAEExperiment(MyVAEModel, config['exp_params'])
    # 创建一个 LearningRateMonitor 回调实例
    lr_monitor = LearningRateMonitor()
    # add 2023-09-07 20:34:57 add memory check
    check_memory(max_attempts=max_attempts)
    device = auto_select_gpu_and_cpu(max_attempts=max_attempts)  # device: e.g. "cuda:0"
    _logger.info("Auto select run on {}".format(device))
    runner = Trainer(logger=tb_logger, log_every_n_steps=1,
                     callbacks=[
                         lr_monitor,
                         ModelCheckpoint(save_top_k=2,
                                         dirpath=os.path.join(tb_logger.log_dir, "checkpoints"), monitor="train_loss",
                                         save_last=True),
                     ],
                     # check_val_every_n_epoch=1, val_check_interval=1,
                     devices=[int(device.split(":")[-1])],
                     accelerator="gpu", max_epochs=train_epoch_num
                     )

    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    if checkpoint_file is None:
        print(f"======= Training {config['model_params']['name']} =======")
        runner.fit(experiment, data)
    else:
        _logger.info(f"ship training with checkpoint file {checkpoint_file}")

    # train data forward the model
    data_predict = SupervisedVAEDataset_onlyPredict(predict_data=train_data, predict_batch_size=len(train_data))
    train_result = runner.predict(experiment, data_predict)
    train_clf_result, train_latent_mu_result, train_latent_log_var_result = train_result[0][0], train_result[0][1], \
        train_result[0][2]
    # ----------------------------------- plot sub result of training process for check  -------------------
    if plot_trainingLossLine:
        _logger.info("Plot training loss line for check.")
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        tags = EventAccumulator(tb_logger.log_dir).Reload().Tags()['scalars']
        _logger.info("All tags in logger: {}".format(tags))
        # Retrieve and print the metric results
        try:
            plot_training_loss_for_tags(tb_logger, ['lr-Adam', 'lr-Adam-1'], special_str="")
        except:
            pass
        try:
            plot_training_loss_for_tags(tb_logger, ['train_loss_epoch', 'train_Reconstruction_loss_epoch', 'train_clf_loss_epoch'], special_str="")
            plot_training_loss_for_tags(tb_logger, ['train_Reconstruction_loss_epoch'], special_str="")
            plot_training_loss_for_tags(tb_logger, ['train_batchEffect_loss_epoch'], special_str="")
        except:
            pass
    else:
        _logger.info("Don't plot training loss line for check.")

    """when we want to get an embedding for specific inputs:
    We either
    1 Feed a hand-written character "9" to VAE, receive a 20 dimensional "mean" vector, then embed it into 2D dimension using t-SNE,
    and finally plot it with label "9" or the actual image next to the point, or
    2 We use 2D mean vectors and plot directly without using t-SNE.
    Note that 'variance' vector is not used for embedding.
    However, its size can be used to show the degree of uncertainty.
    For example a clear '9' would have less variance than a hastily written '9' which is close to '0'."""
    if plot_latentSpaceUmap:
        if time_standard_type == "organdic":  # 2023-11-07 17:10:53 add for Joy project

            cell_type_number_dic = pd.Series(cell_time.broad_celltype_index.values, index=cell_time.broad_celltype).to_dict()
            umap_vae_latent_space(train_latent_mu_result, y_time_nor_train, cell_type_number_dic, special_path_str, config,
                                  special_str="trainData_mu_time_broad_celltype", drop_batch_dim=0)
            umap_vae_latent_space(train_latent_mu_result, donor_index_train, donor_dic, special_path_str, config,
                                  special_str="trainData_mu_donor_Dataset", drop_batch_dim=0)
            mesen_dic = {"False": 0, "True": 1}
            umap_vae_latent_space(train_latent_mu_result, y_time_nor_train.new_tensor(cell_time["mesen"]), mesen_dic, special_path_str, config,
                                  special_str="trainData_mu_donor_mesen", drop_batch_dim=0)
        else:
            umap_vae_latent_space(train_latent_mu_result, y_time_nor_train, label_dic, special_path_str, config,
                                  special_str="trainData_mu_time", drop_batch_dim=0)
            umap_vae_latent_space(train_latent_mu_result, donor_index_train, donor_dic, special_path_str, config,
                                  special_str="trainData_mu_donor", drop_batch_dim=0)
        # umap_vae_latent_space(train_latent_log_var_result, y_time_nor_train, label_dic, special_path_str, config,
        #                       special_str="trainData_logVar_time", drop_batch_dim=0)
        # umap_vae_latent_space(train_latent_log_var_result, donor_index_train, donor_dic, special_path_str, config,
        #                       special_str="trainData_logVar_donor", drop_batch_dim=0)
    train_latent_info_dic = {"mu": train_latent_mu_result,
                             "log_var": train_latent_log_var_result,
                             "label_true": y_time_nor_train,
                             "label_dic": label_dic,
                             "donor_index": donor_index_train,
                             "donor_dic": donor_dic, "total": train_result}
    if train_clf_result.shape[1] == 1:  # time is continues
        train_latent_info_dic["time_continues_bool"] = True
    else:
        train_latent_info_dic["time_continues_bool"] = False
    return sc_expression_train, y_time_nor_train, donor_index_train, runner, experiment, MyVAEModel, train_clf_result, label_dic, train_latent_info_dic


def fineTuning_calRNAvelocity(sc_expression_df, config_file, checkpoint_file, adversarial_bool=False, y_label=None, save_result_path=None,
                              cell_time_info=None, fine_tune_mode="withoutCellType", clf_weight=1,
                              sc_expression_df_add=None, plt_attr=None,
                              testData_dic=None, detT=0.1, batch_size=100000):
    """
    read parameters or weights of model from checkpoint and predict on sc expression dataframe
    return the predict result
    :param sc_expression_df:
    :param config_file:
    :param checkpoint_file:
    :return:
    """
    import yaml
    from pytorch_lightning import Trainer
    from pytorch_lightning import seed_everything
    from TemporalVAE.model_master.experiment_temporalVAE import temporalVAEExperiment
    from TemporalVAE.model_master.experiment_adversarial import VAEXperiment_adversarial
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyPredict
    # make data with gene express min

    with open(config_file, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    config['model_params']['in_channels'] = sc_expression_df.values.shape[1]
    config['model_params']['batch_num'] = 2

    checkpoint = torch.load(checkpoint_file, map_location='cpu')
    # 去掉每层名字前面的 "model."
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        # 去掉前缀 "model."
        if key.startswith('model.'):
            key = key[6:]
        new_state_dict[key] = value
    # MyVAEModel = vae_models[config['model_params']['name']](**config['model_params'])
    MyVAEModel = vae_models["SuperviseVanillaVAE_regressionClfDecoder_mouse_noAdversarial"](**config['model_params'])
    MyVAEModel.load_state_dict(new_state_dict)
    MyVAEModel.eval()
    check_memory()
    # device = auto_select_gpu_and_cpu()
    device = auto_select_gpu_and_cpu(free_thre=5, max_attempts=100000000)  # device: e.g. "cuda:0"
    runner = Trainer(devices=[int(device.split(":")[-1])])
    seed_everything(config['exp_params']['manual_seed'], True)
    #
    x_sc = torch.tensor(sc_expression_df.values, dtype=torch.get_default_dtype()).t()
    data_x = [[x_sc[:, i], 0, 0] for i in range(x_sc.shape[1])]

    # predict batch size will not influence the training
    data_predict = SupervisedVAEDataset_onlyPredict(predict_data=data_x, predict_batch_size=10000)

    if adversarial_bool:
        experiment = VAEXperiment_adversarial(MyVAEModel, config['exp_params'])
    else:
        experiment = temporalVAEExperiment(MyVAEModel, config['exp_params'])
    # z=experiment.predict_step(data_predict,1)
    train_result = runner.predict(experiment, data_predict)
    if len(train_result) > 1:
        print(f'concat the batch predictions with {len(train_result)} batches.')
        #  reclf, mu, log_var, test_loss,recons
        _reclf = np.concatenate([train_result[_i][0] for _i in range(len(train_result))])
        _mu = torch.cat([train_result[_i][1] for _i in range(len(train_result))])
        _log_var = torch.cat([train_result[_i][2] for _i in range(len(train_result))])
        _test_loss = {}
        for _l in train_result:
            for key, value in _l[3].items():
                _test_loss[key] = _test_loss.get(key, 0) + value
        _test_loss = {key: value / len(train_result) for key, value in _test_loss.items()}
        _recons = np.concatenate([train_result[_i][4] for _i in range(len(train_result))])
        train_result = (_reclf, _mu, _log_var, _test_loss, _recons)
    else:
        train_result = train_result[0]
    print(f"fine tune mode {fine_tune_mode}")
    FinetuneVAEModel = vae_models[config['model_params']['name']](**config['model_params'])
    FinetuneVAEModel.load_state_dict(new_state_dict)
    FinetuneVAEModel.eval()
    if (fine_tune_mode == "focusEncoder") and (sc_expression_df_add is None):
        print(f"more feature df is none, only use atlas gene.")
        withCellType, withMoreFeature = False, False

        fine_tune_result_adata, runner_fineTune, experiment_fineTune, testData_result_dic, predict_detT, v = fine_tuning_model_u_s_focusEncoder(train_result[0],
                                                                                                                                                sc_expression_df,
                                                                                                                                                FinetuneVAEModel, config,
                                                                                                                                                save_result_path,
                                                                                                                                                cell_time_info=cell_time_info,
                                                                                                                                                clf_weight=clf_weight,
                                                                                                                                                plt_attr=plt_attr,
                                                                                                                                                detT=detT,
                                                                                                                                                batch_size=batch_size)
        return train_result, fine_tune_result_adata, testData_result_dic, predict_detT, v
    elif (fine_tune_mode == "focusEncoder") and (sc_expression_df_add is not None):
        print(f"more feature df is not none with more {len(sc_expression_df_add.columns)} genes")
        withCellType, withMoreFeature = False, True
        # sc_expression_df_concat = pd.concat((sc_expression_df, sc_expression_df_add), axis=1)
        fine_tune_result_adata, runner_fineTune, experiment_fineTune, testData_result_dic, predict_detT, v = \
            fine_tuning_model_u_s_focusEncoder_moreFeatures(train_result[0],
                                                            sc_expression_df,
                                                            sc_expression_df_add,
                                                            FinetuneVAEModel,
                                                            config,
                                                            save_result_path,
                                                            cell_time_info=cell_time_info,
                                                            clf_weight=clf_weight,
                                                            plt_attr=plt_attr,
                                                            detT=detT,
                                                            batch_size=batch_size)
        return train_result, fine_tune_result_adata, testData_result_dic, predict_detT, v
    else:
        print(f"{fine_tune_mode} error, please check")
        exit(1)
    # spliced:
    sc_expression_df_test, cell_time_info_test = testData_dic["spliced"]["df"], testData_dic["spliced"]["cell_info"]
    fine_tune_test_spliced_result_adata = predict_by_fineTuneModel(sc_expression_df_test, y_label, runner_fineTune, experiment_fineTune,
                                                                   cell_time_info_test, plt_attr,
                                                                   save_result_path, special_file_name_str="test_spliced_",
                                                                   withCellType=withCellType,
                                                                   withMoreFeature=withMoreFeature, withHvg_df=testData_dic["spliced"]["hvg_df"])
    # unspliced:
    sc_expression_df_test, cell_time_info_test = testData_dic["unspliced"]["df"], testData_dic["unspliced"]["cell_info"]
    fine_tune_test_unspliced_result_adata = predict_by_fineTuneModel(sc_expression_df_test, y_label, runner_fineTune, experiment_fineTune,
                                                                     cell_time_info_test, plt_attr,
                                                                     save_result_path, special_file_name_str="test_unspliced_",
                                                                     withCellType=withCellType,
                                                                     withMoreFeature=withMoreFeature, withHvg_df=testData_dic["unspliced"]["hvg_df"])
    testData_result_dic = {"spliced": fine_tune_test_spliced_result_adata, "unspliced": fine_tune_test_unspliced_result_adata}
    return train_result, fine_tune_result_adata, testData_result_dic


def predict_by_fineTuneModel(sc_expression_df_test, y_label, runner_fineTune, experiment_fineTune, cell_time_info_test, plt_attr,
                             save_result_path, special_file_name_str="", withCellType=False, withMoreFeature=False, withHvg_df=None):
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyPredict
    # input sc data change
    # x_sc_test = torch.tensor(sc_expression_df_test.values, dtype=torch.get_default_dtype()).t()
    # data_x_test = [[x_sc_test[:, i], 0, 0] for i in range(x_sc_test.shape[1])]
    # data_predict_test = SupervisedVAEDataset_onlyPredict(predict_data=data_x_test, predict_batch_size=len(data_x_test))
    #
    # test_result = runner.predict(experiment, data_predict_test)
    test_df = pd.DataFrame(index=sc_expression_df_test.index)
    # test_df["pseudotime"] = test_result[0][0]
    test_df["time"] = y_label
    # test_df['predicted_time'] = test_df['pseudotime'].apply(denormalize, args=(8.5, 18.75, -5, 5))
    test_df['normalized_time'] = test_df['time'].apply(normalize, args=(8.5, 18.75, -5, 5))
    # start fine tune model
    x_sc_test_fine = torch.tensor(sc_expression_df_test.values, dtype=torch.get_default_dtype()).t()
    if withCellType:
        one_hot_encoded = pd.get_dummies(cell_time_info_test["cell_type_encoded"]).astype(int).T
        x_sc_test_fine = np.concatenate([x_sc_test_fine, one_hot_encoded.values])
        x_sc_test_fine = torch.tensor(x_sc_test_fine, dtype=torch.get_default_dtype())
    if withMoreFeature:
        x_sc_test_fine = np.concatenate([x_sc_test_fine, withHvg_df.values.T])
        x_sc_test_fine = torch.tensor(x_sc_test_fine, dtype=torch.get_default_dtype())
    data_x_test_fine = [[x_sc_test_fine[:, i], 0, 0] for i in range(x_sc_test_fine.shape[1])]
    data_predict_test_fine = SupervisedVAEDataset_onlyPredict(predict_data=data_x_test_fine, predict_batch_size=len(data_x_test_fine))
    test_result_fine = runner_fineTune.predict(experiment_fineTune, data_predict_test_fine)
    fine_tune_test_latent_mu_result = test_result_fine[0][1]

    test_df["finetune_pseudotime"] = test_result_fine[0][0]
    test_df['finetune_predicted_time'] = test_df['finetune_pseudotime'].apply(denormalize, args=(8.5, 18.75, -5, 5))

    cell_time_info_temp = pd.concat([cell_time_info_test, test_df], axis=1)
    # from draw_images.read_json_plotViolin_oneTimeMulitDonor import plt_umap_byScanpy
    test_adata = anndata.AnnData(X=fine_tune_test_latent_mu_result.cpu().numpy(), obs=cell_time_info_temp)

    test_adata = plt_umap_byScanpy(test_adata, attr_list=plt_attr, save_path=save_result_path, special_file_name_str=special_file_name_str)
    return test_adata


def fine_tuning_model(clf_train_result, sc_expression_df, y_label, MyVAEModel, config, x_sc, save_result_path, cell_time_info=None, clf_weight=1,
                      plt_attr=None):
    """
    without cell type as input
    :param clf_train_result:
    :param sc_expression_df:
    :param y_label:
    :param MyVAEModel:
    :param config:
    :param x_sc:
    :param save_result_path:
    :param cell_time_info:
    :return:
    """
    from pytorch_lightning import Trainer
    from pytorch_lightning import seed_everything
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyPredict
    from TemporalVAE.model_master.experiment_fineTune import VAEXperiment_fineTune
    from pytorch_lightning.loggers import TensorBoardLogger
    # 2023-11-30 10:45:36 fine tune

    predict_donors_df = pd.DataFrame(index=sc_expression_df.index)
    predict_donors_df["pseudotime"] = clf_train_result
    predict_donors_df["time"] = y_label
    predict_donors_df['predicted_time'] = predict_donors_df['pseudotime'].apply(denormalize, args=(8.5, 18.75, -5, 5))
    predict_donors_df['normalized_time'] = predict_donors_df['time'].apply(normalize, args=(8.5, 18.75, -5, 5))

    # set weight of clf  time lower
    print(f"clf weight is {clf_weight}")
    config["exp_params"]["clf_weight"] = clf_weight
    experiment_fineTune = VAEXperiment_fineTune(MyVAEModel, config["exp_params"])
    data_x = [[x_sc[:, i], predict_donors_df["normalized_time"][i], 0] for i in range(x_sc.shape[1])]
    subFold_save_file_path = f"{save_result_path}/finetune/"
    if not os.path.exists(subFold_save_file_path):
        os.makedirs(subFold_save_file_path)
    tb_logger = TensorBoardLogger(save_dir=subFold_save_file_path,
                                  name=config['model_params']['name'], )

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyTrain
    data_predict = SupervisedVAEDataset_onlyTrain(train_data=data_x, train_batch_size=len(data_x))

    # 创建一个 LearningRateMonitor 回调实例
    lr_monitor = LearningRateMonitor()
    # add 2023-09-07 20:34:57 add memory check
    check_memory(max_attempts=100000000)
    device = auto_select_gpu_and_cpu(max_attempts=100000000)  # device: e.g. "cuda:0"
    _logger.info("Auto select run on {}".format(device))
    runner = Trainer(logger=tb_logger, log_every_n_steps=1,
                     callbacks=[
                         lr_monitor,
                         ModelCheckpoint(save_top_k=2,
                                         dirpath=os.path.join(tb_logger.log_dir, "checkpoints"), monitor="train_loss",
                                         save_last=True),
                     ],
                     # check_val_every_n_epoch=1, val_check_interval=1,
                     devices=[int(device.split(":")[-1])],
                     accelerator="gpu", max_epochs=10
                     )
    from pathlib import Path
    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment_fineTune, data_predict)
    # train data forward the model
    data_predict = SupervisedVAEDataset_onlyPredict(predict_data=data_x, predict_batch_size=len(data_x))
    fine_tune_train_result = runner.predict(experiment_fineTune, data_predict)
    fine_tune_train_latent_mu_result = fine_tune_train_result[0][1]

    predict_donors_df["finetune_pseudotime"] = fine_tune_train_result[0][0]
    predict_donors_df['finetune_predicted_time'] = predict_donors_df['finetune_pseudotime'].apply(denormalize, args=(8.5, 18.75, -5, 5))

    cell_time_info_temp = pd.concat([cell_time_info, predict_donors_df], axis=1)
    # from draw_images.read_json_plotViolin_oneTimeMulitDonor import plt_umap_byScanpy
    result_adata = anndata.AnnData(X=fine_tune_train_latent_mu_result.cpu().numpy(), obs=cell_time_info_temp)

    result_adata = plt_umap_byScanpy(result_adata, attr_list=plt_attr, save_path=save_result_path, special_file_name_str="train_")

    return result_adata, runner, experiment_fineTune


def fine_tuning_model_addCellTypeLabel(clf_train_result, sc_expression_df, y_label, MyVAEModel, config, x_sc, save_result_path, cell_time_info=None,
                                       clf_weight=1, plt_attr=None):
    """
    with celltype as input
    :param clf_train_result:
    :param sc_expression_df:
    :param y_label:
    :param MyVAEModel:
    :param config:
    :param x_sc:
    :param save_result_path:
    :param cell_time_info:
    :return:
    """
    from pytorch_lightning import Trainer
    from pytorch_lightning import seed_everything
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyPredict
    from TemporalVAE.model_master.experiment_fineTune import VAEXperiment_fineTune
    from pytorch_lightning.loggers import TensorBoardLogger
    # 2023-12-01 11:30:26 add cell type to model

    cell_type_num = len(np.unique(cell_time_info["cell_type"]))

    predict_donors_df = pd.DataFrame(index=sc_expression_df.index)
    predict_donors_df["pseudotime"] = clf_train_result
    predict_donors_df["time"] = y_label
    predict_donors_df['predicted_time'] = predict_donors_df['pseudotime'].apply(denormalize, args=(8.5, 18.75, -5, 5))
    predict_donors_df['normalized_time'] = predict_donors_df['time'].apply(normalize, args=(8.5, 18.75, -5, 5))
    # encoder change
    encoder_inFeature_num = MyVAEModel.encoder._modules["0"]._modules["0"].in_features
    encoder_outFeature_num = MyVAEModel.encoder._modules["0"]._modules["0"].out_features
    encoder_weight_0 = MyVAEModel.encoder._modules["0"]._modules["0"].weight  # encoder_weight_0 shape as (out_features, in_features)

    MyVAEModel.encoder._modules["0"]._modules["0"] = nn.Linear(encoder_inFeature_num + cell_type_num, encoder_outFeature_num)
    # zero_col = torch.zeros(encoder_weight_0.size(0), cell_type_num)
    zero_col = torch.ones(encoder_weight_0.size(0), cell_type_num)
    temp = torch.nn.Parameter(torch.cat((encoder_weight_0, zero_col), dim=1))
    MyVAEModel.encoder._modules["0"]._modules["0"].weight = temp

    # decoder change. the last layer of decoder is identified as final_layer
    decoder_inFeature_num = MyVAEModel.final_layer._modules["0"].in_features
    decoder_outFeature_num = MyVAEModel.final_layer._modules["0"].out_features
    decoder_weight_lastLayer = MyVAEModel.final_layer._modules["0"].weight  # encoder_weight_0 shape as (out_features, in_features)

    MyVAEModel.final_layer._modules["0"] = nn.Linear(decoder_inFeature_num, decoder_outFeature_num + cell_type_num)
    MyVAEModel.final_layer._modules["1"] = nn.BatchNorm1d(decoder_outFeature_num + cell_type_num)
    # zero_col = torch.zeros(decoder_weight_lastLayer.size(0), cell_type_num)
    zero_col = torch.ones(decoder_weight_lastLayer.size(0), cell_type_num)
    temp = torch.nn.Parameter(torch.cat((decoder_weight_lastLayer, zero_col.T), dim=0))  # shape as (out_features, in_features)
    MyVAEModel.final_layer._modules["0"].weight = temp

    # input sc data change
    # one_hot_encoded = pd.get_dummies(cell_time_info['cell_type']).astype(int).T
    one_hot_encoded = pd.get_dummies(cell_time_info["cell_type_encoded"]).astype(int).T
    x_sc = np.concatenate([x_sc, one_hot_encoded.values])
    x_sc = torch.tensor(x_sc, dtype=torch.get_default_dtype())
    data_x = [[x_sc[:, i], predict_donors_df["normalized_time"][i], 0] for i in range(x_sc.shape[1])]

    # set weight of clf  time lower
    print(f"clf weight is {clf_weight}")
    config["exp_params"]["clf_weight"] = clf_weight
    experiment_fineTune = VAEXperiment_fineTune(MyVAEModel, config["exp_params"])

    subFold_save_file_path = f"{save_result_path}/finetune/"
    if not os.path.exists(subFold_save_file_path):
        os.makedirs(subFold_save_file_path)
    tb_logger = TensorBoardLogger(save_dir=subFold_save_file_path,
                                  name=config['model_params']['name'], )
    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyTrain
    data_predict = SupervisedVAEDataset_onlyTrain(train_data=data_x, train_batch_size=len(data_x))

    # 创建一个 LearningRateMonitor 回调实例
    lr_monitor = LearningRateMonitor()
    # add 2023-09-07 20:34:57 add memory check
    check_memory(max_attempts=100000000)
    device = auto_select_gpu_and_cpu(max_attempts=100000000)  # device: e.g. "cuda:0"
    _logger.info("Auto select run on {}".format(device))
    runner = Trainer(logger=tb_logger, log_every_n_steps=1,
                     callbacks=[
                         lr_monitor,
                         ModelCheckpoint(save_top_k=2,
                                         dirpath=os.path.join(tb_logger.log_dir, "checkpoints"), monitor="train_loss",
                                         save_last=True),
                     ],
                     # check_val_every_n_epoch=1, val_check_interval=1,
                     devices=[int(device.split(":")[-1])],
                     accelerator="gpu", max_epochs=100
                     )
    from pathlib import Path
    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment_fineTune, data_predict)
    # train data forward the model
    data_predict = SupervisedVAEDataset_onlyPredict(predict_data=data_x, predict_batch_size=len(data_x))
    fine_tune_train_result = runner.predict(experiment_fineTune, data_predict)
    fine_tune_train_latent_mu_result = fine_tune_train_result[0][1]

    predict_donors_df["finetune_pseudotime"] = fine_tune_train_result[0][0]
    predict_donors_df['finetune_predicted_time'] = predict_donors_df['finetune_pseudotime'].apply(denormalize, args=(8.5, 18.75, -5, 5))

    cell_time_info_temp = pd.concat([cell_time_info, predict_donors_df], axis=1)
    # from draw_images.read_json_plotViolin_oneTimeMulitDonor import plt_umap_byScanpy
    result_adata = anndata.AnnData(X=fine_tune_train_latent_mu_result.cpu().numpy(), obs=cell_time_info_temp)
    result_adata = plt_umap_byScanpy(result_adata, attr_list=plt_attr, save_path=save_result_path)

    return result_adata, runner, experiment_fineTune


def fine_tuning_model_add_more_feature(clf_train_result, sc_expression_df, sc_expression_df_add, y_label, MyVAEModel, config, x_sc, save_result_path,
                                       cell_time_info=None,
                                       clf_weight=1, plt_attr=None):
    """
    with celltype as input
    :param clf_train_result:
    :param sc_expression_df:
    :param y_label:
    :param MyVAEModel:
    :param config:
    :param x_sc:
    :param save_result_path:
    :param cell_time_info:
    :return:
    """
    from pytorch_lightning import Trainer
    from pytorch_lightning import seed_everything
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyPredict
    from TemporalVAE.model_master.experiment_fineTune import VAEXperiment_fineTune
    from pytorch_lightning.loggers import TensorBoardLogger
    # 2023-12-01 11:30:26 add new features to model
    predict_donors_df = pd.DataFrame(index=sc_expression_df.index)
    predict_donors_df["pseudotime"] = clf_train_result
    predict_donors_df["time"] = y_label
    predict_donors_df['predicted_time'] = predict_donors_df['pseudotime'].apply(denormalize, args=(8.5, 18.75, -5, 5))
    predict_donors_df['normalized_time'] = predict_donors_df['time'].apply(normalize, args=(8.5, 18.75, -5, 5))
    # encoder change
    encoder_inFeature_num = MyVAEModel.encoder._modules["0"]._modules["0"].in_features
    # encoder_outFeature_num = MyVAEModel.encoder._modules["0"]._modules["0"].out_features
    # encoder_weight_0 = MyVAEModel.encoder._modules["0"]._modules["0"].weight  # encoder_weight_0 shape as (out_features, in_features)
    # encoder_bias_0 = MyVAEModel.encoder._modules["0"]._modules["0"].bias  # encoder_weight_0 shape as (out_features, in_features)
    new_layer = nn.Linear(sc_expression_df.shape[1] + sc_expression_df_add.shape[1], encoder_inFeature_num, bias=True)  # add a new layer before encoder
    from collections import OrderedDict
    new_layers = OrderedDict([('new_layer', new_layer)])
    for name, layer in MyVAEModel.encoder._modules["0"].named_children():
        new_layers[name] = layer
    new_encoder_input_sequential = nn.Sequential(new_layers)
    new_encoder_input_sequential._modules["new_layer"].weight.shape
    one_col = torch.ones(encoder_inFeature_num, sc_expression_df.shape[1])
    zero_col = torch.zeros(encoder_inFeature_num, sc_expression_df_add.shape[1])
    temp = torch.nn.Parameter(torch.cat((one_col, zero_col), dim=1))
    new_encoder_input_sequential._modules["new_layer"].weight = temp
    MyVAEModel.encoder._modules["0"] = new_encoder_input_sequential

    # decoder change. the last layer of decoder is identified as final_layer
    decoder_inFeature_num = MyVAEModel.final_layer._modules["0"].in_features
    decoder_outFeature_num = MyVAEModel.final_layer._modules["0"].out_features
    decoder_weight_lastLayer = MyVAEModel.final_layer._modules["0"].weight  # encoder_weight_0 shape as (out_features, in_features)

    MyVAEModel.final_layer._modules["0"] = nn.Linear(decoder_inFeature_num, decoder_outFeature_num + sc_expression_df_add.shape[1])
    MyVAEModel.final_layer._modules["1"] = nn.BatchNorm1d(decoder_outFeature_num + sc_expression_df_add.shape[1])
    zero_col = torch.ones(decoder_weight_lastLayer.size(0), sc_expression_df_add.shape[1])
    temp = torch.nn.Parameter(torch.cat((decoder_weight_lastLayer, zero_col.T), dim=0))  # shape as (out_features, in_features)
    MyVAEModel.final_layer._modules["0"].weight = temp

    # input sc data change
    x_sc = np.concatenate([x_sc, sc_expression_df_add.values.T])
    x_sc = torch.tensor(x_sc, dtype=torch.get_default_dtype())
    data_x = [[x_sc[:, i], predict_donors_df["normalized_time"][i], 0] for i in range(x_sc.shape[1])]

    # set weight of clf  time lower
    print(f"clf weight is {clf_weight}")
    config["exp_params"]["clf_weight"] = clf_weight
    experiment_fineTune = VAEXperiment_fineTune(MyVAEModel, config["exp_params"])

    subFold_save_file_path = f"{save_result_path}/finetune/"
    if not os.path.exists(subFold_save_file_path):
        os.makedirs(subFold_save_file_path)
    tb_logger = TensorBoardLogger(save_dir=subFold_save_file_path,
                                  name=config['model_params']['name'], )
    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyTrain
    data_predict = SupervisedVAEDataset_onlyTrain(train_data=data_x, train_batch_size=len(data_x))

    # 创建一个 LearningRateMonitor 回调实例
    lr_monitor = LearningRateMonitor()
    # add 2023-09-07 20:34:57 add memory check
    check_memory(max_attempts=100000000)
    device = auto_select_gpu_and_cpu(max_attempts=100000000)  # device: e.g. "cuda:0"
    _logger.info("Auto select run on {}".format(device))
    runner = Trainer(logger=tb_logger, log_every_n_steps=1,
                     callbacks=[
                         lr_monitor,
                         ModelCheckpoint(save_top_k=2,
                                         dirpath=os.path.join(tb_logger.log_dir, "checkpoints"), monitor="train_loss",
                                         save_last=True),
                     ],
                     # check_val_every_n_epoch=1, val_check_interval=1,
                     devices=[int(device.split(":")[-1])],
                     accelerator="gpu", max_epochs=100
                     )
    from pathlib import Path
    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment_fineTune, data_predict)
    # train data forward the model
    data_predict = SupervisedVAEDataset_onlyPredict(predict_data=data_x, predict_batch_size=len(data_x))
    fine_tune_train_result = runner.predict(experiment_fineTune, data_predict)
    fine_tune_train_latent_mu_result = fine_tune_train_result[0][1]

    predict_donors_df["finetune_pseudotime"] = fine_tune_train_result[0][0]
    predict_donors_df['finetune_predicted_time'] = predict_donors_df['finetune_pseudotime'].apply(denormalize, args=(8.5, 18.75, -5, 5))

    cell_time_info_temp = pd.concat([cell_time_info, predict_donors_df], axis=1)
    # from draw_images.read_json_plotViolin_oneTimeMulitDonor import plt_umap_byScanpy
    result_adata = anndata.AnnData(X=fine_tune_train_latent_mu_result.cpu().numpy(), obs=cell_time_info_temp)
    result_adata = plt_umap_byScanpy(result_adata, attr_list=plt_attr, save_path=save_result_path)

    return result_adata, runner, experiment_fineTune


def fine_tuning_model_u_s_focusEncoder(clf_train_result, sc_expression_df, MyVAEModel, config, save_result_path,
                                       cell_time_info=None,
                                       clf_weight=1, plt_attr=None, detT=0.1, batch_size=100000):
    """
    with celltype as input
    :param clf_train_result:
    :param sc_expression_df:
    :param y_label:
    :param MyVAEModel:
    :param config:
    :param x_sc:
    :param save_result_path:
    :param cell_time_info:
    :return:
    """
    cell_time_info = cell_time_info.reindex(index=sc_expression_df.index)
    from pytorch_lightning import Trainer
    from pytorch_lightning import seed_everything
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyPredict
    from TemporalVAE.model_master.experiment_fineTune_u_s_focusEncoder import VAEXperiment_fineTune_u_s_focusEncoder
    from pytorch_lightning.loggers import TensorBoardLogger

    # 2023-12-01 11:30:26 add new features to model
    predict_donors_df = pd.DataFrame(index=sc_expression_df.index)
    predict_donors_df["normalized_pseudotime_by_preTrained_mouseAtlas_model"] = clf_train_result  # pseudotime is predicted by mouse-altas pre-train model
    predict_donors_df["physical_time"] = cell_time_info["time_label"]
    predict_donors_df['physical_pseudotime_by_preTrained_mouseAtlas_model'] = predict_donors_df['normalized_pseudotime_by_preTrained_mouseAtlas_model'].apply(denormalize, args=(
        8.5, 18.75, -5, 5))
    predict_donors_df['normalized_physical_time'] = predict_donors_df['physical_time'].apply(normalize, args=(8.5, 18.75, -5, 5))
    # encoder change
    # encoder_inFeature_num = MyVAEModel.encoder._modules["0"]._modules["0"].in_features
    # encoder_outFeature_num = MyVAEModel.encoder._modules["0"]._modules["0"].out_features
    # encoder_weight_0 = MyVAEModel.encoder._modules["0"]._modules["0"].weight  # encoder_weight_0 shape as (out_features, in_features)
    # encoder_bias_0 = MyVAEModel.encoder._modules["0"]._modules["0"].bias  # encoder_weight_0 shape as (out_features, in_features)
    # new_layer = nn.Linear(sc_expression_df.shape[1], sc_expression_df.shape[1], bias=True)  # add a new layer before encoder
    # new_encoder_layers = OrderedDict()
    new_encoder_layers = MyVAEModel.encoder[1:]
    MyVAEModel.q_u_s_encoder = new_encoder_layers
    # for name, layer in MyVAEModel.encoder._modules["0"].named_children():
    #     new_encoder_layers[name] = layer
    # new_encoder_input_sequential = nn.Sequential(new_encoder_layers)
    # new_encoder_input_sequential._modules["splice_layer"].weight.shape
    # zero_col = torch.nn.Parameter(torch.zeros(sc_expression_df.shape[1], sc_expression_df.shape[1]))
    # temp = torch.nn.Parameter(torch.cat((one_col, zero_col), dim=1))
    # new_encoder_input_sequential._modules["spliced_layer"].weight = zero_col
    # MyVAEModel.encoder._modules["0"] = new_encoder_input_sequential

    # new_final_layers = OrderedDict([('unspliced_layer', new_layer)])
    # for name, layer in MyVAEModel.final_layer._modules.items():
    #     new_final_layers[name] = layer
    # new_final_sequential = nn.Sequential(new_final_layers)
    # new_final_sequential._modules["unspliced_layer"].weight = zero_col
    # MyVAEModel.final_layer = new_final_sequential

    # input spliced and unspliced data
    spliced_rownames = list(cell_time_info[cell_time_info["s_or_mrna"] == "spliced"].index)
    spliced_data_dic = {"df": sc_expression_df.loc[spliced_rownames], "cell_info": cell_time_info.loc[spliced_rownames]}
    unspliced_rownames = ["un" + i for i in spliced_rownames]
    unspliced_data_dic = {"df": sc_expression_df.loc[unspliced_rownames], "cell_info": sc_expression_df.loc[unspliced_rownames]}

    finetune_donors_df = predict_donors_df.loc[spliced_rownames]

    spliced_x_sc = torch.tensor(spliced_data_dic["df"].values.T, dtype=torch.get_default_dtype())
    unspliced_x_sc = torch.tensor(unspliced_data_dic["df"].values.T, dtype=torch.get_default_dtype())
    data_x = [[(spliced_x_sc[:, i], unspliced_x_sc[:, i]), finetune_donors_df["normalized_physical_time"][i], 0] for i in range(spliced_x_sc.shape[1])]

    # set weight of clf  time lower
    print(f"clf weight is {clf_weight}")
    config["exp_params"]["clf_weight"] = clf_weight
    print(f"finetune model structure: {MyVAEModel}")
    experiment_fineTune = VAEXperiment_fineTune_u_s_focusEncoder(MyVAEModel, config["exp_params"])

    subFold_save_file_path = f"{save_result_path}/finetune/"
    if not os.path.exists(subFold_save_file_path):
        os.makedirs(subFold_save_file_path)
    tb_logger = TensorBoardLogger(save_dir=subFold_save_file_path,
                                  name=config['model_params']['name'], )
    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyTrain
    from pytorch_lightning.callbacks import EarlyStopping

    data_train = SupervisedVAEDataset_onlyTrain(train_data=data_x, train_batch_size=batch_size)

    # 创建一个 LearningRateMonitor 回调实例
    lr_monitor = LearningRateMonitor()

    # 配置 EarlyStopping 以监控训练损失
    class MyEarlyStopping(EarlyStopping):
        def on_validation_end(self, trainer, pl_module):
            # override this to disable early stopping at the end of val loop
            pass

        def on_train_end(self, trainer, pl_module):
            # instead, do it at the end of training loop
            self._run_early_stopping_check(trainer)

    early_stop_callback = MyEarlyStopping(monitor='train_loss',  # 监控训练损失
                                          min_delta=0.001,  # 变化阈值, 2024-04-19 10:06:57 before min_delta is 0.001
                                          patience=3,  # 耐心周期, # 2024-04-19 01:25:04 before patience is 3
                                          verbose=True,  # 是否打印日志
                                          mode='min'  # 监控指标的目标是最小化
                                          )
    # add 2023-09-07 20:34:57 add memory check
    check_memory(max_attempts=100000000)
    device = auto_select_gpu_and_cpu(max_attempts=100000000)  # device: e.g. "cuda:0"
    _logger.info("Auto select run on {}".format(device))
    runner = Trainer(logger=tb_logger, log_every_n_steps=1,
                     callbacks=[early_stop_callback,
                                lr_monitor,
                                ModelCheckpoint(save_top_k=1,
                                                dirpath=os.path.join(tb_logger.log_dir, "checkpoints"), monitor="train_loss",
                                                save_last=True), ],
                     # check_val_every_n_epoch=1, val_check_interval=1,
                     devices=[int(device.split(":")[-1])],
                     accelerator="gpu", max_epochs=100
                     )
    from pathlib import Path
    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment_fineTune, data_train)
    # train data forward the model

    data_predict = SupervisedVAEDataset_onlyPredict(predict_data=data_x, predict_batch_size=batch_size)

    spliced_unspliced_fine_tune_test_results = runner.predict(experiment_fineTune, data_predict)

    # (s_reclf, s_mu, s_log_var, test_loss, (spliced_recons,unspliced_recons)),(u_reclf, u_mu, u_log_var, test_loss, (spliced_recons,unspliced_recons))
    # if len(spliced_unspliced_fine_tune_test_results) > 1:
    def reconstructure_s_u(spliced_unspliced_fine_tune_test_results, s_or_u):  # for s index=0,for u index=1
        print(f'batch size is {batch_size}, concat the batch predictions.')
        fine_tune_test_result = [i[s_or_u] for i in spliced_unspliced_fine_tune_test_results]
        _s_reclf = np.concatenate([i[0] for i in fine_tune_test_result])
        _s_mu = np.concatenate([i[1] for i in fine_tune_test_result])
        _s_log_var = np.concatenate([i[2] for i in fine_tune_test_result])

        _test_loss = {}
        for _l in fine_tune_test_result:
            for key, value in _l[3].items():
                _test_loss[key] = _test_loss.get(key, 0) + value
        _test_loss = {key: value / len(fine_tune_test_result) for key, value in _test_loss.items()}
        _s_u_recons_tuple = (np.concatenate([i[4][0] for i in fine_tune_test_result]), np.concatenate([i[4][1] for i in fine_tune_test_result]))
        return (_s_reclf, _s_mu, _s_log_var, _test_loss, _s_u_recons_tuple)

    spliced_fine_tune_test_result = reconstructure_s_u(spliced_unspliced_fine_tune_test_results, 0)
    unspliced_fine_tune_test_result = reconstructure_s_u(spliced_unspliced_fine_tune_test_results, 1)
    # else:
    #     spliced_fine_tune_test_result, unspliced_fine_tune_test_result = spliced_unspliced_fine_tune_test_results[0]
    spliced_fine_tune_test_latent_mu_result = spliced_fine_tune_test_result[1]
    unspliced_fine_tune_test_latent_mu_result = unspliced_fine_tune_test_result[1]
    # print(spliced_fine_tune_test_latent_mu_result)
    # print(unspliced_fine_tune_test_latent_mu_result)
    spliced_finetune_donors_df = finetune_donors_df.copy()
    unspliced_finetune_donors_df = finetune_donors_df.copy()
    unspliced_finetune_donors_df.rename(index=lambda x: "un" + x, inplace=True)

    spliced_finetune_donors_df["normalized_pseudotime_by_finetune_model"] = spliced_fine_tune_test_result[0]
    unspliced_finetune_donors_df["normalized_pseudotime_by_finetune_model"] = unspliced_fine_tune_test_result[0]

    spliced_finetune_donors_df['physical_pseudotime_by_finetune_model'] = spliced_finetune_donors_df['normalized_pseudotime_by_finetune_model'].apply(denormalize,
                                                                                                                                                      args=(8.5, 18.75, -5, 5))
    unspliced_finetune_donors_df['physical_pseudotime_by_finetune_model'] = unspliced_finetune_donors_df['normalized_pseudotime_by_finetune_model'].apply(denormalize,
                                                                                                                                                          args=(8.5, 18.75, -5, 5))

    # spliced_finetune_donors_df["s_or_mrna"] = "spliced"
    # unspliced_finetune_donors_df["s_or_mrna"] = "unspliced"
    spliced_finetune_donors_df = spliced_finetune_donors_df.merge(cell_time_info, left_index=True, right_index=True)
    unspliced_finetune_donors_df = unspliced_finetune_donors_df.merge(cell_time_info, left_index=True, right_index=True)
    cell_time_info_temp = pd.concat([spliced_finetune_donors_df, unspliced_finetune_donors_df], axis=0)
    # from draw_images.read_json_plotViolin_oneTimeMulitDonor import plt_umap_byScanpy
    result_adata = anndata.AnnData(
        X=np.concatenate((spliced_fine_tune_test_latent_mu_result, unspliced_fine_tune_test_latent_mu_result), axis=0),
        obs=cell_time_info_temp)
    # plot umap

    fine_tune_test_spliced_result_adata = anndata.AnnData(X=spliced_fine_tune_test_latent_mu_result, obs=spliced_finetune_donors_df)
    fine_tune_test_unspliced_result_adata = anndata.AnnData(X=unspliced_fine_tune_test_latent_mu_result, obs=unspliced_finetune_donors_df)

    testData_result_dic = {"spliced": fine_tune_test_spliced_result_adata, "unspliced": fine_tune_test_unspliced_result_adata}

    predict_detT, v = RNA_velocity(MyVAEModel.clf_input, MyVAEModel.clf_decoder, config, unspliced_fine_tune_test_latent_mu_result, spliced_fine_tune_test_latent_mu_result,
                                   subFold_save_file_path, detT=detT)

    v = pd.DataFrame(data=v, index=spliced_rownames)
    # predict on train data
    return result_adata, runner, experiment_fineTune, testData_result_dic, predict_detT, v


def fine_tuning_model_u_s_focusEncoder_moreFeatures(clf_train_result, sc_expression_df, sc_expression_df_add, MyVAEModel, config, save_result_path,
                                                    cell_time_info=None,
                                                    clf_weight=1, plt_attr=None, detT=0.1, batch_size=10000, ):
    """
    with celltype as input
    :param clf_train_result:
    :param sc_expression_df:
    :param y_label:
    :param MyVAEModel:
    :param config:
    :param x_sc:
    :param save_result_path:
    :param cell_time_info:
    :return:
    """
    print(f"fine tune model is foucus on encoder and with more features, clf weight: {clf_weight}, detT: {detT}, batchsize: {batch_size}")
    cell_time_info = cell_time_info.reindex(index=sc_expression_df.index)
    from pytorch_lightning import Trainer
    from pytorch_lightning import seed_everything
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyPredict
    from TemporalVAE.model_master.experiment_fineTune_u_s_focusEncoder import VAEXperiment_fineTune_u_s_focusEncoder
    from pytorch_lightning.loggers import TensorBoardLogger
    # 2023-12-01 11:30:26 add new features to model
    sc_expression_df_concat = pd.concat((sc_expression_df, sc_expression_df_add), axis=1)
    if not sc_expression_df_concat.index.equals(sc_expression_df.index):
        print(f"Wrong: The indexes of sc and sc_concat is not the same.")
        exit(0)

    predict_donors_df = pd.DataFrame(index=sc_expression_df_concat.index)
    predict_donors_df["normalized_pseudotime_by_preTrained_mouseAtlas_model"] = clf_train_result  # pseudotime is predicted by mouse-altas pre-train model
    predict_donors_df["physical_time"] = cell_time_info["time_label"]
    predict_donors_df['physical_pseudotime_by_preTrained_mouseAtlas_model'] = predict_donors_df['normalized_pseudotime_by_preTrained_mouseAtlas_model'].apply(denormalize, args=(
        8.5, 18.75, -5, 5))
    predict_donors_df['normalized_physical_time'] = predict_donors_df['physical_time'].apply(normalize, args=(8.5, 18.75, -5, 5))
    # set q(u|s) encoder
    MyVAEModel.q_u_s_encoder = MyVAEModel.encoder[1:]
    # encoder change: add a encoder input mou
    from collections import OrderedDict
    # MyVAEModel.encoder._modules["0"]._modules["0"]
    encoder0_inFeature_num = MyVAEModel.encoder._modules["0"]._modules["0"].in_features
    encoder_input_module = nn.Sequential(OrderedDict([('0', nn.Linear(sc_expression_df.shape[1] + sc_expression_df_add.shape[1], encoder0_inFeature_num, bias=True)),
                                                      ("1", nn.BatchNorm1d(encoder0_inFeature_num)),
                                                      ("2", nn.LeakyReLU())]))  # add a new layer before encoder
    # encoder_input_module._modules["0"].weight.shape
    one_col = torch.ones(encoder0_inFeature_num, sc_expression_df.shape[1])
    zero_col = torch.ones(encoder0_inFeature_num, sc_expression_df_add.shape[1])  # 2024-01-24 00:13:28 changge zero col to one col
    temp = torch.nn.Parameter(torch.cat((one_col, zero_col), dim=1))
    encoder_input_module._modules["0"].weight = temp
    MyVAEModel.encoder_input = encoder_input_module

    # decoder change. the last layer of decoder is identified as final_layer
    decoder_inFeature_num = MyVAEModel.final_layer._modules["0"].in_features
    decoder_outFeature_num = MyVAEModel.final_layer._modules["0"].out_features
    decoder_weight_lastLayer = MyVAEModel.final_layer._modules["0"].weight  # encoder_weight_0 shape as (out_features, in_features)

    MyVAEModel.final_layer._modules["0"] = nn.Linear(decoder_inFeature_num, decoder_outFeature_num + sc_expression_df_add.shape[1])
    MyVAEModel.final_layer._modules["1"] = nn.BatchNorm1d(decoder_outFeature_num + sc_expression_df_add.shape[1])
    one_col = torch.ones(decoder_weight_lastLayer.size(0), sc_expression_df_add.shape[1])
    temp = torch.nn.Parameter(torch.cat((decoder_weight_lastLayer, one_col.T), dim=0))  # shape as (out_features, in_features)

    MyVAEModel.final_layer._modules["0"].weight = temp
    print(MyVAEModel)
    # input spliced and unspliced data
    spliced_rownames = list(cell_time_info[cell_time_info["s_or_mrna"] == "spliced"].index)
    spliced_data_dic = {"df": sc_expression_df_concat.loc[spliced_rownames], "cell_info": cell_time_info.loc[spliced_rownames]}
    unspliced_rownames = ["un" + i for i in spliced_rownames]
    unspliced_data_dic = {"df": sc_expression_df_concat.loc[unspliced_rownames], "cell_info": cell_time_info.loc[unspliced_rownames]}

    finetune_donors_df = predict_donors_df.loc[spliced_rownames]

    spliced_x_sc = torch.tensor(spliced_data_dic["df"].values.T, dtype=torch.get_default_dtype())
    unspliced_x_sc = torch.tensor(unspliced_data_dic["df"].values.T, dtype=torch.get_default_dtype())
    data_x = [[(spliced_x_sc[:, i], unspliced_x_sc[:, i]), finetune_donors_df["normalized_physical_time"][i], 0] for i in range(spliced_x_sc.shape[1])]

    # set weight of clf  time lower
    print(f"clf weight is {clf_weight}")
    config["exp_params"]["clf_weight"] = clf_weight
    print(f"finetune model structure: {MyVAEModel}")
    experiment_fineTune = VAEXperiment_fineTune_u_s_focusEncoder(MyVAEModel, config["exp_params"])

    subFold_save_file_path = f"{save_result_path}/finetune/"
    if not os.path.exists(subFold_save_file_path):
        os.makedirs(subFold_save_file_path)
    tb_logger = TensorBoardLogger(save_dir=subFold_save_file_path,
                                  name=config['model_params']['name'], )
    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyTrain
    from pytorch_lightning.callbacks import EarlyStopping

    data_train = SupervisedVAEDataset_onlyTrain(train_data=data_x, train_batch_size=batch_size)

    # 创建一个 LearningRateMonitor 回调实例
    lr_monitor = LearningRateMonitor()

    # 配置 EarlyStopping 以监控训练损失
    class MyEarlyStopping(EarlyStopping):
        def on_validation_end(self, trainer, pl_module):
            # override this to disable early stopping at the end of val loop
            pass

        def on_train_end(self, trainer, pl_module):
            # instead, do it at the end of training loop
            self._run_early_stopping_check(trainer)

    early_stop_callback = MyEarlyStopping(monitor='train_loss',  # 监控训练损失
                                          min_delta=0.001,  # 变化阈值 #2024-04-19 15:28:24 before used 0.001
                                          patience=3,  # 耐心周期
                                          verbose=True,  # 是否打印日志
                                          mode='min'  # 监控指标的目标是最小化
                                          )
    # add 2023-09-07 20:34:57 add memory check
    check_memory(max_attempts=100000000)
    device = auto_select_gpu_and_cpu(free_thre=10, max_attempts=100000000)  # device: e.g. "cuda:0"
    _logger.info("Auto select run on {}".format(device))
    runner = Trainer(logger=tb_logger, log_every_n_steps=1,
                     callbacks=[early_stop_callback,
                                lr_monitor,
                                ModelCheckpoint(save_top_k=1,
                                                dirpath=os.path.join(tb_logger.log_dir, "checkpoints"), monitor="train_loss",
                                                save_last=True), ],
                     # check_val_every_n_epoch=1, val_check_interval=1,
                     devices=[int(device.split(":")[-1])],
                     accelerator="gpu", max_epochs=100
                     )
    from pathlib import Path
    Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment_fineTune, data_train)
    # train data forward the model

    data_predict = SupervisedVAEDataset_onlyPredict(predict_data=data_x, predict_batch_size=batch_size)

    spliced_unspliced_fine_tune_test_results = runner.predict(experiment_fineTune, data_predict)

    # (s_reclf, s_mu, s_log_var, test_loss, (spliced_recons,unspliced_recons)),(u_reclf, u_mu, u_log_var, test_loss, (spliced_recons,unspliced_recons))
    # if len(spliced_unspliced_fine_tune_test_results) > 1:
    def reconstructure_s_u(spliced_unspliced_fine_tune_test_results, s_or_u):  # for s index=0,for u index=1
        print(f'batch size is {batch_size}, concat the batch predictions.')
        fine_tune_test_result = [i[s_or_u] for i in spliced_unspliced_fine_tune_test_results]
        _s_reclf = np.concatenate([i[0] for i in fine_tune_test_result])
        _s_mu = np.concatenate([i[1] for i in fine_tune_test_result])
        _s_log_var = np.concatenate([i[2] for i in fine_tune_test_result])

        _test_loss = {}
        for _l in fine_tune_test_result:
            for key, value in _l[3].items():
                _test_loss[key] = _test_loss.get(key, 0) + value
        _test_loss = {key: value / len(fine_tune_test_result) for key, value in _test_loss.items()}
        _s_u_recons_tuple = (np.concatenate([i[4][0] for i in fine_tune_test_result]), np.concatenate([i[4][1] for i in fine_tune_test_result]))
        return (_s_reclf, _s_mu, _s_log_var, _test_loss, _s_u_recons_tuple)

    spliced_fine_tune_test_result = reconstructure_s_u(spliced_unspliced_fine_tune_test_results, 0)
    unspliced_fine_tune_test_result = reconstructure_s_u(spliced_unspliced_fine_tune_test_results, 1)
    # else:
    #     spliced_fine_tune_test_result, unspliced_fine_tune_test_result = spliced_unspliced_fine_tune_test_results[0]
    spliced_fine_tune_test_latent_mu_result = spliced_fine_tune_test_result[1]
    unspliced_fine_tune_test_latent_mu_result = unspliced_fine_tune_test_result[1]
    # print(spliced_fine_tune_test_latent_mu_result)
    # print(unspliced_fine_tune_test_latent_mu_result)
    spliced_finetune_donors_df = finetune_donors_df.copy()
    unspliced_finetune_donors_df = finetune_donors_df.copy()
    unspliced_finetune_donors_df.rename(index=lambda x: "un" + x, inplace=True)

    spliced_finetune_donors_df["normalized_pseudotime_by_finetune_model"] = spliced_fine_tune_test_result[0]
    unspliced_finetune_donors_df["normalized_pseudotime_by_finetune_model"] = unspliced_fine_tune_test_result[0]

    spliced_finetune_donors_df['physical_pseudotime_by_finetune_model'] = spliced_finetune_donors_df['normalized_pseudotime_by_finetune_model'].apply(denormalize,
                                                                                                                                                      args=(8.5, 18.75, -5, 5))
    unspliced_finetune_donors_df['physical_pseudotime_by_finetune_model'] = unspliced_finetune_donors_df['normalized_pseudotime_by_finetune_model'].apply(denormalize,
                                                                                                                                                          args=(8.5, 18.75, -5, 5))

    # spliced_finetune_donors_df["s_or_mrna"] = "spliced"
    # unspliced_finetune_donors_df["s_or_mrna"] = "unspliced"
    spliced_finetune_donors_df = spliced_finetune_donors_df.merge(cell_time_info, left_index=True, right_index=True)
    unspliced_finetune_donors_df = unspliced_finetune_donors_df.merge(cell_time_info, left_index=True, right_index=True)
    cell_time_info_temp = pd.concat([spliced_finetune_donors_df, unspliced_finetune_donors_df], axis=0)
    # from draw_images.read_json_plotViolin_oneTimeMulitDonor import plt_umap_byScanpy
    result_adata = anndata.AnnData(
        X=np.concatenate((spliced_fine_tune_test_latent_mu_result, unspliced_fine_tune_test_latent_mu_result), axis=0),
        obs=cell_time_info_temp)
    # plot umap

    result_adata = plt_umap_byScanpy(result_adata, attr_list=plt_attr, save_path=save_result_path)
    fine_tune_test_spliced_result_adata = anndata.AnnData(X=spliced_fine_tune_test_latent_mu_result, obs=spliced_finetune_donors_df)
    fine_tune_test_unspliced_result_adata = anndata.AnnData(X=unspliced_fine_tune_test_latent_mu_result, obs=unspliced_finetune_donors_df)

    testData_result_dic = {"spliced": fine_tune_test_spliced_result_adata, "unspliced": fine_tune_test_unspliced_result_adata}

    predict_detT, v = RNA_velocity(MyVAEModel.clf_input, MyVAEModel.clf_decoder, config,
                                   unspliced_fine_tune_test_latent_mu_result, spliced_fine_tune_test_latent_mu_result,
                                   subFold_save_file_path, detT=detT, batch_size=batch_size)

    v = pd.DataFrame(data=v, index=spliced_rownames)
    # predict on train data
    return result_adata, runner, experiment_fineTune, testData_result_dic, predict_detT, v


def RNA_velocity(clf_input, clf_decoder, config, unspliced_fine_tune_test_latent_mu_result, spliced_fine_tune_test_latent_mu_result,
                 subFold_save_file_path, detT=0.1, batch_size=100000):
    print(f"detT is {detT}.")
    print(f"RNA velocity model is use latent space with "
          f"unsplice shape: {unspliced_fine_tune_test_latent_mu_result.shape}, splice shape: {spliced_fine_tune_test_latent_mu_result};"
          f" detT: {detT}, batchsize: {batch_size}.")
    from pytorch_lightning import seed_everything
    seed_everything(config['exp_params']['manual_seed'], True)
    from pytorch_lightning import Trainer
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyPredict, SupervisedVAEDataset_onlyTrain
    from pytorch_lightning.loggers import TensorBoardLogger
    from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.callbacks import EarlyStopping

    from TemporalVAE.model_master.experiment_RNA_velocity import Experiment_RNA_velocity

    config['model_params']['in_channels'] = 101

    RNA_velocity_Model = vae_models["RNA_velocity_u_s_Ft_on_s"](**config['model_params'])

    RNA_velocity_Model.fts_input = clf_input
    RNA_velocity_Model.fts_decoder = clf_decoder
    print(f"RNA velocity model structure: {RNA_velocity_Model}")
    RNA_velocity_data = [[spliced_fine_tune_test_latent_mu_result[i], unspliced_fine_tune_test_latent_mu_result[i], detT] for i in
                         range(spliced_fine_tune_test_latent_mu_result.shape[0])]
    RNA_velocity_train_data = SupervisedVAEDataset_onlyTrain(train_data=RNA_velocity_data, train_batch_size=batch_size)
    # 2024-04-19 12:34:35 test the change of LR init from 0.005 to 0.001
    # config["exp_params"]["LR"]=0.001
    experiment_rna_velocity = Experiment_RNA_velocity(RNA_velocity_Model, config['exp_params'])

    lr_monitor = LearningRateMonitor()

    class MyEarlyStopping(EarlyStopping):
        def on_validation_end(self, trainer, pl_module):
            # override this to disable early stopping at the end of val loop
            pass

        def on_train_end(self, trainer, pl_module):
            # instead, do it at the end of training loop
            self._run_early_stopping_check(trainer)

    early_stop_callback = MyEarlyStopping(monitor='train_loss',  # 监控训练损失
                                          min_delta=0.001,  # 变化阈值
                                          # min_delta=0.001,  # 变化阈值 2024-04-19 10:42:21 before min_delta is 0.001
                                          # min_delta=0.0001,  # 变化阈值
                                          patience=10,  # 耐心周期 # 2024-04-19 12:54:40 patience used to 5
                                          verbose=True,  # 是否打印日志
                                          mode='min'  # 监控指标的目标是最小化
                                          )
    check_memory()
    device = auto_select_gpu_and_cpu(free_thre=10, max_attempts=100000000)
    _logger.info("Auto select run on {}".format(device))
    tb_logger = TensorBoardLogger(save_dir=subFold_save_file_path,
                                  name=config['model_params']['name'], )
    runner_rna_velocity = Trainer(logger=tb_logger, log_every_n_steps=1,
                                  callbacks=[early_stop_callback,
                                             lr_monitor,
                                             ModelCheckpoint(save_top_k=1,
                                                             dirpath=os.path.join(tb_logger.log_dir, "checkpoints_rna_velocity"), monitor="train_loss",
                                                             save_last=True),
                                             ],
                                  # check_val_every_n_epoch=1, val_check_interval=1,
                                  devices=[int(device.split(":")[-1])],
                                  accelerator="gpu", max_epochs=100
                                  )
    from pathlib import Path
    Path(f"{tb_logger.log_dir}/Samples_rna_velocity").mkdir(exist_ok=True, parents=True)
    Path(f"{tb_logger.log_dir}/Reconstructions_rna_velocity").mkdir(exist_ok=True, parents=True)

    runner_rna_velocity.fit(experiment_rna_velocity, RNA_velocity_train_data)
    # train data forward the model
    RNA_velocity_predict_data = SupervisedVAEDataset_onlyPredict(predict_data=RNA_velocity_data, predict_batch_size=batch_size)
    RNA_velocity_results = runner_rna_velocity.predict(experiment_rna_velocity, RNA_velocity_predict_data)
    # predict_detT, v
    if len(RNA_velocity_results) > 1:
        print(f'batch size is {batch_size}, concat the batch predictions.')
        #  reclf, mu, log_var, test_loss,recons
        predict_detT = np.concatenate([_i[0] for _i in RNA_velocity_results])
        v = np.concatenate([_i[1] for _i in RNA_velocity_results])
    else:
        predict_detT, v = runner_rna_velocity.predict(experiment_rna_velocity, RNA_velocity_predict_data)[0]
    return predict_detT, v


def get_parameters_df(local_variables):
    """
    get local variables and return as dataframe
    to save as parameters file
    :param local_variables:
    :return:
    """
    variable_list = []

    # 遍历全局变量，过滤出用户自定义变量
    for var_name, var_value in local_variables.items():
        # 过滤掉特殊变量和模块
        if not var_name.startswith("_") and not callable(var_value):
            variable_list.append({'Variable Name': var_name, 'Value': var_value})

    # 创建 DataFrame，将存储的变量和值添加到 DataFrame 中
    df = pd.DataFrame(variable_list, columns=['Variable Name', 'Value'])
    return df


def predict_newData_preprocess_df(gene_dic, adata_new, min_gene_num, reference_file,
                                  hvg_dic=None, bool_change_geneID_to_geneShortName=True):
    """
    2023-10-24 10:58:39
    return the preprocess of new data, by concat with reference data, and only return the preprocessed new data
    :param gene_dic:
    :param adata_new:
    :param min_gene_num:
    :param reference_file:
    :return:
    """
    print("the original sc expression anndata should be gene as row, cell as column")

    try:
        reference_adata = anndata.read_csv(reference_file, delimiter='\t')
    except:
        reference_adata = anndata.read_csv(reference_file, delimiter=',')
    print("read the mouse atlas anndata with shape (gene, cell): {}".format(reference_adata.shape))
    if bool_change_geneID_to_geneShortName:
        reference_adata.obs_names = [gene_dic.get(name, name) for name in reference_adata.obs_names]
        print("change gene name of mouse atlas data to short gene name")
    # 查找adata1和adata2中duplicate columns, that is the duplicate cell name
    if len(set(adata_new.obs_names) & set(reference_adata.var_names)):
        print(
            f"Error: check the test data and mouse atlas data have cells with same cell name: {set(adata_new.obs_names) & set(reference_adata.var_names)}")
        return
    # here
    draw_venn({"atlas": reference_adata.obs_names, "train": adata_new.var_names})

    adata_concated = anndata.concat([reference_adata.copy(), adata_new.T.copy()], axis=1)
    print("merged sc data and external test dataset with shape (gene, cell): {}".format(adata_concated.shape))

    adata_concated = adata_concated.T  # 基因和cell转置矩阵
    print("Import data, cell number: {}, gene number: {}".format(adata_concated.n_obs, adata_concated.n_vars))

    # 数据数目统计
    sc.pp.filter_cells(adata_concated, min_genes=min_gene_num)
    print("After cell threshold: {}, remain adata shape (cell, gene): {}".format(min_gene_num, adata_concated.shape))
    # new_test_cell_list = list(set(adata_new.obs_names) & set(adata_concated.obs_names))
    _cells = adata_concated.obs_names.intersection(adata_new.obs_names)
    new_test_cell_list = adata_concated.obs_names[adata_concated.obs_names.isin(_cells)].tolist()

    # new_test_cell_list = [_cell for _cell in adata_concated.obs_names if _cell in set(adata_new.obs_names)]
    print(f"remain test adata cell num {len(new_test_cell_list)}")
    sc.pp.normalize_total(adata_concated, target_sum=1e6)
    sc.pp.log1p(adata_concated)
    print("Finish normalize per cell, so that every cell has the same total count after normalization.")

    # sc_expression_df = pd.DataFrame(data=adata_concated.X, columns=adata_concated.var_names, index=list(adata_concated.obs_names))
    denseM = adata_concated.X
    from sklearn.preprocessing import scale
    denseM = scale(denseM.astype(float), axis=0, with_mean=True, with_std=True)
    adata_concated.layers["X_normalized"] = denseM
    print("Finish normalize per gene as Gaussian-dist (0, 1).")

    # sc_expression_df = pd.DataFrame(data=denseM, columns=sc_expression_df.columns, index=sc_expression_df.index)
    adata_new_normalized = adata_concated[new_test_cell_list].copy()

    sc_expression_test_df = pd.DataFrame(data=adata_new_normalized.layers["X_normalized"],
                                         columns=adata_new_normalized.var_names, index=list(adata_new_normalized.obs_names))
    loss_gene_shortName_list = list(set(reference_adata.obs_names) - set(adata_new_normalized.var_names))

    sc_expression_test_df[loss_gene_shortName_list] = 0
    sc_expression_test_df = sc_expression_test_df[reference_adata.obs_names]
    cell_time_df = pd.DataFrame(adata_new[new_test_cell_list].obs)

    if hvg_dic is None:
        return sc_expression_test_df, loss_gene_shortName_list, cell_time_df
    else:
        if isinstance(hvg_dic, dict):
            hvg = list(set([_i for _l in hvg_dic.values() for _i in _l]))
            draw_venn({"hvg": list(hvg), "atlas": sc_expression_test_df.columns})
            add_gene = list(set(hvg) - set(sc_expression_test_df.columns))
            adata_new_add = adata_new[:, add_gene].copy()
            sc.pp.filter_genes(adata_new_add, min_cells=50)
        elif isinstance(hvg_dic, list):
            adata_new_add = adata_new[:, hvg_dic].copy()
        print(" remain adata shape (cell, gene): {}".format(adata_new_add.shape))
        print(f"After gene threshold {50}, remain adata shape (cell, gene): {adata_new_add.shape}")
        sc.pp.normalize_total(adata_new_add, target_sum=1e6)
        sc.pp.log1p(adata_new_add)
        print("Finish normalize per cell, so that every cell has the same total count after normalization.")

        denseM_add = scale(adata_new_add.X.astype(float), axis=0, with_mean=True, with_std=True)
        sc_expression_df_add = pd.DataFrame(data=denseM_add, columns=adata_new_add.var_names, index=list(adata_new_add.obs_names))
        sc_expression_df_add = sc_expression_df_add.reindex(sc_expression_test_df.index)

        return sc_expression_test_df, loss_gene_shortName_list, cell_time_df, sc_expression_df_add


def denormalize(y, min_val, max_val, norm_min, norm_max):
    x = ((y - norm_min) * (max_val - min_val)) / (norm_max - norm_min) + min_val
    return x


def normalize(x, min_val, max_val, norm_min, norm_max):
    y = ((x - min_val) * (norm_max - norm_min)) / (max_val - min_val) + norm_min
    return y


def calHVG_adata(adata, gene_num, method="cell_ranger"):
    print("Calculate hvg gene list use cell_ranger method from scanpy.")
    print(f"use hvg method {method}")
    adata_copy = adata.copy()
    if method == "seurat_v3":
        hvg_cellRanger = sc.pp.highly_variable_genes(adata_copy, flavor=method, n_top_genes=gene_num, inplace=False)
    else:
        sc.pp.normalize_total(adata_copy, target_sum=1e6)
        sc.pp.log1p(adata_copy)
        hvg_cellRanger = sc.pp.highly_variable_genes(adata_copy, flavor=method, n_top_genes=gene_num, inplace=False)
    hvg_cellRanger_list = adata_copy.var.index[hvg_cellRanger["highly_variable"]]
    return hvg_cellRanger_list


def task_kFoldTest(donor_list, sc_expression_df, donor_dic, batch_dic,
                   special_path_str, cell_time, time_standard_type,
                   config, train_epoch_num, _logger, donor_str="donor",
                   checkpoint_file=None, batch_size=100000, adversarial_bool=False,
                   recall_predicted_mu=False,
                   cmap_color="viridis"):
    save_path = _logger.root.handlers[0].baseFilename.replace('.log', '')
    _logger.info(f"start task: k-fold test with {donor_list}.")
    predict_donors_dic = dict()
    if adversarial_bool:
        for fold in range(len(donor_list)):
            predict_donor_dic, test_clf_result, label_dic = one_fold_test_adversarialTrain(fold, donor_list,
                                                                                           sc_expression_df,
                                                                                           donor_dic, batch_dic,
                                                                                           special_path_str, cell_time,
                                                                                           time_standard_type,
                                                                                           config, train_epoch_num,
                                                                                           plot_trainingLossLine=True, plot_tags=['lr-Adam', 'lr-Adam-1'],
                                                                                           plot_latentSpaceUmap=False,
                                                                                           time_saved_asFloat=True, batch_size=batch_size, donor_str=donor_str,
                                                                                           checkpoint_file=checkpoint_file)
            predict_donors_dic.update(predict_donor_dic)
    else:
        kFold_result_recall_dic = dict()
        for fold in range(len(donor_list)):
            gc.collect()
            if recall_predicted_mu:
                predict_donor_dic, test_clf_result, label_dic, train_test_mu_result = one_fold_test(fold, donor_list,
                                                                                                    sc_expression_df,
                                                                                                    donor_dic, batch_dic,
                                                                                                    special_path_str, cell_time,
                                                                                                    time_standard_type,
                                                                                                    config, train_epoch_num,
                                                                                                    plot_trainingLossLine=True,
                                                                                                    plot_latentSpaceUmap=False,
                                                                                                    time_saved_asFloat=True, batch_size=batch_size, donor_str=donor_str,
                                                                                                    checkpoint_file=checkpoint_file,
                                                                                                    recall_predicted_mu=recall_predicted_mu)
                kFold_result_recall_dic[donor_list[fold]] = [predict_donor_dic, test_clf_result, label_dic, train_test_mu_result]

            else:
                predict_donor_dic, test_clf_result, label_dic = one_fold_test(fold, donor_list,
                                                                              sc_expression_df,
                                                                              donor_dic, batch_dic,
                                                                              special_path_str, cell_time,
                                                                              time_standard_type,
                                                                              config, train_epoch_num,
                                                                              plot_trainingLossLine=True,
                                                                              plot_latentSpaceUmap=False,
                                                                              time_saved_asFloat=True, batch_size=batch_size, donor_str=donor_str,
                                                                              checkpoint_file=checkpoint_file,
                                                                              recall_predicted_mu=recall_predicted_mu)
            predict_donors_dic.update(predict_donor_dic)

    predict_donors_df = pd.DataFrame(columns=["pseudotime"])
    for fold in range(len(donor_list)):
        predict_donors_df = pd.concat([predict_donors_df, predict_donors_dic[donor_list[fold]]])
    predict_donors_df['predicted_time'] = predict_donors_df['pseudotime'].apply(denormalize, args=(min(label_dic.keys()) / 100, max(label_dic.keys()) / 100,
                                                                                                   min(label_dic.values()), max(label_dic.values())))
    cell_time = pd.concat([cell_time, predict_donors_df], axis=1)
    cell_time.to_csv(f"{save_path}/k_fold_test_result.csv")

    predict_donors_df['time'] = cell_time['time']
    # calculate_real_predict_corrlation_score(predict_donors_df["predicted_time"],
    #                                         predict_donors_df["time"],
    #                                         only_str=False)
    # color_dic = plot_on_each_test_donor_violin_fromDF(cell_time.copy(), save_path, y_attr="predicted_time", x_attr="time", cmap_color=cmap_color)
    color_dic = plot_violin_240223(predict_donors_df.copy(), save_path,
                                   x_attr="time",
                                   y_attr="predicted_time",
                                   special_file_name="kfold", color_map=cmap_color)
    _logger.info("Finish plot image and fold-test.")
    if recall_predicted_mu:
        return predict_donors_dic, label_dic, kFold_result_recall_dic
    else:
        return predict_donors_dic, label_dic


def series_matrix2csv(file_path: str, prefix: str = None):
    """
    Get a GEO series matrix file describing an experiment and
    parse it into project level and sample level data.
    Parameters
    ----------
    file_path: str
        Path to the local gziped txt file with GEO series matrix.
    prefix: str
        Prefix path to write files to.
    """
    import gzip
    from collections import Counter
    if file_path.endswith('.gz'):
        with gzip.open(file_path, 'rb') as f:
            content = f.read()
    else:
        with open(file_path, 'r') as f:
            content = f.read()
    try:
        lines = content.strip().split("\n")
    except:
        lines = content.decode("utf-8").strip().split("\n")
    # separate lines with only one field (project-related)
    # from lines with >2 fields (sample-related)

    prj_lines = dict()
    sample_lines = dict()
    idx_counts: Counter = Counter()
    col_counts: Counter = Counter()

    for line in lines:
        cols = line.strip().split("\t")
        key = cols[0].replace('"', "")
        if len(cols) == 2:
            if key in idx_counts:
                key = f"{key}_{idx_counts[key] + 1}"
            idx_counts[key] += 1
            prj_lines[key] = cols[1].replace('"', "")
        elif len(cols) > 2:
            if key in col_counts:
                key = f"{key}_{col_counts[key] + 1}"
            col_counts[key] += 1
            sample_lines[key] = [x.replace('"', "") for x in cols[1:]]

    prj = pd.Series(prj_lines)
    prj.index = prj.index.str.replace("!Series_", "")

    samples = pd.DataFrame(sample_lines)
    samples.columns = samples.columns.str.replace("!Sample_", "")

    if prefix is not None:
        prj.to_csv(os.path.join(prefix + ".project_annotation.csv"), index=True)
        samples.to_csv(
            os.path.join(prefix + ".sample_annotation.csv"), index=False
        )

    return prj, samples


def read_rds_file(file_name):
    # import gzip
    # import tempfile
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.conversion import localconverter
    # if file_name.endswith('.gz'):
    #     with gzip.open(file_name, 'rb') as f:
    #         # 将解压后的内容写入临时文件中，以便R可以读取
    #         with tempfile.NamedTemporaryFile(delete=False, mode='wb') as tmp_file:
    #             tmp_file.write(f.read())
    #             file_name = tmp_file.name

    # 启用rpy2的自动pandas转换功能
    pandas2ri.activate()

    # 读取.rds文件
    readRDS = ro.r['readRDS']
    df_rds = readRDS(f'{file_name}')

    # 将R的data.frame转换为pandas DataFrame
    with localconverter(ro.default_converter + pandas2ri.converter):
        df = pandas2ri.rpy2py(df_rds)
    return df


def get_top_gene_perturb_data(cell_info, stage, perturb_data_denor,
                              stage_attr="3stage",
                              top_gene_num=5,
                              top_metric="vote_samples"):
    cell_df = cell_info[cell_info[stage_attr] == stage]
    cell_name = cell_df.index
    pert_data = perturb_data_denor.loc[cell_name]
    print(f"In {stage}, cell number is {pert_data.shape}")
    # if top_metric == "abs_mean":
    #     abs_mean_df = pert_data.apply(lambda col: abs(np.array(col) - np.array(cell_df["predicted_time_denor"])).mean())
    #     top_gene_list = abs_mean_df.nlargest(top_gene_num).index
    if top_metric == "vote_samples":
        column_counts = voteScore_genePerturbation(cell_df, pert_data, top_gene_num, predictedTime_attr="predicted_time_denor")
        top_gene_list = column_counts.head(top_gene_num)
        print(f"With {top_metric} strategy, select {top_gene_num} gene: {top_gene_list}")
        top_gene_list = list(top_gene_list.keys())
    else:
        print(f"Please check top metric, input is {top_metric}")
        exit(1)
    print(f"Gene rank metric: {top_metric}.")
    print(f"In {stage} stage {top_gene_num} genes: {top_gene_list}.")

    plt_gene_pd = pd.DataFrame(columns=["gene", "det_time", "real_time"])
    for _g in top_gene_list:
        temp = {"det_time": np.array(pert_data[_g]) - np.array(cell_df["predicted_time_denor"]),
                "real_time": np.array(cell_df["time"]),
                "gene": _g}
        temp = pd.DataFrame(temp, index=cell_df.index)
        temp = temp.sample(n=int(len(temp) / 10), random_state=42)
        plt_gene_pd = pd.concat([plt_gene_pd, temp], axis=0)
    return plt_gene_pd, top_gene_list, pert_data


def voteScore_genePerturbation(cell_df, perturb_df, top_gene_num, predictedTime_attr="predicted_time_denor"):
    _temp = perturb_df - np.array(cell_df[predictedTime_attr])[:, np.newaxis]
    _temp = abs(_temp)
    top_columns_per_row = _temp.apply(lambda row: row.nlargest(top_gene_num).index.tolist(), axis=1)
    all_top_columns = [col for sublist in top_columns_per_row for col in sublist]
    column_counts = pd.Series(all_top_columns).value_counts()
    return column_counts
def from_adata_randomSelect_cells_equityTimePoint(adata_mu_reference, random_select_n_timePoint=200, random_seed=0):
    from collections import Counter
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
    from collections import Counter
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
    from TemporalVAE.model_master.dataset import SupervisedVAEDataset_onlyPredict
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
    adata_mu_query = anndata.AnnData(X=test_mu_result.cpu().numpy(), obs=cell_time_tyser)
    adata_mu_query.obs['data_type'] = test_donor
    # ----

    # ---- combine low-dim representation of reference and query data
    adata_combined = anndata.concat([adata_mu_reference.copy(), adata_mu_query.copy()], axis=0)

    adata_combined.obs["cell_typeMask4dataset"] = adata_combined.obs.apply(lambda row: 'L & M & P & Z & Xiao & C' if row['dataset_label'] != test_donor else row['cell_type'],
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
                                                          masked_str='L & M & P & Z & Xiao & C', color_palette="tab20",
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
                                                 mask_dataset_label='L & M & P & Z & Xiao & C',
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
                                                          # 'Lv': '#8f5239',
                                                          test_donor: (0.9, 0.9, 0.9, 0.7)},
                                               legend_title="Dataset",
                                               reference_dataset_str=reference_dataset_str,
                                               special_file_str=f"_mask{test_donor}_query{test_donor}{special_file_name}")
    plot_tyser_mapping_to_datasets_attrDataset(adata_combined.copy(), save_path,
                                               attr="data_type", masked_str='L & M & P & Z & Xiao & C',
                                               color_dic={'L & M & P & Z & Xiao & C': (0.9, 0.9, 0.9, 0.7),
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
    #                                                           masked_str='L & M & P & Z & Xiao & C', color_palette="tab20",
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
    return adata_mu_query
