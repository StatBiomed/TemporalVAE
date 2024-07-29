import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Normalization
from tensorflow.keras import metrics
# from tensorflow.keras.callbacks import ModelCheckpoint
from scipy.io import mmread
import numpy as np
from scipy.sparse import vstack
from random import sample
from math import floor
from keras import backend as BK
from scipy.sparse import csr_matrix
import gc
import keras

keras.utils.set_random_seed(0)  # yijun add to make reproduction


######## CUSTOM METRIC FOR counting nuclei in correct bins
class InRightBin(metrics.Metric):
    def __init__(self, name="proportion_correct", **kwargs):
        super().__init__(name=name, **kwargs)
        self.n_correct = self.add_weight(name="nc", initializer="zeros")
        self.total = self.add_weight(name="t", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        error = tf.math.abs(y_true - y_pred)
        ws2 = tf.cast(windows.lookup(tf.cast(y_true, dtype=cs.dtype)),
                      dtype=error.dtype)
        ones_holder = tf.ones_like(error)
        values = tf.where(tf.math.less(error, ws2),
                          ones_holder, tf.zeros_like(error))
        self.n_correct.assign_add(tf.reduce_sum(values))
        self.total.assign_add(tf.reduce_sum(ones_holder))

    def result(self):
        return self.n_correct / self.total

    def reset_state(self):
        self.total.assign(0.0)
        self.n_correct.assign(0.0)


######## CUSTOM METRIC FOR counting nuclei in correct bins

def delete_from_csr(mat, row_indices=[], col_indices=[]):
    """
    Adapted from https://stackoverflow.com/questions/13077527/ +
     is-there-a-numpy-delete-equivalent-for-sparse-matrices
    Remove the rows (denoted by ``row_indices``) and
        columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    rows = []
    cols = []
    if row_indices:
        rows = list(row_indices)
    if col_indices:
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return (mat[row_mask][:, col_mask])
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return (mat[mask])
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return (mat[:, mask])
    else:
        return (mat)


# MAKING VERSIONS OF HELPER FUNCTIONS FOR ATAC
def atac_make_model(run_id, x, out_fn=None, l1=0, l2=0):
    # # this was atac big v3 - 1000 layer
    # # trying v2 with 100 layer
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=(x.shape[1],), sparse=True))
    model.add(layers.Dense(units=10, name='layer1',
                           kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2)))
    # model.add(layers.Dense(units=1000, name='layer2', activation='relu'))
    model.add(layers.Dense(units=100, name='layer2', activation='relu'))
    model.add(layers.Dense(units=60, name='layer3', activation='relu'))
    model.add(layers.Dense(units=20, name='layer4', activation='relu'))
    model.add(layers.Dense(units=1, name='layer5',
                           activation=out_fn))
    return (model)


def atac_open_data(k, validation_split=0.05):
    # Open and data and convert to csr
    fs = [i for i in range(10) if i != k]
    y_train = np.delete(np.concatenate([np.genfromtxt('nobackup/' + str(i + 1) + \
                                                      '_fold_atac_centers.tsv') for i in fs], axis=0), [1, 2, 3], 1)
    x_train = vstack([mmread('nobackup/' + str(i + 1) + \
                             '_fold_atac.mtx') for i in fs]).tocsr()

    # shuffle the training data
    p = np.arange(x_train.shape[0])
    np.random.shuffle(p)
    x_train, y_train = x_train[p,], y_train[p]
    del (p)
    gc.collect()

    # sample validation set and remove from train set
    n_obs = y_train.shape[0]
    valid_idx = sample(range(n_obs), floor(validation_split * n_obs))
    y_validation = y_train[valid_idx]
    x_validation = x_train[valid_idx,]
    y_train = np.delete(y_train, valid_idx)
    x_train = delete_from_csr(x_train, row_indices=valid_idx)

    # go back to coo
    x_train = x_train.tocoo()
    x_validation = x_validation.tocoo()

    # then make all tensorflow sparse, no need for ys
    x_train = tf.sparse.reorder(tf.SparseTensor(
        np.mat([x_train.row, x_train.col]).transpose(),
        x_train.data, x_train.shape))

    x_validation = tf.sparse.reorder(tf.SparseTensor(
        np.mat([x_validation.row, x_validation.col]).transpose(),
        x_validation.data, x_validation.shape))

    # Finally open/process the test fold
    y_test_fold = np.delete(np.genfromtxt('nobackup/' + str(k + 1) + \
                                          '_fold_atac_centers.tsv'), [1, 2, 3], 1)
    x_test_fold = mmread('nobackup/' + str(k + 1) + \
                         '_fold_atac.mtx')
    x_test_fold = tf.sparse.reorder(tf.SparseTensor(
        np.mat([x_test_fold.row, x_test_fold.col]).transpose(),
        x_test_fold.data, x_test_fold.shape))
    return (y_train, x_train, y_test_fold, x_test_fold,
            y_validation, x_validation)


def atac_open_all_data(validation_split=0.05):
    # open everything
    fs = [i for i in range(10)]
    print('opening data')
    gc.collect()
    y_train = np.delete(np.concatenate([np.genfromtxt('nobackup/' + str(i + 1) + \
                                                      '_fold_atac_centers.tsv') for i in fs], axis=0), [1, 2, 3], 1)
    x_train = vstack([mmread('nobackup/' + str(i + 1) + \
                             '_fold_atac.mtx') for i in fs]).tocsr()

    # shuffle
    print('shuffling')
    p = np.arange(x_train.shape[0])
    np.random.shuffle(p)
    x_train, y_train = x_train[p,], y_train[p]
    del (p)
    gc.collect()

    # sample validation set and remove from train set
    print('collecting validation set')
    n_obs = y_train.shape[0]
    valid_idx = sample(range(n_obs), floor(validation_split * n_obs))
    y_validation = y_train[valid_idx]
    x_validation = x_train[valid_idx,]
    y_train = np.delete(y_train, valid_idx)
    x_train = delete_from_csr(x_train, row_indices=valid_idx)

    # back to coo then sparsify
    print('back to coo then sparsify and return')
    gc.collect()
    x_train = x_train.tocoo()
    x_train = tf.sparse.reorder(tf.SparseTensor(
        np.mat([x_train.row, x_train.col]).transpose(),
        x_train.data, x_train.shape))
    x_validation = x_validation.tocoo()
    x_validation = tf.sparse.reorder(tf.SparseTensor(
        np.mat([x_validation.row, x_validation.col]).transpose(),
        x_validation.data, x_validation.shape))
    return (y_train, x_train, y_validation, x_validation)


def atac_save_predictions(run_id):
    time_splits = ['0-2', '1-3', '2-4', '3-7', '4-8', '6-10', '8-12',
                   '10-14', '12-16', '14-18', '16-20']
    model = tf.keras.models.load_model('nobackup/' + run_id,
                                       custom_objects={"custom_loss": custom_loss,
                                                       "out_fn_tanh": out_fn_tanh, "out_fn_sig": out_fn_sig,
                                                       "InRightBin": InRightBin})

    # Run predictions on test fold data
    print('running predictions on test data')
    test_data = mmread('nobackup/11_fold_atac.mtx')
    test_data = tf.sparse.reorder(tf.SparseTensor(
        np.mat([test_data.row, test_data.col]).transpose(),
        test_data.data, test_data.shape))
    np.savetxt('nobackup/11_fold_predictions_' + run_id + '.txt',
               model.predict(test_data))
    gc.collect()

    # loop through time splits then write predictions
    print('running predictions on all data')
    with open('nobackup/all_predictions_' + run_id + '.txt', 'w') as f:
        f.write(run_id + '\n')
    with open('nobackup/all_predictions_' + run_id + '.txt', "a") as f:
        for split in time_splits:
            print('predicting on: ' + split + ' hour window.')
            time_data = mmread('nobackup/' + split + '_all_atac.mtx')
            time_data = tf.sparse.reorder(tf.SparseTensor(
                np.mat([time_data.row, time_data.col]).transpose(),
                time_data.data, time_data.shape))
            np.savetxt(f, model.predict(time_data))


def atac_fit_model(run_id, model, x, y, x_valid, y_valid, loss_fn=None,
                   epochs=50, batch_size=64, my_callbacks=None, verbose=1):
    if (loss_fn is None):
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
                      loss='mean_squared_error', metrics=['mse', InRightBin()])
    else:
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
                      loss=loss_fn, metrics=['mse', InRightBin()])

    gc.collect()  # helps memory to not run out

    # fit the model - need validation to save best model
    history = model.fit(x, y, epochs=epochs, batch_size=batch_size,
                        callbacks=my_callbacks, verbose=verbose,
                        validation_data=(x_valid, y_valid))

    return (history)


######### CUSTOM PROJECTION TO 0-20 hours ############
# Add this as the activation of the final layer
# 'https://stackoverflow.com/questions/49911206/how-to-' +
#     '\restrict-output-of-a-neural-net-to-a-specific-range'
def out_fn_tanh(x, target_min=0, target_max=20):
    x02 = BK.tanh(x) + 1  # x in range(0,2)
    scale = (target_max - target_min) / 2.
    return (x02 * scale + target_min)


def out_fn_sig(x, target_min=0, target_max=20):
    return (BK.sigmoid(x) * (target_max - target_min) + target_min)


######### CUSTOM PROJECTION TO 0-20 hours ############

########################################################
# Custom loss function where we have a 0 if the
# nuclei is within the sampling window, or MSE otherwise
cs = tf.constant([1, 2, 3, 5, 6, 8, 10, 12, 14, 16, 18])
ws = tf.constant([1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])
winit = tf.lookup.KeyValueTensorInitializer(keys=cs, values=ws)
windows = tf.lookup.StaticHashTable(initializer=winit,
                                    default_value=-1, name="window_size")


# if nuclei out of window then MSE from window edge else 0
def custom_loss(ytrue, yhat):
    # ytrue = tf.cast(ytrue, tf.float32)  # 将 y_true 转换为 float32
    # yhat = tf.cast(yhat, tf.float32)  # 确保 y_pred 也是 float32
    error = tf.math.abs(ytrue - yhat)

    ws2 = tf.cast(windows.lookup(tf.cast(ytrue, dtype=cs.dtype)),
                  dtype=error.dtype)
    error2 = tf.math.subtract(error, ws2)
    loss = tf.where(tf.math.less(error, ws2),
                    tf.zeros_like(error), error2)
    return tf.math.reduce_mean(tf.math.square(loss))


########################################################

####### Open kfold data ###################################
def open_data(k):
    fs = [i for i in range(10) if i != k]
    y_train = np.delete(np.concatenate([np.genfromtxt('nobackup/' + str(i + 1) + \
                                                      '_fold_rna_centers.tsv') for i in fs], axis=0), [1, 2, 3], 1)
    x_train = vstack([mmread('nobackup/' + str(i + 1) + \
                             '_fold_rna.mtx') for i in fs]).todense()

    # shuffle the training data
    p = np.arange(x_train.shape[0])
    np.random.shuffle(p)
    x_train, y_train = x_train[p,], y_train[p]
    del (p)
    gc.collect()

    y_validation = np.delete(np.genfromtxt('nobackup/' + str(k + 1) + \
                                           '_fold_rna_centers.tsv'), [1, 2, 3], 1)
    x_validation = mmread('nobackup/' + str(k + 1) + \
                          '_fold_rna.mtx').todense()
    return (y_train, x_train, y_validation, x_validation)


def open_all_data():
    fs = [i for i in range(10)]
    y_train = np.delete(np.concatenate([np.genfromtxt('nobackup/' + str(i + 1) + \
                                                      '_fold_rna_centers.tsv') for i in fs], axis=0), [1, 2, 3], 1)
    x_train = vstack([mmread('nobackup/' + str(i + 1) + \
                             '_fold_rna.mtx') for i in fs]).todense()

    p = np.arange(x_train.shape[0])
    np.random.shuffle(p)
    x_train, y_train = x_train[p,], y_train[p]
    del (p)
    gc.collect()
    return (y_train, x_train)


####### Open kfold data ###################################

######### MODEL MAKING FUNCTION ############
def make_model(run_id, x, normalize=False, out_fn=None, l1=0, l2=0):
    model = tf.keras.Sequential()
    if (normalize):
        normalizer = Normalization(axis=-1)
        normalizer.adapt(np.array(x))
        model.add(normalizer)

    model.add(layers.Dense(units=5, input_shape=(x.shape[1],), name='layer1',
                           kernel_regularizer=tf.keras.regularizers.l1_l2(l1=l1, l2=l2)))
    model.add(layers.Dense(units=100, name='layer2', activation='relu'))
    model.add(layers.Dense(units=50, name='layer3', activation='relu'))
    model.add(layers.Dense(units=20, name='layer4', activation='relu'))
    model.add(layers.Dense(units=1, name='layer5',
                           activation=out_fn))
    return (model)


######### MODEL MAKING FUNCTION ############

######### SAVING PREDICTIONS ############
def save_predictions(run_id, big=None):
    model = tf.keras.models.load_model('nobackup/' + run_id,
                                       # loads all the custom functions we sometimes use
                                       custom_objects={
                                           "custom_loss": custom_loss,
                                           "out_fn_tanh": out_fn_tanh,
                                           "out_fn_sig": out_fn_sig,
                                           "InRightBin": InRightBin,
                                       })

    np.savetxt('nobackup/11_fold_predictions_' + run_id + '.txt',
               model.predict(mmread('nobackup/11_fold_rna.mtx').todense()))

    # open big data in sparse format if not already open
    if (big is None):
        big = mmread('nobackup/all_rna.mtx').tocsr()

    # loop through and write predictions of 50k slices to disk
    with open('nobackup/all_predictions_' + run_id + '.txt', 'w') as f:
        f.write(run_id + '\n')

    gc.collect()
    splits = [i for i in range(0, big.shape[0], 40000)] + [big.shape[0]]
    with open('nobackup/all_predictions_' + run_id + '.txt', "a") as f:
        for i in range(len(splits) - 1):
            j, k = splits[i], splits[i + 1]
            print('predicting from: ' + str(j) + ', to: ' + str(k))
            np.savetxt(f, model.predict(big[j:k, ].todense()))


######### SAVING PREDICTIONS ############

######### MODEL FITTING FUNCTION ############
def fit_model(run_id, model, x, y, loss_fn=None, epochs=50, batch_size=64,
              my_callbacks=None, verbose=1):
    if (loss_fn is None):
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
                      loss='mean_squared_error', metrics=['mse', InRightBin()])
    else:
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
                      loss=loss_fn, metrics=['mse', InRightBin()])

    gc.collect()  # helps memory to not run out
    history = model.fit(x, y, epochs=epochs, batch_size=batch_size,
                        callbacks=my_callbacks, validation_split=0.05, verbose=verbose)

    return (history)
######### MODEL FITTING FUNCTION ############
