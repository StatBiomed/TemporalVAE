from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from scipy.io import mmread
from types import SimpleNamespace as PARS
import os
import gc
from DC_tensorflow_helpers import *

# set up to loop through the bulk samples

######## RNA work
#################### WRITE OUT bulk data PREDICTIONS ##########
for data_file in ['bulk_rna', 'bulk_rna_subsampled']:
    df = 'nobackup/' + data_file + '.mtx'
    print('working on: ' + data_file)
    bulk_data=mmread(df).todense()
    for run_id in ['NNv1', 'NNv2']:
        print('starting: '+run_id)
        model = tf.keras.models.load_model('nobackup/'+run_id,
            custom_objects={"custom_loss": custom_loss,
                "out_fn_tanh": out_fn_tanh, "out_fn_sig": out_fn_sig,
                "InRightBin": InRightBin})
        np.savetxt('nobackup/bulk_predictions_'+run_id+'_'+data_file+'.txt',
            model.predict(bulk_data))
#################### WRITE OUT bulk data PREDICTIONS ##########


######## ATAC work
#################### WRITE OUT bulk data PREDICTIONS ##########
for data_file in ['bulk_atac', 'bulk_atac_subsampled', 'mesodermsc_atac']:
    df = 'nobackup/' + data_file + '.mtx'
    print('working on: ' + data_file)
    bulk_data=mmread(df)
    bulk_data=tf.sparse.reorder(tf.SparseTensor(
                np.mat([bulk_data.row, bulk_data.col]).transpose(),
                bulk_data.data, bulk_data.shape))
    for run_id in ['NNv1', 'NNv2']:
        print('starting: '+run_id)
        model = tf.keras.models.load_model('nobackup/'+run_id,
            custom_objects={"custom_loss": custom_loss,
                "out_fn_tanh": out_fn_tanh, "out_fn_sig": out_fn_sig,
                "InRightBin": InRightBin})
        np.savetxt('nobackup/bulk_predictions_'+run_id+'_'+data_file+'.txt',
            model.predict(bulk_data))
#################### WRITE OUT bulk data PREDICTIONS ##########