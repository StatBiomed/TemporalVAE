from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from scipy.io import mmread
from types import SimpleNamespace as PARS
import os
import gc
from DC_tensorflow_helpers import *

# Loops and builds all test parameters
L1_LIST = [1, 0.1, 0.001, 0.0001, 0.00001, 0.000001, 0]
L2_LIST = [1, 0.1, 0.001, 0.0001, 0.00001, 0.000001, 0]

i=0

runs=[] # 2024-07-19 14:17:21 yijun add
with open('nobackup/k_fold_crossvalidation_runs.txt', 'w') as o:
    o.write('run\tk\tloss\tprojection\tl1\tl2\n')
    for l1 in L1_LIST:
        for l2 in L2_LIST:
            for a_loss in [None, custom_loss]:
                for a_out_fn in [None, out_fn_tanh, out_fn_sig]:
                    ri='rna_v'+str(i)
                    lp='None' if a_loss is None else a_loss.__name__
                    op='None' if a_out_fn is None else a_out_fn.__name__
                    i+=1
                    for k in range(10):
                        runs.append(PARS(run_id=ri, loss_fn=a_loss,
                            out_fn=a_out_fn, l1=l1, l2=l2, k=k))
                        o.write("\t".join([ri,str(k),lp,op,str(l1),
                            str(l2)])+'\n')
#######################################################

# loop model fitting for all folds
runs_by_k={}
lpd={'None':None, 'custom_loss':custom_loss}
opd={'None':None, 'out_fn_tanh':out_fn_tanh, 'out_fn_sig':out_fn_sig}
for k in range(9):
    y0,x0,y1,x1 = open_data(k)
    for run in runs_by_k[str(k)]:
        gc.collect()
        print('starting model: ' + run.run_id)
        m = make_model(run.run_id, x0, out_fn=run.out_fn, l1=run.l1, l2=run.l2)
        h = fit_model(run.run_id, m, x0, y0, loss_fn=run.loss_fn, epochs=50, verbose=0,
            my_callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
        s = m.evaluate(x1, y1, verbose=0)
        a_file='nobackup/k_fold_crossvalidation/'+ run.run_id + '_'+str(k)+'.txt'
        with open(a_file, 'w') as o:
            o.write("\t".join([run.run_id, str(k),
                str(s[0]), str(s[1]), str(s[2])]) + '\n')

    del(x0)
    del(x1)
    gc.collect()

######## THEN rerun model with best parameters on all data and save
## Run and save the best models
best_runs=[
    PARS(run_id='NNv1', loss_fn=None, out_fn=out_fn_sig, l1=0.0001, l2=0.00001),
    # best MSE with custom loss
    PARS(run_id='NNv2', loss_fn=custom_loss, out_fn=out_fn_tanh, l1=0, l2=0.00001)
]

with open('nobackup/best_runs_v1.txt', 'w') as o:
    o.write('run\tloss\tprojection\tl1\tl2\n')
    for run in best_runs:
        lp='None' if run.loss_fn is None else run.loss_fn.__name__
        op='None' if run.out_fn is None else run.out_fn.__name__
        o.write("\t".join([run.run_id,lp,op,str(run.l1), str(run.l2)])+'\n')

y, x = open_all_data()
for run in best_runs:
    print('starting model: ' + run.run_id)
    m = make_model(run.run_id, x, out_fn=run.out_fn)
    h = fit_model(run.run_id, m, x, y, loss_fn=run.loss_fn, epochs=50, verbose=0,
        my_callbacks=[ModelCheckpoint('nobackup/'+run.run_id, save_best_only=True)])

######################## WRITE OUT THE PREDICTIONS ##########
del(y)
del(x)
gc.collect()
big = mmread('nobackup/all_rna.mtx').tocsr()
for m in [run.run_id for run in best_runs]:
    print('starting model: ' + m)
    save_predictions(m, big=big)
################################################################


#################### WRITE OUT bulk data PREDICTIONS ##########
for run_id in ['NNv1', 'NNv2']:
    print('starting: '+run_id)
    model = tf.keras.models.load_model('nobackup/'+run_id,
        custom_objects={"custom_loss": custom_loss,
            "out_fn_tanh": out_fn_tanh, "out_fn_sig": out_fn_sig,
            "InRightBin": InRightBin})
    bulk_data=mmread('nobackup/bulk_rna.mtx').todense()
    np.savetxt('nobackup/bulk_predictions_'+run_id+'.txt',
        model.predict(bulk_data))