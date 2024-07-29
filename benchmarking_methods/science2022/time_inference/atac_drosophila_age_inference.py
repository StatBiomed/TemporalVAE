from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from scipy.io import mmread
from types import SimpleNamespace as PARS
import os
import gc
from DC_tensorflow_helpers import *
import sys

# uncomment if running this as a script
run_info=sys.argv[1]
run_ida,run_idb,k=run_info.split('_')
run_id=run_ida+'_'+run_idb
k=int(k)
print('Starting fold:'+str(k))
print('On run:'+run_id)

# Figures out the run we doing
run=None
lpd={'None':None, 'custom_loss':custom_loss}
opd={'None':None, 'out_fn_tanh':out_fn_tanh, 'out_fn_sig':out_fn_sig}
with open('nobackup/atac_k_fold_crossvalidation_runs_v1.txt', 'r') as o:
    o.readline()
    for line in o:
        ri,k2,lp,op,l1,l2=line.split()
        if ri==run_id and k==int(k2):
            run=PARS(run_id=ri, loss_fn=lpd[lp], out_fn=opd[op],
                    l1=float(l1), l2=float(l2), k=k)

print(run)
## Open the data and start the run
print('starting to open data')
y0,x0,y1,x1,y_valid,x_valid = atac_open_data(k)
print('making model: ' + run.run_id)
m = atac_make_model(run.run_id, x0, out_fn=run.out_fn, l1=run.l1, l2=run.l2)
print('fitting model: ' + run.run_id)
h = atac_fit_model(run.run_id, m, x0, y0, x_valid, y_valid,
    loss_fn=run.loss_fn, epochs=50, verbose=1,
    my_callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
print('evaluating model: ' + run.run_id)
s = m.evaluate(x1, y1, verbose=0)

a_file='nobackup/atac_k_fold_crossvalidation/'+run.run_id+'_'+str(k)+'.txt'
with open(a_file, 'w') as o:
    o.write("\t".join([run.run_id, str(k),
        str(s[0]), str(s[1]), str(s[2])]) + '\n')

# ## Run and save the best models - these all are with output projection functions
best_runs=[
    # best NNv1
    PARS(run_id='NNv1', loss_fn=None, out_fn=out_fn_sig, l1=0.00001, l2=0),
    # best NNv2
    PARS(run_id='NNv1', loss_fn=custom_loss, out_fn=None, l1=0.000001, l2=0.000001)
]
# save info to disk - if save again then rename
with open('nobackup/atac_best_runs_v1.txt', 'w') as o:
    o.write('run\tloss\tprojection\tl1\tl2\n')
    for run in best_runs:
        lp='None' if run.loss_fn is None else run.loss_fn.__name__
        op='None' if run.out_fn is None else run.out_fn.__name__
        o.write("\t".join([run.run_id,lp,op,str(run.l1), str(run.l2)])+'\n')

y, x, y_valid, x_valid = atac_open_all_data() # running here at 1:02 mar 24 - on fit
for run in best_runs:
    print('starting model: ' + run.run_id)
    m = atac_make_model(run.run_id, x, out_fn=run.out_fn)
    h = atac_fit_model(run.run_id, m, x, y, x_valid, y_valid,
        loss_fn=run.loss_fn, epochs=50, verbose=1,
        my_callbacks=[ModelCheckpoint('nobackup/'+run.run_id, save_best_only=True),
            EarlyStopping(patience=5)])

################ WRITE OUT THE PREDICTIONS  ##########
# open best model and save output on full data and test subsample
# # might need to close and then open again to save some memory
del(y)
del(x)
del(x_valid)
del(y_valid)
gc.collect()
for m in [run.run_id for run in best_runs]:
    print('starting model: ' + m)
    # loop through times and save predictions
    atac_save_predictions(m)
################################################################


#######################################################
### IMPORTANT CODE USED TO GO THROUGH AND BUILD RUNS
# programtically adds all the different runs of interest
runs=[]
L1_LIST = [1, 0.1, 0.001, 0.0001, 0.00001, 0.000001, 0]
L2_LIST = [1, 0.1, 0.001, 0.0001, 0.00001, 0.000001, 0]

i=0
with open('nobackup/atac_k_fold_crossvalidation_runs_v1.txt', 'w') as o:
    o.write('run\tk\tloss\tprojection\tl1\tl2\n')
    for l1 in L1_LIST:
        for l2 in L2_LIST:
            for a_loss in [None, custom_loss]:
                for a_out_fn in [None, out_fn_tanh, out_fn_sig]:
                    ri='atac_v'+str(i)
                    lp='None' if a_loss is None else a_loss.__name__
                    op='None' if a_out_fn is None else a_out_fn.__name__
                    i+=1
                    for k in range(10):
                        runs.append(PARS(run_id=ri, loss_fn=a_loss,
                            out_fn=a_out_fn, l1=l1, l2=l2, k=k))
                        o.write("\t".join([ri,str(k),lp,op,str(l1),
                            str(l2)])+'\n')
#######################################################

# if looping through runs
for run in runs_by_k[str(k)]:
    gc.collect()
    print('starting model: ' + run.run_id)
    m = atac_make_model(run.run_id, x0, out_fn=run.out_fn, l1=run.l1, l2=run.l2)
    h = atac_fit_model(run.run_id, m, x0, y0, x_valid, y_valid,
        loss_fn=run.loss_fn, epochs=50, verbose=0,
        my_callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
    s = m.evaluate(x1, y1, verbose=0)
    a_file='nobackup/atac_k_fold_crossvalidation/'+run.run_id+'_'+str(k)+'.txt'
    with open(a_file, 'w') as o:
        o.write("\t".join([run.run_id, str(k),
            str(s[0]), str(s[1]), str(s[2])]) + '\n')