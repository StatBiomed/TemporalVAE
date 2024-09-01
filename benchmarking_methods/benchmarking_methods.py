# -*-coding:utf-8 -*-
"""
@Project ：TemporalVAE 
@File    ：benchmarking_methods.py
@IDE     ：PyCharm 
@Author  ：awa121
@Date    ：2024/8/26 17:24 
"""
import numpy as np

from benchmarking_methods.science2022.time_inference.DC_tensorflow_helpers import *
from tensorflow.keras.callbacks import EarlyStopping
from types import SimpleNamespace as PARS
import gc
def science2022(train_x, train_y,run_id='NNv2',
                loss_fn=custom_loss, out_fn=out_fn_tanh,
                l1=0, l2=0.00001,**kwargs):

    # best_runs = [
    #     PARS(run_id='NNv1', loss_fn=None, out_fn=out_fn_sig, l1=0.0001, l2=0.00001),
    #     best MSE with custom loss
        # PARS(run_id='NNv2', loss_fn=custom_loss, out_fn=out_fn_tanh, l1=0, l2=0.00001)
    # ]
    i=0
    run = PARS(run_id=run_id, loss_fn=loss_fn,out_fn=out_fn, l1=l1, l2=l2)  # 2024-07-19 14:17:21 yijun add
    # with open('nobackup/k_fold_crossvalidation_runs.txt', 'w') as o:
    #     o.write('run\tk\tloss\tprojection\tl1\tl2\n')
    gc.collect()
    print('starting model: ' + run.run_id)
    train_y2=np.array(train_y.tolist()).astype(np.float32)
    m = make_model(run.run_id, train_x, out_fn=run.out_fn, l1=run.l1, l2=run.l2)
    h = fit_model(run.run_id, m, train_x, train_y2, loss_fn=run.loss_fn, epochs=50, verbose=1,
                  my_callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])
    gc.collect()

    if 'test_df' in kwargs:
            return m.predict(kwargs['test_df'])
    else:
        return m
def ot_svm_classifier(train_x, train_y,test_x,test_y):
    #
    # import ot
    train_y=np.array(train_y)
    from skada import JDOTClassifier,JDOTRegressor
    from sklearn.linear_model import LogisticRegression

    X=np.concatenate((train_x,test_x),axis=0)
    y=np.concatenate((train_y,test_y),axis=0)
    domain=np.concatenate((np.ones(len(train_y),dtype=np.int8),-1*(np.ones(len(test_y),dtype=np.int8))))

    # jdot = JDOTClassifier(LogisticRegression(), alpha=0.1, verbose=True)
    # jdot.fit(X, y, sample_domain=domain)
    # ypred = jdot.predict(test_x)
    # jdot = JDOTClassifier()
    jdot = JDOTRegressor()
    try:
        jdot.fit(X, y, sample_domain=domain)
        ypred = jdot.predict(test_x)
    except:
        print("So big train data, random select 1/3 and data down to float16")
        import random
        random.seed(123)
        random_indices = random.sample(range(train_x.shape[0]), int(len(train_x) / 3), )
        train_x_temp = train_x[random_indices, :]
        train_y_temp=train_y[random_indices]

        X = np.concatenate((train_x_temp, test_x), axis=0)
        y = np.concatenate((train_y_temp, test_y), axis=0)
        domain = np.concatenate((np.ones(len(train_y_temp), dtype=np.int8), -1 * (np.ones(len(test_y), dtype=np.int8))))
        import gc
        gc.collect()
        X=X.astype(np.float16)
        y=y.astype(np.float16)
        test_x2=test_x.astype(np.float16)
        jdot.fit(X, y, sample_domain=domain)
        ypred = jdot.predict(test_x2)
    return ypred
def random_forest_regressor(train_x, train_y):
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(max_depth=2, random_state=0)
    model.fit(train_x, train_y)
    return model