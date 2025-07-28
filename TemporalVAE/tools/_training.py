
from ..utils import LogHelper
from ..utils.utils_project import *

import os
import yaml
import numpy as np
import pandas as pd

def train(X_training, cell_time, **kwargs):
    """
    TO BE IMPLEMENTED
    """
    pass

    res = None

    # res = onlyTrain_model(
    #     sc_expression_df_filter, 
    #     donor_dic=batch_dic,
    #     special_path_str="",
    #     cell_time=cell_time_filter,
    #     time_standard_type="embryoneg1to1", 
    #     config=config, 
    #     train_epoch_num=50,
    #     plot_latentSpaceUmap=True, 
    #     plot_trainingLossLine=True, 
    #     time_saved_asFloat=True, 
    #     batch_dic=batch_dic, 
    #     donor_str="dataset_label",
    #     batch_size=100000
    # )

    return res

def predict(trained_model, X_query, **kwargs):
    """
    TO BE IMPLEMENTED
    """
    pass

def cross_validate(X_training, cell_time, stratify_key, **kwargs):
    """
    TO BE IMPLEMENTED
    """
    pass

