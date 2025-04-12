import collections
import copy
import gc
import logging
import math
import os
import pickle
import time
import re

import sys
notebook_dir = os.getcwd()
grandparent_dir = os.path.dirname(os.path.dirname(notebook_dir))
sys.path.append(grandparent_dir)
# sys.path.append('/Users/alanzu/projects/TFs_do_KF_ICL/src/core')
print(sys.path)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fn
from tensordict import TensorDict
from pandas.plotting import table
from datetime import datetime

from core import Config
from models import GPT2, CnnKF
from data_train import set_config_params
from create_plots_with_zero_pred import tf_preds
from linalg_helpers import print_matrix

def getMats(trace_config, seg_starts, multi_sys, preds_tf):
    #returns the one-after matrices of the transformer ouput, true ouput, and average of outputs, and average with 0 predictions
    num_sys = np.size(seg_starts[trace_config])-1
    vec_inds = [seg_starts[trace_config][i] + 1 for i in range(num_sys)]
    q_ind = seg_starts[trace_config][num_sys] + 1
    mats = [multi_sys[trace_config, :, vec_inds[j], -5:] for j in range(num_sys)]
    mat_avg = np.zeros_like(mats[0])
    for i in range(num_sys):
        mat_avg += mats[i]

    mat_avg_w_zero = mat_avg/(num_sys+1) # matrix of numtrials by statedim
    mat_avg /= num_sys # matrix of numtrials by statedim

    transformer_1af = preds_tf[trace_config, :, q_ind] # matrix of numtrials by statedim
    true_1af = multi_sys[trace_config, :, q_ind, -5:] # same dim as transformer_1af
    return transformer_1af, true_1af, mat_avg, mat_avg_w_zero

def getSims(transformer_1af, true_1af, mat_avg, mat_avg_w_zero, payload=None):
    #returns normalized magnitudes and angles of transformer output relative to the true ouput and average output
    transformer_mags = np.linalg.norm(transformer_1af, axis = 1)
    true_mags = np.linalg.norm(true_1af, axis = 1)
    avg_mags = np.linalg.norm(mat_avg, axis = 1)
    avg_w_zero_mags = np.linalg.norm(mat_avg_w_zero, axis = 1)
    mags = {}
    angs = {}
    mags["true"] = transformer_mags/true_mags
    angs["true"] = np.diagonal(transformer_1af@true_1af.T)/(transformer_mags*true_mags)
    mags["avg"] = transformer_mags/avg_mags
    angs["avg"] = np.diagonal(transformer_1af@mat_avg.T)/(transformer_mags*avg_mags)
    mags["avg_w_zero"] = transformer_mags/avg_w_zero_mags
    angs["avg_w_zero"] = np.diagonal(transformer_1af@mat_avg_w_zero.T)/(transformer_mags*avg_w_zero_mags)


    if payload is not None:
        payload_mags = np.linalg.norm(payload, axis = 1)
        mags["payload"] = transformer_mags/payload_mags
        payload_angs = np.diagonal(transformer_1af@payload.T)/(transformer_mags*payload_mags)
        angs["payload"] = payload_angs

    return mags, angs