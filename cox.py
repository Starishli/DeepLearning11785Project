import torch
import os
import time
from lifelines.utils import concordance_index
import sys
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, Dataset
import torch.utils.data.dataloader as dataloader
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import _pickle as pkl
from torch.multiprocessing import Pool
from lifelines import CoxPHFitter
from source.helpers import *
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from source.data import RANDOM_STATE


def cox(file_name):

    df = pd.read_csv(os.path.join(DATA_DIR, file_name))
    raw_x, raw_y = sksurv_data_formatting(df)
    random_state = 7

    x_train, x_test, y_train, y_test = train_test_split(raw_x, raw_y, test_size=0.25, random_state=RANDOM_STATE)

    estimator = CoxPHSurvivalAnalysis()
    estimator.fit(x_train, y_train)

    prediction = estimator.predict(x_test)
    result = concordance_index_censored(y_test["Status"], y_test["Survival_in_days"], prediction)

    return result[0]

