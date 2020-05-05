import os
import numpy as np
import pandas as pd
import torch
import pickle

from sklearn.preprocessing import StandardScaler


def cache_write(file_path, data):
    file_dir = os.path.dirname(file_path)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    with open(file_path, 'wb') as fp:
        pickle.dump(data, fp)


def cache_load(file_path):
    if not os.path.isfile(file_path):
        print('Warning: No such as file to Load')
        return None
    with open(file_path, 'rb') as fp:
        pp = pickle.load(fp)
    return pp


def sksurv_data_formatting(raw_data):
    raw_x = raw_data.iloc[:, :-2].copy()
    raw_y = raw_data.iloc[:, -2:]

    raw_y = np.array([(bool(_y[1]), _y[0]) for _y in raw_y.values],
                     dtype=[("Status", "?"), ("Survival_in_days", "<f8")])

    return raw_x, raw_y


def deepsurv_data_formatting(raw_data):
    raw_x = raw_data.iloc[:, :-2].copy()
    raw_y = raw_data.iloc[:, -2:]

    res_dict = {
        "x": StandardScaler().fit_transform(raw_x).astype("float32"),
        "t": raw_y["lenfol"].values.astype("float32"),
        "e": raw_y["fstat"].values.astype("int32")
    }

    return res_dict





