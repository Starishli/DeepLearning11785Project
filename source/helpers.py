import os
import numpy as np
import pandas as pd

import torch
from sklearn.preprocessing import StandardScaler

from source import DATA_DIR


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
        "x": raw_x.values.astype("float32"),
        "t": raw_y["lenfol"].values.astype("float32"),
        "e": raw_y["fstat"].values.astype("int32")
    }

    return res_dict


def get_dataset(dataset_choice):

    if dataset_choice == 1 :

        data = pd.read_csv(os.path.join(DATA_DIR, 'whas1638.csv'), sep=',')
        train = data[:1310]
        train = data[:1310]
        valid_size = int(data.shape[0] * 0.1)
        valid = train[-valid_size:]
        train = train[:-valid_size]
        test = data[1310:]

        x_train = train[['0', '1', '2', '3', '4', '5']]
        x_valid = valid[['0', '1', '2', '3', '4', '5']]
        x_test = test[['0', '1', '2', '3', '4', '5']]

        y_train = train[['fstat', 'lenfol']]
        y_valid = valid[['fstat', 'lenfol']]
        y_test = test[['fstat', 'lenfol']]

        dataset_name = "WHAS"

    elif dataset_choice == 2:

        data = pd.read_csv(os.path.join(DATA_DIR, 'gbsg2232.csv'), sep=',')
        train = data[:1546]
        valid_size = int(data.shape[0] * 0.1)
        valid = train[-valid_size:]
        train = train[:-valid_size]
        test = data[1546:]
        print("GBSG Dataset: ")
        print("Total dataset size: ", data.shape[0])
        print("Train data size: ", train.shape[0])
        print("Validation data size: ", valid.shape[0])
        print("Test data size: ", test.shape[0])
        x_train = train[['0', '1', '2', '3', '4', '5', '6']]
        x_valid = valid[['0', '1', '2', '3', '4', '5', '6']]
        x_test = test[['0', '1', '2', '3', '4', '5', '6']]

        y_train = train[['fstat', 'lenfol']]
        y_valid = valid[['fstat', 'lenfol']]
        y_test = test[['fstat', 'lenfol']]

        dataset_name = "GBSG"

    elif dataset_choice == 3:

        data = pd.read_csv(os.path.join(DATA_DIR, 'support8873.csv'), sep=',')
        train = data[:1523]
        valid_size = int(data.shape[0] * 0.1)
        valid = train[-valid_size:]
        train = train[:-valid_size]
        test = data[1523:]
        print("METABRIC Dataset: ")
        print("Total dataset size: ", data.shape[0])
        print("Train data size: ", train.shape[0])
        print("Validation data size: ", valid.shape[0])
        print("Test data size: ", test.shape[0])
        x_train = train[['0', '1', '2', '3', '4', '5', '6', '7', '8']]
        x_valid = valid[['0', '1', '2', '3', '4', '5', '6', '7', '8']]
        x_test = test[['0', '1', '2', '3', '4', '5', '6', '7', '8']]

        y_train = train[['fstat', 'lenfol']]
        y_valid = valid[['fstat', 'lenfol']]
        y_test = test[['fstat', 'lenfol']]

        dataset_name = "METABRIC"

    elif dataset_choice == 4:

        data = pd.read_csv(os.path.join(DATA_DIR, 'metabric1904.csv'), sep=',')
        train = data[:7098]
        valid_size = int(data.shape[0] * 0.1)
        valid = train[-valid_size:]
        train = train[:-valid_size]
        test = data[7098:]
        print("SUPPORT Dataset: ")
        print("Total dataset size: ", data.shape[0])
        print("Train data size: ", train.shape[0])
        print("Validation data size: ", valid.shape[0])
        print("Test data size: ", test.shape[0])
        x_train = train[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', ]]
        x_valid = valid[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']]
        x_test = test[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']]

        y_train = train[['fstat', 'lenfol']]
        y_valid = valid[['fstat', 'lenfol']]
        y_test = test[['fstat', 'lenfol']]

        dataset_name = "SUPPORT"

    else:
        print('The dataset input is not valid')
        return

    return x_train, train, x_valid, valid, x_test, test, y_train, y_valid, y_test, dataset_name


def scale_data_to_torch(x_train, train, x_valid, valid, x_test, test):
    scl = StandardScaler()
    x_train = scl.fit_transform(x_train)
    e_train = train['fstat']
    t_train = train['lenfol']

    x_valid = scl.fit_transform(x_valid)
    e_valid = valid['fstat']
    t_valid = valid['lenfol']

    x_test = scl.transform(x_test)
    e_test = test['fstat']
    t_test = test['lenfol']

    x_train = torch.from_numpy(x_train).float()
    e_train = torch.from_numpy(e_train.values).float()
    t_train = torch.from_numpy(t_train.values)

    x_valid = torch.from_numpy(x_valid).float()
    e_valid = torch.from_numpy(e_valid.values).float()
    t_valid = torch.from_numpy(t_valid.values)

    x_test = torch.from_numpy(x_test).float()
    e_test = torch.from_numpy(e_test.values).float()
    t_test = torch.from_numpy(t_test.values)

    return x_train, e_train, t_train, x_valid, e_valid, t_valid, x_test, e_test, t_test


def compute_risk_set(t_train,t_valid,t_test):
    t_ = t_train.cpu().data.numpy()
    risk_set = []
    for i in range(len(t_)):

        risk_set.append([i]+np.where(t_>t_[i])[0].tolist())

    t_ = t_valid.cpu().data.numpy()

    risk_set_valid = []
    for i in range(len(t_)):

        risk_set_valid.append([i]+np.where(t_>t_[i])[0].tolist())


    t_ = t_test.cpu().data.numpy()

    risk_set_test = []
    for i in range(len(t_)):

        risk_set_test.append([i]+np.where(t_>t_[i])[0].tolist())
    return risk_set,risk_set_valid,risk_set_test






