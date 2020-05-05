import os
import numpy as np
import pandas as pd

import torch
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
from sklearn.utils import resample

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


def scale_data_to_torch(x_train, y_train, x_valid, y_valid, x_test, y_test):
    scl = StandardScaler()
    x_train = scl.fit_transform(x_train)
    e_train = y_train['fstat']
    t_train = y_train['lenfol']

    x_valid = scl.fit_transform(x_valid)
    e_valid = y_valid['fstat']
    t_valid = y_valid['lenfol']

    x_test = scl.transform(x_test)
    e_test = y_test['fstat']
    t_test = y_test['lenfol']

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


def get_concordance_index(x, gated_x, t, e, bootstrap=False):
    t = t.detach().cpu().numpy()
    e = e.detach().cpu().numpy()
    softmax = torch.nn.Softmax(dim=1)(gated_x)
    r = x.shape[0]
    soft_computed_hazard = torch.exp(x)
    hard_computed_hazard = soft_computed_hazard[range(r),gated_x.argmax(1)[1]]
    soft_computed_hazard = torch.mul(softmax, soft_computed_hazard)
    soft_computed_hazard = torch.sum(soft_computed_hazard, dim = 1)
    soft_computed_hazard = -1*soft_computed_hazard.detach().cpu().numpy()
    hard_computed_hazard = -1*hard_computed_hazard.detach().cpu().numpy()
    if not bootstrap:
        return concordance_index(t,soft_computed_hazard,e),concordance_index(t,hard_computed_hazard,e)
    else:
        soft_concord, hard_concord = [], []
        for i in range(bootstrap):
            soft_dat_, e_, t_ = resample(soft_computed_hazard, e, t,random_state=i )
            sci = concordance_index(t_,soft_dat_,e_)
            hard_dat_,  e_, t_  = resample(hard_computed_hazard,  e, t ,random_state=i)
            hci = concordance_index(t_,hard_dat_,e_)
            soft_concord.append(sci)
            hard_concord.append(hci)
        return soft_concord, hard_concord






