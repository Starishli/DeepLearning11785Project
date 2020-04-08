import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from source import DATA_DIR
from source.data import RANDOM_STATE
from source.helpers import sksurv_data_formatting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest


def random_survival_forest(file_name, test_size, n_estimators, min_samples_split,
                           min_samples_leaf, max_features, random_state):
    df = pd.read_csv(os.path.join(DATA_DIR, file_name))

    raw_x, raw_y = sksurv_data_formatting(df)

    x_train, x_test, y_train, y_test = train_test_split(raw_x, raw_y, test_size=test_size, random_state=random_state)

    cur_rsf = RandomSurvivalForest(n_estimators=n_estimators,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   max_features=max_features,
                                   n_jobs=-1,
                                   random_state=random_state)
    cur_rsf.fit(x_train, y_train)

    return cur_rsf.score(x_test, y_test)








