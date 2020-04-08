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


df = pd.read_csv(os.path.join(DATA_DIR, "metabric1904.csv"))

raw_x, raw_y = sksurv_data_formatting(df)

x_train, x_test, y_train, y_test = train_test_split(raw_x, raw_y, test_size=0.25, random_state=RANDOM_STATE)

print(x_train.head())

cur_rsf = RandomSurvivalForest(n_estimators=500,
                               min_samples_split=10,
                               min_samples_leaf=15,
                               max_features="sqrt",
                               n_jobs=-1,
                               random_state=RANDOM_STATE)
cur_rsf.fit(x_train, y_train)

print(cur_rsf.score(x_test, y_test))








