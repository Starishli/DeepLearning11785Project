import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from source import DATA_DIR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.ensemble import RandomSurvivalForest


df = pd.read_csv(os.path.join(DATA_DIR, "metabric1904.csv"))

raw_x = df.iloc[:, :9]
raw_y = df.iloc[:, 9:]

raw_y = np.array([(bool(_x[1]), _x[0]) for _x in raw_y])

random_state = 7

x_train, x_test, y_train, y_test = train_test_split(raw_x, raw_y, test_size=0.25, random_state=random_state)

cur_rsf = RandomSurvivalForest(n_estimators=500,
                               min_samples_split=10,
                               min_samples_leaf=15,
                               max_features="sqrt",
                               n_jobs=-1,
                               random_state=random_state)
cur_rsf.fit(x_train, y_train)

print(cur_rsf.score(x_test, y_test))








