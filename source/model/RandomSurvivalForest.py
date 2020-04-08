import os
import pandas as pd

from source import DATA_DIR
from source.data import filename_dict, RANDOM_STATE
from source.helpers import sksurv_data_formatting
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest


def _random_survival_forest(formatted_x, formatted_y, test_size, n_estimators, min_samples_split,
                            min_samples_leaf, max_features, random_state):

    x_train, x_test, y_train, y_test = train_test_split(formatted_x, formatted_y, test_size=test_size,
                                                        random_state=random_state)

    cur_rsf = RandomSurvivalForest(n_estimators=n_estimators,
                                   min_samples_split=min_samples_split,
                                   min_samples_leaf=min_samples_leaf,
                                   max_features=max_features,
                                   n_jobs=-1,
                                   random_state=random_state)
    cur_rsf.fit(x_train, y_train)

    return cur_rsf.score(x_test, y_test)


def random_survival_forest(dataset_name):
    filename = filename_dict[dataset_name]
    df = pd.read_csv(os.path.join(DATA_DIR, filename))

    formatted_x, formatted_y = sksurv_data_formatting(df)

    test_size = 0.25
    n_estimators = 500
    min_samples_split = 10
    min_samples_leaf = 15
    max_features = "sqrt"
    cur_score = _random_survival_forest(formatted_x, formatted_y, test_size, n_estimators, min_samples_split,
                                        min_samples_leaf, max_features, RANDOM_STATE)

    return cur_score











