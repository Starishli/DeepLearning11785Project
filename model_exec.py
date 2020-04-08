import os
import pandas as pd
import numpy as np

from source import DATA_DIR
from source.helpers import sksurv_data_formatting
from source.data import RANDOM_STATE
from source.model.RandomSurvivalForest import random_survival_forest


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_DIR, "metabric1904.csv"))

    formatted_x, formatted_y = sksurv_data_formatting(df)

    test_size = 0.25
    n_estimators = 500
    min_samples_split = 10
    min_samples_leaf = 15
    max_features = "sqrt"
    cur_score = random_survival_forest(formatted_x, formatted_y, test_size, n_estimators, min_samples_split,
                                       min_samples_leaf, max_features, RANDOM_STATE)

    print(cur_score)


