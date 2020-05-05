import os
import pandas as pd

from source.data import deepsurv_hyper_params_dict, filename_dict, RANDOM_STATE
from source import DATA_DIR
from source.helpers import deepsurv_data_formatting
from deepsurv.deep_surv import DeepSurv
from sklearn.model_selection import train_test_split


def deepsurv_trainNtest(name, n_epochs):
    filename = filename_dict[name]
    hyper_params = deepsurv_hyper_params_dict[name]

    cur_network = DeepSurv(**hyper_params)

    df = pd.read_csv(os.path.join(DATA_DIR, filename))
    df = df.drop_duplicates()

    df_train_n_valid, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_STATE)
    df_train, df_valid = train_test_split(df_train_n_valid, test_size=0.2, random_state=RANDOM_STATE)

    train_input_dict = deepsurv_data_formatting(df_train)
    valid_input_dict = deepsurv_data_formatting(df_valid)
    test_input_dict = deepsurv_data_formatting(df_test)

    cur_log = cur_network.train(train_input_dict, valid_input_dict, n_epochs=n_epochs)
    cur_score = cur_network.get_concordance_index(**test_input_dict)

    return cur_score


if __name__ == "__main__":
    dataset_name = "whas"
    score = deepsurv_trainNtest(dataset_name, 500)

    print(score)
