import os
import pandas as pd
import deepsurv

from source.data import RANDOM_STATE
from source import DATA_DIR
from source.helpers import deepsurv_data_formatting
from deepsurv.deep_surv import DeepSurv
from sklearn.model_selection import train_test_split


hidden_layers = [128, 128, 128]
feature_num = 9

hyper_params = {
    "n_in": feature_num,
    "learning_rate": 1e-5,
    "hidden_layers_sizes": hidden_layers,
    "dropout": 0.3,
    "batch_norm": True
}

cur_network = DeepSurv(**hyper_params)

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_DIR, "metabric1904.csv"))

    df_train_n_valid, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_STATE)
    df_train, df_valid = train_test_split(df_train_n_valid, test_size=0.2, random_state=RANDOM_STATE)

    train_input_dict = deepsurv_data_formatting(df_train)
    valid_input_dict = deepsurv_data_formatting(df_valid)
    test_input_dict = deepsurv_data_formatting(df_test)

    cur_log = cur_network.train(train_input_dict, valid_input_dict, n_epochs=500)
    cur_score = cur_network.get_concordance_index(**test_input_dict)

    print(cur_score)
