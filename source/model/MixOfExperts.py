from source.helpers import *
from source.data import RANDOM_STATE, filename_dict
from sklearn.model_selection import train_test_split


def mix_of_experts(name):

    filename = filename_dict[name]
    raw_data = pd.read_csv(os.path.join(DATA_DIR, filename))
    formatted_x, formatted_y = sksurv_data_formatting(raw_data)

    x_train, x_test, y_train, y_test = train_test_split(formatted_x, formatted_y,
                                                        test_size=0.25, random_state=RANDOM_STATE)

    x_train, e_train, t_train, x_valid, e_valid, t_valid, x_test, e_test, t_test = scale_data_to_torch(x_train, y_train,
                                                                                                       x_test, y_test,
                                                                                                       x_test, y_test)
    print("Dataset loaded and scaled")

    risk_set, risk_set_valid, risk_set_test = compute_risk_set(t_train, t_valid, t_test)
    print("Risk set computed")






mix_of_experts("metabric")
