from source.helpers import *
from sklearn.model_selection import train_test_split
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from source.data import RANDOM_STATE, filename_dict


def cox(name):

    filename = filename_dict[name]
    raw_data = pd.read_csv(os.path.join(DATA_DIR, filename))
    formatted_x, formatted_y = sksurv_data_formatting(raw_data)

    x_train, x_test, y_train, y_test = train_test_split(formatted_x, formatted_y,
                                                        test_size=0.25, random_state=RANDOM_STATE)

    estimator = CoxPHSurvivalAnalysis()
    estimator.fit(x_train, y_train)

    prediction = estimator.predict(x_test)
    result = concordance_index_censored(y_test["Status"], y_test["Survival_in_days"], prediction)

    return result[0]

filename = filename_dict["metabric"]
data = pd.read_csv(os.path.join(DATA_DIR, filename))
