from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest


def random_survival_forest(formatted_x, formatted_y, test_size, n_estimators, min_samples_split,
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











