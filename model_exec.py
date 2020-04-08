from source.model.RandomSurvivalForest import random_survival_forest
from source.model.Cox import cox
from source.model.DeepSurv import deepsurv_trainNtest


if __name__ == "__main__":
    dataset_name = "metabric"

    rsf_score = random_survival_forest(dataset_name)
    deepsurv_score = deepsurv_trainNtest(dataset_name)

    print(rsf_score)




