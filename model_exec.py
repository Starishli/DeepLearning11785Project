from source.model.RandomSurvivalForest import random_survival_forest
from source.model.DeepSurv import deepsurv_trainNtest
from source.model.Cox import cox

if __name__ == "__main__":
    dataset_name_list = ["support", "metabric", "whas", "gbsg"]

    for dataset_name in dataset_name_list:
        print("============={}=============".format(dataset_name))
        cox_score = cox(dataset_name)
        print("cox score:", cox_score)

        rsf_score = random_survival_forest(dataset_name)
        print("RSF score:", rsf_score)

        deepsurv_score = deepsurv_trainNtest(dataset_name, 500)
        print("DeepSurv score:", deepsurv_score)
        print("============================")




