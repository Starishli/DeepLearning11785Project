from source.model.CoxMixtureOfExperts import run_all


if __name__ == "__main__":
    # dataset_name_list = ["support", "metabric", "whas", "gbsg"]
    dataset_name_list = ["whas"]

    for dataset_name in dataset_name_list:
        run_all(dataset_name)