from source.model.CoxMixtureOfExperts import run_all


if __name__ == "__main__":
    # dataset_name_list = ["metabric", "whas", "gbsg"]
    dataset_name_list = ["support"]

    for dataset_name in dataset_name_list:
        print("Start all experiments for {}".format(dataset_name))
        run_all(dataset_name)
