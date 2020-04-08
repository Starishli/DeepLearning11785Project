RANDOM_STATE = 7

filename_dict = {
    "metabric": "metabric1904.csv",
    "support": "support8873.csv",
    "gbsg": "gbsg2232.csv"
}

deepsurv_hyper_params_dict = {
    "metabric": {"L2_reg": 10.890986328125,
                 "dropout": 0.160087890625,
                 "learning_rate": 0.010289691253027908,
                 "lr_decay": 0.0041685546875,
                 "momentum": 0.8439658203125,
                 "batch_norm": False,
                 "activation": "selu",
                 "standardize": True,
                 "n_in": 9,
                 "hidden_layers_sizes": [41]},

    "support": {"L2_reg": 8.1196630859375,
                "dropout": 0.2553173828125,
                "learning_rate": 0.047277902766138385,
                "lr_decay": 0.00257333984375,
                "momentum": 0.8589028320312501,
                "batch_norm": False,
                "activation": "selu",
                "standardize": True,
                "n_in": 14,
                "hidden_layers_sizes": [44]},

    "gbsg": {"L2_reg": 6.5512451171875,
             "dropout": 0.6606318359374999,
             "learning_rate": 0.153895727328729,
             "lr_decay": 0.005667089843750001,
             "momentum": 0.88674658203125,
             "batch_norm": False,
             "activation": "selu",
             "standardize": True,
             "n_in": 7,
             "hidden_layers_sizes": [8]}}
