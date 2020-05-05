RANDOM_STATE = 7

filename_dict = {
    "metabric": "metabric1904.csv",
    "support": "support8873.csv",
    "gbsg": "gbsg2232.csv",
    "whas": "whas1638.csv"
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

    "gbsg": {"learning_rate": 0.0004991066534650134,
             "dropout": 0.0783935546875,
             "lr_decay": 0.000746533203125,
             "momentum": 0.8255483398437501,
             "L2_reg": 1.5917993164062498,
             "batch_norm": False,
             "standardize": True,
             "n_in": 7,
             "hidden_layers_sizes": [20, 20, 20],
             "activation": "selu"},

    "whas": {"L2_reg": 2.8032470703125,
             "dropout": 0.0899951171875,
             "learning_rate": 0.023825980458515722,
             "lr_decay": 0.0005817382812499998,
             "momentum": 0.8383100585937501,
             "batch_norm": False,
             "activation": "selu",
             "standardize": True,
             "n_in": 6,
             "hidden_layers_sizes": [40, 40, 40]}
}
