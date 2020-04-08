from source import DATA_DIR
import pandas as pd
import os


def get_dataset(dataset_choice):

    if dataset_choice == 1 :

        data = pd.read_csv(os.path.join(DATA_DIR, 'whas1638.csv'), sep=',')
        train = data[:1310]
        train = data[:1310]
        valid_size = int(data.shape[0] * 0.1)
        valid = train[-valid_size:]
        train = train[:-valid_size]
        test = data[1310:]

        x_train = train[['0', '1', '2', '3', '4', '5']]
        x_valid = valid[['0', '1', '2', '3', '4', '5']]
        x_test = test[['0', '1', '2', '3', '4', '5']]

        y_train = train[['fstat', 'lenfol']]
        y_valid = valid[['fstat', 'lenfol']]
        y_test = test[['fstat', 'lenfol']]

        dataset_name = "WHAS"

    elif dataset_choice == 2:

        data = pd.read_csv(os.path.join(DATA_DIR, 'gbsg2232.csv'), sep=',')
        train = data[:1546]
        valid_size = int(data.shape[0] * 0.1)
        valid = train[-valid_size:]
        train = train[:-valid_size]
        test = data[1546:]
        print("GBSG Dataset: ")
        print("Total dataset size: ", data.shape[0])
        print("Train data size: ", train.shape[0])
        print("Validation data size: ", valid.shape[0])
        print("Test data size: ", test.shape[0])
        x_train = train[['0', '1', '2', '3', '4', '5', '6']]
        x_valid = valid[['0', '1', '2', '3', '4', '5', '6']]
        x_test = test[['0', '1', '2', '3', '4', '5', '6']]

        y_train = train[['fstat', 'lenfol']]
        y_valid = valid[['fstat', 'lenfol']]
        y_test = test[['fstat', 'lenfol']]

        dataset_name = "GBSG"

    elif dataset_choice == 3:

        data = pd.read_csv(os.path.join(DATA_DIR, 'support8873.csv'), sep=',')
        train = data[:1523]
        valid_size = int(data.shape[0] * 0.1)
        valid = train[-valid_size:]
        train = train[:-valid_size]
        test = data[1523:]
        print("METABRIC Dataset: ")
        print("Total dataset size: ", data.shape[0])
        print("Train data size: ", train.shape[0])
        print("Validation data size: ", valid.shape[0])
        print("Test data size: ", test.shape[0])
        x_train = train[['0', '1', '2', '3', '4', '5', '6', '7', '8']]
        x_valid = valid[['0', '1', '2', '3', '4', '5', '6', '7', '8']]
        x_test = test[['0', '1', '2', '3', '4', '5', '6', '7', '8']]

        y_train = train[['fstat', 'lenfol']]
        y_valid = valid[['fstat', 'lenfol']]
        y_test = test[['fstat', 'lenfol']]

        dataset_name = "METABRIC"

    elif dataset_choice == 4:

        data = pd.read_csv(os.path.join(DATA_DIR, 'metabric1904.csv'), sep=',')
        train = data[:7098]
        valid_size = int(data.shape[0] * 0.1)
        valid = train[-valid_size:]
        train = train[:-valid_size]
        test = data[7098:]
        print("SUPPORT Dataset: ")
        print("Total dataset size: ", data.shape[0])
        print("Train data size: ", train.shape[0])
        print("Validation data size: ", valid.shape[0])
        print("Test data size: ", test.shape[0])
        x_train = train[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', ]]
        x_valid = valid[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']]
        x_test = test[['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13']]

        y_train = train[['fstat', 'lenfol']]
        y_valid = valid[['fstat', 'lenfol']]
        y_test = test[['fstat', 'lenfol']]

        dataset_name = "SUPPORT"

    else:
        print('The dataset input is not valid')
        return

    return x_train, train, x_valid, valid, x_test, test, y_train, y_valid, y_test, dataset_name






