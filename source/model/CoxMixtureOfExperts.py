import os
import torch
import numpy as np
import pandas as pd

from torch import nn
from source import DATA_DIR, RESULT_DIR
from source.helpers import cache_write
from source.data import filename_dict, RANDOM_STATE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lifelines.utils import concordance_index
from sklearn.utils import resample


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class ExpertsMixture(object):
    def __init__(self, dataset_name, learning_rate, linear_model, num_hidden_layers, n_epochs=4000):
        filename = filename_dict[dataset_name]
        self.dataset_name = dataset_name

        df = pd.read_csv(os.path.join(DATA_DIR, filename))
        df = df.drop_duplicates()

        df_train_n_valid, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_STATE)
        df_train, df_valid = train_test_split(df_train_n_valid, test_size=0.2, random_state=RANDOM_STATE)

        self.x_train, self.e_train, self.t_train, self.risk_set_train = self._data_preprocess(df_train)
        self.x_valid, self.e_valid, self.t_valid, self.risk_set_valid = self._data_preprocess(df_valid)
        self.x_test, self.e_test, self.t_test, self.risk_set_test = self._data_preprocess(df_test)

        self.n_features = self.x_train.shape[1]

        self.learning_rate = learning_rate
        self.linear_model = linear_model
        self.num_hidden_layers = num_hidden_layers

        self.beta_network = nn.Sequential(nn.Linear(self.n_features, linear_model, bias=False))

        gated_network_layers_sizes = [self.n_features for _ in range(num_hidden_layers)] + [linear_model]
        gated_network_layers = []

        for i in range(len(gated_network_layers_sizes) - 2):
            gated_network_layers.append(nn.Linear(gated_network_layers_sizes[i],
                                                  gated_network_layers_sizes[i + 1], bias=False))
            gated_network_layers.append(nn.ReLU())
        gated_network_layers.append(nn.Linear(gated_network_layers_sizes[-2],
                                              gated_network_layers_sizes[-1], bias=False))

        self.gates_network = nn.Sequential(*gated_network_layers)
        self.optimizer = torch.optim.Adam(list(self.gates_network.parameters())
                                          + list(self.beta_network.parameters()), lr=learning_rate)

        self.n_epochs = n_epochs

        self._to_device()

        self.train_ci_soft = []
        self.train_ci_hard = []

        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []

        self.test_ci_soft = []
        self.test_ci_hard = []

    @staticmethod
    def _data_preprocess(raw_data):
        raw_x = raw_data.iloc[:, :-2].copy()
        raw_y = raw_data.iloc[:, -2:].copy()

        raw_x = StandardScaler().fit_transform(raw_x)
        raw_e = raw_y["fstat"].values
        raw_t = raw_y["lenfol"].values

        risk_set = []

        for i, cur_t in enumerate(raw_t):
            risk_set.append([i] + np.where(raw_t > cur_t)[0].tolist())

        return torch.from_numpy(raw_x).float(), torch.from_numpy(raw_e).float(), \
               torch.from_numpy(raw_t).float(), risk_set

    def _to_device(self):
        self.x_train = self.x_train.to(DEVICE)
        self.x_valid = self.x_valid.to(DEVICE)
        self.x_test = self.x_test.to(DEVICE)

        self.e_train = self.e_train.to(DEVICE)
        self.e_valid = self.e_valid.to(DEVICE)
        self.e_test = self.e_test.to(DEVICE)

        self.t_train = self.t_train.to(DEVICE)
        self.t_valid = self.t_valid.to(DEVICE)
        self.t_test = self.t_test.to(DEVICE)

        self.beta_network = self.beta_network.to(DEVICE)
        self.gates_network = self.gates_network.to(DEVICE)

    def _empty_training_results(self):
        self.train_ci_soft = []
        self.train_ci_hard = []
        self.train_loss = []
        self.valid_loss = []

    def _empty_test_results(self):
        self.test_loss = []
        self.test_ci_soft = []
        self.test_ci_hard = []

    @staticmethod
    def _concordance_index_calc(beta_outputs, gates_outputs, t, e, bootstrap=0):
        t = t.detach().cpu().numpy()
        e = e.detach().cpu().numpy()

        soft_gates = nn.Softmax(dim=1)(gates_outputs)

        soft_hazard = torch.exp(beta_outputs)
        soft_hazard = torch.mul(soft_gates, soft_hazard).sum(dim=1)
        soft_hazard = -1 * soft_hazard.detach().cpu().numpy()

        hard_hazard = torch.exp(beta_outputs)
        hard_hazard = hard_hazard[:, gates_outputs.argmax(1)[1]]
        hard_hazard = -1 * hard_hazard.detach().cpu().numpy()

        if bootstrap == 0:
            return concordance_index(t, soft_hazard, e), concordance_index(t, hard_hazard, e)
        else:
            soft_c_indices, hard_c_indices = [], []
            for i in range(bootstrap):
                soft_dat_, e_, t_ = resample(soft_hazard, e, t, random_state=i)
                sci = concordance_index(t_, soft_dat_, e_)

                hard_dat_, e_, t_ = resample(hard_hazard, e, t, random_state=i)
                hci = concordance_index(t_, hard_dat_, e_)

                soft_c_indices.append(sci)
                hard_c_indices.append(hci)

            return soft_c_indices, hard_c_indices

    @staticmethod
    def _elbo_calc(beta_outputs, gates_outputs, e, risk_set):
        soft_gates = nn.Softmax(dim=1)(gates_outputs)

        l_numerator = torch.mul(soft_gates, beta_outputs).sum(dim=1)
        expected_risks = (torch.exp(beta_outputs) * soft_gates).sum(dim=1)

        l_denominator = [torch.sum(expected_risks[risk_set[i]], dim=0) for i in range(beta_outputs.shape[0])]
        l_denominator = torch.stack(l_denominator, dim=0)
        l_denominator = torch.log(l_denominator)

        likelihoods = l_numerator - l_denominator
        e_index = np.where(e.detach().cpu().numpy() == 1)[0]

        neg_likelihood = -torch.sum(likelihoods[e_index])

        return neg_likelihood

    def model_training(self):
        self._empty_training_results()
        epsilon = 1e-4

        prev_loss_train = 0
        prev_loss_valid = 0

        bad_count = 0

        for epoch in range(self.n_epochs):
            self.gates_network.train()
            self.beta_network.train()

            self.optimizer.zero_grad()

            gates_outputs_train = self.gates_network(self.x_train)
            beta_outputs_train = self.beta_network(self.x_train)

            ci_soft_train, ci_hard_train = self._concordance_index_calc(beta_outputs_train, gates_outputs_train,
                                                                        self.t_train, self.e_train, bootstrap=0)

            self.train_ci_soft.append(ci_soft_train)
            self.train_ci_hard.append(ci_hard_train)

            loss_train = self._elbo_calc(beta_outputs_train, gates_outputs_train, self.e_train, self.risk_set_train) \
                   + (torch.pow(self.beta_network[0].weight, 2)).sum()
            loss_train.backward()
            self.optimizer.step()

            cur_loss_train = loss_train.detach().cpu().numpy()

            self.train_loss.append(cur_loss_train)

            if abs(cur_loss_train - prev_loss_train) < epsilon:
                break
            prev_loss_train = cur_loss_train

            torch.cuda.empty_cache()

            if epoch % 50 == 0:
                print(cur_loss_train)

            # validation
            self.gates_network.eval()
            self.beta_network.eval()

            gates_outputs_valid = self.gates_network(self.x_valid)
            beta_outputs_valid = self.beta_network(self.x_valid)

            loss_valid = self._elbo_calc(beta_outputs_valid, gates_outputs_valid, self.e_valid, self.risk_set_valid)
            cur_loss_valid = loss_valid.detach().cpu().numpy()

            self.valid_loss.append(cur_loss_valid)

            # avoid overfitting
            if cur_loss_valid - prev_loss_valid > epsilon:
                bad_count += 1
                if bad_count > 2:
                    break
            else:
                bad_count = 0

            if epoch % 50 == 0:
                print(cur_loss_valid)

            prev_loss_valid = cur_loss_valid
            torch.cuda.empty_cache()

    def model_testing(self):
        self.gates_network.eval()
        self.beta_network.eval()

        gated_outputs_test = self.gates_network(self.x_test)
        beta_outputs_test = self.beta_network(self.x_test)

        ci_soft_test, ci_hard_test = self._concordance_index_calc(beta_outputs_test, gated_outputs_test,
                                                                  self.t_test, self.e_test, 250)

        self.test_ci_soft.append(ci_soft_test)
        self.test_ci_hard.append(ci_soft_test)

        loss_test = self._elbo_calc(beta_outputs_test, gated_outputs_test, self.e_test, self.risk_set_test)
        cur_loss_test = loss_test.detach().cpu().numpy()

        self.test_loss.append(cur_loss_test)
        torch.cuda.empty_cache()

    def result_output(self):
        result_dict = {"train_loss": self.train_loss, "valid_loss": self.valid_loss, "test_loss": self.test_loss,
                       "c-index-soft": self.train_ci_soft, "c-index-hard": self.train_ci_hard,
                       "c-index-test-soft": self.test_ci_soft, "c-index-test-hard": self.test_ci_hard}

        output_file_name = "{}_{}_{}_{}_{}".format(self.dataset_name, self.learning_rate, self.linear_model,
                                                   self.num_hidden_layers, self.n_epochs)

        cache_write(result_dict, os.path.join(RESULT_DIR, output_file_name))

    def exec(self):
        self.model_training()
        self.model_testing()
        self.result_output()


if __name__ == "__main__":
    test_class = ExpertsMixture("metabric", 0.001, 10, 1)
    test_class.exec()

