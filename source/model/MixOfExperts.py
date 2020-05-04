from source.helpers import *
from source.data import RANDOM_STATE, filename_dict
from sklearn.model_selection import train_test_split
from torch.multiprocessing import Pool
from torch import nn
import time
import _pickle as pkl


def elbo(risk, gated_output, E, risk_set):
    go_sm = nn.Softmax(dim=1)(gated_output)
    lnumerator = torch.mul(go_sm, risk)
    lnumerator = torch.sum(lnumerator, dim=1)
    expected_risks = torch.exp(risk) * go_sm
    expected_risks = torch.sum(expected_risks, dim=1)
    ldenominator = []
    for i in range(risk.shape[0]):
        ldenominator.append(torch.sum(expected_risks[risk_set[i]], dim=0))
    ldenominator = torch.stack(ldenominator, dim=0)
    ldenominator = torch.log(ldenominator)
    likelihoods = lnumerator - ldenominator
    E = np.where(E.cpu().data.numpy() == 1)[0]
    likelihoods = likelihoods[E]
    neg_likelihood = - torch.sum(likelihoods)

    return neg_likelihood


def train_model(gated_network, betas_network, risk_set, x_train, e_train, t_train, risk_set_valid, x_valid, e_valid,
                t_valid,
                optimizer, n_epochs, x_test, e_test, t_test, risk_set_test):
    # Initialize Metrics
    c_index_soft = []
    c_index_hard = []
    train_loss = []
    valid_loss = []
    test_loss = []
    test_c_index_soft = []
    test_c_index_hard = []
    diff = 1e-4
    prev_loss_train = 0
    prev_loss_valid = 0
    bad_cnt = 0
    start = time.time()
    for epoch in range(n_epochs):
        gated_network.train()
        betas_network.train()
        optimizer.zero_grad()
        gated_outputs = gated_network(x_train)
        lsoftmax = nn.LogSoftmax(dim=1)(gated_outputs)
        betas_output = betas_network(x_train)
        ci_train_soft, ci_train_hard = get_concordance_index(betas_output, gated_outputs, t_train, e_train,
                                                             bootstrap=False)
        c_index_soft.append(ci_train_soft)
        c_index_hard.append(ci_train_hard)
        loss = elbo(betas_output, gated_outputs, e_train, risk_set) + (betas_network[0].weight ** 2).sum()
        loss.backward()
        optimizer.step()
        my_loss = loss.cpu().data.numpy()
        train_loss.append(my_loss)
        if abs(my_loss - prev_loss_train) < diff:
            break
        prev_loss_train = my_loss
        torch.cuda.empty_cache()
        ################################################# Validation #######################################################
        gated_network.eval()
        betas_network.eval()
        gated_outputs_valid = gated_network(x_valid)
        lsoftmax_valid = nn.LogSoftmax(dim=1)(gated_outputs_valid)
        betas_output_valid = betas_network(x_valid)
        loss_valid = elbo(betas_output_valid, gated_outputs_valid, e_valid, risk_set_valid)
        my_loss_valid = loss_valid.cpu().data.numpy()
        valid_loss.append(my_loss_valid)
        if my_loss_valid - prev_loss_valid > diff:
            bad_cnt += 1
            if bad_cnt > 2:
                break
        else:
            bad_cnt = 0
        prev_loss_valid = my_loss_valid
        torch.cuda.empty_cache()

        ################################################# Test #############################################################
    gated_network.eval()
    betas_network.eval()
    gated_outputs_test = gated_network(x_test)
    lsoftmax_test = nn.LogSoftmax(dim=1)(gated_outputs_test)
    betas_output_test = betas_network(x_test)
    ci_test_soft, ci_test_hard = get_concordance_index(betas_output_test, gated_outputs_test, t_test, e_test,
                                                       bootstrap=250)
    test_c_index_soft.append(ci_test_soft)
    test_c_index_hard.append(ci_test_hard)
    loss_test = elbo(betas_output_test, gated_outputs_test, e_test, risk_set_test)
    my_loss_test = loss_test.cpu().data.numpy()
    test_loss.append(my_loss_test)
    torch.cuda.empty_cache()

    print('Finished training with %d epochs in %0.2fs' % (epoch + 1, time.time() - start))
    metrics = {}
    metrics['train_loss'] = train_loss
    metrics['valid_loss'] = valid_loss
    metrics['c-index-soft'] = c_index_soft
    metrics['c-index-hard'] = c_index_hard
    metrics['test_loss'] = test_loss
    metrics['c-index-test-soft'] = test_c_index_soft
    metrics['c-index-test-hard'] = test_c_index_hard

    return metrics


def run_experiment(params):
    linear_model, learning_rate, layers_size, seed,data_dict= params
    layers_size = layers_size[:]
    layers_size += [linear_model]
    torch.manual_seed(seed)
    n_in = data_dict["x_train"].shape[1]
    betas_network = nn.Sequential(nn.Linear(n_in, linear_model, bias=False) )
    layers = []
    for i in range(len(layers_size)-2):
        layers.append(nn.Linear(layers_size[i],layers_size[i+1],bias=False ))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(layers_size[-2], layers_size[-1], bias=False))
    gated_network = nn.Sequential(*layers)
    optimizer = torch.optim.Adam(list(gated_network.parameters()) + list(betas_network.parameters()), lr=learning_rate)
    n_epochs = 4000
    metrics = train_model(gated_network, betas_network,
                          data_dict["risk_set"], data_dict["x_train"], data_dict["e_train"], data_dict["t_train"],
                        data_dict["risk_set_valid"], data_dict["x_valid"], data_dict["e_valid"], data_dict["t_valid"],
                          optimizer, n_epochs,
                          data_dict["x_test"],data_dict["e_test"],data_dict["t_test"],data_dict["risk_set_test"])
    return metrics


def mix_of_experts(name):

    filename = filename_dict[name]
    raw_data = pd.read_csv(os.path.join(DATA_DIR, filename))
    formatted_x, formatted_y = sksurv_data_formatting(raw_data)

    x_train, x_test, y_train, y_test = train_test_split(formatted_x, formatted_y,
                                                        test_size=0.25, random_state=RANDOM_STATE)

    x_train, e_train, t_train, x_valid, e_valid, t_valid, x_test, e_test, t_test = scale_data_to_torch(x_train, y_train,
                                                                                                       x_test, y_test,
                                                                                                       x_test, y_test)
    print("Dataset loaded and scaled")

    risk_set, risk_set_valid, risk_set_test = compute_risk_set(t_train, t_valid, t_test)
    print("Risk set computed")

    data_dict = {"x_train": x_train, "e_train": e_train, "t_train": t_train,
                 "x_valid": x_valid, "e_valid": e_valid, "t_valid": t_valid,
                 "x_test": x_test, "e_test": e_test, "t_test": t_test,
                 "risk_set": risk_set, "risk_set_valid": risk_set_valid, "risk_set_test": risk_set_test
                 }

    n_in = x_train.shape[1]
    linear_models = [2, 5, 10, 12]
    learning_rates = [1e-4, 1e-3]
    layer_sizes = [[n_in], [n_in, n_in], [n_in, n_in, n_in], [n_in, 20, 15]]
    data = [data_dict]
    hyperparams = [(linear_model, learning_rate, layer_size, seed, d) for layer_size in layer_sizes for learning_rate in
                   learning_rates
                   for linear_model in linear_models for seed in range(3) for d in data]
    print("Hyperparams initialized")

    p = Pool(50)
    print("Pool created")

    output = p.map(run_experiment, hyperparams)
    p.close()
    p.join()
    print("Models trained. Writing to file")
    filename = name + "_results.pkl"
    f = open(filename, "wb")
    pkl.dump(output, f)
    f.flush()
    f.close()
    print(name, "done")
    print("")

mix_of_experts("metabric")
