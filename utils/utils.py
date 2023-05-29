import math
import sys
import os
import copy
import torch
import numpy as np
import pandas as pd
from copy import deepcopy as dcopy
from torch import nn
from torch.utils.data import DataLoader
from torchtext.data import Batch
import pingouin as pg

import torch.nn.functional as F
from utils.arguments import (adult_args, cifar100_args, cifar_cnn_args,
                             covid_tweet_args, fraud_args, hft_args,
                             mnist_args, mr_args, sst_args, electricity_args,pathmnist_args)

method_name_map = {
    'fifl':'Ours',
    'fedavg':'FedAvg',
    'fedavg_rr':'RR',
    'fedavg_eu':'EU',
    'fedavg_dw':'DW',
    'fedavg_ci':'CIE',
    'standalone':'Standalone',
    'qffl': "q-FFL",
    'cffl': "CFFL",
    # 'full_shapley_cosine':'Full_Shapley_Cosine',
}

method_name_map_sparse = {
    'fifl':'Ours',
    'standalone':'Standalone',
}

metrics = ['{}_best'.format(short_name) for short_name in method_name_map]
metrics += ['{}_best_test_loss'.format(short_name) for short_name in method_name_map]
metrics += ['{}_best_train_loss'.format(short_name) for short_name in method_name_map]

metrics += ['{}_mean'.format(short_name) for short_name in method_name_map]
metrics += ['{}_mean_test_loss'.format(short_name) for short_name in method_name_map]
metrics += ['{}_mean_train_loss'.format(short_name) for short_name in method_name_map]
metrics += ['standalone_vs_{}'.format(short_name) for short_name in method_name_map if short_name != 'standalone']
metrics += ['{}_rs_vs_test_accs'.format(short_name) for short_name in method_name_map if short_name != 'standalone']
metrics += ['{}_rs_vs_test_losses'.format(short_name) for short_name in method_name_map if short_name != 'standalone']
metrics += ['{}_rs_vs_train_losses'.format(short_name) for short_name in method_name_map if short_name != 'standalone']


# targeted_acc_keys = ['_{}'.format(short_name) for short_name in method_name_map if short_name != 'standalone']

'''
metrics  = ['standalone_best', 'FIFL_best', 'fedavg_best', 'qffl_best', 'fsg_best', 
'krum_best', 'signsgd_best', 'median_best',
'standalone_vs_FIFL', 'standalone_vs_fedavg', 'standalone_vs_signsgd', 'standalone_vs_qffl',
 'standalone_vs_fsg','standalone_vs_krum', 'standalone_vs_median']
'''

def averge_models(models, device=None):
    final_model = copy.deepcopy(models[0])
    if device:
        models = [model.to(device) for model in models]
        final_model = final_model.to(device)

    averaged_parameters = aggregate_gradient_updates([list(model.parameters()) for model in models], mode='mean')
    
    for param, avg_param in zip(final_model.parameters(), averaged_parameters):
        param.data = avg_param.data
    return final_model

def scale_grad(grad, scale):
    for param in grad:
        param.data = param.data * scale
    return grad

def translate_grad(grad, distance):
    for param in grad:
        param.data += distance
    return grad


def compute_grad_update(old_model, new_model, device=None):
    # maybe later to implement on selected layers/parameters
    if device:
        old_model, new_model = old_model.to(device), new_model.to(device)
    return [(new_param.data - old_param.data) for old_param, new_param in zip(old_model.parameters(), new_model.parameters())]

def compute_distance_percentage(model, ref_model):
    percents, dists  = [], []
    for layer, ref_layer in zip(model.parameters(), ref_model.parameters()):
        dist = torch.norm(layer - ref_layer)
        dists.append(dist.item())
        percents.append( (torch.div(dist, torch.norm(ref_layer))).item() )

    return percents, dists

def add_gradient_updates(grad_update_1, grad_update_2, weight = 1.0):
    assert len(grad_update_1) == len(
        grad_update_2), "Lengths of the two grad_updates not equal"
    
    for param_1, param_2 in zip(grad_update_1, grad_update_2):
        param_1.data += param_2.data * weight

def aggregate_gradient_updates(grad_updates, R, device=None, mode='sum', credits=None, shard_sizes=None):
    if grad_updates:
        len_first = len(grad_updates[0])
        assert all(len(i) == len_first for i in grad_updates), "Different shapes of parameters. Cannot aggregate."
    else:
        return

    grad_updates_ = [copy.deepcopy(grad_update) for i, grad_update in enumerate(grad_updates) if i in R]

    if device:
        for i, grad_update in enumerate(grad_updates_):
            grad_updates_[i] = [param.to(device) for param in grad_update]

    if credits is not None:
        credits = [credit for i, credit in enumerate(credits) if i in R]
    if shard_sizes is not None:
        shard_sizes = [shard_size for i,shard_size in enumerate(shard_sizes) if i in R]

    aggregated_gradient_updates = []
    if mode=='mean':
        # default mean is FL-avg: weighted avg according to nk/n
        if shard_sizes is None:
            shard_sizes = torch.ones(len(grad_updates))

        for i, (grad_update, shard_size) in enumerate(zip(grad_updates_, shard_sizes)):
            grad_updates_[i] = [(shard_size * update) for update in grad_update]
        for i in range(len(grad_updates_[0])):
            aggregated_gradient_updates.append(torch.stack(
                [grad_update[i] for grad_update in grad_updates_]).mean(dim=0))

    elif mode =='sum':
        for i in range(len(grad_updates_[0])):
            aggregated_gradient_updates.append(torch.stack(
                [grad_update[i] for grad_update in grad_updates_]).sum(dim=0))

    elif mode == 'credit-sum':
        # first changes the grad_updates altogether
        for i, (grad_update, credit) in enumerate(zip(grad_updates_, credits)):
            grad_updates_[i] = [(credit * update) for update in grad_update]

        # then compute the credit weight sum
        for i in range(len(grad_updates_[0])):
            aggregated_gradient_updates.append(torch.stack(
                [grad_update[i] for grad_update in grad_updates_]).sum(dim=0))

    return aggregated_gradient_updates

def add_update_to_model(model, update, weight=1.0, device=None):
    if not update: return model
    if device:
        model = model.to(device)
        update = [param.to(device) for param in update]
            
    for param_model, param_update in zip(model.parameters(), update):
        param_model.data += weight * param_update.data
    return model

def compare_models(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False # two models have different weights
    return True


def sign(grad):
    return [torch.sign(update) for update in grad]

def flatten(grad_update):
    return torch.cat([update.data.view(-1) for update in grad_update])

def unflatten(flattened, normal_shape):
    grad_update = []
    for param in normal_shape:
        n_params = len(param.view(-1))
        grad_update.append(torch.as_tensor(flattened[:n_params]).reshape(param.size())  )
        flattened = flattened[n_params:]

    return grad_update

def l2norm(grad):
    return torch.sqrt(torch.sum(torch.pow(flatten(grad), 2)))

def cosine_similarity(grad1, grad2, normalized=False):
    """
    Input: two sets of gradients of the same shape
    Output range: [-1, 1]
    """

    cos_sim = F.cosine_similarity(flatten(grad1), flatten(grad2), 0, 1e-10) 
    if normalized:
        return (cos_sim + 1) / 2.0
    else:
        return cos_sim

def cosine_similarity_modified(coalition_grad, coalition_grad_majority, grad_all, grad_all_majority, normalized=False, Lambda=0):
    sign_cossim = F.cosine_similarity(coalition_grad_majority, grad_all_majority, 0, 1e-10) 
    modu_cossim = F.cosine_similarity(coalition_grad, grad_all, 0, 1e-10)

    return Lambda * sign_cossim  + (1 - Lambda) * modu_cossim


from math import pi
def angular_similarity(grad1, grad2):
    return 1 - torch.div(torch.acovs(cosine_similarity(grad1, grad2)), pi)

def evaluate(model, eval_loader, device, data_prepper=None, loss_fn=None, verbose=False, label_flip=None):
    model.eval()
    model = model.to(device)
    correct = 0
    total = 0
    loss = 0

    target_correct = 0
    target_total = 0
    attack_success = 0

    with torch.no_grad():
        for i, batch in enumerate(eval_loader):

            if isinstance(batch, Batch):
                batch_data, batch_target = batch.text, batch.label
                # batch_data.data.t_(), batch_target.data.sub_(1)  # batch first, index align
                batch_data = batch_data.permute(1, 0)
            else:
                batch_data, batch_target = batch[0], batch[1]

            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            outputs = model(batch_data)

            if loss_fn:
                loss += loss_fn(outputs, batch_target)
            else:
                loss = None
            
            if isinstance(loss_fn,torch.nn.MSELoss):
                if hasattr(data_prepper, "mean_y"):
                    mean_y, std_y = data_prepper.mean_y, data_prepper.std_y
                    batch_target_tmp = batch_target * std_y + mean_y
                    outputs_tmp = outputs * std_y + mean_y
                    correct += torch.sum(torch.abs((batch_target_tmp - outputs_tmp) / batch_target_tmp))
                else:
                    correct += torch.sum(torch.abs((batch_target - outputs) / batch_target))
            else:
                correct += (torch.max(outputs, 1)[1].view(batch_target.size()).data == batch_target.data).sum()
            total += len(batch_target)

            if label_flip:
                classes = label_flip.split('-')
                from_class, to_class = int(classes[0]), int(classes[1])
                indices = batch_target==from_class
                target_total += sum(indices)

                target_class_outputs, target_class_labels = outputs[indices], batch_target[indices]
                target_correct += (torch.max(target_class_outputs, 1)[1].view(target_class_labels.size()).data == target_class_labels.data).sum()

                attack_success += (torch.max(target_class_outputs, 1)[1].view(target_class_labels.size()).data == to_class).sum()

        accuracy =  correct.float() / total
        if loss_fn:
            loss /= total

        if label_flip:
            target_accuracy = target_correct.float() / target_total
            attack_success_rate =  attack_success.float() / target_total.item()
            # print("For attack class: ", label_flip)
            # print("Correct:", target_correct.item(), "Total: ", target_total.item(), "Accuracy: ", target_accuracy.item())
            # print("Successful attacks:", attack_success.item(), "Attack success rate:",)
            return loss, target_accuracy, attack_success_rate
    
    if verbose:
        if isinstance(loss_fn,torch.nn.MSELoss):
            print("Loss: {:.6f}. MAPE: {:.4%}.".format(loss, accuracy))
        else:
            print("Loss: {:.6f}. Accuracy: {:.4%}.".format(loss, accuracy))
    return loss, accuracy

from torchtext.data import Batch
def train_model(model, loader, loss_fn, optimizer, device, E=1, **kwargs):

    model.train()
    for e in range(E):
        # running local epochs
        for _, batch in enumerate(loader):
            if isinstance(batch, Batch):
                data, label = batch.text, batch.label
                data = data.permute(1, 0)
                # data.data.t_(), label.data.sub_(1)  # batch first, index align
            else:
                data, label = batch[0], batch[1]

        data, label = data.to(device), label.to(device)

        # for data, label in loader:
            # data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        pred = model(data)
        # print(pred, label)
        loss_fn(pred, label).backward()

        # specifically used for pruning to get a subnetwork
        if 'mask' in kwargs:
            # set the grad for the parameters in the mask to be zero
            for layer_zero_mask, layer in zip(kwargs['mask'], model.parameters()):				
                if torch.all(layer_zero_mask == 0):
                    print("zero mask is empty")
                    print("original grad is:", layer.grad)

                layer.grad.data = torch.mul(layer.grad.data, layer_zero_mask.data)
                # copy = layer.grad.view(-1)

                # copy.data = torch.mul(copy.data, layer_zero_mask.data)
                # [layer_zero_mask] = 0
                
                # layer.grad.data = copy.reshape(layer.grad.size())
                if len(layer_zero_mask) == 0:
                    print("zero mask is empty")
                    print("masked grad is:", layer.grad)

        optimizer.step()

    if 'scheduler' in kwargs: kwargs['scheduler'].step()
    
    return model


from itertools import chain, combinations
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

from math import factorial as f
def choose(n, r):
    return f(n) // f(r) // f(n-r)

def clip_gradient_update(grad_update, grad_clip):
    """
    Return a copy of clipped grad update 

    """
    return [torch.clamp(param.data, min=-grad_clip, max=grad_clip) for param in grad_update]



def mask_grad_update_by_order(grad_update, mask_order=None, mask_percentile=None, mode='all'):

    if mode == 'all':
        # mask all but the largest <mask_order> updates (by magnitude) to zero
        all_update_mod = torch.cat([update.data.view(-1).abs()
                                    for update in grad_update])
        if not mask_order and mask_percentile is not None:
            mask_order = int(len(all_update_mod) * mask_percentile)
        
        if mask_order == 0:
            return mask_grad_update_by_magnitude(grad_update, float('inf'))
        else:
            topk, indices = torch.topk(all_update_mod, mask_order)
            return mask_grad_update_by_magnitude(grad_update, topk[-1])

    elif mode == 'layer': # layer wise largest-values criterion
        grad_update = copy.deepcopy(grad_update)

        mask_percentile = max(0, mask_percentile)
        for i, layer in enumerate(grad_update):
            layer_mod = layer.data.view(-1).abs()
            if mask_percentile is not None:
                mask_order = math.ceil(len(layer_mod) * mask_percentile)

            if mask_order == 0:
                grad_update[i].data = torch.zeros(layer.data.shape, device=layer.device)
            else:
                topk, indices = torch.topk(layer_mod, min(mask_order, len(layer_mod)-1))
                if len(topk) > 0:																																										
                    grad_update[i].data[layer.data.abs() < topk[-1]] = 0
        return grad_update

def mask_grad_update_by_magnitude(grad_update, mask_constant):

    # mask all but the updates with larger magnitude than <mask_constant> to zero
    # print('Masking all gradient updates with magnitude smaller than ', mask_constant)
    grad_update = copy.deepcopy(grad_update)
    for i, update in enumerate(grad_update):
        grad_update[i].data[update.data.abs() < mask_constant] = 0
    return grad_update

def mask_grad_update_by_indices(grad_update, indices=None):
    """
    Mask the grad.data to be 0, if the position is not in the list of indices
    If indicies is empty, mask nothing.
    
    Arguments: 
    grad_update: as in the shape of the model parameters. A list of tensors.
    indices: a tensor of integers, corresponding to the specific individual scalar values in the grad_update, 
    as if the entire grad_update is flattened.

    e.g. 
    grad_update = [[1, 2, 3], [3, 2, 1]]
    indices = [4, 5]
    returning masked grad_update = [[0, 0, 0], [0, 2, 1]]
    """

    grad_update = copy.deepcopy(grad_update)
    if indices is None or len(indices)==0: return grad_update

    #flatten and unflatten
    flattened = torch.cat([update.data.view(-1) for update in grad_update])	
    masked = torch.zeros_like(torch.arange(len(flattened)), device=flattened.device).float()
    masked.data[indices] = flattened.data[indices]

    pointer = 0
    for m, update in enumerate(grad_update):
        size_of_update = torch.prod(torch.tensor(update.shape)).long()
        grad_update[m].data = masked[pointer: pointer + size_of_update].reshape(update.shape)
        pointer += size_of_update
    return grad_update


import numpy as np
np.random.seed(1111)


def random_split(sample_indices, m_bins, equal=True):
    sample_indices = np.asarray(sample_indices)
    if equal:
        indices_list = np.array_split(sample_indices, m_bins)
    else:
        split_points = np.random.choice(
            n_samples - 2, m_bins - 1, replace=False) + 1
        split_points.sort()
        indices_list = np.split(sample_indices, split_points)

    return indices_list

import random
from itertools import permutations

def compute_shapley(grad_updates, federated_model, test_loader, device, Max_num_sequences=50):
    num_participants = len(grad_updates)
    all_sequences = list(permutations(range(num_participants)))
    if len(all_sequences) > Max_num_sequences:
        random.shuffle(all_sequences)
        all_sequences = all_sequences[:Max_num_sequences]

    test_loss_prev, test_acc_prev = evaluate(federated_model, test_loader, device, verbose=False)
    prev_contribution = test_acc_prev.data
    
    marginal_contributions = torch.zeros((num_participants))
    for sequence in all_sequences:
        running_model = copy.deepcopy(federated_model)
        curr_contributions = []
        for participant_id in sequence:
            running_model = add_update_to_model(running_model, grad_updates[participant_id])
            test_loss, test_acc = evaluate(running_model, test_loader, device, verbose=False)
            contribution = test_acc.data

            if not curr_contributions:
                marginal_contributions[participant_id] +=  contribution - prev_contribution
            else:
                marginal_contributions[participant_id] +=  contribution - curr_contributions[-1]

            curr_contributions.append(contribution)

    return marginal_contributions / len(all_sequences)

class Logger(object):
    def __init__(self, file_p):
        self.terminal = sys.stdout
        self.log = file_p
    def write(self, message):
        self.terminal.write(message)
        if (not self.log.closed):
            self.log.write(message)


    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def softmax(x, beta = 1.0):

    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(beta * (x - np.max(x)))
    return e_x / e_x.sum()

def kldivergence(dis_a, dis_b):
    """Compute KL divergence for two probability distribution"""
    try:
        log_a = np.log(dis_a)
        log_b = np.log(dis_b)
        return np.sum(np.multiply(dis_a,log_a-log_b))
    except:
        return np.inf

def distance_metric(dis_a, dis_b, dtype = "inf"):
    """Compute different distance metrics for two vector"""
    try:
        if dtype == "inf":
            return np.linalg.norm(dis_a-dis_b,np.inf)
        elif dtype == "l1":
            return np.linalg.norm(dis_a-dis_b,1)
        elif dtype == "l2":
            return np.linalg.norm(dis_a-dis_b,2)
        elif dtype == "kl":
            return kldivergence(dis_a, dis_b)
        else:
            return None
    except:
        return np.inf

def reverse_rank_list(rank_list):
    new_list = dcopy(rank_list)
    order = np.argsort(new_list)
    list_len = len(new_list)
    for i in range(list_len):
        if i >= list_len - 1 - i:
            break
        tmp = new_list[order[i]]
        new_list[order[i]] = new_list[order[list_len - 1 - i]]
        new_list[order[list_len - 1 - i]] = tmp
    return new_list


# utility functions for covid_tweet dataset

# split the dataset into multi-time-interval dataset
def split_multi_time_interval_dataset(data_input, dataset_days=60, data_step_size=5):
    data = data_input.copy()

    date_series = np.unique(data['date'])
    total_day_cnt = len(date_series)

    multi_time_dataset = []
    for d in range(0,total_day_cnt-dataset_days, data_step_size):
        min_date = date_series[d]
        max_date = min_date + np.timedelta64(dataset_days,'D')
        time_interval_dataset = data[(data['date'] < max_date) 
                                & (data['date'] >= min_date)] \
                                .reset_index(drop=True)
        multi_time_dataset.append(time_interval_dataset)
    return multi_time_dataset
    
# train, valid split
def split_train_valid(data_input, valid_ratio=0.3):

    data = data_input.copy()

    valid_set = data.sample(frac=valid_ratio)
    train_set = data.drop(valid_set.index)

    valid_set, train_set = valid_set.reset_index(drop=True), train_set.reset_index(drop=True)

    valid_set_aggregated = valid_set.groupby(['date']).mean()
    return valid_set_aggregated, train_set


# transform table data to X and y samples
def transform_table_to_Xy(data_input, training_days=30, std=np.array([1]), mean=np.array([0])):
    data = data_input.copy()
    X = []
    y = []
    for k in range(len(data) - training_days - 1):
        X.append(np.expand_dims(data.iloc[k:(k+training_days),:]
                            [['valence_intensity', 'fear_intensity',	'anger_intensity',
                            'happiness_intensity', 'sadness_intensity']].values, axis=0))
        y.append(data.iloc[(k+training_days),:][['valence_intensity']].values)
    X,y = (np.concatenate(X,axis=0) - mean)/std, np.concatenate(y,axis=0)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# randomly split the data into N clients
def split_data_N_clients(data_input, N = 30, training_days = 30):

    data = data_input.copy()

    n_patition_data = []
    for p in range(N):
        if p == N-1:
            sampled_client_data = data.copy()
        else:
            sampled_client_data = data.sample(frac=1.0/N)
            data = data.drop(sampled_client_data.index)
        sampled_client_data = sampled_client_data.groupby(['date']).mean()
        X,y = transform_table_to_Xy(sampled_client_data, training_days)
        n_patition_data.append((X,y))
    return n_patition_data

# split the last n entry as the validation data we want to predict
def split_n_day_valid(data_input, valid_days=5):
    data = copy.deepcopy(data_input)

    train_set = []
    valid_set = []

    for i in range(len(data)):
        train_data = (data[i][0][:-valid_days], data[i][1][:-valid_days])
        valid_data = (data[i][0][-valid_days:], data[i][1][-valid_days:])
        train_set.append(train_data)
        valid_set.append(valid_data)
    return train_set, valid_set

def aggregate_date_data(data_input):
    return data_input.groupby(['date']).mean()


#-------------------------High Frequency Trading Data Processing---------------------------------------


def split_dataset_to_streams_hft(ori_data, stream_steps, task_number=0):
    stream_data_size = len(ori_data) // stream_steps
    stream_datasets = [(ori_data[(i*stream_data_size):((i+1)*stream_data_size), 0:144]
                        , ori_data[(i*stream_data_size):((i+1)*stream_data_size), (144+task_number)]-1)
                       for i in range(stream_steps)]
                       
    # convert to torch tensor
    stream_datasets_torch = [(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)) for (X,y) in stream_datasets]

    return stream_datasets_torch


def split_dataset_to_N_clients_hft(dataset, N = 30):
    data = copy.deepcopy(dataset)

    data_size = len(data[0])

    # shuffle the data
    random_index = np.random.choice(range(data_size), data_size,replace=False)

    split_indexs = np.array_split(random_index, N)

    return split_indexs


def split_datasets_to_stream_mnist_cifar(dataset, stream_step):
    randon_indices = np.random.choice(np.arange(len(dataset)), len(dataset), replace=False)
    stream_indices = np.array_split(randon_indices, stream_step)
    stream_datasets = []
    for indice in stream_indices:
        dataset_now = (dataset.data[indice], dataset.targets[indice])
        stream_datasets.append(dataset_now)
    return stream_datasets

def split_indices(indices, num_part):
    randon_indices = np.random.choice(indices, len(indices), replace=False)
    splited_indices = np.array_split(randon_indices, num_part)
    return splited_indices


def performance_summary(args, path):
    """
    Function to compute the performance of different fairness framework
    """
    # compute the average accuracy/loss/modeldiff for each clients in stage 2
    clients_perf_data = pd.read_csv(os.path.join(path,"valid.csv"))
    clients_modeldiff_data = pd.read_csv(os.path.join(path,"model_diff.csv"))
    clients_staleness_data = pd.read_csv(os.path.join(path,"staleness.csv"))
    
    clients_avg_acc = []
    clients_avg_loss = []
    clients_model_diff = []
    clients_staleness = []
    for i in range(args['n_participants']):
        clients_avg_acc.append(np.mean(clients_perf_data['{}_accu'.format(i)]))
        clients_avg_loss.append(np.mean(clients_perf_data['{}_loss'.format(i)]))
        clients_model_diff.append(np.mean(clients_modeldiff_data['{}_model_diff_l2norm'.format(i)]))
        try:
            clients_staleness.append(len(clients_staleness_data)/np.sum(clients_staleness_data['client {}'.format(i)]))
        except:
            clients_staleness.append(len(clients_staleness_data)/np.sum(clients_staleness_data["{}".format(i)]))
    
    args['clients_avg_acc'] = clients_avg_acc
    args['clients_avg_loss'] = clients_avg_loss
    args['clients_model_diff'] = clients_model_diff
    args['clients_staleness'] = clients_staleness
    
    # get the noise level/shapley value/selection prob of different clients
    fraction_of_noise_label = args["fraction_of_noise_label"]
    try:
        shapley_value = args["valid_criteria_score"]
    except:
        shapley_value = args["data_value_score"]
    selection_prob = args["selection_probability"]

    # correlation between noise-shapley value
    noi_shapley = np.corrcoef(fraction_of_noise_label, shapley_value)
    noi_acc = np.corrcoef(fraction_of_noise_label, clients_avg_acc)
    noi_loss = np.corrcoef(fraction_of_noise_label, clients_avg_loss)
    noi_model_diff = np.corrcoef(fraction_of_noise_label, clients_model_diff)

    shapley_staleness = np.corrcoef(clients_staleness, shapley_value)
    shapley_modeldiff = np.corrcoef(clients_model_diff, shapley_value)
    shapley_loss = np.corrcoef(clients_avg_loss, shapley_value)
    shapley_acc = np.corrcoef(clients_avg_acc, shapley_value)

    args['cor_noi_shapley'] = noi_shapley[0,1]
    args['cor_noi_acc'] = noi_acc[0,1]
    args['cor_noi_loss'] = noi_loss[0,1]
    args['cor_noi_model_diff'] = noi_model_diff[0,1]
    
    args['cor_shapley_staleness'] = shapley_staleness[0,1]
    args['cor_shapley_modeldiff'] = shapley_modeldiff[0,1]
    args['cor_shapley_loss'] = shapley_loss[0,1]
    args['cor_shapley_acc'] = shapley_acc[0,1]
    
    return args

def performance_summary_standalone(args, path):
    """
    Function to compute the performance of different fairness framework
    """
    # compute the average accuracy/loss/modeldiff for each clients in stage 2
    clients_perf_data = pd.read_csv(os.path.join(path,"valid.csv"))

    clients_avg_acc = []
    clients_avg_loss = []

    for i in range(args['n_participants']):
        clients_avg_acc.append(np.mean(clients_perf_data['{}_accu'.format(i)]))
        clients_avg_loss.append(np.mean(clients_perf_data['{}_loss'.format(i)]))

    args['clients_avg_acc'] = clients_avg_acc
    args['clients_avg_loss'] = clients_avg_loss
    
    # get the noise level/shapley value/selection prob of different clients
    fraction_of_noise_label = args["fraction_of_noise_label"]

    # correlation between noise-shapley value
    noi_acc = np.corrcoef(fraction_of_noise_label, clients_avg_acc)
    noi_loss = np.corrcoef(fraction_of_noise_label, clients_avg_loss)

    args['cor_noi_acc'] = noi_acc[0,1]
    args['cor_noi_loss'] = noi_loss[0,1]
    
    return args


def performance_summary_other(args, path):
    """
    Function to compute the performance of different fairness framework
    """
    # compute the average accuracy/loss/modeldiff for each clients in stage 2
    clients_perf_data = pd.read_csv(os.path.join(path,"valid.csv"))
    clients_modeldiff_data = pd.read_csv(os.path.join(path,"model_diff.csv"))
    
    clients_avg_acc = []
    clients_avg_loss = []
    clients_model_diff = []
    for i in range(args['n_participants']):
        clients_avg_acc.append(np.mean(clients_perf_data['{}_accu'.format(i)]))
        clients_avg_loss.append(np.mean(clients_perf_data['{}_loss'.format(i)]))
        clients_model_diff.append(np.mean(clients_modeldiff_data['{}_model_diff_l2norm'.format(i)]))
    
    args['clients_avg_acc'] = clients_avg_acc
    args['clients_avg_loss'] = clients_avg_loss
    args['clients_model_diff'] = clients_model_diff
    
    # get the noise level/shapley value/selection prob of different clients
    fraction_of_noise_label = args["fraction_of_noise_label"]

    # correlation between noise-shapley value
    noi_acc = np.corrcoef(fraction_of_noise_label, clients_avg_acc)
    noi_loss = np.corrcoef(fraction_of_noise_label, clients_avg_loss)
    noi_model_diff = np.corrcoef(fraction_of_noise_label, clients_model_diff)


    args['cor_noi_acc'] = noi_acc[0,1]
    args['cor_noi_loss'] = noi_loss[0,1]
    args['cor_noi_model_diff'] = noi_model_diff[0,1]
    
    return args

def performance_summary_rl(args, path):
    """
    Function to compute the performance of Fed RL
    """
    # compute the average accuracy/loss/modeldiff for each clients in stage 2
    clients_perf_data = pd.read_csv(os.path.join(path,"valid.csv"))
    clients_modeldiff_data = pd.read_csv(os.path.join(path,"model_diff.csv"))
    
    clients_avg_score = []
    clients_avg_step = []
    clients_model_diff = []
    for i in range(args['n_participants']):
        clients_avg_score.append(np.mean(clients_perf_data['{}_score'.format(i)]))
        clients_avg_step.append(np.mean(clients_perf_data['{}_step'.format(i)]))
        clients_model_diff.append(np.mean(clients_modeldiff_data['{}_model_diff_l2norm'.format(i)]))
    
    args['clients_avg_score'] = clients_avg_score
    args['clients_avg_step'] = clients_avg_step
    args['clients_model_diff'] = clients_model_diff
    
    # get the noise level of different clients
    fraction_of_noise_label = args["noise_level"]

    # correlation between noise value and performance
    noi_score = np.corrcoef(fraction_of_noise_label, clients_avg_score)
    noi_step = np.corrcoef(fraction_of_noise_label, clients_avg_step)
    noi_model_diff = np.corrcoef(fraction_of_noise_label, clients_model_diff)

    args['cor_noi_score'] = noi_score[0,1]
    args['cor_noi_step'] = noi_step[0,1]
    args['cor_noi_model_diff'] = noi_model_diff[0,1]

    # shapley value related stats
    try:
        clients_staleness_data = pd.read_csv(os.path.join(path,"staleness.csv"))
        clients_staleness = []
        for i in range(args['n_participants']):
            try:
                clients_staleness.append(len(clients_staleness_data)/np.sum(clients_staleness_data['client {}'.format(i)]))
            except:
                clients_staleness.append(len(clients_staleness_data)/np.sum(clients_staleness_data["{}".format(i)]))        
        try:
            shapley_value = args["valid_criteria_score"]
        except:
            shapley_value = args["data_value_score"]        
        
        selection_prob = args["selection_probability"]
        args['clients_staleness'] = clients_staleness
        shapley_staleness = np.corrcoef(clients_staleness, shapley_value)
        shapley_modeldiff = np.corrcoef(clients_model_diff, shapley_value)
        shapley_score = np.corrcoef(clients_avg_score, shapley_value)
        shapley_step = np.corrcoef(clients_avg_step, shapley_value)
        noi_shapley = np.corrcoef(fraction_of_noise_label, shapley_value)
        args['cor_shapley_staleness'] = shapley_staleness[0,1]
        args['cor_noi_shapley'] = noi_shapley[0,1]
        args['cor_shapley_modeldiff'] = shapley_modeldiff[0,1]
        args['cor_shapley_score'] = shapley_score[0,1]
        args['cor_shapley_step'] = shapley_step[0,1]
    except Exception as e:
        print(e)
    return args


# ----------------- RL utility function -------------
def synchronized_epsilon(agents):
    epsilons = [tmp.epsilon for tmp in agents]
    return min(epsilons)


def cosine_gradient_linear(data_value_score, server_model, server_gradient, device_gradients, weights, n_participant, t, device):
	data_value_round = torch.zeros(n_participant,device=device)
	for i in range(n_participant):
		set_minus_i = set(list(range(n_participant)))
		set_minus_i.remove(i)
		set_minus_i = list(set_minus_i)

		for picked_s in range(n_participant):# iterate size from {0,1,...,n-1}

			sampled_S = random.sample(set_minus_i, picked_s)

			coalition_grad = [torch.zeros(param.shape).to(device) for param in server_model.parameters()]

			for j in sampled_S:
				add_gradient_updates(coalition_grad, device_gradients[j], weights[j])
			before = cosine_similarity(server_gradient, coalition_grad)

			add_gradient_updates(coalition_grad, device_gradients[i], weights[i])
			after = cosine_similarity(server_gradient, coalition_grad)

			data_value_round[i] += (after - before) / n_participant # N different sizes

	data_value_score = (t/(t+1)) * data_value_score + (1/(t+1)) * data_value_round
	return data_value_score, data_value_round

def perm_generator(seq):
	seen = set()
	length = len(seq)
	while True:
		perm = tuple(random.sample(seq, length))
		if perm not in seen:
			seen.add(perm)
			yield perm

def get_perm_sampled_sets(candidate_i, seq, num_set):
	perm_g = perm_generator(seq)
	perm_samples = [next(perm_g) for _ in range(num_set)]
	result_sets = []
	for i in range(num_set):
		tmp_perm = perm_samples[i]
		c_index = tmp_perm.index(candidate_i)
		result_sets.append(tmp_perm[:c_index])
	return result_sets

def multi_round_perm_sampling(data_value_score, pre_server_model, device_gradients, weights, n_participant, t, valid_loader, loss_fn, device, simulate_time = 5):
	data_value_round = torch.zeros(n_participant,device=device)
	simulate_time = simulate_time * n_participant
	for i in range(n_participant):
		sampled_sets = get_perm_sampled_sets(i,range(n_participant),simulate_time)
		for sampled_S in sampled_sets:
			sampled_S = list(sampled_S)
			sampled_S_plus_i = sampled_S + [i]

			new_weights_minus_i = weights/(sum(weights[sampled_S])+1e-30)
			new_weights_plus_i = weights/(sum(weights[sampled_S_plus_i])+1e-30)

			coalition_grad_minus_i = [torch.zeros(param.shape).to(device) for param in pre_server_model.parameters()]
			coalition_grad_plus_i = [torch.zeros(param.shape).to(device) for param in pre_server_model.parameters()]

			for j in sampled_S:
				add_gradient_updates(coalition_grad_minus_i, device_gradients[j], new_weights_minus_i[j])

			for j in sampled_S_plus_i:
				add_gradient_updates(coalition_grad_plus_i, device_gradients[j], new_weights_plus_i[j])

			coalition_model_minus_i = dcopy(pre_server_model)
			add_update_to_model(coalition_model_minus_i, coalition_grad_minus_i)
			loss_minus_i, accuracy_minus_i = evaluate(coalition_model_minus_i, valid_loader, loss_fn=loss_fn, device=device)

			coalition_model_plus_i = dcopy(pre_server_model)
			add_update_to_model(coalition_model_plus_i, coalition_grad_plus_i)
			loss_plus_i, accuracy_plus_i = evaluate(coalition_model_plus_i, valid_loader, loss_fn=loss_fn, device=device)

			data_value_round[i] += (accuracy_plus_i-accuracy_minus_i)/simulate_time
	data_value_round = data_value_round/(torch.sum(data_value_round)+1e-30)
	data_value_score = (t/(t+1)) * data_value_score + np.power(0.98,t+1)*(1/(t+1)) * data_value_round
	return data_value_score, data_value_round

def fed_loo(data_value_score, pre_server_model, device_gradients, weights, n_participant, t, valid_loader, loss_fn, device, accuracy):
	server_model_now = dcopy(pre_server_model)
	data_value_round = torch.zeros(n_participant,device=device)
	for i in range(n_participant):
		server_model_now.load_state_dict(pre_server_model.state_dict())

		server_gradient_loo =  [torch.zeros(param.shape).to(device) for param in pre_server_model.parameters()]

		set_minus_i_loo = list(set(list(range(n_participant))) - set([i]))

		new_weights = weights/sum(weights[set_minus_i_loo])

		for k in set_minus_i_loo:
			add_gradient_updates(server_gradient_loo, device_gradients[k], new_weights[k])

		add_update_to_model(server_model_now, server_gradient_loo)

		loss_loo, accuracy_loo = evaluate(server_model_now, valid_loader, loss_fn=loss_fn, verbose=True, device=device)
		data_value_round[i] = accuracy - accuracy_loo
	data_value_score = (t/(t+1)) * data_value_score + (1/(t+1)) * data_value_round
	return data_value_score, data_value_round


def data_valuation(data_value_score, server_model, pre_server_model, server_gradient, device_gradients, weights, n_participant, t, valid_loader, loss_fn, device, accuracy, dv_method):
	if dv_method == "cos_grad":
		return cosine_gradient_linear(data_value_score, server_model, server_gradient, device_gradients, weights, n_participant, t, device)
	elif dv_method == "fed_loo":
		return fed_loo(data_value_score, pre_server_model, device_gradients, weights, n_participant, t, valid_loader, loss_fn, device, accuracy)
	elif dv_method == "mr":
		return multi_round_perm_sampling(data_value_score, pre_server_model, device_gradients, weights, n_participant, t, valid_loader, loss_fn, device, simulate_time=5)
	else:
		raise Exception('No data valuation method: {}'.format(dv_method))

def hotelling_t2_ht(dv_round_recorder, alpha=0.5, shift=20, up_sample_num=None, verbose=True):
	if up_sample_num == None:
		up_sample_num = shift
	dv_round_values = pd.DataFrame(dv_round_recorder).values
	total_size, n_dim = dv_round_values.shape
	if total_size <= shift + 1:
		return False, 0

	sample_size = min(total_size - shift,up_sample_num)
	sampling_one = dcopy(dv_round_values[-sample_size:])
	sampling_two = dcopy(dv_round_values[-(shift + sample_size):-shift])

	upsamle_one_idx = np.random.choice(np.arange(sample_size), up_sample_num - sample_size)
	upsamle_two_idx = np.random.choice(np.arange(sample_size), up_sample_num - sample_size)
	
	sampling_one = np.concatenate([sampling_one] + [sampling_one[upsamle_one_idx]])
	sampling_two = np.concatenate([sampling_two] + [sampling_two[upsamle_two_idx]])
	
	# n_one, _ =  sampling_one.shape
	# n_two, _ = sampling_two.shape

	# delta = np.mean(sampling_one, axis=0) - np.mean(sampling_two, axis=0)
	# cov_m_one = np.cov(sampling_one, rowvar=False)
	# cov_m_two = np.cov(sampling_two, rowvar=False)
	# S_pooled = ((n_one-1)*cov_m_one + (n_two-1)*cov_m_two)/(n_one+n_one-2)
	# t2_statistics = ((n_one*n_two)/(n_one+n_two)) * np.matmul(np.matmul(delta.transpose(), np.linalg.inv(S_pooled)), delta)
	# transformed_statistics = t2_statistics * (n_one+n_two-n_dim-1)/(n_dim*(n_one+n_two-2))
	# if transformed_statistics < 0: # only happen when number of samples less than the feature dimension
	# 	return False, 0
	
	# F_distribution = f_dis(n_dim, n_one+n_two-n_dim-1)
	# p_value = 1 - F_distribution.cdf(transformed_statistics)

	ht_result = pg.multivariate_ttest(sampling_one, sampling_two)
	p_value = ht_result['pval'][0]

	stopping = False
	if p_value > alpha:
		stopping = True
	if verbose:
		print("P values {:.10f}".format(p_value))
	return stopping, p_value

def hotelling_t2_onesample_ht(dv_round_recorder, alpha=0.5, shift = 5, sample_num=10, verbose=True):
	dv_round_values = pd.DataFrame(dv_round_recorder).values
	total_size, n_dim = dv_round_values.shape
	if total_size <= sample_num:
		return False, 0

	# sampling_one = dcopy(dv_round_values[-sample_num:])
	sampling_one = dcopy(dv_round_values)
	sampling_mean = np.mean(dcopy(dv_round_values[:-(sample_num - shift)]),axis=0)
	# sampling_mean = dcopy(dv_round_values[:-(sample_num - shift)])

	# n_one, _ =  sampling_one.shape
	# n_two, _ = sampling_two.shape

	# delta = np.mean(sampling_one, axis=0) - np.mean(sampling_two, axis=0)
	# cov_m_one = np.cov(sampling_one, rowvar=False)
	# cov_m_two = np.cov(sampling_two, rowvar=False)
	# S_pooled = ((n_one-1)*cov_m_one + (n_two-1)*cov_m_two)/(n_one+n_one-2)
	# t2_statistics = ((n_one*n_two)/(n_one+n_two)) * np.matmul(np.matmul(delta.transpose(), np.linalg.inv(S_pooled)), delta)
	# transformed_statistics = t2_statistics * (n_one+n_two-n_dim-1)/(n_dim*(n_one+n_two-2))
	# if transformed_statistics < 0: # only happen when number of samples less than the feature dimension
	# 	return False, 0
	
	# F_distribution = f_dis(n_dim, n_one+n_two-n_dim-1)
	# p_value = 1 - F_distribution.cdf(transformed_statistics)

	ht_result = pg.multivariate_ttest(sampling_one, sampling_mean)
	p_value = ht_result['pval'][0]

	stopping = False
	if p_value > alpha:
		stopping = True
	if verbose:
		print("P values {:.10f}".format(p_value))
	return stopping, p_value

def selection_proba_processing(selection_probs):
    # check if nan exits in probability 
    if np.isnan(selection_probs).any():
        selection_probs = [1/len(selection_probs)]*len(selection_probs)
        print("Probability exists NaN!!!!!!!!")
    # make sure there is no absolute-zero probability in p
    selection_probs = [(tmp+1e-30)/(sum(selection_probs)+len(selection_probs)*1e-30) for tmp in selection_probs]
    return selection_probs

def selection_proba_score_type(selection_probs, score_type):
    if score_type == "random":
        probs_new = [1/n_participants] * n_participants
    elif score_type == "reverse":
        probs_new = reverse_rank_list(selection_probs)
    else:
        probs_new = dcopy(selection_probs)
    return probs_new

def update_args(cmd_args):
    if cmd_args.dataset == 'mnist':
        args = copy.deepcopy(mnist_args)

    elif cmd_args.dataset == 'cifar10':
        args = copy.deepcopy(cifar_cnn_args)

    elif cmd_args.dataset == 'cifar100':
        args = copy.deepcopy(cifar100_args)

    elif cmd_args.dataset == 'sst':
        args = copy.deepcopy(sst_args)
        participant_iterations = [[5, 8000]]

    elif cmd_args.dataset == 'mr':
        args = copy.deepcopy(mr_args)
        participant_iterations = [[5, 8000]]

    elif cmd_args.dataset == 'covid_tweet':
        args = copy.deepcopy(covid_tweet_args)

    elif cmd_args.dataset == 'hft':
        args = copy.deepcopy(hft_args)

    elif cmd_args.dataset == 'fraud':
        args = copy.deepcopy(fraud_args)
    
    elif cmd_args.dataset == 'electricity':
        args = copy.deepcopy(electricity_args)

    elif cmd_args.dataset == 'pathmnist':
        args = copy.deepcopy(pathmnist_args)

    else:
        pass
    return args