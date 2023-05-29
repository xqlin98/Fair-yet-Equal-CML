import argparse
import datetime
import os
import pickle
import sys
import time
from collections import defaultdict
from copy import deepcopy as dcopy
from os.path import join as oj

import numpy as np
import pandas as pd
import torch
from torchvision import models

from utils.Data_Prepper import Data_Prepper
from utils.utils import (Logger, add_gradient_updates, add_update_to_model,
                         compute_grad_update, data_valuation, evaluate,
                         hotelling_t2_onesample_ht, l2norm,
                         performance_summary, selection_proba_processing,
                         selection_proba_score_type, softmax, train_model,
                         update_args)

''' Parse all arguments '''
parser = argparse.ArgumentParser(description='Process which dataset to run')
parser.add_argument('-d', '--dataset', help='Dataset name'
                    , type=str, required=True)
parser.add_argument('-b', '--beta', help='Beta values for softmax'
                    , type=float, default=150)
parser.add_argument('-r', '--ratio', help='Selection ratio in stage 2'
                    , type=float, default=0.4)
parser.add_argument('-st', '--score_type', help='Use which score to do stage 2'
                    , choices=["random", "reverse", "proportion"], type=str, default="proportion")
parser.add_argument('-split', '--split', help='Type of data partition',
                    nargs='?', choices=['uniform', 'classimbalance', 'powerlaw'], type=str, default='uniform')
parser.add_argument('-T1', '--T1', help='Number of iterations of stage 1'
                    , nargs='?', type=int, default=0)
parser.add_argument('-T2', '--T2', help='Number of iterations of stage 2'
                    , nargs='?', const=0, type=int, default=0)
parser.add_argument('-T', '--T', help='Number of total iteratios'
                    , nargs='?', const=0, type=int, default=0)
parser.add_argument('-E', '--E', help='Number of local iterations'
                    , nargs='?', const=3, type=int, default=3)
parser.add_argument('-lr', '--manual_lr', help='Learning rate'
                    , nargs='?', const=-1, type=float, default=0)
parser.add_argument('-n', '--exp_name', help='Name of the experiment'
                    , nargs='?',  type=str, default="")
parser.add_argument('-dv', '--dv_method', help='Method used to do data valuation'
                    , nargs='?', choices=['fed_loo', 'cos_grad', 'mr'],  type=str, default="cos_grad")
parser.add_argument('-noise', '--noisy_type', help='Noise attack type'
                    , nargs='?', choices=['normal', 'label_noise_different', 'feature_noise_different', 'powerlaw', 'missing_values',
                    'nonstatinary_label_noise_inc', 'nonstatinary_label_noise_dec', 'nonstatinary_label_noise_both'], type=str, default="normal")
parser.add_argument('-nc', '--n_participants', help='Number of participants'
                    , nargs='?',  type=int, default=30)
parser.add_argument('-max_noise', '--max_noise', help='maximum level of noise'
                    , nargs='?',  type=float, default=0.9)
parser.add_argument('-alpha', '--alpha', help='Threshold for hypothesis testing'
                    , type=float, default=0.7)
parser.add_argument('-samp_num', '--samp_num', help='Look ahead samples for hypothesis testing'
                    , type=int, default=15)
parser.add_argument('-samp_dim', '--samp_dim', help='Number of participants used in hypothesis testing'
                    , type=int, default=10)

cmd_args = parser.parse_args()
print(cmd_args)
args = update_args(cmd_args)

args.update(vars(cmd_args))

# ''' Set up experiment arguments             '''
E = args['E']
T = args['T']
T1 = args['T1']
T2 = args['T2']
split = args["split"]
score_type = args["score_type"]
ratio = args["ratio"]
beta = args["beta"]
sample_size_cap = 0
n_participants = args["n_participants"]
if args['manual_lr'] != 0 :
    args['lr'] = args['manual_lr']
else:
    args['lr'] = args['lr'] / 5 if n_participants > 5 else args['lr']
args['momentum'] = 0.9
optimizer_fn = args['optimizer_fn']
loss_fn = args['loss_fn']
device = torch.device("cuda")

# ''' Set up entire experiment directory            '''
str_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M')
exp_dir = 'Exp_{}_{}'.format(cmd_args.exp_name, str_time)
exp_dir = oj(args['dataset'], exp_dir)
results_dir = 'Benchmark'
os.makedirs(oj(results_dir, exp_dir), exist_ok=True)

''' Set up individual/specific experiment directory and logger '''
exp_spec = 'Stream_{}_{}_{:.2f}_{:.2f}_T1-{}_T2-{}_B{}_E{}_lr{}_N{}_D{}'.format(args['dataset'], args['split'][:3], beta, ratio,
    T1, T2, args['batch_size'], E, str(args['lr']).replace('.',''), n_participants, sample_size_cap)

individual_exp_dir = oj(results_dir, exp_dir, exp_spec)
os.makedirs(individual_exp_dir, exist_ok=True)

log = open(oj(individual_exp_dir, 'log'), "w")
sys.stdout = Logger(log)
print("Experimental settings are: ", args, '\n')
print("Logging to the : {}.".format(oj(individual_exp_dir, 'log')))

""" Setting up the data loader and model/optimizer for each clients """
data_prepper = Data_Prepper(
    args['dataset'], train_batch_size=args['batch_size'], n_participants=n_participants, sample_size_cap=sample_size_cap
    , device=device, args_dict=args)

train_loaders, valid_loader, test_loader = data_prepper.get_stream_data_loader()
print("Stream size is: ",data_prepper.stream_step)
shard_sizes = torch.tensor(data_prepper.shard_sizes).float()
relative_shard_sizes = torch.div(shard_sizes, torch.sum(shard_sizes))
weights = relative_shard_sizes
print("Shard sizes are: ", shard_sizes.tolist())

server_model = args['model_fn'](device=device).cuda()
init_backup = dcopy(server_model)

models, optimizers = [], []
for i in range(args['n_participants']):
    local_model = dcopy(server_model).cuda()
    try:
        local_optimizer = args['optimizer_fn'](local_model.parameters(), lr=args['lr'], momentum=args['momentum'] )
    except:
        local_optimizer = args['optimizer_fn'](local_model.parameters(), lr=args['lr'])

    models.append(local_model)
    optimizers.append(local_optimizer)


''' Start federated training - stage 1 ''' 
# define experimental recorder
pre_server_model = dcopy(server_model)
server_perfs = defaultdict(list)
dv_recorder = defaultdict(list)
dv_round_recorder = defaultdict(list)
hypothesis_test_recorder = defaultdict(list)

local_perfs = defaultdict(list)
valid_perfs = defaultdict(list)
fed_perfs = defaultdict(list)
model_diffs = defaultdict(list)

historical_gradients = [[torch.zeros(param.shape).to(device) for param in server_model.parameters()] for i in range(n_participants)]

data_value_score = torch.zeros(n_participants,device=device)

sampled_dims = np.random.choice(np.arange(n_participants), args["samp_dim"], replace=False)

# if T1 = 0 then use KL to determine the stopping
if T1 == 0 & T == 0:
    T1 = sys.maxsize
else:
    T1 = T

for t in range(T1):

    server_gradient =  [torch.zeros(param.shape).to(device) for param in server_model.parameters()]
    device_gradients = [[torch.zeros(param.shape).to(device) for param in server_model.parameters()] for i in range(args['n_participants'])]

    for i, (model, optimizer, loader) in enumerate(zip(models, optimizers, train_loaders)):

        model.load_state_dict(server_model.state_dict())

        model = train_model(model, loader, loss_fn, optimizer, device=device, E=E)

        gradient = compute_grad_update(server_model, model)
        model.load_state_dict(server_model.state_dict())

        add_gradient_updates(device_gradients[i], gradient, weight=weights[i])

        add_gradient_updates(server_gradient, gradient, weight=weights[i])

        add_gradient_updates(historical_gradients[i], gradient, weight=weights[i])

    pre_server_model = dcopy(server_model)
    add_update_to_model(server_model, server_gradient)
    
    # predication on validation set
    print("Validation performance at round : ", t+1 )
    loss, accuracy = evaluate(server_model, valid_loader, data_prepper=data_prepper, loss_fn=loss_fn, verbose=True, device=device)
    server_perfs['_loss'].append(loss.item())
    server_perfs['_accu'].append(accuracy.item())
    for i in range(n_participants):
        valid_perfs[str(i)+'_loss'].append(loss.item())
        valid_perfs[str(i)+'_accu'].append(accuracy.item())

    # data valuation on each clients
    data_value_score, data_value_round = data_valuation(data_value_score, server_model, pre_server_model, server_gradient,
                                                        device_gradients, weights, n_participants, t, valid_loader, loss_fn, device, accuracy, dv_method=args["dv_method"])

    data_value_score_d, data_value_round_d = data_value_score.cpu().detach().numpy(), data_value_round.cpu().detach().numpy()
    for i in range(n_participants):
        dv_recorder[i].append(data_value_score_d[i])
        dv_round_recorder[i].append(data_value_round_d[i])
    
    # hotelling's T2 hypothesis testing
    dv_round_df = pd.DataFrame(dv_round_recorder)
    stop_flag, p_value = hotelling_t2_onesample_ht(dv_round_df.iloc[:,sampled_dims],
                                                    alpha=args["alpha"], shift=0,sample_num=args["samp_num"], verbose=False)
    hypothesis_test_recorder["stop_flag"].append(stop_flag)
    hypothesis_test_recorder["p_value"].append(p_value)

    # determine the stopping of stage 1
    if stop_flag and cmd_args.T1==0:
        break

    # reload the dataset to the next batch of data streaming
    train_loaders, valid_loader, test_loader = data_prepper.get_stream_data_loader()

''' Show the information computed from stage 1 '''
T1 = t + 1
print("Step 1 train stop at: {}\n\n".format((t+1)))
print("Data values are: ", data_value_score.data)
data_value_score_ = torch.div(data_value_score, torch.max(data_value_score))
print("Data values are(max norm): ", data_value_score_.data)
print("Beginning of rewarding stage:")
print("-"*60)
param_counts = [ len(param.view(-1)) for param in server_model.parameters() ]
print("Parameter counts for each layer:", param_counts)

''' Start federated training stage 2 ''' 
# processing the selection probability 
data_value_score = data_value_score.cpu().detach().numpy()
selection_probs = softmax(data_value_score, beta=beta)
selection_probs = selection_proba_processing(selection_probs)
print("Beta value:{:.2f}:", beta)
print("Throughout ratio:{:.2f}:", ratio)
print("Selection probability linear:", selection_probs)
probs_new = selection_proba_score_type(selection_probs, score_type)

# preparing parameters
size_S = int(ratio * n_participants) # randomly picked devices to reward
if T2 == 0:
    T2 = T
staleness_status = {tmp:[0]*T2 for tmp in range(n_participants)}

for t in range(T1, T2):

    # reload dataset
    train_loaders, valid_loader, test_loader = data_prepper.get_stream_data_loader()

    selected_S = np.random.choice(range(n_participants), size=size_S, replace=False, p=probs_new)

    server_gradient =  [torch.zeros(param.shape).to(device) for param in server_model.parameters()]

    for i in selected_S:
        staleness_status[i][t-T1] = 1

        model, optimizer, loader = models[i], optimizers[i], train_loaders[i]

        model.load_state_dict(server_model.state_dict())

        model = train_model(model, loader, loss_fn, optimizer, device=device, E=E)

        gradient = compute_grad_update(server_model, model)
        model.load_state_dict(server_model.state_dict())

        add_gradient_updates(server_gradient, gradient, weight=weights[i])

    add_update_to_model(server_model, server_gradient)

    # predication on validation set
    print("Validation performance at round : ", t+1)
    loss, accuracy = evaluate(server_model, valid_loader, data_prepper=data_prepper, loss_fn=loss_fn, verbose=True, device=device)
    server_perfs['_loss'].append(loss.item())
    server_perfs['_accu'].append(accuracy.item())

    for i, model in enumerate(models):
        loss, accuracy = evaluate(model, valid_loader, data_prepper=data_prepper, loss_fn=loss_fn, device=device)
        valid_perfs[str(i)+'_loss'].append(loss.item())
        valid_perfs[str(i)+'_accu'].append(accuracy.item())

        fed_loss, fed_accu = 0, 0

        loss, accuracy = evaluate(model, train_loaders[i], data_prepper=data_prepper, loss_fn=loss_fn, device=device)
        local_perfs[str(i)+'_loss'].append(loss.item())
        local_perfs[str(i)+'_accu'].append(accuracy.item())

        model_diff_norm = l2norm(compute_grad_update(server_model, model)).item()
        model_diffs[str(i)+"_model_diff_l2norm"].append(model_diff_norm)


''' Compiling results and plotting  '''
pd.DataFrame(server_perfs).to_csv(oj(individual_exp_dir,'server.csv'), index=False)
pd.DataFrame(fed_perfs).to_csv(oj(individual_exp_dir,'fed.csv'), index=False)
pd.DataFrame(valid_perfs).to_csv(oj(individual_exp_dir,'valid.csv'), index=False)
pd.DataFrame(local_perfs).to_csv(oj(individual_exp_dir,'local.csv'), index=False)
pd.DataFrame(staleness_status).to_csv(oj(individual_exp_dir,'staleness.csv'), index=False)
pd.DataFrame(dv_recorder).to_csv(oj(individual_exp_dir,'dv_recorder.csv'), index=False)
pd.DataFrame(dv_round_recorder).to_csv(oj(individual_exp_dir, 'dv_round_recorder.csv'), index=False)
pd.DataFrame(hypothesis_test_recorder).to_csv(oj(individual_exp_dir, 'hypothesis_test_recorder.csv'), index=False)
pd.DataFrame(model_diffs).to_csv(oj(individual_exp_dir, 'model_diff.csv'), index=False)

# write the sys settings down
args["data_value_score"] = data_value_score
args["num_noisy_clients"] = data_prepper.num_noisy_clients
args["fraction_of_noise_label"] = data_prepper.fraction_of_noise_label
args["stage1_stop_step"] = T1
args["selection_probability"] = probs_new

# compute the performance of the framework
from new_plot import plot

try:
    args = performance_summary(args=args, path=individual_exp_dir)
    plot(oj(individual_exp_dir))
except:
    pass

with open(oj(individual_exp_dir, 'settings_dict.txt'), 'w') as file:
    [file.write(key + ' : ' + str(value) + '\n') for key, value in args.items()]

with open(oj(individual_exp_dir, 'settings_dict.pickle'), 'wb') as f:
    pickle.dump(args, f)

with open(oj(individual_exp_dir, 'complete.txt'), 'w') as file:
    file.write('complete')

log.close() # close the writer of log

