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

from utils.arguments import breakout_args, pong_args, spaceinvader_args
from utils.RL_Environment import AtariEnv, DDQNSolver, DDQNTrainer
from utils.utils import (Logger, add_gradient_updates, add_update_to_model,
                         compute_grad_update, data_valuation,
                         hotelling_t2_onesample_ht, l2norm,
                         performance_summary_rl, selection_proba_processing,
                         selection_proba_score_type, softmax,
                         synchronized_epsilon)

''' Parse cmd arguments '''
parser = argparse.ArgumentParser(description='Process which dataset to run')
parser.add_argument('-d', '--dataset', help='Dataset name'
                    , type=str, required=True)
parser.add_argument('-b', '--beta', help='Beta values for softmax'
                    , type=float, default=30)
parser.add_argument('-r', '--ratio', help='Selection ratio in stage 2'
                    , type=float, default=0.4)
parser.add_argument('-st', '--score_type', help='Use which score to do stage 2'
                    , choices=["random", "reverse", "proportion"], type=str, default="proportion")
parser.add_argument('-T1', '--T1', help='Number of iterations of stage 1'
                    , nargs='?', type=int, default=0)
parser.add_argument('-T2', '--T2', help='Number of iterations of stage 2'
                    , nargs='?', const=0, type=int, default=0)
parser.add_argument('-T', '--T', help='Number of overall iterations'
                    , nargs='?', const=600, type=int, default=600)
parser.add_argument('-n', '--exp_name', help='Name of the experiment'
                    , nargs='?',  type=str, default="")
parser.add_argument('-dv', '--dv_method', help='Method used to do data valuation'
                    , nargs='?', choices=['fed_loo', 'cos_grad', 'mr'],  type=str, default="cos_grad")
parser.add_argument('-a', '--attack', help='Noise attack type'
                    , nargs='?', choices=['normal', 'reward_noise', 'state_noise','memory_size','exploration'], type=str, default="normal")
parser.add_argument('-nc', '--n_participants', help='Number of participants'
                    , nargs='?',  type=int, default=5)
parser.add_argument('-alpha', '--alpha', help='Threshold for hypothesis testing'
                    , type=float, default=0.95)
parser.add_argument('-samp_num', '--samp_num', help='Look ahead samples for hypothesis testing'
                    , type=int, default=20)
parser.add_argument('-samp_dim', '--samp_dim', help='Number of participants used in hypothesis testing'
                    , type=int, default=5)

cmd_args = parser.parse_args()
print(cmd_args)
if cmd_args.dataset == 'Breakout':
    args = dcopy(breakout_args)
elif cmd_args.dataset == 'SpaceInvaders':
    args = dcopy(spaceinvader_args)
elif cmd_args.dataset == 'Pong':
    args = dcopy(pong_args)
else:
    pass

args.update(vars(cmd_args))

''' Set up experiment arguments             '''
E = args['E']
T1 = args['T1']
T2 = args['T2']
T = args['T']
split = args["split"]
score_type = args["score_type"]
ratio = args["ratio"]
beta = args["beta"]
sample_size_cap = 0
optimizer_fn = args['optimizer_fn']
loss_fn = args['loss_fn']
n_participants = args['n_participants']

if args['attack'] in ["reward_noise", "state_noise"]:
    if args['n_participants'] == 5:
        if args['dataset'] in ["Pong"]:
            noise_level = [0.2, 0.1, 0.05, 0, 0]
        else:
            noise_level = [0.6, 0.4, 0.2, 0, 0]
    elif args['n_participants'] == 10:
        if args['dataset'] in ["Pong"]:
            noise_level = [0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.06, 0.04, 0.02, 0]
        else:
            noise_level = [0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05,  0]
elif args['attack'] == "memory_size":
    if args['n_participants'] == 5:
        noise_level = [0.05, 0.1, 0.15, 0.35, 0.35]
    elif args['n_participants'] == 10:
        noise_level = [0.01, 0.02, 0.04, 0.08, 0.1, 0.15, 0.15, 0.15, 0.15, 0.15]
elif args['attack'] == "exploration":
    if args['n_participants'] == 5:
        noise_level = np.arange(0,1,0.2)
    elif args['n_participants'] == 10:
        noise_level = np.arange(0,1,0.1)
else:
    noise_level = [0] * args['n_participants']
args["noise_level"] = noise_level
device = torch.device("cuda")


''' Set up entire experiment directory            '''
str_time = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H:%M')
exp_dir = 'Exp_{}_{}'.format(cmd_args.exp_name, str_time)
exp_dir = oj(args['dataset'], exp_dir)
results_dir = 'FedRL-2stage'
os.makedirs(oj(results_dir, exp_dir), exist_ok=True)

''' Set up individual/specific experiment directory and logger '''
exp_spec = 'FedRL_{}_{}_{:.2f}_{:.2f}_T1-{}_T2-{}_B{}_E{}_lr{}_N{}_D{}'.format(args['dataset'], args['split'][:3], beta, ratio,
    T1, T2, args['batch_size'], E, str(args['lr']).replace('.',''), n_participants, sample_size_cap)

individual_exp_dir = oj(results_dir, exp_dir, exp_spec)
os.makedirs(individual_exp_dir, exist_ok=True)

log = open(oj(individual_exp_dir, 'log'), "w")
sys.stdout = Logger(log)
print("Experimental settings are: ", args, '\n')
print("Logging to the : {}.".format(oj(individual_exp_dir, 'log')))

""" Setting up the data loader and model/optimizer for each clients """
env = AtariEnv(args['dataset'])
n_action_space = env.action_space
server_model = args['model_fn'](4,84,84, n_action_space,device=device).cuda()
init_backup = dcopy(server_model)

# setting up the RL agents and model
models, optimizers, agents = [], [], []
for i in range(n_participants):
    local_model = dcopy(server_model).cuda()
    local_optimizer = args['optimizer_fn'](local_model.parameters(), lr=args['lr'])

    agent = DDQNTrainer(action_space=n_action_space, model=local_model,
                        optimizer=local_optimizer, device=device, args=args, noise=args["attack"], noise_level=noise_level[i])

    models.append(local_model)
    optimizers.append(local_optimizer)
    agents.append(agent)

eval_agent = DDQNSolver(n_action_space, dcopy(server_model).cuda(), device, args=args)

shard_sizes = torch.tensor([1]*n_participants)
relative_shard_sizes = torch.div(shard_sizes, torch.sum(shard_sizes))
weights = relative_shard_sizes
print("Shard sizes are: ", shard_sizes.tolist())

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
if T1 == 0:
    T1 = sys.maxsize

# pre-play each agent first to initilize the memory to certrain capacity
for agent in agents:
    env.agent_play(agent, int(args["replay_start_size"]/n_participants))

for t in range(T1):

    server_gradient =  [torch.zeros(param.shape).to(device) for param in server_model.parameters()]
    device_gradients = [[torch.zeros(param.shape).to(device) for param in server_model.parameters()] for i in range(args['n_participants'])]

    # get synchronized exporation ratio
    epsilon = synchronized_epsilon(agents)
    print("T1 step {}: Epsilon: {:.4f}".format(t, epsilon))

    for i, local_agent in enumerate(agents):
        
        # distribute the model to the agents
        local_agent._assign_epsilon(epsilon, n_participants)
        local_agent._assign_model(server_model.state_dict())

        env.agent_play(local_agent, args['train_step'])

        model = local_agent.get_model()

        gradient = compute_grad_update(server_model, model)
        model.load_state_dict(server_model.state_dict())

        add_gradient_updates(device_gradients[i], gradient, weight=weights[i])

        add_gradient_updates(server_gradient, gradient, weight=weights[i])

        add_gradient_updates(historical_gradients[i], gradient, weight=weights[i])

    add_update_to_model(server_model, server_gradient)

    # predication on validation
    print("Validation performance at round : ", t+1 )
    epsilon = synchronized_epsilon(agents)
    eval_agent._assign_epsilon(epsilon)
    eval_agent._assign_model(server_model.state_dict())
    score, step = env.agent_play(eval_agent, None, args['eval_run'])
    server_perfs['_score'].append(score)
    server_perfs['_step'].append(step)
    print("Server model score {:.2f}, step: {:.2f}".format(score,step))

    # data valuation on each clients
    data_value_score, data_value_round = data_valuation(data_value_score, server_model, pre_server_model, server_gradient,
                                                        device_gradients, weights, n_participants, t, None, loss_fn, device, None, dv_method=args["dv_method"])

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

if T != 0:
    T2 = T - T1
# preparing parameters
size_S = int(ratio * n_participants) # randomly picked devices to reward
staleness_status = {"client {}".format(tmp):[0]*T2 for tmp in range(n_participants)}

for t in range(T1, T1+T2):

    # randomly sample devices to reward them
    selected_S = np.random.choice(range(n_participants), size=size_S, replace=False, p=probs_new)

    server_gradient =  [torch.zeros(param.shape).to(device) for param in server_model.parameters()]

    # get synchronized exporation ratio
    epsilon = synchronized_epsilon(agents)

    for i in selected_S:
        staleness_status['client {}'.format(i)][t-T1] = 1
        
        # distribute model to agents
        local_agent = agents[i]
        local_agent._assign_epsilon(epsilon, len(selected_S))
        local_agent._assign_model(server_model.state_dict())

        env.agent_play(local_agent, args['train_step'])

        model = local_agent.get_model()

        gradient = compute_grad_update(server_model, model)
        model.load_state_dict(server_model.state_dict())

        add_gradient_updates(server_gradient, gradient, weight=weights[i])

    add_update_to_model(server_model, server_gradient)

    print("Validation performance at round : ", t+1)
    epsilon = synchronized_epsilon(agents)
    eval_agent._assign_epsilon(epsilon)
    eval_agent._assign_model(server_model.state_dict())
    score, step = env.agent_play(eval_agent, None, args['eval_run'])
    print("Server model score {:.2f}, step: {:.2f}".format(score,step))
    server_perfs['_score'].append(score)
    server_perfs['_step'].append(step)

    for i, local_model in enumerate(models):
        epsilon = synchronized_epsilon(agents)
        eval_agent._assign_epsilon(epsilon)
        eval_agent._assign_model(local_model.state_dict())
        score, step = env.agent_play(eval_agent, None, args['eval_run'])    
        valid_perfs[str(i)+'_score'].append(score)
        valid_perfs[str(i)+'_step'].append(step)

        fed_loss, fed_accu = 0, 0

        model_diff_norm = l2norm(compute_grad_update(server_model, local_model)).item()
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
args["stage1_stop_step"] = T1
args["selection_probability"] = probs_new

# compute the performance of the framework
try:
    args = performance_summary_rl(args=args, path=oj(individual_exp_dir))
except Exception as e:
    print(e)

with open(oj(individual_exp_dir, 'settings_dict.txt'), 'w') as file:
    [file.write(key + ' : ' + str(value) + '\n') for key, value in args.items()]

with open(oj(individual_exp_dir, 'settings_dict.pickle'), 'wb') as f:
    pickle.dump(args, f)

from utils.new_plot import plot_rl

try:
    plot_rl(oj(individual_exp_dir))
except Exception as e:
    print(e)

with open(oj(individual_exp_dir, 'complete.txt'), 'w') as file:
    file.write('complete')

log.close() # close the writer of log

with open(oj(individual_exp_dir, 'settings_dict.txt'), 'w') as file:
    [file.write(key + ' : ' + str(value) + '\n') for key, value in args.items()]

with open(oj(individual_exp_dir, 'settings_dict.pickle'), 'wb') as f:
    pickle.dump(args, f)
