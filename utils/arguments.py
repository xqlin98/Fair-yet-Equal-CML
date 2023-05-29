import torch
from torch import nn, optim
from copy import deepcopy as dcopy

from utils.models_defined import DQN, MEDNET, LogisticRegression, MLP, MNIST_LogisticRegression,MLP_FRAUD,\
MLP_Net, MLP_HFT, CNN_Net, RNN, RNN_IMDB, CNN_Text, ResNet18, ResNet18_torch, CNNCifar, CNNCifar_10, CNNCifar_100, CNNCifar_TF, CNN_Cifar100_BN, AlexNet, VGG11,RNN_TS, RNN_ELEC

use_cuda = True
cuda_available = torch.cuda.is_available()


def update_gpu(args):
	if 'cuda' in str(args['device']):
		args['device'] = torch.device('cuda:{}'.format(args['gpu']))
	if torch.cuda.device_count() > 0:
		args['device_ids'] = [device_id for device_id in range(torch.cuda.device_count())]
	else:
		args['device_ids'] = []



adult_args = {
	# system parameters
	'gpu': 0,
	'device': torch.device("cuda" if cuda_available and use_cuda else "cpu"),
	# setting parameters
	'dataset': 'adult',
	'sample_size_cap': 4000,
	'n_participants': 5,
	'split': 'powerlaw',
	'batch_size': 16,
	'train_val_split_ratio': 0.9,
	'alpha': 0.8,

	# model parameters
	'model_fn': LogisticRegression, #LogisticRegression, MLP
	'optimizer_fn': optim.SGD,
	'loss_fn': nn.NLLLoss(),  #CrossEntropyLoss NLLLoss
	  # only used during pretraining for rffl models, no decay
	'lr': 3e-2, # initial lr, with decay
	'lr_decay':0.977,  #0.977**100 ~= 0.1

	# training parameters
	'iterations': 50,
	'E': 2,

	'reputation_fade':1,
	'alpha_decay':True,
}

mnist_args = {
	# system parameters
	'save_gpu':False,
	'gpu': 0,
	'device': torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"),
	# setting parameters
	# 'stream_step':700,
	'stream_step':250,
	'dataset': 'mnist',
	'sample_size_cap': 6000,
	'n_participants': 5,
	'split': 'powerlaw', #or 'classimbalance'

	'batch_size' : 32, 
	'train_val_split_ratio': 0.9,
	'alpha': 0.95,
	'Gamma': 0.5,

	# model parameters
	'model_fn': CNN_Net, #MLP_Net, CNN_Net, MNIST_LogisticRegression
	'optimizer_fn': optim.SGD,
	'loss_fn': nn.NLLLoss(), 
	'lr': 0.01,
	'gamma':0.977,
	'lr_decay':0.977,  #0.977**100 ~= 0.1

	# fairness/training parameters
	'iterations': 60,
	'E':2
}

sst_args = {
	# system parameters
	'gpu': 0,
	'device': torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"),
	'save_gpu': False,
	# setting parameters
	'dataset': 'sst',
	'sample_size_cap': 5000,
	'n_participants': 5,
	'split': 'powerlaw', #or 'powerlaw' classimbalance
	'batch_size' : 256, 

	'train_val_split_ratio': 0.9,
	'alpha': 0.95,
	'Gamma': 1,


	# model parameters
	'model_fn': CNN_Text,
	'embed_num': 20000,
	'embed_dim': 300,
	'class_num': 5,
	'kernel_num': 128,
	'kernel_sizes': [3,3,3],
	'static':False,

	'optimizer_fn': optim.Adam,
	'loss_fn': nn.NLLLoss(), 
	'lr': 1e-4,
	# 'grad_clip':1e-3,
	'gamma':0.977,
	'lr_decay':0.977,  #0.977**100 ~= 0.1


	# training parameters
	'iterations': 100,
	'E': 2,
}


mr_args = {
	# system parameters
	'gpu': 0,
	'device': torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"),
	'save_gpu': False,
	# setting parameters
	'dataset': 'mr',
	'sample_size_cap': 5000,
	'n_participants': 5,
	'split': 'powerlaw', #or 'powerlaw' classimbalance

	'batch_size' : 128, 
	'train_val_split_ratio': 0.9,
	'alpha': 0.95,
	'Gamma':1,

	# model parameters
	'model_fn': CNN_Text,
	'embed_num': 20000,
	'embed_dim': 300,
	'class_num': 2,
	'kernel_num': 128,
	'kernel_sizes': [3,3,3],
	'static':False,

	'optimizer_fn': optim.Adam,
	'loss_fn': nn.NLLLoss(), 
	'lr': 5e-5,
	# 'grad_clip':1e-3,
	'gamma':0.977,
	'lr_decay':0.977,  #0.977**100 ~= 0.1


	# training parameters
	'iterations': 100,
	'E': 2,
}


cifar_cnn_args = {
	# system parameters
	'gpu': 0,
	'device': torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"),
	'log_interval':20,
	# setting parameters
	'stream_step':300,
	'dataset': 'cifar10',
	'sample_size_cap': 20000,
	'n_participants': 10,
	'split': 'powerlaw', #or 'classimbalance'

	'batch_size' : 128, 
	'train_val_split_ratio': 0.8,
	'alpha': 0.95,
	'Gamma': 0.15,

	# model parameters
	'model_fn': CNNCifar_10, #ResNet18_torch, CNNCifar_TF
	#'model_fn': ResNet18,
	# 'optimizer_fn': optim.SGD,
	'optimizer_fn': optim.Adam,
	# 'loss_fn': nn.NLLLoss(),#  nn.CrossEntropyLoss(), 
	'loss_fn': nn.CrossEntropyLoss(),
	 # only used during pretraining for rffl models, no decay
	# 'lr': 0.015,
	'lr':0.001,
	'gamma':0.977,
	'lr_decay':0.977,  #0.977**100 ~= 0.1


	# training parameters
	'iterations': 200,
	'E': 2,
}

cifar100_args = {
	# system parameters
	'gpu': 0,
	'device': torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"),
	'log_interval': 20,
	
	# setting parameters
	'stream_step':150,
	'dataset': 'cifar100',
	'sample_size_cap': 20000,
	'n_participants': 5,
	'split': 'powerlaw', #or 'classimbalance'
	'batch_size' : 128, 
	'train_val_split_ratio': 0.8,
	'alpha': 5,

	# model parameters
	'model_fn': ResNet18_torch, #ResNet18_torch, CNN_Cifar100_BN, AlexNet
	'optimizer_fn': optim.Adam, # optim.SGD,
	'loss_fn': nn.NLLLoss(),#  nn.CrossEntropyLoss(),  nn.NLLLoss()
	 # only used during pretraining for rffl models, no decay
	'lr': 0.0005,
	'lr_decay': 0.977,   #0.955**100 ~= 0.01


	# training parameters
	'iterations': 300,
	'E': 1,
}

covid_tweet_args = {
	# system parameters
	'gpu': 0,
	'device': torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"),
	'log_interval': 20,
	
	# setting parameters
	'dataset': 'covid_tweet',
	'dataset_days':25,
	'training_days':10,
	'valid_days':5,
	'data_step_size':4,
	'sample_size_cap': 20000,
	'n_participants': 5,
	'split': 'powerlaw', #or 'classimbalance'
	'batch_size' : 15, 
	'alpha': 5,

	# model parameters
	'model_fn': RNN_TS, #ResNet18_torch, CNN_Cifar100_BN
	'optimizer_fn': optim.Adam, # optim.SGD,optim.Adam
	'loss_fn': nn.MSELoss(),#  nn.CrossEntropyLoss(), 
	 # only used during pretraining for rffl models, no decay
	# 'lr': 0.007,
	'lr': 0.001,
	'lr_decay': 0.977,   #0.955**100 ~= 0.01


	# training parameters
	'iterations': 300,
	'E': 1,
}

hft_args = {
	# system parameters
	'gpu': 0,
	'device': torch.device("cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"),
	'log_interval': 20,
	
	# setting parameters
	'dataset': 'hft',
	'stream_step': 250,
	'dataset_day': 3,
	'task_number': 0,
	'n_participants': 30,
	'batch_size' : 5, 
	'alpha': 5,

	# model parameters
	'model_fn': MLP_HFT, #ResNet18_torch, CNN_Cifar100_BN
	'optimizer_fn': optim.Adam, # optim.SGD,optim.Adam
	'loss_fn': nn.NLLLoss(),#  nn.CrossEntropyLoss(), 
	 # only used during pretraining for rffl models, no decay
	# 'lr': 0.007,
	'lr': 0.001,
	'lr_decay': 0.977,   #0.955**100 ~= 0.01


	# training parameters
	'iterations': 300,
	'E': 2,
}

fraud_args = {
	# system parameters
	'save_gpu':False,
	'gpu': 0,
	'device': torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"),
	# setting parameters
	'stream_step':400,
	'dataset': 'fraud',
	'sample_size_cap': 6000,
	'n_participants': 5,
	'split': 'powerlaw', #or 'classimbalance'

	'batch_size' : 32, 
	'train_val_split_ratio': 0.9,
	'alpha': 0.95,
	'Gamma': 0.5,

	# model parameters
	'model_fn': MLP_FRAUD, #MLP_Net, CNN_Net, MNIST_LogisticRegression
	'optimizer_fn': optim.SGD,
	'loss_fn': nn.NLLLoss(), 
	'lr': 0.001,
	'gamma':0.977,
	'lr_decay':0.977,  #0.977**100 ~= 0.1

	# fairness/training parameters
	'iterations': 60,
	'E':2
}

electricity_args = {
	# system parameters
	'save_gpu':False,
	'gpu': 0,
	'device': torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"),
	# setting parameters
	# 'stream_step':700,
	'stream_step':500,
	'dataset': 'electricity',
	'sample_size_cap': 6000,
	'n_participants': 5,
	'split': 'powerlaw', #or 'classimbalance'

	'batch_size' : 32, 
	'train_val_split_ratio': 0.9,
	'alpha': 0.95,
	'Gamma': 0.5,

	# model parameters
	'model_fn': RNN_ELEC, #MLP_Net, CNN_Net, MNIST_LogisticRegression
	'optimizer_fn': optim.SGD,
	'loss_fn': nn.MSELoss(), 
	'lr': 0.0005,
	'gamma':0.977,
	'lr_decay':0.977,  #0.977**100 ~= 0.1

	# fairness/training parameters
	'iterations': 60,
	'E':2
}

pathmnist_args = {
	# system parameters
	'save_gpu':False,
	'gpu': 0,
	'device': torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"),
	# setting parameters
	# 'stream_step':700,
	'stream_step':450,
	'dataset': 'pathmnist',
	'sample_size_cap': 6000,
	'n_participants': 5,
	'split': 'powerlaw', #or 'classimbalance'

	'batch_size' : 128, 
	'train_val_split_ratio': 0.9,
	'alpha': 0.95,
	'Gamma': 0.5,

	# model parameters
	'model_fn': MEDNET, #MLP_Net, CNN_Net, MNIST_LogisticRegression
	'optimizer_fn': optim.Adam,
	'loss_fn': nn.CrossEntropyLoss(), 
	'lr': 0.01,
	'gamma':0.977,
	'lr_decay':0.977,  #0.977**100 ~= 0.1

	# fairness/training parameters
	'iterations': 60,
	'E':2
}

fedrl_common_args = {
	# system parameters
	'save_gpu':False,
	'gpu': 0,
	'device': torch.device("cuda" if torch.cuda.is_available() and use_cuda else "cpu"),
	# setting parameters
	
	# rl parameters

	# original parameters -v0.0
	'training_frequency': 2,
	'target_network_update_frequency': 2000,
	'train_run': 600,
	'train_step': 2000,
	'eval_run':5,
	'exploration_steps': 850000,

	# # parameters -v1.0
	# 'training_frequency': 10,
	# 'target_network_update_frequency': 500,
	# 'train_step': 500,
	# 'eval_run':1,
	# 'exploration_steps': 850000,

	'gamma': 0.99,
	'memory_size': 500000,
	'model_persistence_update_frequency': 50000,
	'replay_start_size': 50000,
	'exploration_max': 1,
	'exploration_min': 0.1,
	'exploration_test': 0.02,
	
	# environment parameters
	'total_run_limit': 10000,
	'clip': True,

	# 'n_participants': 5,
	'split': 'powerlaw', #or 'classimbalance'

	'batch_size' : 32, 
	'train_val_split_ratio': 0.9,
	'alpha': 0.95,
	'eps':0.01,

	# model parameters
	'model_fn': DQN, #MLP_Net, CNN_Net, MNIST_LogisticRegression
	'optimizer_fn': optim.Adam,
	'loss_fn': nn.NLLLoss(), 
	'lr': 0.0001/5,
	# 'gamma':0.977,
	'lr_decay':0.977,  #0.977**100 ~= 0.1

	# fairness/training parameters
	'iterations': 60,
	'E':2
}

fedrl_gog_args = {
	# original parameters -v0.0
	'training_frequency': 2,
	'target_network_update_frequency': 2000,
	'train_run': 600,
	'train_step': 2000,
	'eval_run':1,
	'exploration_steps': 850000,
}

breakout_args = dcopy(fedrl_common_args)
breakout_args.update(
{
	# setting parameters
	'dataset': 'Breakout'
}
)


spaceinvader_args = dcopy(fedrl_common_args)
spaceinvader_args.update(
{
	# setting parameters
	'dataset': 'SpaceInvaders'
}
)

pong_args = dcopy(fedrl_common_args)
pong_args.update(
{
	# setting parameters
	'dataset': 'Pong'
}
)