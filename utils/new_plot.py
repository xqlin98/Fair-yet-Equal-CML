import os
from os.path import join as oj

import sys
import json
import copy
import time
import datetime
import random
from itertools import product
from collections import defaultdict
from math import ceil
import pandas as pd
# from utils.models_defined import ResNet18_torch, ResNet18

import matplotlib.pyplot as plt

import pickle

def plot_adv(logdir, FONTSIZE=16):

	# pickle - loading 
	with open(oj(logdir,'settings_dict.pickle'), 'rb') as f: 
		args = pickle.load(f)

	dataset = args['dataset']
	N = args['n_participants']
	A = args['n_adversaries']
	attack = args['attack']

	if attack == 'lf':

		adv_df = pd.read_csv(oj(logdir,'adv_lf.csv'))
		target_accs = [col for col in adv_df if 'target' in col]

		for col in target_accs:
			player_index = col.replace('_target_accu', '')
			plt.plot(range(len(adv_df)), adv_df[col], label=player_index)

		plt.title('{dataset} H{N}A{A}'.format(dataset=dataset.upper(), N=str(N), A=str(A)), 
			fontsize=FONTSIZE, fontweight='bold')
		plt.xlabel("Communication Rounds", fontsize=FONTSIZE, fontweight='bold')
		plt.ylabel("Target Accuracy", fontsize=FONTSIZE, fontweight='bold')

		if N > 10:
			plt.legend(loc='lower right', ncol=2)
		else:
			plt.legend(loc='lower right')
		plt.tight_layout()

		plt.savefig(oj(logdir, 'target_accs.png'))
		plt.clf()


		atk_succs = [col for col in adv_df if 'attack' in col]
		for col in atk_succs:
			player_index = col.replace('_attack_success', '')
			plt.plot(range(len(adv_df)), adv_df[col], label=player_index)

		plt.title('{dataset} H{N}A{A}'.format(dataset=dataset.upper(), N=str(N), A=str(A)), 
			fontsize=FONTSIZE, fontweight='bold')
		plt.xlabel("Communication Rounds", fontsize=FONTSIZE, fontweight='bold')
		plt.ylabel("Attack Success", fontsize=FONTSIZE, fontweight='bold')
		
		if N > 10:
			plt.legend(loc='lower right', ncol=2)
		else:
			plt.legend(loc='lower right')
		plt.tight_layout()

		plt.savefig(oj(logdir, 'atk_success.png'))
		plt.clf()

	if 'dist_all_layer.csv' in os.listdir(logdir):

		dist_all_layer_df  = pd.read_csv(oj(logdir,'dist_all_layer.csv'))
		fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
		for j, metric in enumerate(['dist', "perc"]):
			
			metric_cols = [col for col in dist_all_layer_df.columns if metric in col]

			for player_index, col in enumerate(dist_all_layer_df[metric_cols]):

				if int(player_index) < N:
					
					axs[j].plot(range(len(dist_all_layer_df)), dist_all_layer_df[col], label='H'+str(player_index))
				elif int(player_index) < N + A :

					axs[j].plot(range(len(dist_all_layer_df)), dist_all_layer_df[col], label='A'+str(player_index), linestyle='dashdot', marker='+')
				else: #init back up distance
					axs[j].plot(range(len(dist_all_layer_df)), dist_all_layer_df[col], label='Init', linestyle='--', marker='o')

				# axs[j].set_title(metric.replace('dist','Distance').replace('perc', 'Percentage'))
			axs[j].set_xlabel("Communication Rounds", fontsize=FONTSIZE, fontweight='bold')
			
			axs[j].set_ylabel(metric.replace('dist','$L_2$ Distance').replace('perc', 'Relative'), 
				fontsize=FONTSIZE, fontweight='bold')
	
		plt.suptitle('{dataset} Distance to Server Model'.format(dataset=dataset.upper()), 
			fontsize=FONTSIZE, fontweight='bold')

		if N + A > 10:
			axs[0].legend(loc='lower right', ncol=2)
		else:
			axs[0].legend(loc='lower right')

		plt.tight_layout()
		plt.savefig(oj(logdir, 'dist_all_layer.png'))
		plt.clf()


	if 'rs.csv' in os.listdir(logdir):

		rs_df = pd.read_csv(oj(logdir,'rs.csv'))

		for player_index in rs_df:

			if player_index == 'R_threshold':
				plt.plot(range(len(rs_df)), rs_df[player_index], label='Threshold', linestyle='dashed', color='k')

			elif int(player_index) < N:
				# honest participant
				plt.plot(range(len(rs_df)), rs_df[player_index], label='H'+player_index)
			else:
				plt.plot(range(len(rs_df)), rs_df[player_index], label='A'+player_index, linestyle='dashdot', marker='+')

		plt.title('{dataset} H{N}A{A}'.format(dataset=dataset.upper(), N=str(N), A=str(A)), 
			fontsize=FONTSIZE, fontweight='bold')
		plt.xlabel("Communication Rounds", fontsize=FONTSIZE, fontweight='bold')
		plt.ylabel("Reputations", fontsize=FONTSIZE, fontweight='bold')

		if N + A > 10:
			plt.legend(loc='lower left', ncol=2)
		else:
			plt.legend(loc='lower left')
		# plt.legend()
		plt.tight_layout()

		plt.savefig(oj(logdir, 'reputations.png'))
		plt.clf()

	if 'qs.csv' in os.listdir(logdir):

		qs_df = pd.read_csv(oj(logdir,'qs.csv'))
		for player_index in qs_df:
			if int(player_index) < N:
				# honest participant
				plt.plot(range(len(qs_df)), qs_df[player_index], label='H'+player_index)
			else:
				plt.plot(range(len(qs_df)), qs_df[player_index], label='A'+player_index, linestyle='dashdot', marker='+')

		plt.title('{dataset} H{N}A{A}'.format(dataset=dataset.upper(), N=str(N), A=str(A)), 
			fontsize=FONTSIZE, fontweight='bold')
		plt.xlabel("Communication Rounds", fontsize=FONTSIZE, fontweight='bold')
		plt.ylabel("Quotas", fontsize=FONTSIZE, fontweight='bold')


		if N+A > 10:
			plt.legend(loc='lower left', ncol=2)
		else:
			plt.legend(loc='lower left')

		plt.tight_layout()
		plt.savefig(oj(logdir, 'quotas.png'))
		# plt.show()

		plt.clf()


def plot(results_dir):


	with open(oj(results_dir,'settings_dict.pickle'), 'rb') as f: 
		args = pickle.load(f) 


	for j, metric in enumerate(['accu', "loss"]):
		fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
		for i, name in enumerate(['valid', 'local']):
			path = oj(results_dir, name+'.csv')
			df = pd.read_csv(path)
			metric_cols = [col for col in df.columns if metric in col]
			
			[axs[i].plot(df[col], label=str(player_index+1)) for player_index,col in enumerate(metric_cols)]

			axs[i].set_title(name.capitalize())

		if metric == 'accu':
			axs[0].legend(loc='lower right', ncol= ceil(len(metric_cols)/10) )
		elif metric == 'loss':
			axs[0].legend(loc='upper right', ncol= ceil(len(metric_cols)/10) )

		name = 'Accuracy' if metric == 'accu' else 'Loss'

		fig.suptitle('Agent Performance - ' + name)

		plt.tight_layout()
		plt.savefig(oj(results_dir, name+'.png'))
		plt.clf()
		plt.close()
		# plt.show()
	

	if 'model_diff.csv' in os.listdir(results_dir):

		model_diff_norm_df  = pd.read_csv(oj(results_dir,'model_diff.csv'))

		metric_cols = [col for col in model_diff_norm_df.columns if 'model_diff_l2norm' in col]

		fig = plt.figure(figsize=(4.5, 4))

		for player_index, col in enumerate(model_diff_norm_df[metric_cols]):

			plt.plot(range(len(model_diff_norm_df)), model_diff_norm_df[col], label=str(player_index+1))


		plt.xlabel("Iterations")
		plt.ylabel("L2 model diff")

		if args['n_participants'] > 10:
			plt.legend(loc='lower right', ncol=2)
		else:
			plt.legend(loc='lower right')

		plt.tight_layout()
		plt.savefig(oj(results_dir, 'model_diff.png'))
		plt.clf()
		plt.close()

	try:
		path = oj(results_dir, 'param_counts.csv')
		param_df = pd.read_csv(path)

		path = oj(results_dir, 'sparsity_levels.csv')
		sparsity_df = pd.read_csv(path)

		fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

		ax1.plot(param_df)
		ax1.set_title("Parameter counts")
		ax1.legend(param_df.columns, loc='lower left', ncol= ceil(len(param_df.columns)/10) )

		ax2.plot(sparsity_df)
		ax2.set_title("Sparsity levels")

		fig.suptitle('Model sparsity analysis')
		plt.tight_layout()
		plt.savefig(oj(results_dir, 'model_sparsity.png'))
		plt.clf()
		plt.close()

	except:
		pass

	try:
		path = oj(results_dir, 'server.csv')
		df = pd.read_csv(path)
	except:
		return

	fig, ax = plt.subplots()
	ax.plot(df['_accu'], label='Accuracy')
	ax.tick_params(axis='y')

	# Generate a new Axes instance, on the twin-X axes (same position)
	ax2 = ax.twinx()
	# Plot exponential sequence, set scale to logarithmic and change tick color
	ax2.plot(df['_loss'], color='orange', label='Loss')
	ax2.tick_params(axis='y')

	# ask matplotlib for the plotted objects and their labels
	lines, labels = ax.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax.legend(lines + lines2, labels + labels2, loc=2)

	# plt.legend()
	plt.title('Server Performance')
	plt.tight_layout()
	plt.savefig(oj(results_dir, 'Server.png'))
	plt.close()
	return

def plot_rl(results_dir):


	with open(oj(results_dir,'settings_dict.pickle'), 'rb') as f: 
		args = pickle.load(f) 


	for j, metric in enumerate(['score', "step"]):
		fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
		for i, name in enumerate(['valid']):
			path = oj(results_dir, name+'.csv')
			df = pd.read_csv(path)
			metric_cols = [col for col in df.columns if metric in col]
			
			[axs[i].plot(df[col], label=str(player_index+1)) for player_index,col in enumerate(metric_cols)]

			axs[i].set_title(name.capitalize())
		
		axs[0].legend(loc='lower right', ncol= ceil(len(metric_cols)/10) )

		name = 'Score' if metric == 'score' else 'Step'

		fig.suptitle('Agent Performance - ' + name)

		plt.tight_layout()
		plt.savefig(oj(results_dir, name+'.png'))
		plt.clf()
		plt.close()
		# plt.show()
	

	if 'model_diff.csv' in os.listdir(results_dir):

		model_diff_norm_df  = pd.read_csv(oj(results_dir,'model_diff.csv'))

		metric_cols = [col for col in model_diff_norm_df.columns if 'model_diff_l2norm' in col]

		fig = plt.figure(figsize=(4.5, 4))

		for player_index, col in enumerate(model_diff_norm_df[metric_cols]):

			plt.plot(range(len(model_diff_norm_df)), model_diff_norm_df[col], label=str(player_index+1))


		plt.xlabel("Iterations")
		plt.ylabel("L2 model diff")

		if args['n_participants'] > 10:
			plt.legend(loc='lower right', ncol=2)
		else:
			plt.legend(loc='lower right')

		plt.tight_layout()
		plt.savefig(oj(results_dir, 'model_diff.png'))
		plt.clf()
		plt.close()

	try:
		path = oj(results_dir, 'param_counts.csv')
		param_df = pd.read_csv(path)

		path = oj(results_dir, 'sparsity_levels.csv')
		sparsity_df = pd.read_csv(path)

		fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

		ax1.plot(param_df)
		ax1.set_title("Parameter counts")
		ax1.legend(param_df.columns, loc='lower left', ncol= ceil(len(param_df.columns)/10) )

		ax2.plot(sparsity_df)
		ax2.set_title("Sparsity levels")

		fig.suptitle('Model sparsity analysis')
		plt.tight_layout()
		plt.savefig(oj(results_dir, 'model_sparsity.png'))
		plt.clf()
		plt.close()

	except:
		pass

	try:
		path = oj(results_dir, 'server.csv')
		df = pd.read_csv(path)
	except:
		return

	fig, ax = plt.subplots()
	ax.plot(df['_score'], label='Score')
	ax.tick_params(axis='y')

	# Generate a new Axes instance, on the twin-X axes (same position)
	ax2 = ax.twinx()
	# Plot exponential sequence, set scale to logarithmic and change tick color
	ax2.plot(df['_step'], color='orange', label='Step')
	ax2.tick_params(axis='y')

	# ask matplotlib for the plotted objects and their labels
	lines, labels = ax.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax.legend(lines + lines2, labels + labels2, loc=2)

	# plt.legend()
	plt.title('Server Performance')
	plt.tight_layout()
	plt.savefig(oj(results_dir, 'Server.png'))
	plt.close()
	return

if __name__ == '__main__':

	# plot('reward_gradients-betas/mnist/P10-CLA/1000 r0s1')
	directory = 'fairFL_2stages/FedAvg/mnist/Exp_2021-09-21-13:53/mnist_pow_T1-10_T2-10_B32_E2_lr015_N5_D3000'
	plot(directory)
	# plot_adv(directory)
	# for rootdir, dirs, files in os.walk(directory):
		# if 'r0s1' in rootdir or 'r1s1' in rootdir or 'r1s0' in rootdir:		
			# plot_adv(rootdir)
