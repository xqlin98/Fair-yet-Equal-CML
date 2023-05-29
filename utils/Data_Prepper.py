import copy
import os
import random
import argparse
import numpy as np
from numpy.core.numeric import indices
import pandas as pd

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision.datasets import CIFAR10, CIFAR100
from torchtext.data import Field, LabelField, BucketIterator

from utils.Custom_Dataset import Custom_Dataset
from utils.utils import transform_table_to_Xy, split_datasets_to_stream_mnist_cifar, split_indices, random_split

from medmnist.dataset import PathMNIST

class Data_Prepper:
	def __init__(self, name, train_batch_size, n_participants, 
		sample_size_cap=-1, test_batch_size=100, valid_batch_size=None, 
		train_val_split_ratio=0.8, device=None, args_dict=None):
		self.args = None
		self.args_dict = args_dict
		self.name = name
		self.device = device
		self.n_participants = n_participants
		self.sample_size_cap = sample_size_cap
		self.train_val_split_ratio = train_val_split_ratio
		self.reload_dataset_source = None
		self.reload_checkpoint = 0
		self.data_statistics = None

		self.init_batch_size(train_batch_size, test_batch_size, valid_batch_size)

		if name in ['sst', 'mr', 'imdb']:
			parser = argparse.ArgumentParser(description='CNN text classificer')
			# self.args = parser.parse_args()

			self.args  = {}

			self.train_datasets, self.validation_dataset, self.test_dataset = self.prepare_dataset(name)

			self.valid_loader = BucketIterator(self.validation_dataset, batch_size = 500, sort_key=lambda x: len(x.text), device=self.device  )
			self.test_loader = BucketIterator(self.test_dataset, batch_size = 500, sort_key=lambda x: len(x.text), device=self.device)

			# self.args.embed_num = len(self.args.text_field.vocab)
			# self.args.class_num = len(self.args.label_field.vocab)
			
			self.args['embed_dim'] = self.args_dict['embed_dim']
			self.args['kernel_num'] = self.args_dict['kernel_num']
			self.args['kernel_sizes'] = self.args_dict['kernel_sizes']
			self.args['static'] = self.args_dict['static']
			
			train_size = sum([len(train_dataset) for train_dataset in self.train_datasets])
			if self.n_participants > 5:
				print("Splitting all {} train data to {} parties. Caution against this due to the limited training size.".format(train_size, self.n_participants))
			print("Model embedding arguments:", self.args)
			print('------')
			print("Train to split size: {}. Validation size: {}. Test size: {}".format(train_size, len(self.validation_dataset), len(self.test_dataset)))
			print('------')

		elif name in  ['tiny_imagenet', 'tiny_imagenet_224']:
			self.train_folder, self.test_folder = self.prepare_dataset(name)
			train_indices, val_indices = get_train_valid_indices(len(self.train_folder), self.train_val_split_ratio, self.sample_size_cap)

			self.train_indices = train_indices
			self.val_indices = val_indices

			print('------')
			print("Train to split size: {}. Validation size: {}. Test size: {}".format(len(self.train_indices), len(self.val_indices), len(self.test_folder)))
			print('------')

			self.valid_loader = DataLoader(self.train_folder, batch_size=self.test_batch_size, sampler=SubsetRandomSampler(self.val_indices))
			self.test_loader = DataLoader(self.test_folder, batch_size=self.test_batch_size)
		
		elif name in  ['covid_tweet','hft']:
			multi_time_dataset = self.prepare_dataset(name)

			self.distribute_dataset()
			self.add_noise_dataset()
			
			# _ = self.reload_dataset()


		else:
			self.train_dataset, self.validation_dataset, self.test_dataset = self.prepare_dataset(name)

			print('------')
			print("Train to split size: {}. Validation size: {}. Test size: {}".format(len(self.train_dataset), len(self.validation_dataset), len(self.test_dataset)))
			print('------')

			self.distribute_dataset()
			self.add_noise_dataset()

			self.valid_loader = DataLoader(self.validation_dataset, batch_size=self.test_batch_size)
			self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size)

	def reload_dataset(self):
		if self.args_dict['dataset'] == 'hft':
			if self.reload_checkpoint < len(self.reload_dataset_source)-1:
				train_data = self.reload_dataset_source[self.reload_checkpoint]
			else:
				raise RuntimeError("Exhuasted dataset in HFT dataset")
			self.reload_checkpoint += 1

			self.fixed_time_interval_data = train_data

			from utils.utils import split_dataset_to_N_clients_hft
			
			# prepare training data for n clients
			indices_list = split_dataset_to_N_clients_hft(dataset=train_data, N=self.args_dict['n_participants'])

			# prepare validation data in server
			server_valid_data = self.reload_dataset_source[self.reload_checkpoint+1]

			# get the train loader
			self.train_dataset = Custom_Dataset(train_data[0], train_data[1], device=self.device)

			# count the shard size for each clients
			shard_sizes = [len(tmp) for tmp in indices_list]
			self.shard_sizes = shard_sizes

			noisy_type = self.args_dict['noisy_type']

			# noise manipulation
			if noisy_type == "label_noise_different":
				num_noisy_clients = int(self.n_participants * 1.0)
				fraction_of_noise_label = np.arange(0,1,1/num_noisy_clients) * self.args_dict["max_noise"]

				label_nums = 3

				for c in range(num_noisy_clients):
					num_noisy_samples = int(shard_sizes[c] * fraction_of_noise_label[c])
					noisy_indices = np.random.choice(indices_list[c],num_noisy_samples,replace=False)

					self.train_dataset.targets[noisy_indices] = torch.tensor(np.random.choice(range(label_nums), num_noisy_samples),dtype=torch.int64,device=self.device)

			elif noisy_type == "feature_noise_different":
				num_noisy_clients = int(self.n_participants * 1.0)
				fraction_of_noise_label = np.arange(0,1,1/num_noisy_clients) * self.args_dict["max_noise"]
				std_scale = 0.8

				shape_of_data = list(self.train_dataset.data.shape[1:])
				for c in range(num_noisy_clients):
					num_noisy_samples = int(len(indices_list[c]) * fraction_of_noise_label[c])
					noisy_indices = np.random.choice(indices_list[c],num_noisy_samples,replace=False)

					random_size = [num_noisy_samples] + shape_of_data
					self.train_dataset.data[noisy_indices] = self.train_dataset.data[noisy_indices] + torch.tensor(np.random.normal(0,std_scale,random_size),dtype=torch.float32,device=self.device)
			else:
				num_noisy_clients = 0
				fraction_of_noise_label = 0
				pass
			self.num_noisy_clients = num_noisy_clients
			self.fraction_of_noise_label = fraction_of_noise_label

			self.train_loaders = [DataLoader(self.train_dataset, batch_size=self.args_dict['batch_size'], sampler=SubsetRandomSampler(indices)) for indices in indices_list]
			self.valid_loader = DataLoader(Custom_Dataset(*server_valid_data, device=self.device), batch_size=1)
			self.test_loader = None

		elif self.args_dict['dataset'] == 'covid_tweet':

			if self.reload_checkpoint < len(self.reload_dataset_source):
				data = self.reload_dataset_source[self.reload_checkpoint]
			else:
				raise RuntimeError("Exhuasted dataset in covid tweet dataset")
			self.reload_checkpoint += 1

			from utils.utils import split_data_N_clients, split_multi_time_interval_dataset, \
											transform_table_to_Xy, aggregate_date_data, split_n_day_valid
			self.fixed_time_interval_data = data

			# prepare training data for n clients
			n_client_data = split_data_N_clients(self.fixed_time_interval_data, 
								N = self.args_dict['n_participants'], training_days = self.args_dict['training_days'])
			n_client_train_data, n_client_valid_data = split_n_day_valid(n_client_data, valid_days=self.args_dict['valid_days'])

			# prepare validation data in server
			server_data = aggregate_date_data(self.fixed_time_interval_data)
			server_data = transform_table_to_Xy(server_data,
						training_days=self.args_dict['training_days'],
						mean=self.data_statistics[0],std=self.data_statistics[1])
			[server_train_data], [server_valid_data] = split_n_day_valid([server_data], valid_days=self.args_dict['valid_days'])

			# get train loader
			train_X_cat, train_y_cat = torch.cat([tmp[0] for tmp in n_client_train_data]),torch.cat([tmp[1] for tmp in n_client_train_data])
			self.train_dataset = Custom_Dataset(train_X_cat, train_y_cat, device=self.device)
			indices_list = []
			shard_sizes = []
			off_set = 0
			for i in range(len(n_client_train_data)):
				data_len = len(n_client_train_data[i][1])
				indices_list.append(list(range(off_set, off_set+data_len)))
				shard_sizes.append(data_len)
				off_set += data_len
			self.shard_sizes = shard_sizes
			
			noisy_type = self.args_dict['noisy_type']

			# noise manipulation
			if noisy_type == "label_noise_different":
				num_noisy_clients = int(self.n_participants * 1.0)
				fraction_of_noise_label = np.arange(0,1,1/num_noisy_clients) * self.args_dict["max_noise"]

				for c in range(num_noisy_clients):
					num_noisy_samples = int(len(indices_list[c]) * fraction_of_noise_label[c])
					noisy_indices = np.random.choice(indices_list[c],num_noisy_samples,replace=False)

					self.train_dataset.targets[noisy_indices] = torch.tensor(np.random.rand(num_noisy_samples) * 5,dtype=torch.float32,device=self.device)

			elif noisy_type == "feature_noise_different":
				num_noisy_clients = int(self.n_participants * 1.0)
				fraction_of_noise_label = np.arange(0,1,1/num_noisy_clients) * self.args_dict["max_noise"]
				std_scale = 0.8

				shape_of_data = list(self.train_dataset.data.shape[1:])
				for c in range(num_noisy_clients):
					num_noisy_samples = int(len(indices_list[c]) * fraction_of_noise_label[c])
					noisy_indices = np.random.choice(indices_list[c],num_noisy_samples,replace=False)

					random_size = [num_noisy_samples] + shape_of_data
					self.train_dataset.data[noisy_indices] = self.train_dataset.data[noisy_indices] + torch.tensor(np.random.normal(0,std_scale,random_size),dtype=torch.float32,device=self.device)
			else:
				num_noisy_clients = 0
				fraction_of_noise_label = 0
				pass
			self.num_noisy_clients = num_noisy_clients
			self.fraction_of_noise_label = fraction_of_noise_label

			self.train_loaders = [DataLoader(self.train_dataset, batch_size=self.args_dict['batch_size'], sampler=SubsetRandomSampler(indices)) for indices in indices_list]
			self.valid_loader = DataLoader(Custom_Dataset(*server_valid_data, device=self.device), batch_size=1)
			self.test_loader = None
			
		elif self.args_dict['dataset'] in ['mnist', 'cifar10', 'cifar100']:
			if self.reload_checkpoint < len(self.reload_dataset_source):
				train_dataset = self.reload_dataset_source[self.reload_checkpoint]
				validation_dataset = self.reload_dataset_source[self.reload_checkpoint + 1]
				self.train_dataset = Custom_Dataset(train_dataset[0], train_dataset[1], device=self.device)
				self.validation_dataset = Custom_Dataset(validation_dataset[0], validation_dataset[1], device=self.device)
			else:
				raise RuntimeError("Exhuasted dataset in covid tweet dataset")
			self.reload_checkpoint += 1

			# split the new stream data into n parts
			from utils.utils import random_split
			indices_list = random_split(sample_indices=list(range(len(self.train_dataset))), m_bins=self.args_dict['n_participants'], equal=True)

			self.train_datasets = [Custom_Dataset(self.train_dataset.data[indices],self.train_dataset.targets[indices])  for indices in indices_list]

			self.shard_sizes = [len(indices) for indices in indices_list]
			
			noisy_type = self.args_dict['noisy_type']
			# noise manipulation
			if noisy_type == "label_noise":
				num_noisy_clients = int(self.n_participants * 0.3)
				fraction_of_noise_label = 0.1

				label_nums = max(self.train_dataset.targets) + 1

				for c in range(num_noisy_clients):
					num_noisy_samples = int(len(indices_list[c]) * fraction_of_noise_label)
					noisy_indices = np.random.choice(indices_list[c],num_noisy_samples,replace=False)

					self.train_dataset.targets[noisy_indices] = torch.tensor(np.random.choice(range(label_nums), num_noisy_samples),dtype=torch.int64,device=self.device)
			elif noisy_type == "label_noise_different":
				num_noisy_clients = int(self.n_participants * 1.0)
				fraction_of_noise_label = np.arange(0,1,1/num_noisy_clients) * self.args_dict["max_noise"]

				label_nums = max(self.train_dataset.targets) + 1

				for c in range(num_noisy_clients):
					num_noisy_samples = int(len(indices_list[c]) * fraction_of_noise_label[c])
					noisy_indices = np.random.choice(indices_list[c],num_noisy_samples,replace=False)

					self.train_dataset.targets[noisy_indices] = torch.tensor(np.random.choice(range(label_nums), num_noisy_samples),dtype=torch.int64,device=self.device)

			elif noisy_type == "feature_noise_different":
				num_noisy_clients = int(self.n_participants * 1.0)
				fraction_of_noise_label = np.arange(0,1,1/num_noisy_clients) * self.args_dict["max_noise"]
				std_scale = 1.2

				shape_of_data = list(self.train_dataset.data.shape[1:])
				for c in range(num_noisy_clients):
					num_noisy_samples = int(len(indices_list[c]) * fraction_of_noise_label[c])
					noisy_indices = np.random.choice(indices_list[c],num_noisy_samples,replace=False)

					random_size = [num_noisy_samples] + shape_of_data
					self.train_dataset.data[noisy_indices] = self.train_dataset.data[noisy_indices] + torch.tensor(np.random.normal(0,std_scale,random_size),dtype=torch.float32,device=self.device)

			elif noisy_type == "feature_noise":
				num_noisy_clients = int(self.n_participants * 0.3)
				fraction_of_noise_label = 0.1
				std_scale = 0.8

				shape_of_data = list(self.train_dataset.data.shape[-3:])
				for c in range(num_noisy_clients):
					num_noisy_samples = int(len(indices_list[c]) * fraction_of_noise_label)
					noisy_indices = np.random.choice(indices_list[c],num_noisy_samples,replace=False)

					random_size = [num_noisy_samples] + shape_of_data
					self.train_dataset.data[noisy_indices] = self.train_dataset.data[noisy_indices] + torch.tensor(np.random.normal(0,std_scale,random_size),dtype=torch.float32,device=self.device)
					
			elif noisy_type == "backdoor":
				num_noisy_clients = int(self.n_participants * 0.3)
				fraction_of_noise_label = 0.3
				if self.name in ["mnist","cifar10","cifar100"]:
					label_nums = max(self.train_dataset.targets) + 1

					shape_of_img = list(self.train_dataset.data.shape[-2:])
					pixel_list = np.array([[shape_of_img[0]-4,shape_of_img[0]-4],[shape_of_img[0]-5,shape_of_img[0]-3],[shape_of_img[0]-3,shape_of_img[0]-5],[shape_of_img[0]-3,shape_of_img[0]-3]])
					for c in range(num_noisy_clients):
						num_noisy_samples = int(len(indices_list[c]) * fraction_of_noise_label)
						noisy_indices = np.random.choice(indices_list[c],num_noisy_samples,replace=False)

						for pixel in pixel_list:
							self.train_dataset.data[noisy_indices,:,pixel[0],pixel[1]] = torch.ones_like(self.train_dataset.data[noisy_indices,:,pixel[0],pixel[1]]) * 5
						self.train_dataset.targets[noisy_indices]  = self.train_dataset.targets[noisy_indices] + 1
					self.train_dataset.targets[self.train_dataset.targets == label_nums] = 0 # trim the label
			else:
				num_noisy_clients = 0
				fraction_of_noise_label = 0
				pass
			self.num_noisy_clients = num_noisy_clients
			self.fraction_of_noise_label = fraction_of_noise_label

			self.train_loaders = [DataLoader(self.train_dataset, batch_size=self.args_dict['batch_size'], sampler=SubsetRandomSampler(indices)) for indices in indices_list]
			self.valid_loader = DataLoader(self.validation_dataset, batch_size=self.args_dict['batch_size'])
			self.test_loader = None
			
		return self.train_loaders, self.valid_loader, self.test_loader

	def init_batch_size(self, train_batch_size, test_batch_size, valid_batch_size):
		self.train_batch_size = train_batch_size
		self.test_batch_size = test_batch_size
		self.valid_batch_size = valid_batch_size if valid_batch_size else test_batch_size

	def get_valid_loader(self):
		return self.valid_loader

	def get_test_loader(self):
		return self.test_loader

	def get_train_loaders(self, n_participants, split='powerlaw', batch_size=None, noisy_type = "normal"):
		if not batch_size:
			batch_size = self.train_batch_size

		if self.name in ['sst', 'mr', 'imdb']:
			# sst, mr, imdb split is different from other datasets, so return here				

			self.train_loaders = [BucketIterator(train_dataset, batch_size=self.train_batch_size, device=self.device, sort_key=lambda x: len(x.text),train=True) for train_dataset in self.train_datasets]
			self.shard_sizes = [(len(train_dataset)) for train_dataset in self.train_datasets]
			return self.train_loaders

		elif self.name in ['tiny_imagenet', 'tiny_imagenet_224']:
			
			if split == 'classimbalance':
				pass

			elif split == 'powerlaw':
				indices_list = powerlaw(self.train_indices, n_participants)
			else:
				# default uniform
				from utils.utils import random_split
				indices_list = random_split(sample_indices=self.train_indices, m_bins=n_participants, equal=True)				

			self.shard_sizes = [len(indices) for indices in indices_list]
			participant_train_loaders = [DataLoader(self.train_folder, batch_size=batch_size, sampler=SubsetRandomSampler(indices), num_workers=4) for indices in indices_list]
			self.train_loaders = participant_train_loaders
			return participant_train_loaders

		else:

			if split == 'classimbalance':
				if self.name not in ['mnist','cifar10']:
					raise NotImplementedError("Calling on dataset {}. Only mnist and cifar10 are implemnted for this split".format(self.name))

				n_classes = 10			
				data_indices = [torch.nonzero(self.train_dataset.targets == class_id).view(-1).tolist() for class_id in range(n_classes)]
				class_sizes = np.linspace(1, n_classes, n_participants, dtype='int')
				print("class_sizes for each party", class_sizes)
				party_mean = self.sample_size_cap // self.n_participants

				from collections import defaultdict
				party_indices = defaultdict(list)
				for party_id, class_sz in enumerate(class_sizes):	
					classes = range(class_sz) # can customize classes for each party rather than just listing
					each_class_id_size = party_mean // class_sz
					# print("party each class size:", party_id, each_class_id_size)
					for i, class_id in enumerate(classes):
						# randomly pick from each class a certain number of samples, with replacement 
						selected_indices = random.choices(data_indices[class_id], k=each_class_id_size)

						# randomly pick from each class a certain number of samples, without replacement 
						'''
						NEED TO MAKE SURE THAT EACH CLASS HAS MORE THAN each_class_id_size for no replacement sampling
						selected_indices = random.sample(data_indices[class_id],k=each_class_id_size)
						'''
						party_indices[party_id].extend(selected_indices)

						# top up to make sure all parties have the same number of samples
						if i == len(classes) - 1 and len(party_indices[party_id]) < party_mean:
							extra_needed = party_mean - len(party_indices[party_id])
							party_indices[party_id].extend(data_indices[class_id][:extra_needed])
							data_indices[class_id] = data_indices[class_id][extra_needed:]

				indices_list = [party_index_list for party_id, party_index_list in party_indices.items()] 

			elif split == 'disjoint':

				if self.name not in ['mnist','cifar10']:
					raise NotImplementedError("Calling on dataset {}. Only mnist and cifar10 are implemnted for this split".format(self.name))

				n_classes, n_classes_each = 10, 2 			
				data_indices = [torch.nonzero(self.train_dataset.targets == class_id).view(-1).tolist() for class_id in range(n_classes)]
				print("Each gets {} out of {} classes of data.".format(n_classes_each, n_classes))

				from collections import defaultdict
				party_indices = defaultdict(list)
				data_size = self.sample_size_cap // self.n_participants
				for i in range(self.n_participants):
					classes =  [(n_classes_each*i)%n_classes, (n_classes_each*i +1)%n_classes]
					# print("selected classes for agent {} is {}".format(i, classes))
					for class_id in classes:

						selected_indices = random.choices(data_indices[class_id], k=data_size // n_classes_each)
						party_indices[i].extend(selected_indices)

				indices_list = [party_index_list for party_id, party_index_list in party_indices.items()] 

			elif split == 'powerlaw':	
				indices_list = powerlaw(list(range(len(self.train_dataset))), n_participants)

			elif split in ['uniform','equal']:
				from utils.utils import random_split
				indices_list = random_split(sample_indices=list(range(len(self.train_dataset))), m_bins=n_participants, equal=True)
			
			elif split == 'random':
				from utils.utils import random_split
				indices_list = random_split(sample_indices=list(range(len(self.train_dataset))), m_bins=n_participants, equal=False)

			# from collections import Counter
			# for indices in indices_list:
			# 	print(Counter(self.train_dataset.targets[indices].tolist()))
			
			# individual trainining datasets created from the overall extracted dataset: self.train_dataset
			# this is so we can construct differentially private loaders
			self.train_datasets = [Custom_Dataset(self.train_dataset.data[indices],self.train_dataset.targets[indices])  for indices in indices_list]

			self.shard_sizes = [len(indices) for indices in indices_list]

			# noise manipulation
			if noisy_type == "label_noise":
				num_noisy_clients = int(self.n_participants * 0.3)
				fraction_of_noise_label = 0.1

				label_nums = max(self.train_dataset.targets) + 1

				for c in range(num_noisy_clients):
					num_noisy_samples = int(len(indices_list[c]) * fraction_of_noise_label)
					noisy_indices = np.random.choice(indices_list[c],num_noisy_samples,replace=False)

					self.train_dataset.targets[noisy_indices] = torch.tensor(np.random.choice(range(label_nums), num_noisy_samples),dtype=torch.int64,device=self.device)
			elif noisy_type == "label_noise_different":
				num_noisy_clients = int(self.n_participants * 1.0)
				fraction_of_noise_label = np.arange(0,1,1/num_noisy_clients)

				label_nums = max(self.train_dataset.targets) + 1

				for c in range(num_noisy_clients):
					num_noisy_samples = int(len(indices_list[c]) * fraction_of_noise_label[c])
					noisy_indices = np.random.choice(indices_list[c],num_noisy_samples,replace=False)

					self.train_dataset.targets[noisy_indices] = torch.tensor(np.random.choice(range(label_nums), num_noisy_samples),dtype=torch.int64,device=self.device)

			elif noisy_type == "feature_noise":
				num_noisy_clients = int(self.n_participants * 0.3)
				fraction_of_noise_label = 0.1
				std_scale = 0.8

				shape_of_data = list(self.train_dataset.data.shape[-3:])
				for c in range(num_noisy_clients):
					num_noisy_samples = int(len(indices_list[c]) * fraction_of_noise_label)
					noisy_indices = np.random.choice(indices_list[c],num_noisy_samples,replace=False)

					random_size = [num_noisy_samples] + shape_of_data
					self.train_dataset.data[noisy_indices] = self.train_dataset.data[noisy_indices] + torch.tensor(np.random.normal(0,std_scale,random_size),dtype=torch.float32,device=self.device)
			elif noisy_type == "backdoor":
				num_noisy_clients = int(self.n_participants * 0.3)
				fraction_of_noise_label = 0.3
				if self.name in ["mnist","cifar10","cifar100"]:
					label_nums = max(self.train_dataset.targets) + 1

					shape_of_img = list(self.train_dataset.data.shape[-2:])
					pixel_list = np.array([[shape_of_img[0]-4,shape_of_img[0]-4],[shape_of_img[0]-5,shape_of_img[0]-3],[shape_of_img[0]-3,shape_of_img[0]-5],[shape_of_img[0]-3,shape_of_img[0]-3]])
					for c in range(num_noisy_clients):
						num_noisy_samples = int(len(indices_list[c]) * fraction_of_noise_label)
						noisy_indices = np.random.choice(indices_list[c],num_noisy_samples,replace=False)

						for pixel in pixel_list:
							self.train_dataset.data[noisy_indices,:,pixel[0],pixel[1]] = torch.ones_like(self.train_dataset.data[noisy_indices,:,pixel[0],pixel[1]]) * 5
						self.train_dataset.targets[noisy_indices]  = self.train_dataset.targets[noisy_indices] + 1
					self.train_dataset.targets[self.train_dataset.targets == label_nums] = 0 # trim the label
			else:
				num_noisy_clients = 0
				fraction_of_noise_label = 0
				pass
			self.num_noisy_clients = num_noisy_clients
			self.fraction_of_noise_label = fraction_of_noise_label

			participant_train_loaders = [DataLoader(self.train_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(indices)) for indices in indices_list]
			self.train_loaders = participant_train_loaders
			return participant_train_loaders

	def get_stream_data_loader(self):
		if self.reload_checkpoint >= self.stream_step - 1:
			raise RuntimeError("Exhuasted dataset in {}".format(self.args_dict['dataset']))

		if self.args_dict['dataset'] == 'covid_tweet':
			# get the train indices and validation indices
			train_indices_list = [client_indices[self.reload_checkpoint] for client_indices in  self.n_client_stream_indices]

			# get the data loader
			self.train_loaders = [DataLoader(self.noise_train_dataset, batch_size=self.args_dict['batch_size'], sampler=SubsetRandomSampler(indices)) for indices in train_indices_list]
			self.valid_loader = DataLoader(self.stream_valid_data[self.reload_checkpoint], batch_size=self.args_dict['batch_size'])
			self.test_loader = None
			
		elif self.args_dict['dataset'] in ['mnist', 'cifar10', 'cifar100','hft', 'fraud', 'electricity','pathmnist']:
			# get the train indices and validation indices
			train_indices_list = [client_indices[self.reload_checkpoint] for client_indices in  self.n_client_stream_indices]
			valid_indice = np.concatenate([client_indices[self.reload_checkpoint + 1] for client_indices in  self.n_client_stream_indices])
			
			# get the data loader
			self.train_loaders = [DataLoader(self.noise_train_dataset, batch_size=self.args_dict['batch_size'], sampler=SubsetRandomSampler(indices)) for indices in train_indices_list]
			self.valid_loader = DataLoader(self.stream_train_dataset, batch_size=self.args_dict['batch_size'], sampler=SubsetRandomSampler(valid_indice))
			self.test_loader = None
		
		self.reload_checkpoint += 1
			
		return self.train_loaders, self.valid_loader, self.test_loader

	def add_noise_dataset(self):
		name = self.args_dict['dataset']
		max_noise = self.args_dict['max_noise']
		noisy_type = self.args_dict['noisy_type']
		indices_list = self.n_client_indices

		self.noise_train_dataset = copy.deepcopy(self.stream_train_dataset)

		# noise manipulation
		if noisy_type == "label_noise_different":
			num_noisy_clients = int(self.n_participants * 1.0)
			fraction_of_noise_label = np.arange(0,1,1/num_noisy_clients) * self.args_dict["max_noise"]

			label_nums = max(self.noise_train_dataset.targets) + 1

			for c in range(num_noisy_clients):
				num_noisy_samples = int(len(indices_list[c]) * fraction_of_noise_label[c])
				noisy_indices = np.random.choice(indices_list[c],num_noisy_samples,replace=False)

				if name in ['covid_tweet']:
					self.noise_train_dataset.targets[noisy_indices] = self.noise_train_dataset.targets[noisy_indices] + torch.tensor(np.random.rand(num_noisy_samples),dtype=torch.float32,device=self.device) * 0.5
				elif name in ['electricity']:
					# self.noise_train_dataset.targets[noisy_indices] = self.noise_train_dataset.targets[noisy_indices] + torch.tensor(np.random.normal(loc=0.5, scale=1.0,size=num_noisy_samples),dtype=torch.float32,device=self.device)
					self.noise_train_dataset.targets[noisy_indices] = -self.noise_train_dataset.targets[noisy_indices]
				else:
					self.noise_train_dataset.targets[noisy_indices] = torch.tensor(np.random.choice(range(label_nums), num_noisy_samples),dtype=torch.int64,device=self.device)

		elif noisy_type == "feature_noise_different":
			num_noisy_clients = int(self.n_participants * 1.0)
			fraction_of_noise_label = np.arange(0,1,1/num_noisy_clients) * self.args_dict["max_noise"]
			std_scale = 1.2

			shape_of_data = list(self.noise_train_dataset.data.shape[1:])
			for c in range(num_noisy_clients):
				num_noisy_samples = int(len(indices_list[c]) * fraction_of_noise_label[c])
				noisy_indices = np.random.choice(indices_list[c],num_noisy_samples,replace=False)

				random_size = [num_noisy_samples] + shape_of_data

				if name in ['electricity']:
					# self.noise_train_dataset.data[noisy_indices] = self.noise_train_dataset.data[noisy_indices] + torch.tensor(np.random.normal(1,1,random_size),dtype=torch.float32,device=self.device)
					self.noise_train_dataset.data[noisy_indices] = -self.noise_train_dataset.data[noisy_indices]
				else:
					self.noise_train_dataset.data[noisy_indices] = self.noise_train_dataset.data[noisy_indices] + torch.tensor(np.random.normal(0,std_scale,random_size),dtype=torch.float32,device=self.device)
		
		elif "nonstatinary_label_noise" in noisy_type:
			total_step = self.args_dict['T1']
			self.n_client_stream_indices
			num_noisy_clients = int(self.n_participants * 1.0)
			fraction_of_noise_label = np.arange(0,1,1/num_noisy_clients) * self.args_dict["max_noise"]

			label_nums = max(self.noise_train_dataset.targets) + 1
			
			# increasing noise level or decreasing noise level
			if "inc" in noisy_type:
				max_noise = 2 * max_noise
				stationary_clients = range(1,num_noisy_clients)
				nons_noise_level = np.arange(0,max_noise,max_noise/total_step)
				nons_client = 0
			elif "dec" in noisy_type:
				max_noise = 2 * max_noise
				stationary_clients = range(num_noisy_clients-1)
				nons_noise_level = np.arange(max_noise,0,-max_noise/total_step)
				nons_client = num_noisy_clients - 1
			elif "both" in noisy_type:
				# max_noise = 2 * max_noise
				incre_step = 50
				stationary_step = total_step - incre_step
				stationary_clients = range(1,num_noisy_clients-1)
				nons_noise_level_inc = list(np.arange(0,max_noise,max_noise/incre_step)) + [max_noise]*stationary_step
				print(nons_noise_level_inc)
				nons_noise_level_dec = list(np.arange(max_noise,0,-max_noise/incre_step)) + [0]*stationary_step
				nons_client_inc = [0]
				nons_client_dec = [num_noisy_clients - 1]
			
			for c in stationary_clients:
				num_noisy_samples = int(len(indices_list[c]) * fraction_of_noise_label[c])
				noisy_indices = np.random.choice(indices_list[c],num_noisy_samples,replace=False)

				if name in ['covid_tweet', 'electricity']:
					self.noise_train_dataset.targets[noisy_indices] = self.noise_train_dataset.targets[noisy_indices] + torch.tensor(np.random.rand(num_noisy_samples),dtype=torch.float32,device=self.device) * 0.5
				else:
					self.noise_train_dataset.targets[noisy_indices] = torch.tensor(np.random.choice(range(label_nums), num_noisy_samples),dtype=torch.int64,device=self.device)
			
			def continous_random_samples(idx_set, ratio):
				num_idx = len(idx_set)
				select_tf = np.random.choice([0,1],size=num_idx,p=[1-ratio,ratio])
				return list(np.array(idx_set)[np.nonzero(select_tf)]), np.sum(select_tf)
			# nonstationary noise
			for step in range(total_step):
				noisy_indices = []
				num_noisy_samples = 0
				if "both" in noisy_type:
					for client in nons_client_inc:
						noisy_indices_tmp,num_noisy_samples_tmp = continous_random_samples(self.n_client_stream_indices[client][step], nons_noise_level_inc[step])
						noisy_indices.extend(noisy_indices_tmp)
						num_noisy_samples += num_noisy_samples_tmp
					for client in nons_client_dec:
						noisy_indices_tmp,num_noisy_samples_tmp = continous_random_samples(self.n_client_stream_indices[client][step], nons_noise_level_dec[step])
						noisy_indices.extend(noisy_indices_tmp)
						num_noisy_samples += num_noisy_samples_tmp
				else:
					num_noisy_samples = int(len(self.n_client_stream_indices[nons_client][step]) * nons_noise_level[step])
					noisy_indices = np.random.choice(self.n_client_stream_indices[nons_client][step],num_noisy_samples,replace=False)
				if name in ['covid_tweet', 'electricity']:
					self.noise_train_dataset.targets[noisy_indices] = self.noise_train_dataset.targets[noisy_indices] + torch.tensor(np.random.rand(num_noisy_samples),dtype=torch.float32,device=self.device) * 0.5
				else:
					self.noise_train_dataset.targets[noisy_indices] = torch.tensor(np.random.choice(range(label_nums), num_noisy_samples),dtype=torch.int64,device=self.device)
		elif noisy_type == "missing_values":
			num_noisy_clients = int(self.n_participants * 1.0)
			fraction_of_noise_label = np.arange(0,1,1/num_noisy_clients) * self.args_dict["max_noise"]

			frac_missing = 0.5
			shape_of_data = list(self.noise_train_dataset.data.shape[1:])
			for c in range(num_noisy_clients):
				num_noisy_samples = int(len(indices_list[c]) * fraction_of_noise_label[c])
				noisy_indices = np.random.choice(indices_list[c],num_noisy_samples,replace=False)
				random_size = [num_noisy_samples] +  shape_of_data

				# if name in ['electricity']:
				# 	# self.noise_train_dataset.data[noisy_indices] = self.noise_train_dataset.data[noisy_indices] + torch.tensor(np.random.normal(1,1,random_size),dtype=torch.float32,device=self.device)
				# 	self.noise_train_dataset.data[noisy_indices] = -self.noise_train_dataset.data[noisy_indices]
				# else:
				self.noise_train_dataset.data[noisy_indices] = self.noise_train_dataset.data[noisy_indices].mul_(torch.tensor((np.random.uniform(0,1,random_size) > frac_missing).astype(np.float32),dtype=torch.float32,device=self.device))

		elif noisy_type == "powerlaw":
			num_noisy_clients = self.n_participants
			fraction_of_noise_label = self.fraction_of_noise_label
		else:
			num_noisy_clients = 0
			fraction_of_noise_label = 0
			pass
		self.num_noisy_clients = num_noisy_clients
		self.fraction_of_noise_label = fraction_of_noise_label

	def distribute_dataset(self):
		name = self.args_dict['dataset']

		if name in ["mnist", "cifar10", "hft", 'fraud', 'electricity','pathmnist']:
			if name != 'hft':
				# split the data indices into streaming format
				stream_indices = random_split(sample_indices=list(range(len(self.stream_train_dataset))), m_bins=self.args_dict['stream_step'], equal=True)
				self.stream_indices = stream_indices

			# split the data indices into N clients 
			split = self.args_dict['split']
			n_client_stream_indices = [[] for _ in range(self.n_participants)]

			for i in range(self.args_dict['stream_step']):
				if split == 'powerlaw' or self.args_dict['noisy_type'] == 'powerlaw':	
					indices_list, noise_level = powerlaw(self.stream_indices[i], self.n_participants)
					self.fraction_of_noise_label = -noise_level # more noise here means more data points
				elif split in ['uniform','equal']:
					indices_list = random_split(sample_indices=self.stream_indices[i], m_bins=self.n_participants, equal=True)
				
				elif split == 'random':
					indices_list = random_split(sample_indices=self.stream_indices[i], m_bins=self.n_participants, equal=False)
				
				# store the indices
				for j in range(self.n_participants):
					n_client_stream_indices[j].append(indices_list[j])
			
			# store the corresponding indices
			self.n_client_stream_indices = n_client_stream_indices
			self.n_client_indices = [np.concatenate(tmp) for tmp in n_client_stream_indices]
			self.shard_sizes = [len(tmp[0]) for tmp in self.n_client_stream_indices]
			self.stream_step = self.args_dict['stream_step']

		elif name in ['covid_tweet']:
			
			# gather all the split to a whole dataset and compute n client stream indices
			offset = 0
			n_client_stream_indices = [[] for i in range(self.n_participants)]

			all_data_X, all_data_y = [], []
			for i in range(self.n_participants):
				client_i_data = self.n_client_stream_train_data[i]

				for j in range(len(client_i_data)):
					X, y = client_i_data[j]
					n_client_stream_indices[i].append(np.arange(offset,(offset+len(X))))
					all_data_X.append(X)
					all_data_y.append(y)
					offset += len(X)
			
			# construct a whole dataset
			train_X, train_y = torch.cat(all_data_X, dim=0), torch.cat(all_data_y, dim=0)
			self.stream_train_dataset = Custom_Dataset(train_X, train_y, device=self.device)

			# store the corresponding indices
			self.n_client_stream_indices = n_client_stream_indices
			self.n_client_indices = [np.concatenate(tmp) for tmp in n_client_stream_indices]
			self.shard_sizes = [len(tmp[0]) for tmp in self.n_client_stream_indices]

		

	def prepare_dataset(self, name='adult'):
		if name == 'adult':
			from utils.load_adult import get_train_test

			train_data, train_target, test_data, test_target = get_train_test()

			X_train = torch.tensor(train_data.values, requires_grad=False).float()
			y_train = torch.tensor(train_target.values, requires_grad=False).long()
			X_test = torch.tensor(test_data.values, requires_grad=False).float()
			y_test = torch.tensor(test_target.values, requires_grad=False).long()

			print("X train shape: ", X_train.shape)
			print("y train shape: ", y_train.shape)
			pos, neg =(y_train==1).sum().item() , (y_train==0).sum().item()
			print("Train set Positive counts: {}".format(pos),"Negative counts: {}.".format(neg), 'Split: {:.2%} - {:.2%}'.format(1. * pos/len(X_train), 1.*neg/len(X_train)))
			print("X test shape: ", X_test.shape)
			print("y test shape: ", y_test.shape)
			pos, neg =(y_test==1).sum().item() , (y_test==0).sum().item()
			print("Test set Positive counts: {}".format(pos),"Negative counts: {}.".format(neg), 'Split: {:.2%} - {:.2%}'.format(1. * pos/len(X_test), 1.*neg/len(X_test)))

			train_indices, valid_indices = get_train_valid_indices(len(X_train), self.train_val_split_ratio, self.sample_size_cap)

			train_set = Custom_Dataset(X_train[train_indices], y_train[train_indices], device=self.device)
			validation_set = Custom_Dataset(X_train[valid_indices], y_train[valid_indices], device=self.device)
			test_set = Custom_Dataset(X_test, y_test, device=self.device)

			return train_set, validation_set, test_set

		elif name == 'mnist':

			train = FastMNIST('datasets', train=True, download=True)
			test = FastMNIST('datasets', train=False, download=True)

			train_indices, valid_indices = get_train_valid_indices(len(train), self.train_val_split_ratio, self.sample_size_cap)

			train_set = Custom_Dataset(train.data[train_indices], train.targets[train_indices], device=self.device)
			# validation_set = Custom_Dataset(train.data[valid_indices],train.targets[valid_indices] , device=self.device)
			test_set = Custom_Dataset(test.data, test.targets, device=self.device)
			validation_set = test_set

			self.stream_train_dataset = Custom_Dataset(train.data, train.targets, device=self.device)
			del train, test

			return train_set, validation_set, test_set

		elif name == 'pathmnist':
			from torchvision import transforms
			data_transform = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(mean=[.5], std=[.5])
			])
			train = PathMNIST(split='train', transform=data_transform, download=False, root="./datasets")
			# test = PathMNIST(split='test', transform=data_transform, download=False, root="./datasets")

			train_imgs, train_labels = [], []
			for i in range(len(train)):
				tmp_img, tmp_label = train[i]
				train_imgs.append(tmp_img)
				train_labels.append(tmp_label)
			train_imgs = torch.stack(train_imgs)
			train_labels = torch.tensor(train_labels, dtype=torch.int64).reshape([-1])

			train_set = Custom_Dataset(train_imgs, train_labels, device=self.device)
			validation_set = train_set
			test_set = train_set

			self.stream_train_dataset = Custom_Dataset(train_imgs, train_labels, device=self.device)

			del train
			return train_set, validation_set, test_set

		elif name == 'cifar10':	
			from torchvision import transforms
			transform_train = transforms.Compose([
				# transforms.RandomCrop(32, padding=4),
				# transforms.RandomHorizontalFlip(),
				# transforms.ToTensor(),
				# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			])

			transform_test = transforms.Compose([
				# transforms.ToTensor(),
				# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
				transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			])
			
			train = FastCIFAR10('datasets', train=True, download=True)#, transform=transform_train)
			test = FastCIFAR10('datasets', train=False, download=True)#, transform=transform_test)

			# transformation on data
			train.data = transform_train(train.data)
			test.data = transform_test(test.data)

			train_indices, valid_indices = get_train_valid_indices(len(train), self.train_val_split_ratio, self.sample_size_cap)
			
			train_set = Custom_Dataset(train.data[train_indices], train.targets[train_indices], device=self.device)
			# validation_set = Custom_Dataset(train.data[valid_indices],train.targets[valid_indices] , device=self.device)
			test_set = Custom_Dataset(test.data, test.targets, device=self.device)
			validation_set = test_set

			self.stream_train_dataset = Custom_Dataset(train.data, train.targets, device=self.device)
			del train, test

			return train_set, validation_set, test_set
		
		elif name == 'cifar100':
			from torchvision import transforms			

			# transform_train = transforms.Compose([
			# 	# transforms.RandomCrop(32, padding=4),
			# 	# transforms.RandomHorizontalFlip(),
			# 	# transforms.ToTensor(),
			# 	# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			# 	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			# ])

			# transform_test = transforms.Compose([
			# 	# transforms.ToTensor(),
			# 	# transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
			# 	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
			# ])

			stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
			transform_train = transforms.Compose([
				transforms.RandomHorizontalFlip(),
				transforms.RandomCrop(32,padding=4,padding_mode="reflect"),
				transforms.ToTensor(),
				transforms.Normalize(*stats)
			])

			transform_test = transforms.Compose([
				transforms.ToTensor(),
				transforms.Normalize(*stats)
			])

			train = CIFAR100(root='datasets', train=True, download=True, transform=transform_train)
			test = CIFAR100(root='datasets', train=False, download=False, transform=transform_test)

			from torch import Tensor
			train.targets = Tensor(train.targets).long()
			test.targets = Tensor(test.targets).long()

			from torch import from_numpy
			train.data = from_numpy(train.data).permute(0, 3, 1, 2).float()
			test.data = from_numpy(test.data).permute(0, 3, 1, 2).float()

			train_indices, valid_indices = get_train_valid_indices(len(train), self.train_val_split_ratio, self.sample_size_cap)

			train_set = Custom_Dataset(train.data[train_indices], train.targets[train_indices], device=self.device)
			validation_set = Custom_Dataset(train.data[valid_indices],train.targets[valid_indices] , device=self.device)
			test_set = Custom_Dataset(test.data, test.targets, device=self.device)

			self.stream_train_dataset = Custom_Dataset(train.data, train.targets, device=self.device)
			del train, test

			return train_set, validation_set, test_set

		elif name == "tiny_imagenet":

			pretrained_224 = False

			dataset_dir = "datasets/tiny-imagenet-200"
			train_dir = os.path.join(dataset_dir, 'train')
			val_dir = os.path.join(dataset_dir, 'val', 'images')
			kwargs = {'num_workers': 8, 'pin_memory': True}

			'''
			Separate validation images into separate sub folders
			'''
			val_dir = os.path.join(dataset_dir, 'val')
			img_dir = os.path.join(val_dir, 'images')

			fp = open(os.path.join(val_dir, 'val_annotations.txt'), 'r')
			data = fp.readlines()
			val_img_dict = {}
			for line in data:
				words = line.split('\t')
				val_img_dict[words[0]] = words[1]
			fp.close()

			# Create folder if not present and move images into proper folders
			for img, folder in val_img_dict.items():
				newpath = (os.path.join(img_dir, folder))
				if not os.path.exists(newpath):
					os.makedirs(newpath)
				if os.path.exists(os.path.join(img_dir, img)):
					os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))

			# Pre-calculated mean & std on imagenet:
			# norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
			# For other datasets, we could just simply use 0.5:
			# norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
			
			print('Preparing tiny_imagenet data ...')
			# Normalization
			from torchvision import transforms			
			norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) \
				if pretrained_224 else transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

			# Normal transformation
			if pretrained_224:
				train_trans = [transforms.RandomHorizontalFlip(), transforms.RandomResizedCrop(224), 
								transforms.ToTensor()]
				val_trans = [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), norm]
			else:
				train_trans = [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
				val_trans = [transforms.ToTensor(), norm]

			print('Preparing tiny_imagenet pytorch ImageFolders ...')
			from torchvision import datasets
			train_folder = datasets.ImageFolder(train_dir, transform=transforms.Compose(train_trans + [norm]))
			test_folder = datasets.ImageFolder(val_dir, transform=transforms.Compose(val_trans))

			return train_folder, test_folder

		elif name == 'covid_tweet':
			
			try:
				data = pd.read_csv("./datasets/tweetid_userid_keyword_sentiments_emotions_Singapore.csv")
			except: 
				raise RuntimeError("Covid Tweet dataset did not exists")

			# transform time stamp
			data['tweet_timestamp'] = pd.to_datetime(data['tweet_timestamp'], format="%Y-%m-%d %H-%M-%S")
			data['date'] = data['tweet_timestamp'].dt.floor('d')

			# compute the std and mean of features
			data_values = data[['valence_intensity', 'fear_intensity',	'anger_intensity',
							'happiness_intensity', 'sadness_intensity']].values
			data_mean = np.mean(data_values,axis=0)
			data_std = np.std(data_values, axis=0)
			self.data_statistics = [data_mean, data_std]

			from utils.utils import split_multi_time_interval_dataset
			
			# split dataset to streaming data
			multi_time_dataset = split_multi_time_interval_dataset(data
								, dataset_days=self.args_dict['dataset_days'], data_step_size=self.args_dict['data_step_size'])

			# store the multi-time interval dataset
			self.reload_dataset_source = multi_time_dataset

			from utils.utils import split_data_N_clients, split_multi_time_interval_dataset, \
											transform_table_to_Xy, aggregate_date_data, split_n_day_valid

			n_client_stream_train_data = [[] for _ in range(self.n_participants)]
			stream_valid_data =[]

			# split streaming data into n clients
			for i in range(len(multi_time_dataset)):
				# split training data to n clients
				stream_now = multi_time_dataset[i]
				n_client_data = split_data_N_clients(stream_now, 
									N = self.args_dict['n_participants'], training_days = self.args_dict['training_days'])
				n_client_train_data, n_client_valid_data = split_n_day_valid(n_client_data, valid_days=self.args_dict['valid_days'])

				# prepare validation data in server
				server_data = aggregate_date_data(stream_now)
				server_data = transform_table_to_Xy(server_data,
							training_days=self.args_dict['training_days'],
							mean=self.data_statistics[0],std=self.data_statistics[1])
				[server_train_data], [server_valid_data] = split_n_day_valid([server_data], valid_days=self.args_dict['valid_days'])

				# store the splited n client streaming data
				for j in range(self.n_participants):
					n_client_stream_train_data[j].append(n_client_train_data[j])
				stream_valid_data.append(Custom_Dataset(*server_valid_data, device=self.device))

			self.n_client_stream_train_data = n_client_stream_train_data
			self.stream_valid_data = stream_valid_data
			self.stream_step = len(multi_time_dataset)

			return multi_time_dataset
		
		elif name == 'electricity':
			try:
				data = np.load("./datasets/electricity.npz")
			except:
				raise RuntimeError("Electricity dataset did not exists")
			X, y = torch.tensor(data["X"],dtype=torch.float32), torch.tensor(data["y"],dtype=torch.float32)

			mean_y, std_y = torch.tensor(data["mean_y"],dtype=torch.float32), torch.tensor(data["std_y"],dtype=torch.float32)
			mean_y.to(self.device), std_y.to(self.device)

			self.mean_y = mean_y
			self.std_y = std_y

			train_set = Custom_Dataset(X, y, device=self.device)
			validation_set, test_set = train_set, train_set

			self.stream_train_dataset = Custom_Dataset(X, y, device=self.device)
			del X, y

			return train_set, validation_set, test_set

		elif name == 'hft':
			# read the data from txt file
			try:
				data = np.genfromtxt("./datasets/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore/NoAuction_Zscore_Training/Train_Dst_NoAuction_ZScore_CF_{}.txt"
										.format(self.args_dict['dataset_day'])).transpose()
			except: 
				raise RuntimeError("High Frequency Trading dataset did not exists")
			
			from utils.utils import split_dataset_to_streams_hft

			# split the whole dataset into streaming datasets
			stream_datasets = split_dataset_to_streams_hft(ori_data=data,stream_steps=self.args_dict['stream_step']
															,task_number=self.args_dict['task_number'])
			
			# collect to a whole dataset and compute the streaming indices
			offset = 0
			stream_indices = []
			for i in range(len(stream_datasets)):
				X,y = stream_datasets[i]
				stream_indices.append(np.arange(offset, (offset+len(X))))
				offset += len(X)

			train_X, train_y = torch.cat([X for (X,_) in stream_datasets]), torch.cat([y for (_,y) in stream_datasets])
			self.stream_train_dataset = Custom_Dataset(train_X, train_y, device=self.device)
			self.stream_indices = stream_indices

			return stream_datasets

		elif name == "sst":
			import torchtext.data as data
			text_field = data.Field(lower=True)
			from torch import long as torch_long
			label_field = LabelField(dtype = torch_long, sequential=False)


			import torchtext.datasets as datasets
			train_data, validation_data, test_data = datasets.SST.splits(text_field, label_field, root='datasets' ,fine_grained=True)

			if self.args_dict['split'] == 'uniform':
				from utils.utils import random_split
				indices_list = random_split(sample_indices=list(range(len(train_data))), m_bins=self.n_participants, equal=True)
			else:
				indices_list = powerlaw(list(range(len(train_data))), self.n_participants)
			ratios = [len(indices) / len(train_data) for indices in indices_list]

			train_datasets = split_torchtext_dataset_ratios(train_data, ratios)

			text_field.build_vocab(*(train_datasets + [validation_data, test_data]))
			label_field.build_vocab(*(train_datasets + [validation_data, test_data]))

			self.args['embed_num'] = len(text_field.vocab)
			self.args['class_num'] = len(label_field.vocab)
			
			# self.args.text_field = text_field
			# self.args.label_field = label_field

			return train_datasets, validation_data, test_data

		elif name == 'mr':

			import torchtext.data as data
			from utils import mydatasets

			text_field = data.Field(lower=True)
			from torch import long as torch_long
			label_field = LabelField(dtype = torch_long, sequential=False)
			# label_field = data.Field(sequential=False)

			train_data, dev_data = mydatasets.MR.splits(text_field, label_field, root='datasets', shuffle=False)

			validation_data, test_data = dev_data.split(split_ratio=0.5, random_state = random.seed(1234))
			
			if self.args_dict['split'] == 'uniform':
				from utils.utils import random_split
				indices_list = random_split(sample_indices=list(range(len(train_data))), m_bins=self.n_participants, equal=True)
			else:
				indices_list = powerlaw(list(range(len(train_data))), self.n_participants)
			
			ratios = [len(indices) / len(train_data) for indices in  indices_list]

			train_datasets = split_torchtext_dataset_ratios(train_data, ratios)

			text_field.build_vocab( *(train_datasets + [validation_data, test_data] ))
			label_field.build_vocab( *(train_datasets + [validation_data, test_data] ))


			self.args['embed_num'] = len(text_field.vocab)
			self.args['class_num'] = len(label_field.vocab)

			return train_datasets, validation_data, test_data

		elif name == 'imdb':

			from torch import long as torch_long
			# text_field = Field(tokenize = 'spacy', preprocessing = generate_bigrams) # generate_bigrams takes about 2 minutes
			text_field = Field(tokenize = 'spacy')
			label_field = LabelField(dtype = torch_long)

			dirname = 'datasets'

			from torch.nn.init import normal_
			from torchtext import datasets


			train_data, test_data = datasets.IMDB.splits(text_field, label_field) # 25000, 25000 samples each

			# use 5000 out of 25000 of test_data as the test_data
			test_data, remaining = test_data.split(split_ratio=0.2 ,random_state = random.seed(1234))
			
			# use 5000 out of the remaining 2000 of test_data as valid data
			valid_data, remaining = remaining.split(split_ratio=0.25 ,random_state = random.seed(1234))

			# train_data, valid_data = train_data.split(split_ratio=self.train_val_split_ratio ,random_state = random.seed(1234))

			indices_list = powerlaw(list(range(len(train_data))), self.n_participants)
			ratios = [len(indices) / len(train_data) for indices in  indices_list]

			train_datasets = split_torchtext_dataset_ratios(train_data, ratios)

			MAX_VOCAB_SIZE = 25_000

			text_field.build_vocab(*(train_datasets + [valid_data, test_data] ), max_size = MAX_VOCAB_SIZE, vectors = "glove.6B.100d",  unk_init = normal_)
			label_field.build_vocab( *(train_datasets + [valid_data, test_data] ))

			# INPUT_DIM = len(text_field.vocab)
			# OUTPUT_DIM = 1
			# EMBEDDING_DIM = 100

			PAD_IDX = text_field.vocab.stoi[text_field.pad_token]


			self.args['embed_num'] = text_field.vocab
			self.args['class_num'] = len(label_field.vocab)
			self.args['pad_idx'] = PAD_IDX

			return train_datasets, valid_data, test_data
		
		elif name == 'fraud':
			
			try:
				data = np.load('./datasets/ieee-fraud-detection/numeric_dataset.npz')
			except:
				raise RuntimeError("Fraud detection dataset did not exists")
			X, y = data['X_train'], data['y_train']

			# subsampling on unbalanced data 2:1
			fraud_indices = np.where(y == 1)[0]
			normal_indices = np.where(y == 0)[0]
			num_fraud = len(fraud_indices)

			normal_sampled_indices = np.random.choice(normal_indices,2*num_fraud, replace=False)

			# new sampled dataset
			dataset_indices = np.concatenate([fraud_indices, normal_sampled_indices])
			np.random.shuffle(dataset_indices)
			train = Custom_Dataset(torch.tensor(X[dataset_indices],dtype=torch.float32), torch.tensor(y[dataset_indices],dtype=torch.int64), device=self.device)

			train_indices, valid_indices = get_train_valid_indices(len(train), self.train_val_split_ratio, self.sample_size_cap)

			train_set = Custom_Dataset(train.data[train_indices], train.targets[train_indices], device=self.device)
			validation_set = Custom_Dataset(train.data[valid_indices],train.targets[valid_indices] , device=self.device)
			# test_set = Custom_Dataset(test.data, test.targets, device=self.device)

			self.stream_train_dataset = Custom_Dataset(train.data, train.targets, device=self.device)
			del train

			return train_set, validation_set, validation_set

from torchvision.datasets import MNIST
class FastMNIST(MNIST):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)		
		
		self.data = self.data.unsqueeze(1).float().div(255)
		from torch.nn import ZeroPad2d
		pad = ZeroPad2d(2)
		self.data = torch.stack([pad(sample.data) for sample in self.data])

		self.targets = self.targets.long()

		self.data = self.data.sub_(self.data.mean()).div_(self.data.std())
		# self.data = self.data.sub_(0.1307).div_(0.3081)
		# Put both data and targets on GPU in advance
		self.data, self.targets = self.data, self.targets
		print('MNIST data shape {}, targets shape {}'.format(self.data.shape, self.targets.shape))

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, target = self.data[index], self.targets[index]

		return img, target

from torchvision.datasets import CIFAR10, CIFAR100
class FastCIFAR10(CIFAR10):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		
		# Scale data to [0,1]
		from torch import from_numpy
		self.data = from_numpy(self.data)
		self.data = self.data.float().div(255)
		self.data = self.data.permute(0, 3, 1, 2)

		self.targets = torch.Tensor(self.targets).long()


		# https://github.com/kuangliu/pytorch-cifar/issues/16
		# https://github.com/kuangliu/pytorch-cifar/issues/8
		for i, (mean, std) in enumerate(zip((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))):
			self.data[:,i].sub_(mean).div_(std)

		# Put both data and targets on GPU in advance
		self.data, self.targets = self.data, self.targets
		print('CIFAR10 data shape {}, targets shape {}'.format(self.data.shape, self.targets.shape))

	def __getitem__(self, index):
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target class.
		"""
		img, target = self.data[index], self.targets[index]

		return img, target

def powerlaw(sample_indices, n_participants, alpha=1.65911332899, shuffle=False):
	# the smaller the alpha, the more extreme the division
	if shuffle:
		random.seed(1234)
		random.shuffle(sample_indices)

	from scipy.stats import powerlaw
	import math
	party_size = int(len(sample_indices) / n_participants)
	b = np.linspace(powerlaw.ppf(0.3, alpha), powerlaw.ppf(0.99, alpha), n_participants)
	shard_sizes = list(map(math.ceil, b/sum(b)*party_size*n_participants))
	indices_list = []
	accessed = 1
	for participant_id in range(n_participants):
		indices_list.append(sample_indices[accessed:accessed + shard_sizes[participant_id]])
		accessed += shard_sizes[participant_id]
		if len(indices_list[-1]) == 0:
			indices_list[-1] = sample_indices[0:1]
	new_shard_sizes = [len(tmp) for tmp in indices_list]
	return indices_list, np.array(new_shard_sizes)/np.sum(new_shard_sizes)


def get_train_valid_indices(n_samples, train_val_split_ratio, sample_size_cap=None):
	indices = list(range(n_samples))
	random.seed(1234)
	random.shuffle(indices)
	split_point = int(n_samples * train_val_split_ratio)
	train_indices, valid_indices = indices[:split_point], indices[split_point:]
	if sample_size_cap is not None:
		train_indices = indices[:min(split_point, sample_size_cap)]

	return  train_indices, valid_indices 


def split_torchtext_dataset_ratios(data, ratios):
	train_datasets = []
	while len(ratios) > 1:

		split_ratio = ratios[0] / sum(ratios)
		ratios.pop(0)
		train_dataset, data = data.split(split_ratio=split_ratio, random_state=random.seed(1234))
		train_datasets.append(train_dataset)
	train_datasets.append(data)
	return train_datasets


def generate_bigrams(x):
	n_grams = set(zip(*[x[i:] for i in range(2)]))
	for n_gram in n_grams:
		x.append(' '.join(n_gram))
	return x