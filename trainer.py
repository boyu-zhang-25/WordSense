import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.nn import Parameter
from torch.nn import MSELoss, CrossEntropyLoss, CosineEmbeddingLoss
import torch.nn.utils.rnn as rnn_utils

import pandas as pd
import numpy as np
import math

from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm_n
import itertools

from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
from model import *
from nltk.corpus import wordnet as wn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device: {}'.format(device))

class Trainer(object):

	def __init__(self, 
				optimizer_class = torch.optim.Adam,
				optim_wt_decay = 0.,
				epochs = 5,
				regularization = None,
				loss_type = 'cos',
				all_senses = None,
				all_supersenses = None, 
				elmo_class = None, # for sense vector in the model
				file_path = "",
				device = device,
				**kwargs):

		## Training parameters
		self.epochs = epochs
		self.elmo_class = elmo_class

		## optimizer 
		self.optimizer = optimizer_class
		self.optim_wt_decay = optim_wt_decay
		
		# taget word index and senses list
		self.all_senses = all_senses
		self.all_supersenses = all_supersenses

		self._init_kwargs = kwargs
		self.device = device

		# loss to calculate the similarity betwee two tensors
		if loss_type == 'mse':
			self.loss = MSELoss().to(self.device)
		else:
			self.loss = CosineEmbeddingLoss().to(self.device)
		
		'''
		if regularization == "l1":
			self.regularization = L1Loss()
		elif regularization == "smoothl1":
			self.regularization = SmoothL1Loss()
		else:
			self.regularization = None
		'''
		self.best_model_file =  file_path + "word_sense_model_.pth"		
		'''
		if self.regularization:
			self.regularization = self.regularization.to(self.device)
		'''

	# generate new model
	def _initialize_trainer_model(self):
		# print(len(self.all_senses.keys()))
		self._model = Model(device = self.device, all_senses = self.all_senses, elmo_class = self.elmo_class, all_supersenses = self.all_supersenses)
		self._model = self._model.to(self.device)

		print("#############   Model Parameters   ##############")
		for name, param in self._model.named_parameters():     
			if param.requires_grad:
				print(name, param.size())
		print("##################################################")
	
	# supports batch input
	def train(self, train_X, train_Y, train_idx, dev_X, dev_Y, dev_idx, **kwargs):

		# train_Y is the annotator response
		self.train_X, self.train_Y = train_X, train_Y
		self.dev_X, self.dev_Y = dev_X, dev_Y
			
		self._initialize_trainer_model()  
		# old = self._model.definition_embeddings['spring'].clone().detach()

		# trainer setup
		parameters = [p for p in self._model.parameters() if p.requires_grad]
		optimizer = self.optimizer(parameters, weight_decay = self.optim_wt_decay, **kwargs)

		num_train = len(self.train_X)
		# num_dev = len(self.dev_X)
		
		# dev_accs = []
		best_loss = float('inf')
		best_r = -float('inf')
		train_losses = []
		dev_losses = []
		dev_rs = []
		bad_count = 0
		
		for epoch in range(self.epochs):
			
			# loss for the current iteration
			batch_losses = []

			# Turn on training mode which enables dropout.
			self._model.train()			
			tqdm.write("[Epoch: {}/{}]".format((epoch + 1), self.epochs))
			
			# time print
			pbar = tqdm_n(total = num_train)
			s_list = []

			# SGD batch = 1
			for idx, sentence in enumerate(self.train_X):
				
				# Zero grad
				optimizer.zero_grad()

				# the target word
				word_idx = train_idx[idx]
				word_lemma = '____' + sentence[word_idx]

				# model output
				sense_vec = self._model.forward(sentence, word_idx)
				# print(sense_vec)
				# s_list.append(sense_vec)

				# calculate loss pair-wise: sense vector and definition vector
				# accumulative loss
				loss = torch.zeros(1).to(self.device)

				# check all definitions in the annotator response for the target word
				for i, response in enumerate(self.train_Y[idx]):

					# slice the particular definition for gradient calculation
					definition_vec = self._model.definition_embeddings[word_lemma][:, i].view(1, -1)
						
					# find the supersense
					synset = self.all_senses[word_lemma][i]
					supersense = wn.synset(synset).lexname().replace('.', '_')
					supersense_vec = self._model.supersense_embeddings[supersense].view(1, -1)

					if response:

						# if annotator response is True: increase the cosine similarity
						# loss between sense embeddings and the definition embeddings
						loss += self.loss(sense_vec, definition_vec, torch.ones(1).to(self.device))

						# loss between the supersense and the sensen embeddings
						loss += self.loss(sense_vec, supersense_vec, torch.ones(1).to(self.device))

						# loss between the supersense and the definition embeddings
						# they should always be similar
						loss += self.loss(definition_vec, supersense_vec, torch.ones(1).to(self.device))

					else:

						# if annotator response is False
						# decrease the cosine similarity
						loss += self.loss(sense_vec, definition_vec, -torch.ones(1).to(self.device))
						loss += self.loss(sense_vec, supersense_vec, -torch.ones(1).to(self.device))

						# loss between the supersense and the definition embeddings
						# they should always be similar
						loss += self.loss(definition_vec, supersense_vec, torch.ones(1).to(self.device))

				# individual definition tensor gradient update
				# also backprop the accumulative loss for the predicted sense embeddings
				loss.backward()
				optimizer.step()

				'''
				for o in range(len(s_list)):
					for g in range(len(s_list)):
						print("o: {}, g: {}\n{}".format(o, g, torch.eq(s_list[o], s_list[g])))
				'''

				# record training loss for each example
				current_loss = loss.detach().item()
				batch_losses.append(current_loss)
				pbar.update(1)
					
			pbar.close()
			
			# calculate the training loss of the current epoch
			curr_train_loss = np.mean(batch_losses)
			print("Epoch: {}, Mean Training Loss: {}".format(epoch + 1, curr_train_loss))
			train_losses.append(curr_train_loss)

			# dev loss of the current epoch
			curr_dev_loss = np.mean(self.dev_loss(dev_X, dev_Y, dev_idx))
			print("Epoch: {}, Mean Dev Loss: {}".format(epoch + 1, curr_dev_loss))
			dev_losses.append(curr_dev_loss)

			# save the best model by dev 
			if curr_dev_loss < best_loss:
				with open(self.best_model_file, 'wb') as f:
					torch.save(self._model.state_dict(), f)
				best_loss = curr_dev_loss
			
			# early stopping
			'''
			if epoch:
				if (abs(curr_train_loss - train_losses[-1]) < 0.0001):
					break
			'''

		# print(torch.eq(old, self._model.definition_embeddings['spring']))
		return train_losses, dev_losses, dev_rs

	# for dev and test
	def dev_loss(self, dev_X, dev_Y, dev_idx):

		dev_losses = []

		for idx, sentence in enumerate(dev_X):

			# the target word
			word_idx = dev_idx[idx]
			word_lemma = '_____' + sentence[word_idx]

			# model output
			sense_vec = self._model.forward(sentence, word_idx)
			loss = 0.0

			# only count the loss for known words
			if self.all_senses.get(word_lemma, 'not_exist') != 'not_exist':

				# check all definitions in the annotator response for the target word
				for i, response in enumerate(dev_Y[idx]):

					# slice the particular definition for gradient calculation
					definition_vec = self._model.definition_embeddings[word_lemma][:, i].view(1, -1)
						
					# find the supersense
					synset = self.all_senses[word_lemma][i]
					supersense = wn.synset(synset).lexname().replace('.', '_')
					supersense_vec = self._model.supersense_embeddings[supersense].view(1, -1)

					if response:
						loss += self.loss(sense_vec, definition_vec, torch.ones(1).to(self.device))
						loss += self.loss(sense_vec, supersense_vec, torch.ones(1).to(self.device))
						loss += self.loss(definition_vec, supersense_vec, torch.ones(1).to(self.device))

					else:
						loss += self.loss(sense_vec, definition_vec, -torch.ones(1).to(self.device))
						loss += self.loss(sense_vec, supersense_vec, -torch.ones(1).to(self.device))
						loss += self.loss(definition_vec, supersense_vec, torch.ones(1).to(self.device))

				# record training loss for each example
				dev_loss = loss.detach().item()
				dev_losses.append(dev_loss)

			else:
				print('new word in dev: {}\n sentence:{}'.format(word_lemma, sentence))
				# dev_losses.append(-1)

		return dev_losses

