import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
import numpy as np
import math

from collections import Iterable, defaultdict
import itertools

from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder

# model for fine-tuning with ELMo
# support mini-batch
class Model(nn.Module):
	def __init__(self, 
				all_senses = None,
				# all_definitions = None, 
				output_size = 300, # output size of each sense vector [300, 1]
				embedding_size = 1024, # ELMo embedding size
				elmo_class = None,
				tuned_embed_size = 256,
				mlp_dropout = 0.3,
				lstm_hidden_size = 256,
				# encode_hidden_size = [512], # 1 hidden layer for definitio encoder
				# encode_input_size = 1024, # encode of definitions from WordNet
				# train_batch_size = 64,
				MLP_sizes = [512], # 1 hidden layer for fine-tuning sense vector
				device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
		super().__init__()
		
		# all senses and all definitions for all words
		# useful for all purposes
		self.all_senses = all_senses
		# self.all_definitions =  all_definitions
		self.elmo_class = elmo_class
		self.device = device

		# for dimension reduction 
		self.tuned_embed_size = tuned_embed_size 
		self.embedding_size = embedding_size

		# sizes of the fine-tuning MLP and LSTM
		self.MLP_sizes = MLP_sizes 
		self.output_size = output_size
		self.lstm_hidden_size = lstm_hidden_size

		## initialize fine-tuning MLP layers
		self.layers = nn.ModuleDict()
		self.mlp_dropout = nn.Dropout(mlp_dropout) 

		# dimension reduction for elmo
		# 3 * 1024 ELMo -> 1 * 256 
		self.dimension_reduction_MLP = nn.Linear(self.embedding_size * 3, self.tuned_embed_size)

		# construct a LSTM on top of ELMo
		self.wsd_lstm = nn.LSTM(self.tuned_embed_size, self.lstm_hidden_size, num_layers = 2, bidirectional = True)

		# build a 2-layer MLP on top of LSTM for fine-tuning
		self._init_MLP(self.tuned_embed_size * 2, self.MLP_sizes, self.output_size, param = "word_sense")

		# randomly initialize all vectors for definition embeddings
		def_dict = self._init_definition_embeddings(self.output_size, param = "definition_embedding")
		self.definition_embeddings = nn.ParameterDict(def_dict)

	# initialize all the definition embeddings for all words
	# put into a matrix for each word
	def _init_definition_embeddings(self, output_size, param = None):

		def_dict = {}

		for word in self.all_senses.keys():

			def_tuple = tuple([torch.randn(output_size, 1) for m in range(len(self.all_senses[word]))])
			def_matrix = nn.Parameter(torch.cat(def_tuple, 1))
			def_dict[word] = def_matrix

		return def_dict

	def _init_MLP(self, input_size, hidden_sizes, output_size, param = None):
		'''
		Initialize a 2-layer MLP on top of ELMo
		w1: input_size * hidden_sizes[0]
		w2: hidden_sizes[0] * output_size
		'''

		# dict for fine-tuning MLP structures
		self.layers[param] = nn.ModuleList()

		# initialize MLP
		for h in hidden_sizes:

			layer = torch.nn.Linear(input_size, h)
			layer = layer.to(self.device)

			# append to the fine-tuning MLP
			self.layers[param].append(layer)
			# update dimension
			input_size = h

			# ReLU activation after linear layer
			layer = nn.ReLU()
			layer = layer.to(self.device)
			self.layers[param].append(layer)            

		output_layer = torch.nn.Linear(input_size, output_size)
		output_layer = output_layer.to(self.device)
		self.layers[param].append(output_layer)
		
	def _get_embedding(self, sentences):
		'''
			@param: a list of list of words (sentences)

			@return: 
			ELMo embeddins of every sentences (for all three ELMo layers each 1024 dim)
			concatenates them to make a 3072 dim embedding vector
			reduces the dimension to a lower number (256)
		''' 

		# get ELMo embeddings of the sentences
		# torch.Size([number of sentences, 3 (layers), max sentence length, 1024 (word vector length)])
		embeddings, masks = self.elmo_class.batch_to_embeddings(sentences)
		# print('\nOriginal ELMo embeddings size of word sense: {}, mask: {}'.format(embeddings.size(), masks.size()))

		# pass to CUDA
		embeddings = embeddings.to(self.device)
		masks = masks.to(self.device)

		# old: [batch_size, num_layers, max_sent_len, word_embedding_size]
		# new: [max_sent_len, batch_size, num_layers, word_embedding_size]
		batch_size = embeddings.size()[0]
		max_length = embeddings.size()[2]
		embeddings = embeddings.permute(2, 0, 1, 3)
		# masks = masks.permute(1, 0)

		# concatenate 3 layers and reshape
		# [max_sent_len, batch_size, 3 * 1024]
		embeddings = embeddings.contiguous().view(max_length, batch_size, -1)

		# Tune embeddings into lower dim
		# masks = masks.unsqueeze(2).repeat(1, 1, self.tuned_embed_size).byte()
		
		# 3 * 1024 -> 256 by dimension reduction
		embeddings = self._tune_embeddings(embeddings, param = 'word_sense')
		# embeddings = embeddings * masks.float()

		# print('Word sense embedding size after dimension reduction: {}, mask: {}'.format(embeddings.size(), masks.size()))

		return embeddings, masks

	# get the embeddings for literal definitions in WordNet
	# usesless
	def _get_embedding_def_old(self, word_lemma):
		'''
			@param: a list target word in the dataset
			@return:
			a list of 
			all definition embeddings for all target words
		'''
		embedding_def_results = []
		masks_def_results = []

		for word in word_lemma:

			# all definitions for the current target word in a list
			definitions = self.all_definitions[word]
			# print('\ndef size: {}'.format(len(definitions)))

			# get ELMo embeddings of the definitions
			embeddings_def, masks_def = self.elmo_class.batch_to_embeddings(definitions)
			# print('ELMo embeddings_def size: {}'.format(embeddings_def.size()))

			# pass to CUDA
			embeddings_def = embeddings_def.to(self.device)
			masks_def = masks_def.to(self.device)

			# old: [batch_size, num_layers, max_sent_len, word_embedding_size]
			# new: [max_sent_len, batch_size, num_layers, word_embedding_size]
			batch_size = embeddings_def.size()[0]
			max_length = embeddings_def.size()[2]
			embeddings_def = embeddings_def.permute(2, 0, 1, 3)
			masks_def = masks_def.permute(1, 0)
			# print('permute embeddings_def size: {}'.format(embeddings_def.size()))

			# concatenate 3 layers and reshape
			# [max_sent_len, batch_size, 3 * 1024]
			embeddings_def = embeddings_def.contiguous().view(max_length, batch_size, -1)
			# print('con embeddings_def size: {}'.format(embeddings_def.size()))

			# Tune embeddings into lower dim
			masks_def = masks_def.unsqueeze(2).repeat(1, 1, self.tuned_embed_size).byte()
		
			# 3 * 1024 -> 256 by dimension reduction
			embeddings_def = self._tune_embeddings(embeddings_def, param = 'definition')
			embeddings_def = embeddings_def * masks_def.float()
			# print('embeddings_def size: {}'.format(embeddings_def.size()))

			# append to the whole list
			embedding_def_results.append(embeddings_def)
			masks_def_results.append(masks_def)

		return embedding_def_results, masks_def_results

	# get the embeddings for literal definitions in WordNet
	def get_embedding_def(self, word_lemma):
		'''
			@param: a list target word in the dataset
			@return:
			a list of all definition embeddings for all target words
		'''
		embedding_def_results = [self.definition_embeddings[word] for word in word_lemma]
		return embedding_def_results

	# forward propagation selected sentence and definitions
	def forward(self, sentences, word_idx):
		
		# preserve word lemma for future use
		word_lemma = [sentences[i][word_idx[i]] for i in range(len(word_idx))]
		# print("word lemma: {}".format(word_lemma))

		# get the dimension-reduced ELMo embeddings
		embeddings, masks = self._get_embedding(sentences)

		# Run a Bi-LSTM and get the sense embeddings
		# (seq_len, batch, num_directions * hidden_size)
		embeddings, (hn, cn) = self.wsd_lstm(embeddings)

		# convert masked tokens to zero after passing through Bi-lstm
		# bilstm_masks = masks.repeat(1, 1, 2)
		# [sentence_length, batch_size, word_vector_length]
		# embeddings = embeddings * bilstm_masks.float()
		# print('\nWord sense embedding size after bi-LSTM: {}'.format(embeddings.size()))

		# Extract the new word embeddings by index
		# batch_size words, each has length 512
		new_word_embs = self._extract_word(embeddings, word_idx)
		
		# Run fine-tuning MLP on new word embeddings and get sense embeddings
		# batch_size words, each has length 10 for 10 possible senses
		sense_embedding = self._run_fine_tune_MLP(new_word_embs, word_lemma, param = 'word_sense')

		# run definition encoder
		# definition_embedding = self._encode_definitions(definition_embedding, word_lemma, param = 'definition')

		# get the definition embeddings
		# definition_embedding = self.get_embedding_def(word_lemma)

		return sense_embedding
		
	def _extract_word(self, embeddings, word_idx):
		'''
		Extract the new word embeddings by index
		'''
		batch_size = embeddings.size()[1]
		new_word_embs = []
		for i in range(batch_size):
			embeddings[word_idx[i], i, :] = embeddings[word_idx[i], i, :].to(self.device)
			new_word_embs.append(embeddings[word_idx[i], i, :])

		return new_word_embs

	# 3 * 1024 -> 256 by dimension reduction
	def _tune_embeddings(self, embeddings, param = None):

		if param == 'definition':
			return torch.tanh(self.encode_dimension_reduction(embeddings))
		else:
			return torch.tanh(self.dimension_reduction_MLP(embeddings))
	
	def _run_fine_tune_MLP(self, new_word_embs, word_lemma, param = None):
		
		'''
		Runs MLP on all word embeddings
		'''
		results = []
		for idx, word_vec in enumerate(new_word_embs):

			# run the fine-tuning MLP and get the sense vector
			for layer in self.layers[param]:
				word_vec = layer(word_vec)

			results.append(word_vec)
			print('\nWord lemma: {}\nWord sense embedding size: {}\nAll its senses: {}'.format(word_lemma[idx], word_vec.size(), self.all_senses[word_lemma[idx]]))

		return results

	# used to convert WordNet definitions to vectors
	# useless
	def _encode_definitions(self, definition_embeddings, word_lemma, param = None):
		
		'''
		Runs MLP to get definition embeddings
		Warning: code hard to read
		index is mess!!!!!!!!
		'''
		results = []

		# all the definitions of all input words
		for idx, definitions in enumerate(definition_embeddings):

			def_of_one_word = []

			# every definitions of a word from WordNet
			for i in range(definitions.size()[1]):

				def_embs = []

				# every word in the definition
				for j in range(definitions.size()[0]):

					def_vec = definitions[j, i, :]

					# run the fine-tuning MLP and get the sense vector
					for layer in self.layers[param]:
						def_vec = layer(def_vec)
						def_vec = self.mlp_dropout(def_vec)

					def_embs.append(def_vec)

				def_of_one_word.append(def_embs)
				print('\nWord lemma: {}\nCurrent definition Index (according to WordNet): {} \nTotal definition embedding size: {}'.format(word_lemma[idx], i, (len(def_embs), len(def_embs[0]))))

			results.append(def_of_one_word)

		return results
