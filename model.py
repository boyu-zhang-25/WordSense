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
class Model(nn.Module):
	def __init__(self, 
				all_senses = None,
				output_size = 300, # output size of each sense vector [300, 1]
				embedding_size = 1024, # ELMo embedding size
				elmo_class = None,
				tuned_embed_size = 256,
				mlp_dropout = 0,
				lstm_hidden_size = 256,
				MLP_sizes = [512], # 1 hidden layer for fine-tuning sense vector
				device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
		super().__init__()
		
		# all senses and all definitions for all words
		# useful for all purposes
		self.all_senses = all_senses
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
		self.dimension_reduction_MLP = nn.Linear(self.embedding_size * 3, self.tuned_embed_size).to(self.device)

		# construct a LSTM on top of ELMo
		self.wsd_lstm = nn.LSTM(self.tuned_embed_size, self.lstm_hidden_size, num_layers = 2, bidirectional = True).to(self.device)

		# build a 2-layer MLP on top of LSTM for fine-tuning
		self._init_MLP(self.tuned_embed_size * 2, self.MLP_sizes, self.output_size, param = "word_sense")

		# randomly initialize all vectors for definition embedding
		def_dict = self._init_definition_embeddings(self.output_size, param = "definition_embedding")
		self.definition_embeddings = nn.ParameterDict(def_dict)

	# initialize all the definition embedding for all words
	# put into a matrix for each word
	def _init_definition_embeddings(self, output_size, param = None):

		def_dict = {}

		for word in self.all_senses.keys():

			def_tuple = tuple([torch.randn(output_size, 1) for m in range(len(self.all_senses[word]))])
			def_matrix = nn.Parameter(torch.cat(def_tuple, 1), requires_grad = True)
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
		
	def _get_embedding(self, sentence):
		'''
			@param: a list of words (a sentence)

			@return: 
			ELMo embeddins of the sentence (for all three ELMo layers each 1024 dim)
			concatenates them to make a 3072 dim embedding vector
			reduces the dimension to a lower number (256)
		''' 

		# get ELMo embedding of the sentence
		# torch.Size([3 (layers), sentence length, 1024 (word vector length)])
		embedding = torch.from_numpy(self.elmo_class.embed_sentence(sentence))
		print('119: {}'.format(embedding.requires_grad))

		# pass to CUDA
		embedding = embedding.to(self.device)

		# old: [3 (layers), sentence_length, 1024 (word vector length)]
		# new: [sentence_length, 3 (layers), word_embedding_size]
		embedding = embedding.permute(1, 0, 2)
		print('126: {}'.format(embedding.requires_grad))

		# concatenate 3 layers and reshape
		# [sentence_length, batch_size, 3 * 1024]
		batch_size = 1
		sentence_length = embedding.size()[0]
		embedding = embedding.contiguous().view(sentence_length, batch_size, -1)
		print('132: {}'.format(embedding.requires_grad))
		
		# [sentence_length, batch_size, 256]
		embedding = self._tune_embeddings(embedding, param = 'word_sense')
		print('139: {}'.format(embedding.requires_grad))
		# print('em require: {}'.format(embedding.requires_grad))

		return embedding

	# get the embedding for literal definitions in WordNet
	# usesless
	'''
	def _get_embedding_def_old(self, word_lemma):

		embedding_def_results = []
		masks_def_results = []

		for word in word_lemma:

			# all definitions for the current target word in a list
			definitions = self.all_definitions[word]
			# print('\ndef size: {}'.format(len(definitions)))

			# get ELMo embedding of the definitions
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

			# Tune embedding into lower dim
			masks_def = masks_def.unsqueeze(2).repeat(1, 1, self.tuned_embed_size).byte()
		
			# 3 * 1024 -> 256 by dimension reduction
			embeddings_def = self._tune_embeddings(embeddings_def, param = 'definition')
			embeddings_def = embeddings_def * masks_def.float()
			# print('embeddings_def size: {}'.format(embeddings_def.size()))

			# append to the whole list
			embedding_def_results.append(embeddings_def)
			masks_def_results.append(masks_def)

		return embedding_def_results, masks_def_results
	'''
	'''
	# get the embedding for literal definitions in WordNet
	def get_embedding_def(self, word_lemma):

		embedding_def_results = [self.definition_embeddings[word] for word in word_lemma]
		return embedding_def_results
	'''

	# forward propagation selected sentence and definitions
	def forward(self, sentence, word_idx):
		
		# preserve word lemma for future use
		word_lemma = sentence[word_idx]

		# get the dimension-reduced ELMo embedding
		embedding = self._get_embedding(sentence)
		# print('em2 require: {}'.format(embedding.requires_grad))

		# Run a Bi-LSTM and get the sense embedding
		# (seq_len, batch, num_directions * hidden_size)
		embedding_new, (hn, cn) = self.wsd_lstm(embedding)
		print('213: {}'.format(embedding_new.requires_grad))

		# Extract the new word embedding by index
		word_embedding = embedding_new[word_idx, :, :]
		print('217: {}'.format(word_embedding.requires_grad))

		# Run fine-tuning MLP on new word embedding and get sense embedding
		# batch_size words, each has length 10 for 10 possible senses
		sense_embedding = self._run_fine_tune_MLP(word_embedding, word_lemma, param = 'word_sense')
		sense_embedding = sense_embedding.view(self.output_size, -1)
		print('223: {}'.format(sense_embedding.requires_grad))

		return sense_embedding
	'''		
	def _extract_word(self, embedding, word_idx):

		batch_size = embedding.size()[1]
		new_word_embs = nn.ParameterList()
		for i in range(batch_size):
			embedding[word_idx[i], i, :] = embedding[word_idx[i], i, :].to(self.device)
			new_word_embs.append(embedding[word_idx[i], i, :])

		return new_word_embs
	'''

	# 3 * 1024 -> 256 by dimension reduction
	def _tune_embeddings(self, embedding, param = None):

		if param == 'definition':
			return torch.tanh(self.encode_dimension_reduction(embedding))
		else:
			return torch.tanh(self.dimension_reduction_MLP(embedding))
	
	def _run_fine_tune_MLP(self, word_embedding, word_lemma, param = None):
		
		'''
		Runs MLP on the target word embedding
		'''
		for layer in self.layers[param]:
			word_embedding = layer(word_embedding)
			print('253 loop: {}'.format(word_embedding.requires_grad))

		# print('\nWord lemma: {}\nWord sense embedding size: {}\nAll its senses: {}'.format(word_lemma, word_embedding.size(), self.all_senses[word_lemma]))
		return word_embedding

	# used to convert WordNet definitions to vectors
	# useless
	'''
	def _encode_definitions(self, definition_embeddings, word_lemma, param = None):
		
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
				# print('\nWord lemma: {}\nCurrent definition Index (according to WordNet): {} \nTotal definition embedding size: {}'.format(word_lemma[idx], i, (len(def_embs), len(def_embs[0]))))

			results.append(def_of_one_word)

		return results
	'''