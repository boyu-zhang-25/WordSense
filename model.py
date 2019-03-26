import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D  

from torch.nn import Parameter
from torch.nn import MSELoss, L1Loss, SmoothL1Loss, CrossEntropyLoss
from torch.distributions.binomial import Binomial
import torch.nn.utils.rnn as rnn_utils

import pandas as pd
import numpy as np
import math
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import precision_score, f1_score, recall_score

from tqdm import tqdm
from tqdm import tqdm_notebook as tqdm_n

from collections import Iterable, defaultdict
import itertools

from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder

# model for fine-tuning with ELMo
class Model(torch.nn.Module):
	def __init__(self, 
				output_size = 1,
				embedding_size = 1024,
				elmo_class = None,
				tuned_embed_size = 256,
				mlp_dropout = 0,
				rnn_hidden_size = 300,
				train_batch_size = 64,
				MLP_sizes = [300,300],
				embeddings = None,
				device = torch.device(type="cpu")):
		super().__init__()
		
		self.output_size = output_size
		self.elmo_class = elmo_class
		self.tuned_embed_size = tuned_embed_size 
		self.embedding_size = embedding_size
		self.device = device
		self.MLP_sizes = MLP_sizes
		self.rnn_hidden_size = rnn_hidden_size
		self.embeddings_data = embeddings

		## initialize MLP layers
		self.linear_maps = nn.ModuleDict()
		self.mlp_dropout =  nn.Dropout(mlp_dropout) 

		## initialize embedding-tuning MLP for elmo
		# 3 * 1024 -> 256 by MLP
		self.tuned_embed_MLP = nn.Linear(self.embedding_size * 3, self.tuned_embed_size)
		self.rnn = nn.LSTM(self.tuned_embed_size, self.rnn_hidden_size, num_layers = 2, bidirectional = True)
		self._init_MLP(self.tuned_embed_size * 2, self.MLP_sizes, self.output_size, param="factuality")
		
	def _init_MLP(self, input_size, hidden_sizes, output_size, param=None):
		'''
		Initialise MLP
		'''
		self.linear_maps[param] = nn.ModuleList()

		for h in hidden_sizes:
			linmap = torch.nn.Linear(input_size, h)
			linmap = linmap.to(self.device)
			self.linear_maps[param].append(linmap)
			input_size = h

		linmap = torch.nn.Linear(input_size, output_size)
		linmap = linmap.to(self.device)
		self.linear_maps[param].append(linmap)
		
	def _get_embedding(self, sentences):
		'''
		   Return the embeddings for all input sentences
		   @param: a list of list of words (sentences) 
		   @return: ELMo embeddins of every sentences (for all three ELMo layers each 1024 dim)
		   concatenates them to make a 3072 dim embedding vector
		   reduces the dimension to a lower number for e. 256 by passing it to an MLP layer.
		''' 

		# get ELMo embeddings of the sentences
		embeddings, masks = self.elmo_class.batch_to_embeddings(sentences)
		# torch.Size([number of sentences, 3 (layers), max sentence length, 1024 (word vector length)])
		print(embeddings.size())

		# pass to CUDA
		embeddings = embeddings.to(self.device)
		masks = masks.to(self.device)

		## Concatenate ELMO's 3 layers
		batch_size = embeddings.size()[0]
		max_length = embeddings.size()[2]
		embeddings = embeddings.permute(0,2,1,3) #dim0=batch_size, dim1=num_layers, dim2=sent_len, dim3=embedding-size
		embeddings = embeddings.contiguous().view(batch_size, max_length, -1)
			
		## Tune embeddings into lower dim:
		masks = masks.unsqueeze(2).repeat(1, 1, self.tuned_embed_size).byte()
		
		# 1024 -> 256 by MLP dimension reduction
		embeddings = self._tune_embeddings(embeddings)
		embeddings = embeddings*masks.float()

		return embeddings, masks

	def forward(self, sentences, indexes):
		
		embeddings, masks = self._get_inputs(sentences)

		##Run a Bi-LSTM:
		embeddings, (hn, cn) = self.rnn(embeddings)

		##convert masked tokens to zero after passing through Bi-lstm
		bilstm_masks = masks.repeat(1,1,2)
		embeddings = embeddings*bilstm_masks.float()

		## Extract index-span embeddings:
		span_input = self._extract_span_inputs(embeddings, indexes)
		#print("Span input shape: {}".format(span_input.shape))
		
		## Run MLP through the input:
		y_hat = self._run_regression(span_input, param="factuality", activation='relu')

		#y_hat = torch.exp(y_hat)*6 - 3.0
		
		return y_hat
		
	def _extract_span_inputs(self, embeddings, span_idxs):
		'''
		Extract embeddings for a span in the sentence
		
		For a mini-batch, keeps the length of span equal to the length 
		max span in that batch
		'''
		batch_size = embeddings.size()[0]
		span_lengths = [len(x) for x in span_idxs]
		max_span_len = max(span_lengths)
		
		if self.vocab: #glove
			span_embeds = torch.zeros((batch_size, max_span_len, self.embedding_size*2), 
								  dtype=torch.float, device=self.device)
		else: #elmo
			span_embeds = torch.zeros((batch_size, max_span_len, self.tuned_embed_size*2), 
								  dtype=torch.float, device=self.device)
		
		for sent_idx in range(batch_size):
			m=0
			for span_idx in span_idxs[sent_idx]:
				span_embeds[sent_idx][m] = embeddings[sent_idx][span_idx]
				m+=1
				
		return span_embeds
	
	def _tune_embeddings(self, embeddings):
		return torch.tanh(self.tuned_embed_MLP(embeddings))
	
	def _run_regression(self, h_last, param=None, activation=None):
		
		'''
		Runs MLP on input
		
		Note that for n hidden layers, there would be n+1 linear_maps
		'''

		for i, linear_map in enumerate(self.linear_maps[param]):
			if i:
				if activation == "sigmoid":
					h_last = torch.sigmoid(h_last)
					h_last = self.mlp_dropout(h_last)
				elif activation == "relu":
					h_last = F.relu(h_last)
					h_last = self.mlp_dropout(h_last)                  
				else:  ##else tanh
					h_last = torch.tanh(h_last)
					h_last = self.mlp_dropout(h_last)

			h_last = linear_map(h_last)
			
		return h_last