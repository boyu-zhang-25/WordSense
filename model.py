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
				output_size = 10, # max 10 possible different senses
				embedding_size = 1024,
				elmo_class = None,
				tuned_embed_size = 256,
				mlp_dropout = 0,
				rnn_hidden_size = 300,
				train_batch_size = 64,
				MLP_sizes = [300, 300],
				embeddings = None,
				device = torch.device(type="cpu")):
		super().__init__()
		
		self.output_size = output_size
		self.elmo_class = elmo_class
		self.tuned_embed_size = tuned_embed_size 
		self.embedding_size = embedding_size
		self.device = device
		self.MLP_sizes = MLP_sizes # sizes of the fine-tuning MLP
		self.rnn_hidden_size = rnn_hidden_size
		self.embeddings_data = embeddings

		## initialize MLP layers
		self.layers = nn.ModuleDict()
		self.mlp_dropout =  nn.Dropout(mlp_dropout) 

		## initialize embedding-tuning MLP for elmo
		# for each sentence:
		# 3 * 1024 ELMo -> 1 * 256 by 1-layer MLP
		self.tuned_embed_MLP = nn.Linear(self.embedding_size * 3, self.tuned_embed_size)

		# construct a LSTM on top of ELMo
		self.rnn = nn.LSTM(self.tuned_embed_size, self.rnn_hidden_size, num_layers = 2, bidirectional = True)

		# build a 3-layer MLP on top of ELMo for fine-tuning
		self._init_MLP(self.tuned_embed_size * 2, self.MLP_sizes, self.output_size, param = "WSD")
		
	def _init_MLP(self, input_size, hidden_sizes, output_size, param = None):
		'''
		Initialize a 3-layer MLP on top of ELMo
		w1: input_size * hidden_sizes[0]
		w2: hidden_sizes[0] * hidden_sizes[1]
		w3: hidden_sizes[1] * output_size
		'''

		# dict for fine-tuning MLP structures
		self.layers[param] = nn.ModuleList()

		# initialize all 3 layers
		for h in hidden_sizes:

			layer = torch.nn.Linear(input_size, h)
			layer = layer.to(self.device)

			# append to the fine-tuning MLP
			self.layers[param].append(layer)
			# update dimension
			input_size = h

		# the output layer structure
		layer = torch.nn.Linear(input_size, output_size)
		layer = layer.to(self.device)

		self.layers[param].append(layer)
		
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
		print('Original embeddings size: {}'.format(embeddings.size()))

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
		
		embeddings, masks = self._get_embedding(sentences)

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
		
		Note that for n hidden layers, there would be n+1 layers
		'''

		for i, linear_map in enumerate(self.layers[param]):
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

class BaseTrainer(object):
	'''

	data_name = 'megaverid' or 'ithappen'

	'''
	def __init__(self, 
				 optimizer_class = torch.optim.Adam,
				 optim_wt_decay=0.,
				 epochs=3,
				 train_batch_size = 64,
				 data_name = None,
				 pretrain_data_name = None,
				 predict_batch_size = 128,
				 pretraining=False,
				 regularization = None,
				 file_path = "",
				device = torch.device(type="cpu"),
				**kwargs):

		## Training parameters
		self.epochs = epochs
		self.train_batch_size = train_batch_size
		self.predict_batch_size = predict_batch_size
		self.pretraining = pretraining
		self.data_name = data_name
		self.pretrain_data_name = pretrain_data_name

		## optimizer 
		self._optimizer_class = optimizer_class
		self.optim_wt_decay = optim_wt_decay
		
		self._init_kwargs = kwargs
		self.device = device
		
		if regularization == "l1":
			self.regularization = L1Loss()
		elif regularization == "smoothl1":
			self.regularization = SmoothL1Loss()

		else:
			self.regularization = None


		## Model file name
		self.best_model_file =  file_path + "model_" + data_name + \
									"_" + str(optim_wt_decay) + \
									"_" + "pre_" + str(pretrain_data_name) + \
									"_" + str(regularization) + "_.pth"

		self.smooth_loss = SmoothL1Loss().to(self.device)
		self.l1_loss = L1Loss().to(self.device)

		if self.regularization:
			self.regularization = self.regularization.to(self.device)


	def _initialize_trainer_model(self):
		self._model = BaseModel(device=self.device,
											 **self._init_kwargs)
		
		self._model = self._model.to(self.device)
		

	def _custom_loss(self, predicted, actual, pretrain_x, pretrain_actual):
		'''

		Inputs:
		```````1. predicted: model predicted values
				2. actual: actual values
				3. pretrain_data: 
				4. pretrain_preds: predictions on pretrain_data based on original pretraining

		'''
		actual_torch = torch.from_numpy(np.array(actual)).float().to(self.device)

		domain_loss = self.smooth_loss(predicted.squeeze(), actual_torch)

		if not self.pretraining: 
			return domain_loss

		else:        
			pretrain_curr_preds  = self.predict_grad(pretrain_x)
			pretrain_actual_preds = torch.from_numpy(np.array(pretrain_actual)).float().to(self.device)
			generic_loss = self.regularization(pretrain_curr_preds, pretrain_actual_preds)

			beta = 0.5
			print("Domain_loss: {} Beta: {}, generic_loss_beta: {}".format(domain_loss.item(), beta, beta*generic_loss.item()))
		return domain_loss + beta*generic_loss

	
	def fit(self, train_X, train_Y, dev, pretrain_x, pretrain_actual, **kwargs):
		self._X,  self._Y = train_X, train_Y
		
		self.pretrain_x = pretrain_x
		self.pretrain_actual = pretrain_actual

		if self.data_name != "megaverid":
			dev_x, dev_y = dev
			
		self._initialize_trainer_model() 
		
		print("########## .   Model Parameters   ##############")
		for name,param in self._model.named_parameters():     
			if param.requires_grad:
				print(name, param.shape)
		print("##############################################") 

		parameters = [p for p in self._model.parameters() if p.requires_grad]
		optimizer = self._optimizer_class(parameters, 
											weight_decay = self.optim_wt_decay,
										**kwargs)
		
		total_obs = len(self._X)
		#dev_obs = len(dev_x)
		
		#dev_accs = []
		best_loss = float('inf')
		best_r = -float('inf')
		train_losses = []
		dev_losses = []
		dev_rs = []
		bad_count = 0
		
		for epoch in range(self.epochs):
			batch_losses = []
			# Turn on training mode which enables dropout.
			self._model.train()
			
			bidx_i = 0
			bidx_j =self.train_batch_size
			
			tqdm.write("Running Epoch: {}".format(epoch+1))
			
			#time print
			pbar = tqdm_n(total = total_obs//self.train_batch_size)
			
			while bidx_j < total_obs:
				words = [words for words, spans in self._X[bidx_i:bidx_j]]
				spans = [spans for words, spans in self._X[bidx_i:bidx_j]]
				
				##Zero grad
				optimizer.zero_grad()

				##Calculate Loss
				model_out  = self._model(words, spans)   
				
				if self.pretraining:
					curr_loss = self._custom_loss(model_out, self._Y[bidx_i:bidx_j], pretrain_x, pretrain_actual)
				else:
					curr_loss = self._custom_loss(model_out, self._Y[bidx_i:bidx_j], None, None)
					
				batch_losses.append(curr_loss.detach().item())
				
				##Backpropagate
				curr_loss.backward()

				#plot_grad_flow(self._model.named_parameters())
				optimizer.step()
				bidx_i = bidx_j
				bidx_j = bidx_i + self.train_batch_size
				
				if bidx_j >= total_obs:
					words = [words for words, spans in self._X[bidx_i:bidx_j]]
					spans = [spans for words, spans in self._X[bidx_i:bidx_j]]
					##Zero grad
					optimizer.zero_grad()

					##Calculate Loss
					model_out  = self._model(words, spans)   
					
					if self.pretraining:
						curr_loss = self._custom_loss(model_out, self._Y[bidx_i:bidx_j], pretrain_x, pretrain_actual)
					else:
						curr_loss = self._custom_loss(model_out, self._Y[bidx_i:bidx_j], None, None)
					
					batch_losses.append(curr_loss.detach().item())
					##Backpropagate
					curr_loss.backward()

					#plot_grad_flow(self.named_parameters())
					optimizer.step()
					
				pbar.update(1)
					
			pbar.close()
			
			#print(batch_losses)
			curr_train_loss = np.mean(batch_losses)
			print("Epoch: {}, Mean Train Loss across batches: {}".format(epoch+1, curr_train_loss))
			
			if self.data_name == "megaverid":
				if curr_train_loss < best_loss:
					with open(self.best_model_file, 'wb') as f:
						torch.save(self._model.state_dict(), f)
					best_loss = curr_train_loss
				
				## Stop training when loss converges
				if epoch:
					if (abs(curr_train_loss - train_losses[-1]) < 0.0001):
						break

				train_losses.append(curr_train_loss)

			else:
				curr_dev_loss, curr_dev_preds = self.predict(dev_x, dev_y)
				curr_dev_r = pearsonr(curr_dev_preds.cpu().numpy(), dev_y)
				print("Epoch: {}, Mean Dev Loss across batches: {}, pearsonr: {}".format(epoch+1, 
																						curr_dev_loss,
																						curr_dev_r[0]))
				
				# if curr_dev_loss < best_loss:
				#     with open(self.best_model_file, 'wb') as f:
				#         torch.save(self._model.state_dict(), f)
				#     best_loss = curr_dev_loss


				if curr_dev_r[0] > best_r:
					with open(self.best_model_file, 'wb') as f:
						torch.save(self._model.state_dict(), f)
					best_r = curr_dev_r[0]
			

				# if epoch:
				#     if curr_dev_loss > dev_losses[-1]:
				#         bad_count+=1
				#     else:
				#         bad_count=0

				if epoch:
					if curr_dev_r[0] < dev_rs[-1]:
						bad_count+=1
					else:
						bad_count=0

				if bad_count >=3:
					break

				dev_rs.append(curr_dev_r[0])
				dev_losses.append(curr_dev_loss)
				train_losses.append(curr_train_loss)
			

		# print("Epoch: {}, Converging-Loss: {}".format(epoch+1, curr_mean_loss))

		return train_losses, dev_losses, dev_rs

	def predict_grad(self, data_x):
		'''
		Predictions with gradients and computation graph intact
		'''     
		bidx_i = 0
		bidx_j = self.predict_batch_size
		total_obs = len(data_x)
		yhat = torch.zeros(total_obs).to(self.device)

		while bidx_j < total_obs:
			words = [words for words, spans in data_x[bidx_i:bidx_j]]
			spans = [spans for words, spans in data_x[bidx_i:bidx_j]]
		
			##Calculate Loss
			model_out  = self._model(words, spans)   
			yhat[bidx_i:bidx_j] = model_out.squeeze()
			
			bidx_i = bidx_j
			bidx_j = bidx_i + self.train_batch_size
			
			if bidx_j >= total_obs:
				words = [words for words, spans in data_x[bidx_i:bidx_j]]
				spans = [spans for words, spans in data_x[bidx_i:bidx_j]]
				
				##Calculate Loss
				model_out  = self._model(words, spans)   
				yhat[bidx_i:bidx_j] = model_out.squeeze()
				
		return yhat

	def predict(self, data_x, data_y, loss=None):
		'''
		Predict loss, and prediction values for whole data_x
		'''
		# Turn on evaluation mode which disables dropout.
		self._model.eval()
		batch_losses = []
		
		with torch.no_grad():  
			bidx_i = 0
			bidx_j = self.predict_batch_size
			total_obs = len(data_x)
			yhat = torch.zeros(total_obs).to(self.device)

			while bidx_j < total_obs:
				words = [words for words, spans in data_x[bidx_i:bidx_j]]
				spans = [spans for words, spans in data_x[bidx_i:bidx_j]]
			
				##Calculate Loss
				model_out  = self._model(words, spans)   
				yhat[bidx_i:bidx_j] = model_out.squeeze()

				if self.pretraining:
					curr_loss = self._custom_loss(model_out, data_y[bidx_i:bidx_j], self.pretrain_x, self.pretrain_actual)
				else:
					if loss=="l1":
						actual_torch = torch.from_numpy(np.array(data_y[bidx_i:bidx_j])).float().to(self.device)
						curr_loss = self.l1_loss(model_out.squeeze(), actual_torch)
					else:
						curr_loss = self._custom_loss(model_out, data_y[bidx_i:bidx_j], None, None)

				batch_losses.append(curr_loss.detach().item())
				
				bidx_i = bidx_j
				bidx_j = bidx_i + self.train_batch_size
				
				if bidx_j >= total_obs:
					words = [words for words, spans in data_x[bidx_i:bidx_j]]
					spans = [spans for words, spans in data_x[bidx_i:bidx_j]]
					
					##Calculate Loss
					model_out  = self._model(words, spans)   
					yhat[bidx_i:bidx_j] = model_out.squeeze()
					if self.pretraining:
						curr_loss = self._custom_loss(model_out, data_y[bidx_i:bidx_j], self.pretrain_x, self.pretrain_actual)
					else:
						if loss=="l1":
							actual_torch = torch.from_numpy(np.array(data_y[bidx_i:bidx_j])).float().to(self.device)
							curr_loss = self.l1_loss(model_out.squeeze(), actual_torch)
						else:
							curr_loss = self._custom_loss(model_out, data_y[bidx_i:bidx_j], None, None)
					batch_losses.append(curr_loss.detach().item())
				

		return np.mean(batch_losses), yhat.detach()