# getting familiar with ELMo with pytorch for fine-tuning
import h5py
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids

h5py_file = h5py.File("elmo_layers.hdf5", 'r')

# the 3-layer embedding of the first sentence
# "The cryptocurrency space is now figuring out to have the highest search on Google globally ."
embedding = h5py_file.get("0")

# [3, 16, 1024]
# 3 layers
# 16 words
# 1024 embedding size for each word
print(embedding)

assert(len(embedding) == 3) # one layer for each vector
assert(len(embedding[0]) == 16) # one entry for each word in the first sentence
assert(len(embedding[0][0]) == 1024) # each word has a embedding of 1024

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

# Compute two different representations for each token.
# Each representation is a linear weighted combination for the
# 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
elmo = Elmo(options_file, weight_file, 1, dropout=0)

# 2 target sentences
sentences = [['Word', 'sense', 'disambiguation', '.'], ['WSD', '.']]

# use batch_to_ids to convert sentences to character ids
# size = [2, 4, 50]
# [number of sentences (batch size), max sentence length, max word length]
character_ids = batch_to_ids(sentences)
# print(character_ids)
print(character_ids.size())

embeddings = elmo(character_ids)

# number of representations specified in line 28
print(len(embeddings['elmo_representations']))

# size = [2, 4, 1024]
# 2: number of sentences
# 4: max sentence length
# 1024: vector length for each word
print(embeddings['elmo_representations'][0].size())
