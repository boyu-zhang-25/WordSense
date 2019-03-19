import h5py
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids

h5py_file = h5py.File("elmo_layers.hdf5", 'r')
embedding = h5py_file.get("0")

print(embedding)

assert(len(embedding) == 3) # one layer for each vector
assert(len(embedding[0]) == 16) # one entry for each word in the source sentence