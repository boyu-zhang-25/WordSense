from util import parse_data, get_raw_sentences
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids
from allennlp.commands.elmo import ElmoEmbedder
from model import Model

# test the model
def main():

	'''
	parse all the data
	wsd_data[idx]: ordered dic {['split', 'train'], ['Annotator.ID', '0'], ...['Display.Position', '6']}
	train_data[sentence_idx]: a list of tokenized sentence at idx from EUD
	train_data[sentence_idx][word_idx]: an ordered dict for the word at this index
	in the form: OrderedDict([('id', 14), ('form', 'the'), ('lemma', 'the'), ...])
	same for dev and test
	Notice: the index in EUD is 1-based!!!
	'''

	# parse the data
	wsd_data, train_data, test_data, dev_data = parse_data()

	# return the raw sentences from the EUD for train, test, and dev
	train_sentences, train_word_sense, train_word_index, test_sentences, test_word_sense, test_word_index, dev_sentences, dev_word_sense, dev_word_index = get_raw_sentences(wsd_data, train_data, test_data, dev_data)

	# ELMo setup
	# ELMo is tuned to lower dimension (256) by MLP in Model
	elmo = ElmoEmbedder()
	model = Model(elmo_class = elmo)

	# get the ELMo embeddings
	embeddings, masks = model._get_embedding(train_sentences)

if __name__ == '__main__':
	main()