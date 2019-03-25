from util import parse_data
import torch
from allennlp.modules.elmo import Elmo, batch_to_ids

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
	wsd_data, train_data, test_data, dev_data = parse_data()

	# ELMo setup
	options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
	weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
	'''
	Compute two different representations for each token.
	Each representation is a linear weighted combination for the
	3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
	'''
	elmo = Elmo(options_file, weight_file, 1, dropout=0)

	# get the first 10 words
	train_sentences = []
	train_word_index = []
	for i in range(1):

		# get the original sentence from EUD
		sentence_id = wsd_data[i].get('Sentence.ID')

		# the index in EUD is 1-based!!!
		sentence_number = int(sentence_id.split()[-1]) - 1
		word_index = int(wsd_data[i].get('Arg.Token')) - 1
		word_lemma = wsd_data[i].get('Arg.Lemma')
		sentence = train_data[sentence_number]
		# print(sentence)
		assert(sentence[word_index].get('lemma') == word_lemma)

		# the clean sentence in list
		clean_sentence = [word_dict.get('lemma') for word_dict in sentence]
		print(clean_sentence)
		print(len(clean_sentence))
		train_sentences.append(clean_sentence)
		train_word_index.append(word_index)

	# get the ELMo of the first 10 words
	# use batch_to_ids to convert sentences to character ids
	# size = [1, 48, 50]
	# [number of sentences (batch size), max sentence length, max word length]
	character_ids = batch_to_ids(train_sentences)
	# print(character_ids)
	print(character_ids.size())

	embeddings = elmo(character_ids)

	# number of representations specified in line 27
	print(len(embeddings['elmo_representations']))

	# size = [1, 48, 1024]
	# 1: number of sentences
	# 4: max sentence length
	# 1024: vector length for each word
	print(embeddings['elmo_representations'][0].size())

if __name__ == '__main__':
	main()