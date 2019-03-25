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

	# test on the first 10 words
	all_sentences = []
	for i in range(1):

		# get the original sentence from EUD
		sentence_id = wsd_data[i].get('Sentence.ID')

		# the index in EUD is 1-based!!!
		sentence_number = int(sentence_id.split()[-1]) - 1
		word_index = int(wsd_data[i].get('Arg.Token')) - 1
		word_lemma = wsd_data[i].get('Arg.Lemma')
		sentence = []

		if "train" in sentence_id:

			sentence = train_data[sentence_number]
			# print(sentence)
			assert(sentence[word_index].get('lemma') == word_lemma)

		elif "test" in sentence_id:

			sentence = test_data[sentence_number]
			# print(sentence)
			assert(sentence[word_index].get('lemma') == word_lemma)

		else:

			sentence = dev_data[sentence_number]
			# print(sentence)
			assert(sentence[word_index].get('lemma') == word_lemma)

		all_sentences.append(sentence)
		print(all_sentences)

if __name__ == '__main__':
	main()