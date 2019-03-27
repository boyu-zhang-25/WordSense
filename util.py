import csv
import math
import itertools
from io import open
from conllu import parse_incr

'''
Copyright@
White, A. S., D. Reisinger, K. Sakaguchi, T. Vieira, S. Zhang, R. Rudinger, K. Rawlins, & B. Van Durme. 2016. [Universal decompositional semantics on universal dependencies](http://aswhite.net/media/papers/white_universal_2016.pdf). To appear in *Proceedings of the Conference on Empirical Methods in Natural Language Processing 2016*.
'''

# print the structure of the fine-tuning MLP for illustration
def print_fine_tuning_MLP(model, param):

	print('******************* fine-tuning MLP structure ***********************')

	print('Current Task: {}'.format(param))
	module_dict = model.layers[param]

	for module in module_dict:
		print(module)

	for param in model.parameters():
		if param.requires_grad:
			print(param.size())

	print(model)
	print('**********************************************************************')

# return the raw sentences from the EUD for train, test, and dev
def get_raw_sentences(wsd_data, train_data, test_data, dev_data, sen_num):

	# get the raw sentences, target word index, and target word sense
	train_sentences = []
	train_word_index = []
	train_word_sense = []
	test_sentences = []
	test_word_index = []
	test_word_sense = []
	dev_sentences = []
	dev_word_index = []
	dev_word_sense = []

	# for test purpose: only load specific amount of data
	for i in range(sen_num):

		# get the original sentence from EUD
		sentence_id = wsd_data[i].get('Sentence.ID')

		# the index in EUD is 1-based!!!
		sentence_number = int(sentence_id.split(' ')[-1]) - 1
		# print('sentence id {} i {}'.format(sentence_id, i))
		word_index = int(wsd_data[i].get('Arg.Token')) - 1
		word_lemma = wsd_data[i].get('Arg.Lemma')
		word_sense = wsd_data[i].get('Synset')

		if "train" in sentence_id: 
			sentence = train_data[sentence_number]
			# print(sentence)
			assert(sentence[word_index].get('lemma') == word_lemma)

			# the clean sentence in list
			clean_sentence = [word_dict.get('lemma') for word_dict in sentence]
			# print(clean_sentence)
			# print(len(clean_sentence))
			train_sentences.append(clean_sentence)
			train_word_index.append(word_index)
			train_word_sense.append(word_sense)

		elif "test" in sentence_id:
			sentence = test_data[sentence_number]
			# print(sentence)
			assert(sentence[word_index].get('lemma') == word_lemma)

			# the clean sentence in list
			clean_sentence = [word_dict.get('lemma') for word_dict in sentence]
			# print(clean_sentence)
			# print(len(clean_sentence))
			test_sentences.append(clean_sentence)
			test_word_index.append(word_index)
			test_word_sense.append(word_sense)

		else:
			sentence = dev_data[sentence_number]
			# print(sentence)
			assert(sentence[word_index].get('lemma') == word_lemma)

			# the clean sentence in list
			clean_sentence = [word_dict.get('lemma') for word_dict in sentence]
			# print(clean_sentence)
			# print(len(clean_sentence))
			dev_sentences.append(clean_sentence)
			dev_word_index.append(word_index)
			dev_word_sense.append(word_sense)

	print('Parsed {} sentences'.format(sen_num))
	print('******************* Data Example ***********************')
	print('Sentence: {}'.format(train_sentences[0]))
	print('Target Word Index: {}'.format(train_word_index[0]))
	print('Target Word Sense (index in WordNet 3.1): {}'.format(train_word_sense[0]))
	print('********************************************************')

	return train_sentences, train_word_sense, train_word_index, test_sentences, test_word_sense, test_word_index, dev_sentences, dev_word_sense, dev_word_index
			
# parse the WSD dataset first
# and retrieve all sentences from the EUD
def parse_data():
	
	# parse the WSD dataset
	wsd_data = []

	# read in tsv by White et. al., 2016
	with open('data/wsd/wsd_eng_ud1.2_10262016.tsv', mode = 'r') as wsd_file:

		tsv_reader = csv.DictReader(wsd_file, delimiter = '\t')

		# store the data
		for row in tsv_reader:

			# each data vector
			wsd_data.append(row)

		# make sure all data are parsed
		print('Parsed {} word sense data from White et. al., 2016.'.format(len(wsd_data)))

	# parse the EUD-EWT conllu files and retrieve the sentences
	train_file = open("data/UD_English-EWT/en_ewt-ud-train.conllu", "r", encoding="utf-8")
	train_data = list(parse_incr(train_file))
	print('Parsed {} training data from UD_English-EWT/en_ewt-ud-train.conllu.'.format(len(train_data)))
	# 'spring' as the first example in White et. al., 2016
	# print(train_data[1363])

	test_file = open("data/UD_English-EWT/en_ewt-ud-test.conllu", "r", encoding="utf-8")
	test_data = list(parse_incr(test_file))
	print('Parsed {} testing data from UD_English-EWT/en_ewt-ud-test.conllu'.format(len(test_data)))

	dev_file = open("data/UD_English-EWT/en_ewt-ud-dev.conllu", "r", encoding="utf-8")
	dev_data = list(parse_incr(dev_file))
	print('Parsed {} dev data from UD_English-EWT/en_ewt-ud-dev.conllu'.format(len(dev_data)))

	return wsd_data, train_data, test_data, dev_data

# test utilities
def main():
	wsd_data, train_data, test_data, dev_data = parse_data()

if __name__ == '__main__':
	main()