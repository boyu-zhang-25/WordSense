import csv
import math
import itertools
from io import open
from conllu import parse_incr

'''
Copyright@
White, A. S., D. Reisinger, K. Sakaguchi, T. Vieira, S. Zhang, R. Rudinger, K. Rawlins, & B. Van Durme. 2016. [Universal decompositional semantics on universal dependencies](http://aswhite.net/media/papers/white_universal_2016.pdf). To appear in *Proceedings of the Conference on Empirical Methods in Natural Language Processing 2016*.
'''

# parse the WSD dataset first
# and retrieve all sentences from the EUD
def parse_data():
    
    # parse the WSD dataset
    wsd_data = []

    # read in tsv by White et. al., 2016
    with open('data/wsd/wsd_eng_ud1.2_10262016.tsv', mode = 'r') as wsd_file:

        tsv_reader = csv.DictReader(wsd_file)

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
    print(train_data[1363])

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