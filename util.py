import csv
import math
import itertools

'''
Copyright@
White, A. S., D. Reisinger, K. Sakaguchi, T. Vieira, S. Zhang, R. Rudinger, K. Rawlins, & B. Van Durme. 2016. [Universal decompositional semantics on universal dependencies](http://aswhite.net/media/papers/white_universal_2016.pdf). To appear in *Proceedings of the Conference on Empirical Methods in Natural Language Processing 2016*.
'''

# parse the WSD dataset
def parse_tsv():
    
    # parse the WSD dataset
    data_num = 0
    data = []

    # read in tsv
    with open('data/wsd/wsd_eng_ud1.2_10262016.tsv', mode = 'r') as wsd_data:

        tsv_reader = csv.DictReader(wsd_data)

        # store the data
        for row in tsv_reader:

            # each data vector
            data.append(row)
            data_num += 1

        # make sure all data are parsed
        print('Parsed {} word sense data.'.format(data_num))
        # print(data[0])
        # print(data[1])

# test utilities
def main():
    parse_tsv()

if __name__ == '__main__':
    main()