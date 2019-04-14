
# coding: utf-8

# In[1]:


import csv
import math
import string
import itertools
from io import open
from conllu import parse_incr
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import wordnet as wn
import numpy as np


# In[2]:


# parse the WSD dataset first

'''
Copyright@
White, A. S., D. Reisinger, K. Sakaguchi, T. Vieira, S. Zhang, R. Rudinger, K. Rawlins, & B. Van Durme. 2016. 
[Universal decompositional semantics on universal dependencies]
(http://aswhite.net/media/papers/white_universal_2016.pdf). 
To appear in *Proceedings of the Conference on Empirical Methods in Natural Language Processing 2016*.
'''

def parse_wsd_data():

    # parse the WSD dataset
    wsd_data = []

    # read in tsv by White et. al., 2016
    with open('data/wsd/wsd_eng_ud1.2_10262016.tsv', mode = 'r') as wsd_file:

        tsv_reader = csv.DictReader(wsd_file, delimiter = '\t')      

        # store the data: ordered dict row
        for row in tsv_reader:                                

            # each data vector
            wsd_data.append(row)

        # make sure all data are parsed
        print('Parsed {} word sense data from White et. al., 2016.'.format(len(wsd_data)))

    return wsd_data


# In[3]:


# get the raw wsd data
wsd_data = parse_wsd_data()


# In[4]:


'''
return: 
all senses for each word 
all definitions for each word
all supersenses
from the EUD for train, test, and dev dataset
index provided by WSD dataset by White et. al.
'''
# get all the senses and definitions for each word from WSD dataset
# order of senses and definitions are in order
def get_all_senses_and_definitions(wsd_data):

    # all senses for each word in train and dev
    # supersense is shared 
    all_senses = {}
    all_definitions = {}
    all_supersenses = {}

    # all senses for each word in test
    all_test_senses = {}
    all_test_definitions = {}
    
    # only get the senses for train and dev set
    for i in range(len(wsd_data)):
        
        # get the original sentence from EUD
        sentence_id = wsd_data[i].get('Sentence.ID')
        
        # get the definitions for the target word from EUD
        definition = wsd_data[i].get('Sense.Definition').split(' ')
        
        # the index in EUD is 1-based!!!
        sentence_number = int(sentence_id.split(' ')[-1]) - 1
        word_index = int(wsd_data[i].get('Arg.Token')) - 1
        word_lemma = wsd_data[i].get('Arg.Lemma')
        word_sense = wsd_data[i].get('Synset')
        response = wsd_data[i].get('Sense.Response')
        
        # senses for train and dev
        # preserve unknown words
        if wsd_data[i].get('Split') != 'test':
        
            # supersense-> (word_lemma, word_sense) dictionary
            super_s = wn.synset(word_sense).lexname().replace('.', '_')
            if all_supersenses.get(super_s, 'not_exist') != 'not_exist':
                all_supersenses[super_s].add((word_lemma, word_sense))
            else:
                all_supersenses[super_s] = {(word_lemma, word_sense)}

            # if the word already exits: add the new sense to the list
            # else: creata a new list for the word
            if all_senses.get(word_lemma, 'not_exist') != 'not_exist':
                if word_sense not in all_senses[word_lemma]:
                    all_senses[word_lemma].append(word_sense)
            else:
                all_senses[word_lemma] = []
                all_senses[word_lemma].append(word_sense)            
            
            if all_definitions.get(word_lemma,'not_exist') != 'not_exist':
                if definition not in all_definitions[word_lemma]: 
                    all_definitions[word_lemma].append(definition)
            else:
                all_definitions[word_lemma] = []
                all_definitions[word_lemma].append(definition)
                
        else:

            # all the senses and definitions for test words
            if all_test_senses.get(word_lemma, 'not_exist') != 'not_exist':
                if word_sense not in all_test_senses[word_lemma]:
                    all_test_senses[word_lemma].append(word_sense)
            else:
                all_test_senses[word_lemma] = []
                all_test_senses[word_lemma].append(word_sense)            
            
            if all_test_definitions.get(word_lemma,'not_exist') != 'not_exist':
                if definition not in all_test_definitions[word_lemma]: 
                    all_test_definitions[word_lemma].append(definition)
            else:
                all_test_definitions[word_lemma] = []
                all_test_definitions[word_lemma].append(definition)            
        
    return all_senses, all_definitions, all_supersenses, all_test_senses, all_test_definitions


# In[5]:


# get all the senses and definitions
all_senses, all_definitions, all_supersenses, all_test_senses, all_test_definitions = get_all_senses_and_definitions(wsd_data)


# In[6]:


# test for the WordNet NLTK API
'''
The specific Synset method is lexname, e.g. wn.synsets('spring')[0].lexname(). 
That should make it really easy to get the suspersenses.
And if you have the synset name–e.g. 'spring.n.01'
you can access the supersense directly: wn.synset('spring.n.01').lexname().
Which returns 'noun.time'.
And wn.synset('spring.n.02').lexname() returns 'noun.artifact'

for idx, d in enumerate(all_definitions['spring']):
    print(d)
    print(wn.synset(all_senses['spring'][idx]).lexname())

for _ in wn.synsets('spring'):
    print(_.lexname())
'''


# In[7]:


# read the train, dev, test datasets from processed files
# check the 'data_loader.ipynb' for details
def read_file():
    
    train_X = []
    train_X_num = 0
    train_Y = []
    train_Y_num = 0
    test_X = []
    test_X_num = 0
    test_Y = []
    test_Y_num = 0
    dev_X = []
    dev_X_num = 0
    dev_Y = []
    dev_Y_num = 0
    
    train_word_idx = []
    test_word_idx = []
    dev_word_idx = []
    
    # read in csv
    with open('data/train_X.tsv', mode = 'r') as data_file:
        
        csv_reader = csv.reader(data_file, delimiter = '\t')

        # store the data
        for row in csv_reader:

            train_X.append(row)
            train_X_num += 1

        # make sure all data are parsed
        print(f'Parsed {train_X_num} data points for train_X.')

    with open('data/train_Y.tsv', mode = 'r') as data_file:
        
        csv_reader = csv.reader(data_file)

        # store the data
        for row in csv_reader:

            row = list(map(int, row))
            train_Y.append(row)
            train_Y_num += 1

        # make sure all data are parsed
        print(f'Parsed {train_Y_num} data points for train_Y.')
        
    with open('data/train_word_idx.tsv', mode = 'r') as data_file:
        
        csv_reader = csv.reader(data_file)

        # store the data
        for row in csv_reader:

            row = list(map(int, row))
            train_word_idx = (row)

        # make sure all data are parsed
        print(f'Parsed {len(train_word_idx)} data points for train_word_idx.')

    with open('data/dev_X.tsv', mode = 'r') as data_file:
        
        csv_reader = csv.reader(data_file, delimiter = '\t')

        # store the data
        for row in csv_reader:

            dev_X.append(row)
            dev_X_num += 1

        # make sure all data are parsed
        print(f'Parsed {dev_X_num} data points for dev_X.')

    with open('data/dev_Y.tsv', mode = 'r') as data_file:
        
        csv_reader = csv.reader(data_file)

        # store the data
        for row in csv_reader:

            row = list(map(int, row))
            dev_Y.append(row)
            dev_Y_num += 1

        # make sure all data are parsed
        print(f'Parsed {dev_Y_num} data points for dev_Y.')
        
    with open('data/dev_word_idx.tsv', mode = 'r') as data_file:
        
        csv_reader = csv.reader(data_file)

        # store the data
        for row in csv_reader:

            row = list(map(int, row))
            dev_word_idx = (row)

        # make sure all data are parsed
        print(f'Parsed {len(dev_word_idx)} data points for dev_word_idx.')
        
    with open('data/test_X.tsv', mode = 'r') as data_file:
        
        csv_reader = csv.reader(data_file, delimiter = '\t')

        # store the data
        for row in csv_reader:

            test_X.append(row)
            test_X_num += 1

        # make sure all data are parsed
        print(f'Parsed {test_X_num} data points for test_X.')

    with open('data/test_Y.tsv', mode = 'r') as data_file:
        
        csv_reader = csv.reader(data_file)

        # store the data
        for row in csv_reader:

            row = list(map(int, row))
            test_Y.append(row)
            test_Y_num += 1

        # make sure all data are parsed
        print(f'Parsed {test_Y_num} data points for test_Y.')
        
    with open('data/test_word_idx.tsv', mode = 'r') as data_file:
        
        csv_reader = csv.reader(data_file)

        # store the data
        for row in csv_reader:

            row = list(map(int, row))
            test_word_idx = (row)

        # make sure all data are parsed
        print(f'Parsed {len(test_word_idx)} data points for test_word_idx.')    
        
    return train_X, train_Y, test_X, test_Y, dev_X, dev_Y, train_word_idx, test_word_idx, dev_word_idx


# In[8]:


# get all the structured data
train_X, train_Y, test_X, test_Y, dev_X, dev_Y, train_word_idx, test_word_idx, dev_word_idx = read_file()

# test on one word
word_choice = 'level'

new_train_X = []
new_train_Y = []
new_train_idx = []
distri_train = np.zeros(len(all_test_senses[word_choice]))
# stst = 0
for index, sen in enumerate(train_X):
    
    if sen[train_word_idx[index]] == word_choice:
        new_train_idx.append(train_word_idx[index])
        new_train_X.append(sen)
        new_train_Y.append(train_Y[index])
        distri_train += np.asarray(train_Y[index])
        # summ = train_Y[index][0] + train_Y[index][1]
        # if summ != 2:
            # stst += 1
# print('stst: {}'.format(stst))        
print('distri of train: {}'.format(distri_train))
        
new_test_X = []
new_test_Y = []
new_test_idx = []
distri_test = np.zeros(len(all_test_senses[word_choice]))
for index, sen in enumerate(test_X):
    
    if sen[test_word_idx[index]] == word_choice:
        new_test_idx.append(test_word_idx[index])
        new_test_X.append(sen)
        new_test_Y.append(test_Y[index])
        # print(test_Y[index])
        distri_test += np.asarray(test_Y[index])
print('distri of test: {}'.format(distri_test))

new_dev_X = []
new_dev_Y = []
new_dev_idx = []
distri_dev = np.zeros(len(all_test_senses[word_choice]))
for index, sen in enumerate(dev_X):
        
    if sen[dev_word_idx[index]] == word_choice:
        new_dev_idx.append(dev_word_idx[index])
        new_dev_X.append(sen)
        new_dev_Y.append(dev_Y[index])
        distri_dev += np.asarray(dev_Y[index])
print('distri of dev: {}'.format(distri_dev))

target_senses = all_senses[word_choice]
new_all_senses = {word_choice : target_senses}
target_def = all_definitions[word_choice]
new_all_def = {word_choice : target_def}

# limit the supersense to only the test word
new_all_supersenses = {}
for supersense in all_supersenses.keys():
    for tuples in all_supersenses[supersense]:
        
        if tuples[0] == word_choice:
            if new_all_supersenses.get(supersense, 'e') != 'e':
                new_all_supersenses[supersense].add((word_choice, tuples[1]))
            else:
                new_all_supersenses[supersense] = {(word_choice, tuples[1])}


# In[9]:


from model import *
from trainer import *

from allennlp.commands.elmo import ElmoEmbedder
elmo = ElmoEmbedder()


# In[10]:


# trainer
epochs = 30

# test on one word
# trainer = Trainer(epochs = epochs, elmo_class = elmo, all_senses = all_senses, all_supersenses = all_supersenses)
trainer = Trainer(epochs = epochs, elmo_class = elmo, all_senses = new_all_senses, all_supersenses = new_all_supersenses)


# In[11]:


# train the model
# train_losses, dev_losses, dev_rs = trainer.train(train_X, train_Y, train_word_idx, dev_X, dev_Y, dev_word_idx)

# small test on only one word
train_losses, dev_losses, dev_rs = trainer.train(new_train_X, new_train_Y, new_train_idx, new_dev_X, new_dev_Y, new_dev_idx)


# In[12]:


# plot the learning curve
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

with open('train_loss.tsv', mode = 'w') as loss_file:
        
    csv_writer = csv.writer(loss_file)
    csv_writer.writerow(train_losses)

    
with open('dev_loss.tsv', mode = 'w') as loss_file:
        
    csv_writer = csv.writer(loss_file)
    csv_writer.writerow(dev_losses)


# In[13]:


plt.figure(1)
rc('text', usetex = True)
rc('font', family='serif')
plt.grid(True, ls = '-.',alpha = 0.4)
plt.plot(train_losses, ms = 4, marker = 's', label = "Train Loss")
plt.legend(loc = "best")
title = "Cosine Similarity Loss (number of examples: " + str(len(new_train_X)) + ")"
plt.title(title)
plt.ylabel('Loss')
plt.xlabel('Number of Iteration')
plt.tight_layout()
plt.savefig('train_loss.png')


# In[14]:


plt.figure(2)
rc('text', usetex = True)
rc('font', family='serif')
plt.grid(True, ls = '-.',alpha = 0.4)
plt.plot(dev_losses, ms = 4, marker = 'o', label = "Dev Loss")
plt.legend(loc = "best")
title = "Cosine Similarity Loss (number of examples: " + str(len(new_dev_X)) + ")"
plt.title(title)
plt.ylabel('Loss')
plt.xlabel('Number of Iteration')
plt.tight_layout()
plt.savefig('dev_loss.png')


# In[19]:


# test the model
# modify to test only one word for now
cos = nn.CosineSimilarity(dim = 0, eps = 1e-6).to(self.device)
correct_count = 0
known_test_size = 0
unknown_test_size = 0
unknown_correct_count = 0

# overall accuracy
for test_idx, test_sen in enumerate(new_test_X):
    
    test_lemma = test_sen[new_test_idx[test_idx]]
    test_emb = trainer._model.forward(test_sen, new_test_idx[test_idx]).to(self.device)
    all_similarity = []
    
    # if it is a new word
    # only test on the supersense
    if new_all_senses.get(test_lemma, 'e') == 'e':
        
        unknown_test_size += 1
        test_result = ''
        best_sim = -float('inf')
        
        for n, new_s in enumerate(all_test_senses[test_lemma]):
            
            new_super = wn.synset(new_s).lexname().replace('.', '_')
            super_vec = trainer._model.supersense_embeddings[new_super].view(trainer._model.output_size, -1).to(self.device)
            cos_sim = cos(test_emb, super_vec)
            
            if cos_sim > best_sim:
                test_result = new_super
                best_sim = cos_sim
                
        correct_super = []
        for q, respon in new_test_Y[test_idx]:
            if respon:
                correct_s = wn.synset(all_test_senses[test_lemma][q]).lexname().replace('.', '_')
                correct_super.append(correct_s)            
        if test_result in correct_super:
            unknown_correct_count += 1
        
    else:
        
        # if it is a known word
        known_test_size += 1
        
        for k, sense in enumerate(new_all_senses[test_lemma]):
            definition_vec = trainer._model.definition_embeddings[test_lemma][:, k].view(trainer._model.output_size, -1).to(self.device)
            cos_sim = cos(test_emb, definition_vec)
            all_similarity.append(cos_sim)
        # print(all_similarity)
        test_result = all_similarity.index(max(all_similarity))
        # print("result index: {}".format(test_result))
        if new_test_Y[test_idx][test_result] == 1:
            correct_count += 1

print('test size for known words: {}'.format(known_test_size))
print('accuracy for known words: {}'.format(correct_count / known_test_size))

print('test size for unknown words: {}'.format(unknown_test_size))
# print('accuracy for unknown words: {}'.format(unknown_correct_count / unknown_test_size))

