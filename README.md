# Multi-Label Word Sense Tagging with Supersense Back-off

Code for the paper *Multi-Label Word Sense Tagging with Supersense Back-off*. 

Preparing for EMNLP 2019.

## Work in progress.

Please check the `test.ipynb` for results so far. 

# Supervisor and Authors

Director and Supervisor: [Prof. Aaron Steven White](http://aaronstevenwhite.io/)

and me :)

[The Formal And CompuTational Semantics lab (FACTS.lab)](http://factslab.io/)

University of Rochester, Spring 2019

# Intro

Many words have multiple potential senses. Ambiguous words like “bank” – which has both “financial institution” and “side of a river” senses – are a typical case of this; words that undergo regular polysemy are another. Utilizing distributed embeddings to represent words and contexts are shown to be a promising future solution to word sense disambiguation (WSD) tasks.

Previous work on WSD are based on computing independent distributed embeddings for the contexts of the target ambiguous word in large corpus and clustering them, which is often referred to as *multi-prototype* ([Mooney et. al, 2010](https://www.aclweb.org/anthology/N10-1013)). [Neelakantan et al. (2014)](https://arxiv.org/abs/1504.06654) was the first to extend the multi-prototype models to joint learing: 'the crucial difference is that word sense discrimination and learning embeddings are performed jointly by predicting the sense of the word using the current parameter estimates.'

Here we propose a whole new model for WSD. Instead of clustering contextual embeddings to represent different sense, for each context with a target word, we fused some knowledge-based approach to our distributional model -- the literal sense definition from the [WordNet 3.1](https://wordnet.princeton.edu/). We first project the target word embedding to the vector space of meaning (ELMo + MLP), and then encode the right answer of the word sense (from WordNet) to the same vector space of meaning by a simple seq2seq model. Second, we compute the distance between these two vectors, calculating the loss. Third, we perform a gradient update on the embedding of both the word embedding and the truth embedding. 

The goal of this joint optimization is to increase the similarity between the word embedding and the truth embedding. We jointly improved:

1. The ability of the model to project words sense to meaning space accurately. 

2. The ability of the model to locate accurate ground truth embedding for definitions in the WordNet. 

The metaphor here is that *step 1* is assigning word to a cluster, and *step 2* is correct the mean of the clusters. Specifically, BP for the sense embedding on top of ELMo will be a regular optimization, while the gradient descent for ground truth encoder shall be updated with a normalized loss per example. This is means one particular word contributes certain amount of correction normalized by the size of all words, which is similar to calculating new means in Expectation-Maximization algorithms. 

Supersense:

# Corpora

1. The [Universal Decompositional Semantics Word Sense dataset](http://decomp.io/projects/word-sense/): it contains annotations of nouns in context for word sense based on the senses listed in the WordNet.

# Requirements


# Copyrights
This project is under the GNU AFFERO GENERAL PUBLIC LICENSE.

# Reference
@inproceedings{Peters:2018,
  author={Peters, Matthew E. and  Neumann, Mark and Iyyer, Mohit and Gardner, Matt and Clark, Christopher and Lee, Kenton and Zettlemoyer, Luke},
  title={Deep contextualized word representations},
  booktitle={Proc. of NAACL},
  year={2018}
}

@inproceedings{Gardner2017AllenNLP,
  title={{AllenNLP}: A Deep Semantic Natural Language Processing Platform},
  author={Matt Gardner and Joel Grus and Mark Neumann and Oyvind Tafjord
    and Pradeep Dasigi and Nelson F. Liu and Matthew Peters and
    Michael Schmitz and Luke S. Zettlemoyer},
  year={2018},
  booktitle={ACL workshop for NLP Open Source Software}
}
