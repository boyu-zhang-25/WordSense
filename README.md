# Multi-Label Word Sense Tagging with Supersense Back-off

Code for the paper *Multi-Label Word Sense Tagging with Supersense Back-off*. 

## Work in progress.

Please check the `demo.ipynb` for results so far. 

|Number of Epoch|Number of Training Data|Number of Dev Data|Accuracy for Known Words|Accuracy for Unknown Words (supersense level)|
|---|---|---|---|---|
| 10  | 1000 | 200-ish  |  36.7%  |  39.5% |
| 20  | 10000 | 7000-ish  | 62.1%  | 69.3% |


# Supervisor and Authors

Director and Supervisor: [Prof. Aaron Steven White](http://aaronstevenwhite.io/)

and me :)

[The Formal And CompuTational Semantics lab (FACTS.lab)](http://factslab.io/)

University of Rochester, Spring 2019

# Corpora

The [Universal Decompositional Semantics Word Sense dataset](http://decomp.io/projects/word-sense/): it contains annotations of nouns in context for word sense based on the senses listed in the WordNet.

The Universal Decompositional Semantics Dataset ([White et. al., 2016](https://www.aclweb.org/anthology/D16-1177)) contains 439312 sentences with target words extracted from the [Universal Dependency of English dataset by Stanford University](https://universaldependencies.org/). Annotators were hired to label the correct sense of each target word in the given sentences, and the choices of senses are provided by the WordNet. Below is an example:

![data_example](https://github.com/cristianoBY/WordSense/blob/master/data_example.png)

The sample sentence and index of the target word are given in the first place. Punctuations shall not be removed since they have important effects when generating contextual information. In the annotator response vector, a `1` at index `i` means the ith definition of the target word in the WordNet is the correct sense in the given sentence, and vice versa for `0`. Each of the senses also has a serial number, e.g., `spring.n.01`. For the above example, the annotator marked the first sense as 1 (true), and the corresponding WordNet definition is `the season of growth.` The other senses are 0 and not suitable for this sentence.

# Abstraction

## Intro

Many words have multiple potential senses. Ambiguous words like “bank” – which has both “financial institution” and “side of a river” senses – are a typical case of this; words that undergo regular polysemy are another. Utilizing distributed embeddings to represent words and contexts are shown to be a promising future solution to word sense disambiguation (WSD) tasks.

Previous work on WSD are based on computing independent distributed embeddings for the contexts of the target ambiguous word in large corpus and clustering them, which is often referred to as *multi-prototype* ([Mooney et. al, 2010](https://www.aclweb.org/anthology/N10-1013)). [Neelakantan et al. (2014)](https://arxiv.org/abs/1504.06654) was the first to extend the multi-prototype models to joint learing: 'the crucial difference is that word sense discrimination and learning embeddings are performed jointly by predicting the sense of the word using the current parameter estimates.'

However, treating WSD as a classification problems has some drawbacks caused by the nature of the ambiguity of words: for the word ‘spring’, it has 6 different senses according to the dataset (originally defined by the WordNet), while the word ‘job’ has more than 10 definitions. Thus, we have to custom an individual output layer for each word, and the size of each of the customed layer equals to the number of senses the corresponding word has. This output layer is designed to represent the probability distributions over all target word senses. 

Therefore, for each output layer, the available training samples are only the sentences containing the target word, i.e., two words do not share the same MLP output layer. However, for example, the dataset only containing 100 sentences with the target word ‘spring’, and this is clearly not enough for training an individual output weight matrix for ‘spring’. One alternative is to build a universal output layer with the size of all senses of all words in the dataset, but this would be computationally intractable. 

Here we propose a whole new model for WSD. Instead of clustering and classifying contextual word embeddings, for each context with a target word, we decided to use an alternative idea: sense embedding. This is not a new idea ([Camacho-Collados et. al, 2018](https://arxiv.org/abs/1805.04032); [Neelakantan et. al, 2015](https://arxiv.org/abs/1504.06654)), but it will avoid the problems in classification. We modified the output layer to be a universal layer with an output size [300, 1], treating this as the sense embedding but do not apply any activation for classification. Thus the values in the tensor are not bounded between 1 and 0 like the probability but have the potential to explore the whole vector space. After that, for a example word with an annotator response [1, 0, 0], we randomly initialize 3 vectors for each element in this response, treating them as the sense vector of definitions, laying in the same vector space as the output tensor of our model. In the rest of this report, I will refer the sense vector of definitions as definition embedding and the output tensor as predicted embedding. 

## Model

First, for one word with *n* senses, we can generate one predicted embedding (by ELMo and MLP) and *n* definition embeddings (randomly initialized). Instead of clustering contextual embeddings to represent different sense, we then compute the distance and cosine similarity between the predicted embedding with each of the definition embeddings, calculating the loss. Third, we perform a gradient update on the embedding of both the predicted embedding and the definition embedding.

The goal of this joint optimization is to increase the similarity between the predicted embedding and the definition embedding. For the definitions with annotator response ‘1’, we pull the predicted embedding closer to them, and for the definitions with ‘0’s, we push the predicted embedding further away from them. The gradient update is calculated by the mean square loss and the cosine similarity. To push away, we simply negated the loss. 

We jointly improved:

1. The ability of the model to project word sense to meaning space accurately.
2. The ability of the model to locate accurate ground truth embedding for definitions from the WordNet.

The metaphor here is that *step 1* is assigning a word to a cluster, and *step 2* is correcting the mean of the clusters. Specifically, backpropagation for the predicted embedding will be a ordinary optimization, i.e., accumulated across all its definitions. For definition embeddings, we only update it w.r.t. the gradient of the predicted embedding. This means that one particular definition contributes only a small amount of correction normalized by the size of all definitions, which is similar to calculating new means in Expectation-Maximization algorithms. 

After the training process, we are expecting to have a model that can generate accurate word sense embeddings and have a bunch of definition embeddings that lay with certain patterns (contextual similarity, unit-type polysemy, etc.) in the vector space. The exact location of the definition vectors may vary since they are randomly initialized, but the important properties we get is the relative position of definitions embeddings w.r.t. the predicted sense embeddings, even between definitions embeddings themselves. These relative positions imply semantic relationships between different sensors. An illustrated graph is given below:


For a given word, we can test the performance of the model by:
```
1. Generating the predicted embedding.
2. Masking out the annotator responses.
3. Comparing the distances and similarities between the predicted embeddings and all its definition embeddings. 
4. Picking the most similar one.  
5. Checking if the pick has an annotator response of 1.
```
As you may notice, this model only works for words within the lexicon of the WordNet. Otherwise, we do not have the definition embeddings.

## Supersense:

After the joint optimization of the above model, we are seeking a solution to the problem of unknown word aforementioned. We will adopt an idea called ‘supersense’ ([Ciaramit, 2003](https://dl.acm.org/citation.cfm?id=1119377)). Supersense, or Lexicographer, is a universal ontology across all vocabularies. It acts as a ‘genralization’ and ‘abstraction’ of word senses and is not affected by unseen words. 

We will randomly initialize a [300, 1] vector for each supersense as supersense embedding. And when we are optimizing step 1 and step 2, we also optimize the embedding of supersense in the same way. Here is an intuitive example (let’s forget about ambiguity for a while): suppose the model knows `apple`, `banana`, and `orange`, and it also knows that these three words belong to the supersense `fruit` (this deterministic knowledge comes from artificial ontology like WordNet). Thus, for the input `apple`, the predicted embedding shall not only be closer to the actual `apple` sense embedding, but also closer to the `fruit` embedding. After training, if we input an unseen word `durian`, we will get nothing because we do not have the definition embedding for ‘durian’. But! We do have the universal supersense `fruit`, and our trained model is supposed to put `durian` close to `fruit`. Thus, our model would say: ‘oh I do not know what exactly it is, but at least I know it is a fruit!’

By using the WordNet supersense or any other ontology, which are usually much smaller than the vocabulary size, deterministic knowledge (`apple` belongs to `fruit`) can be integrated to statistical embedding, providing stronger reasoning and symbolic knowledge. This will improve the robustness of our word sense model.

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
