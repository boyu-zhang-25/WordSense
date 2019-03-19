# WSD

WSD: Word Sense Disambiguation

Models for capturing the difference between rule-governed and non-rule-governed word sense ambiguity from contextual embeddings. 

Preparing for EMNLP 2019.

##Work in progress.

# Organization, Supervisor, and Authors

[The Formal And CompuTational Semantics lab (FACTS.lab)](http://factslab.io/)

Director and Supervisor: [Prof. Aaron Steven White](http://aaronstevenwhite.io/)

University of Rochester, Spring 2019

# Intro
Many words have multiple potential senses. Ambiguous words like “bank” – which has both “financial institution” and “side of a river” senses – are a typical case of this; words that undergo regular polysemy are another. A classic case of this is what is the unit-type ambiguity – e.g. “book” can be used to refer either to a particular physical instantiation of a book or to the abstract contents of a book. This latter sort of ambiguity seems to be rule-governed in an important way – e.g. many objects that contain informational content undergo it.

This project is to investigate how to capture the difference between rule governed ambiguity and non-rule governed ambiguity in a computational model. This model will predict word senses and genericity based on the model’s hidden states and build a notion of regular polysemy rule into the relationship between different word senses. 

# Corpora

1. The [Universal Decompositional Semantics Word Sense dataset](): it contains annotations of nouns in context for word sense based on the senses listed in WordNet, 

2. The [Universal Decompositional Semantics Generics dataset]()

# Copyrights
This project is under the GNU AFFERO GENERAL PUBLIC LICENSE.




