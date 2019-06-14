# Word Sense Tagging Over Continuous Space with Supersense Back-off

**This project is a part of the upcoming TACL paper *Decoding Word Sense with Tree LSTM*.**

**Please check the `abstract.pdf` for the model design, test results, and other details.**

**I have a updated version of senses encoder (more efficient, supports batch) for my new project in-progress. And it also uses to the SemCor + OMNIST WSD dataset. If you are interested, please feel free to contact me**

|Number of Epoch|Number of Training Data|Number of Dev Data|Accuracy for Known Words|Accuracy for Unknown Words (supersense level)|
|---|---|---|---|---|
| 45  | 15000 | 7118 (both known and unknown words)  | 62.9% (out of 6440 known words)  | 70.8% (out of 678 unknown words)|

The average number of senses for an ambiguous word in the dataset: 4.3

# Authors

[Prof. Aaron Steven White](http://aaronstevenwhite.io/) and me :)

[The Formal And CompuTational Semantics lab (FACTS.lab)](http://factslab.io/)

University of Rochester, Spring 2019

I am glad and greatful that my course project can be this interesting!!

# Requirements

# Copyrights
This project is under the GNU AFFERO GENERAL PUBLIC LICENSE.

