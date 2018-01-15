## Introduction

This is an experimental yet fully functional TensorFlow reimplementation of the document embedding algorithm Doc2Vec by Le *et al.*. It is implemented by subclassing from the `Word2Vec` class in [Word2Vec](https://github.com/chao-ji/Word2Vec) by which the Doc2Vec algorithm was inspired. Both model architectures, Distributed Bag-of-Words (PV-DBOW) and Distrubuted Memory (PV-DM), as well as both training algorithms, Negative sampling and Hierarchical Softmax are supported.

## Reference
1. Q Le, T Mikolov - Distributed Representations of Sentences and Documents, ICML 2014
2. Gensim implementation by Radim Řehůřek, https://radimrehurek.com/gensim/models/doc2vec.html
