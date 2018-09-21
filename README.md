
Doc2Vec (a.k.a. Paragraph2Vec) is a Word2Vec-inspired, simple yet effective model to obtain real-valued, distributed representation of unlabeled documents (sentences, paragraphs, articles etc.). Here is a tensorflow implementation of Doc2Vec. It allows you to train in both PV-DBOW (distributed bag of words, similar to skip-gram in Word2Vec) and PV-DM (distributed memory, similar to bag-of-words in Word2Vec) architectures. It can also run in inference mode where you only learn the embeddings of unseen documents (not present in the training data) given a trained model.

### Usage
To install Doc2Vec, do
```
git clone git@github.com:chao-ji/tf-doc2vec.git doc2vec
```
, then`cd` into `doc2vec`.

Because the Doc2Vec model is subclassed from Word2Vec, you would also need to install Word2Vec:

```
git clone git@github.com:chao-ji/tf-word2vec.git word2vec
```
You should have `word2vec` in your current working directory. Now you should be able to do `import word2vec` if word2vec is correctly installed.

##### Training and inference
Two python executables are provided from training (`run_training`) and making inference (`run_inference`). 

To perform training, you need to have your training data formatted as text files where each line holds a single **document** in which words are space-separated.

Example:
```
  python run_training.py \
    --filenames=/PATH/TO/FILE/file1.txt,/PATH/TO/FILE/file2.txt \
    --out_dir=/PATH/TO/OUT_DIR/ \
    --arch=PV-DM
```
where `filenames` is a comma-separated list of file paths of training data, and `out_dir` points to the directory where outputs are saved.

To perform inference (i.e. inference mode), you need to give, other than `filenames`, the path of the file containing documents whose embeddings are to be inferred, through `filenames_infer`.

Example:
```
  python run_training.py \
    --filenames=/PATH/TO/FILE/file1.txt,/PATH/TO/FILE/file2.txt \
    --filenames_infer=/PATH/TO/FILE/infer_file.txt \
    --out_dir=/PATH/TO/OUT_DIR/ \
    --arch=PV-DM
```

### Combinations of training modes
As in Word2Vec, Doc2Vec can be trained in two architectures (`PV-DBOW` and `PV-DM`), in conjunction with two training algorithms: negative sampling and hierarchical softmax. In addition, `PV-DBOW` has the option of training word documents or not, and `PV-DM` has the option of concatenating word and document embeddings instead of taking an average (See the Doc2Vec paper in Ref). So there are 8 combinations of training modes in total:

|PV-DM|PV-DBOW|
|-|-|
|negative sampling, `dm_concat=True`|negative sampling, `dm_train_words=True`|
|negative sampling, `dm_concat=False`|negative sampling, `dm_train_words=False`|
|hierarchical softmax, `dbow_train_words=True`|hierarchical softmax, `dbow_train_words=True`|
|hierarchical softmax, `dbow_train_words=False`|hierarchical softmax, `dbow_train_words=False`|

### Example: classifying IMDB movie reviews that are represented as real-valued vectors 

[IMDB movie review dataset](ai.stanford.edu/~amaas/data/sentiment/) contains 100k movie reviews, which are split into training set (25k, with labels), test set (25k, with labels), and unlabeled set (50k). A doc2vec model (`PV-DBOW`) is trained on the combined training set and unlabeld set (75k in total), with default parameter setting (run `python run_running.py --help` to see the default), which results in the vector representation of of 75k documents (we only need the 25k from training set to train the classifier).

Next the trained doc2vec model learns the vector representation of reviews in the test set (`python run_inferece --filenames_infer=/PATH/TO/FILENAMES_INFER`).

Given the vector representation of 25k labeled reviews in the training set, and 25k labeled reviews in the test set, a logistic regression is trained on training set and validated on test set (validation accuracy: 0.87).

### Reference
1. Q Le, T Mikolov - Distributed Representations of Sentences and Documents, ICML 2014
2. Gensim implementation by Radim Řehůřek, https://radimrehurek.com/gensim/models/doc2vec.html
