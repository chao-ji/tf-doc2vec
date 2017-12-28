from word2vec import *

class Doc2Vec(Word2Vec):
  def __init__(self, dbow_train_words=True, **kwargs):
    super(Doc2Vec, self).__init__(**kwargs)
    self.dbow_train_words = dbow_train_words

  def _get_tarcon_generator(self, sents_iter):
    return (tarcon for id_, sent in sents_iter for tarcon in self._tarcon_per_sent(sent, id_))

  def _tarcon_per_target(self, sent_trimmed, word_index, id_):
    target = sent_trimmed[word_index]
    reduced_size = self._random_state.randint(self.window)
    before = map(lambda i: sent_trimmed[i],
              xrange(max(word_index - self.window + reduced_size, 0), word_index)) 
    after = map(lambda i: sent_trimmed[i],
              xrange(word_index + 1, min(word_index + 1 + self.window - reduced_size, len(sent_trimmed))))
    contexts = before + after

    if contexts:
      if self.hidden_layer_toggle: # skip gram/dbow
        if self.dbow_train_words:
          for context in contexts:
            yield target, context
        yield id_, target
      else: # cbow/dm
        yield target, contexts + [id_]

  def _tarcon_per_sent(self, sent, id_):
    sent_trimmed = [self.vocab[word].index for word in sent if self._keep_word(word)]

    for word_index in xrange(len(sent_trimmed)):
      for tarcon in self._tarcon_per_target(sent_trimmed, word_index, id_):
        yield tarcon

    self._sents_covered += 1
    self._progress = self._sents_covered / float(self.total_sents)

  def build_vocab(self, sents):
    super(Doc2Vec, self).build_vocab(sents)
    self._docvec_tags = ["Document_%d" % i for i in xrange(len(sents))]

  def initialize_variables(self):
    docvec_tags = self._docvec_tags
    def seeded_vector(seed_string):
      random = np.random.RandomState(hash(seed_string) & 0xffffffff)
      return (random.rand(self.size) - 0.5) / self.size

    syn0_val = np.empty((self.vocabulary_size+len(docvec_tags), self.size), dtype=np.float32)
    for i in xrange(self.vocabulary_size):
      syn0_val[i] = self._seeded_vector(self.index2word[i] + str(self.seed))
    for i in xrange(len(docvec_tags)):
      syn0_val[i+self.vocabulary_size] = self._seeded_vector("%d %s" % (self.seed, docvec_tags[i])) 

    self._syn0 = tf.Variable(syn0_val, dtype=tf.float32)
    self._syn1 = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.size],
                                stddev=1.0/np.sqrt(self.size)), dtype=tf.float32)
    self._biases = tf.Variable(tf.zeros([self.vocabulary_size]), dtype=tf.float32)
    inputs = tf.placeholder(dtype=tf.int64, shape=[None] if self.hidden_layer_toggle else [None, 2])
    labels = tf.placeholder(dtype=tf.int64, shape=[None] if self.output_layer_toggle else [None, 3])
    
    return inputs, labels

  def _get_sent_iter(self, sents):
    return itertools.chain(*itertools.tee(enumerate(sents, self.vocabulary_size), self.epochs))

  def _save_embedding(self, syn0_final):
    self.word_embedding = syn0_final[:self.vocabulary_size]
    self.doc_embedding = syn0_final[self.vocabulary_size:]
    return WordEmbeddings(self.word_embedding, self.vocab, self.index2word)
