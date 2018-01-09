from word2vec import *

class Doc2Vec(Word2Vec):
  def __init__(self, dbow_train_words=True, dm_concat=False, **kwargs):
    super(Doc2Vec, self).__init__(**kwargs)
    self.dbow_train_words = dbow_train_words
    self.dm_concat = dm_concat
    self._concat_mode = (not self.hidden_layer_toggle) and self.dm_concat
    self._inference_mode = False 
 
  def _get_tarcon_generator(self, sents_iter):
    return (tarcon for id_, sent in sents_iter for tarcon in self._tarcon_per_sent(sent, id_))

  def _cbow_ns(self, batch):
    if self._concat_mode:
      return np.concatenate(batch[1]), np.array(batch[0])
    else: 
      segment_ids = np.repeat(xrange(len(batch[0])), map(len, batch[1]))
      return np.array([np.concatenate(batch[1]), segment_ids]).T, np.array(batch[0])

  def _cbow_hs(self, batch):
    paths = [np.array([self.vocab[self.index2word[i]].point, self.vocab[self.index2word[i]].code]).T for i in batch[0]]
    code_lengths = map(len, paths)
    labels = np.vstack(paths)
    contexts_repeated = np.repeat(batch[1], code_lengths, axis=0)
    if self._concat_mode:
      inputs = np.concatenate(contexts_repeated)
    else:
      contexts_repeated_segment_ids = np.repeat(xrange(len(contexts_repeated)), map(len, contexts_repeated))
      inputs = np.array([np.concatenate(contexts_repeated), contexts_repeated_segment_ids]).T
    return inputs, labels

  def _tarcon_per_target(self, index_list, word_index, id_):
    target = index_list[word_index]
    reduced_size = 0 if self._concat_mode else self._random_state.randint(self.window)
    left = self._words_to_left(index_list, word_index, reduced_size)
    right = self._words_to_right(index_list, word_index, reduced_size)
    contexts = left + right

    if contexts:
      if self.hidden_layer_toggle: # PV-DBOW/skip gram
        if self.dbow_train_words:
          for context in contexts:
            yield target, context
        yield id_, target
      else: # PV-DM/cbow
        yield target, left + [id_] + right 

  def _tarcon_per_sent(self, sent, id_):
    sent_subsampled = [self.vocab[word].index for word in sent if self._keep_word(word)]
#    print sent_subsampled
    if self._concat_mode: 
      sent_subsampled_padded = ([self.null_word.index] * self.window) + \
        sent_subsampled + ([self.null_word.index] * self.window)
      for word_index in xrange(self.window, len(sent_subsampled) + self.window):
        for tarcon in self._tarcon_per_target(sent_subsampled_padded, word_index, id_):
          yield tarcon 
    else:
      for word_index in xrange(len(sent_subsampled)):
        for tarcon in self._tarcon_per_target(sent_subsampled, word_index, id_):
          yield tarcon

    if not self._inference_mode:
      self._sents_covered += 1
      self._progress = self._sents_covered / float(self.total_sents)

  def build_vocab(self, sents):
    super(Doc2Vec, self).build_vocab(sents)
    self._docvec_tags = ["Document_%d" % i for i in xrange(len(sents))]
    self.num_docs = len(sents)

    if self._concat_mode:
      key, self.null_word = "\0", VocabWord(count=1, index=len(self.vocab), keep_prob=0., fraction=0., word="\0")
      self._unigram_count.append(self.null_word.count) 
      self.vocab[key] = self.null_word
      self.vocab_size = len(self.vocab)
      self.index2word.append(key)

  def _get_init_val_syn0(self):
    if not hasattr(self, "_init_val_syn0"):
      docvec_tags = self._docvec_tags
      init_val_syn0 = np.empty((self.vocab_size + self.num_docs + 1, self.size), dtype=np.float32)
      for i in xrange(self.vocab_size):
        init_val_syn0[i] = self._seeded_vector(self.index2word[i] + str(self.seed))
      for i in xrange(self.num_docs):
        init_val_syn0[i + self.vocab_size] = self._seeded_vector("%d %s" % (self.seed, docvec_tags[i]))
      init_val_syn0[-1] = self._seeded_vector("%d %s" % (self.seed, "Document_test"))
      self._init_val_syn0 = init_val_syn0
    return self._init_val_syn0

  def create_variables(self):
    init_val_syn0 = self._get_init_val_syn0();  print init_val_syn0[:self.vocab_size]; print init_val_syn0[self.vocab_size:-1]
    syn1_rows = self.vocab_size if self.output_layer_toggle else self.vocab_size - 1
    syn1_cols = self.size * (2 * self.window + 1) if self._concat_mode else self.size

    if not self._inference_mode:
      with tf.variable_scope("model"):
        self._syn0_word = tf.get_variable("syn0_word", initializer=init_val_syn0[:self.vocab_size], dtype=tf.float32)
        self._syn0_doc = tf.get_variable("syn0_doc", initializer=init_val_syn0[self.vocab_size:-1], dtype=tf.float32)
        self._syn0 = tf.concat([self._syn0_word, self._syn0_doc], axis=0)
        self._syn1 = tf.get_variable("syn1",
          initializer=tf.truncated_normal([syn1_rows, syn1_cols], 
          stddev=1.0/np.sqrt(self.size)), dtype=tf.float32)
        self._biases = tf.get_variable("biases", initializer=tf.zeros([syn1_rows]), dtype=tf.float32)
    else:
      with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
        self._syn0_word = tf.get_variable("syn0_word", initializer=init_val_syn0[:self.vocab_size], dtype=tf.float32)
        self._syn0_doc_test = tf.get_variable("syn0_doc_test", initializer=init_val_syn0[-1:], dtype=tf.float32)

        self._syn0_word = tf.stop_gradient(self._syn0_word)

        self._syn0 = tf.concat([self._syn0_word, self._syn0_doc_test], axis=0)
        self._syn1 = tf.get_variable("syn1",
          initializer=tf.truncated_normal([syn1_rows, syn1_cols],
          stddev=1.0/np.sqrt(self.size)), dtype=tf.float32)

        self._syn1 = tf.stop_gradient(self._syn1)
        
        self._biases = tf.get_variable("biases", initializer=tf.zeros([syn1_rows]), dtype=tf.float32)

        self._biases = tf.stop_gradient(self._biases)

    inputs = tf.placeholder(dtype=tf.int64)
    labels = tf.placeholder(dtype=tf.int64)
    return inputs, labels

  def _input_to_hidden(self, syn0, inputs):
    if self._concat_mode:
      return tf.reshape(tf.nn.embedding_lookup(syn0, inputs), [-1, self.size * (2 * self.window + 1)]) 
    else:
      return super(Doc2Vec, self)._input_to_hidden(syn0, inputs)

  def _get_sent_iter(self, sents):
    return itertools.chain(*itertools.tee(enumerate(sents, start=self.vocab_size), self.epochs))

  def _wrap_syn0(self, syn0_final):
    return WordVectors(syn0_final[:self.vocab_size], self.vocab, self.index2word), \
            DocVectors(syn0_final[self.vocab_size:], self._docvec_tags, self._raw_corpus)

  def infer_vector(self, sent, sess, alpha=0.1, min_alpha=1e-4, steps=10):
    self._inference_mode = True
    id_ = self.vocab_size

    sents_iter = itertools.chain(*itertools.tee([(id_, sent)], steps))
    batch_iter = self.generate_batch(sents_iter)
    progress = tf.placeholder(dtype=tf.float32, shape=[])
    lr = tf.maximum(alpha * (1 - progress) + min_alpha * progress, min_alpha)

    inputs, labels = self.create_variables()
    loss = self.loss_ns(inputs, labels) if self.output_layer_toggle \
            else self.loss_hs(inputs, labels)

    train_step = self._get_train_step(lr, loss)
    sess.run(self._syn0_doc_test.initializer)

    for step, batch in enumerate(batch_iter):
      feed_dict = {inputs: batch[0], labels: batch[1], progress: float(step) / steps}

      _, loss_val, lr_val = sess.run([train_step, loss, lr], feed_dict)
    vec = self._syn0_doc_test.eval() 

    return vec


class DocVectors(object):
  def __init__(self, syn0_final, docvec_tags, sents):
    self.syn0_final = syn0_final
    self.docvec_tags = docvec_tags
    self.sents = sents
    self.doc_dict = dict([(tag, (index, sent)) for index, (tag, sent)
      in enumerate(zip(self.docvec_tags, sents))]) 
  
  def __contains__(self, docvec_tag):
    return docvec_tag in self.doc_dict

  def __getitem__(self, docvec_tag):
    return self.syn_final[self.doc_dict[docvec_tag][0]]
