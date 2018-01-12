from word2vec import *

class Doc2Vec(Word2Vec):
  """Trains Doc2Vec model. The model can be trained on two model architectures 
  "Distributed bag of words" (`PV-DBOW`) or "Distributed memory" (`PV-DM`) and by two training algorithms 
  "negative sampling" (`ns`) or "hierarchical softmax" (`hs`). 

  `PV-DBOW` is implemeted as skip gram `sg` in Word2Vec; `PV-DM` is implemented as continuous bag of words
  `cbow` in Word2Vec.

  As in Word2Vec model, `hidden_layer_toggle` is used to switch between `PV-DBOW` and `PV-DM`, 
  `output_layer_toggle` is used to switch between `ns` and `hs`.
  """
  def __init__(self, dbow_train_words=True, dm_concat=False, **kwargs):
    super(Doc2Vec, self).__init__(**kwargs)
    self.dbow_train_words = dbow_train_words                              # If `True`, train on word-to-word examples in `PV-DBOW` architecture
    self.dm_concat = dm_concat                                            # If `True`, word and document vectors are concatenated (as opposed to averaged) as a single context vector; Valid only in `PV-DM` architecture
    self._concat_mode = (not self.hidden_layer_toggle) and self.dm_concat # Inidcates if model is in concatenate mode (See `dm_concat`)
    self._inference_mode = False                                          # Indicates if model is in inference mode (word vectors and output weights `syn1`, `biases` are freezed at the time of training).
 
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

    self._sents_covered += 1
    self._progress = self._sents_covered / float(self._total_sents)

  def build_vocab(self, sents):
    """Build vocabulary"""
    super(Doc2Vec, self).build_vocab(sents)
    if self._concat_mode:
      key, self.null_word = "\0", VocabWord(count=1, index=len(self.vocab), keep_prob=0., fraction=0., word="\0")
      self._unigram_count.append(self.null_word.count) 
      self.vocab[key] = self.null_word
      self.vocab_size = len(self.vocab)
      self.index2word.append(key)

  def _get_syn0_init_val(self, wordtags, doctags):
    syn0_w_init_val = np.vstack([self._seeded_vector(wordtags[i] + str(self.seed)) 
      for i in xrange(len(wordtags))]).astype(np.float32)
    syn0_d_init_val = np.vstack([self._seeded_vector("%d %s" % (self.seed, doctags[i]))
      for i in xrange(len(doctags))]).astype(np.float32)
    return syn0_w_init_val, syn0_d_init_val

  def create_variables(self, syn0_w_init_val, syn0_d_init_val, inference_mode):
    """Define trainable variables"""
    syn1_rows = self.vocab_size if self.output_layer_toggle else self.vocab_size - 1
    syn1_cols = self.size * (2 * self.window + 1) if self._concat_mode else self.size    

    with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
      syn0_w = tf.get_variable("syn0_w", initializer=syn0_w_init_val, dtype=tf.float32)
      syn1 = tf.get_variable("syn1", 
        initializer=tf.truncated_normal([syn1_rows, syn1_cols], 
        stddev=1.0/np.sqrt(self.size)), dtype=tf.float32)
      biases = tf.get_variable("biases", initializer=tf.zeros([syn1_rows]),
        dtype=tf.float32)
    syn0_d = tf.Variable(initial_value=syn0_d_init_val, name="syn0_d")
    syn0 = tf.concat([syn0_w, syn0_d], axis=0)
    var_list = [syn0_d] if inference_mode else [syn0_w, syn0_d, syn1, biases]
    return syn0, syn1, biases, var_list

  def _input_to_hidden(self, syn0, inputs):
    if self._concat_mode:
      return tf.reshape(tf.nn.embedding_lookup(syn0, inputs), [-1, self.size * (2 * self.window + 1)]) 
    else:
      return super(Doc2Vec, self)._input_to_hidden(syn0, inputs)

  def _get_sent_iter(self, sents):
    return itertools.chain(*itertools.tee(enumerate(sents, start=self.vocab_size), self.epochs))

  def _get_train_step(self, lr, loss, var_list):
    sgd = tf.train.GradientDescentOptimizer(lr)
    if self.clip_gradient:
      gradients, variables = zip(*sgd.compute_gradients(loss, var_list=var_list))
      gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
      return sgd.apply_gradients(zip(gradients, variables))
    else:
      return sgd.minimize(loss, var_list=var_list)

  def train(self, sents, doctags, sess, inference_mode=False):
    """Learn document vectors.

    If `inference_mode` is `True`, word vectors and output weights are fixed during training. 
    Args:
      `sents`: an iterable of list of strings
       `doctags`: an iterable of strings, with same length as `sents`
      `inference_mode`: bool
      `sess`: TensorFlow session

    Returns:
      `WordVectors` instance

     """
    if not hasattr(self, "vocab"):
      if inference_mode:
        raise ValueError("Model is not yet trained for inference mode.")
      self.build_vocab(sents)
      if not self.output_layer_toggle:
        self.create_binary_tree()

    sents_iter = self._get_sent_iter(sents)
    batch_iter = self.generate_batch(sents_iter)

    self._reset(sents)

    syn0_w_init_val, syn0_d_init_val = self._get_syn0_init_val(self.index2word, doctags)
    self.syn0, self.syn1, self.biases, var_list = self.create_variables(syn0_w_init_val, syn0_d_init_val, inference_mode)
    inputs, labels = tf.placeholder(dtype=tf.int64), tf.placeholder(dtype=tf.int64)
    progress = tf.placeholder(dtype=tf.float32)
    lr = tf.maximum(self.alpha * (1 - progress) + self.min_alpha * progress, self.min_alpha)
    loss = self.build_graph(inputs, labels, self.syn0, self.syn1, self.biases)
    train_step = self._get_train_step(lr, loss, var_list)
    sess.run([var.initializer for var in var_list])

    self._do_train(batch_iter, inputs, labels, progress, lr, loss, train_step, sess)

    syn0_final = self.syn0.eval()
    syn0_w_final, syn0_d_final = syn0_final[:self.vocab_size], syn0_final[self.vocab_size:]
    if self.norm_embed:
      syn0_final = syn0_final / np.linalg.norm(syn0_final, axis=1)
    return WordVectors(syn0_w_final, self.vocab, self.index2word), \
      DocVectors(syn0_d_final, doctags, sents)


class DocVectors(object):
  """Trained doc2vec model. Stores the document tags, tag-to-document mapping, and 
  final document embeddings"""
  def __init__(self, syn0_final, doctags, sents):
    self.syn0_final = syn0_final
    self.doctags = doctags
    self.sents = sents
    self.doc_dict = dict([(tag, (index, sent)) for index, (tag, sent)
      in enumerate(zip(self.doctags, sents))]) 
  
  def __contains__(self, doctag):
    return doctag in self.doc_dict

  def __getitem__(self, doctag):
    return self.syn_final[self.doc_dict[doctag][0]]
