from word2vec import *

class Doc2Vec(Word2Vec):
  def __init__(self, dbow_train_words=True, **kwargs):
    super(Doc2Vec, self).__init__(**kwargs)
    self.dbow_train_words = dbow_train_words

  def generate_batch(self, sents_iter):
    vocab, index2word = self.vocab, self.index2word

    def _sg_ns(batch):
      return np.array(batch[0]), np.array(batch[1])
    def _cbow_ns(batch):
      segment_ids = np.repeat(xrange(len(batch[0])), map(len, batch[1]))
      return np.array([np.concatenate(batch[1]), segment_ids]).T, np.array(batch[0])
    def _sg_hs(batch):
      tmp = [np.array([vocab[index2word[i]].point, vocab[index2word[i]].code]).T for i in batch[1]]
      code_lengths = map(len, tmp)
      segment_ids = np.repeat(xrange(len(batch[0])), code_lengths).reshape((-1, 1))
      labels = np.hstack([np.vstack(tmp), segment_ids])
      inputs = np.repeat(batch[0], code_lengths)
      return inputs, labels
    def _cbow_hs(batch):
      tmp = [np.array([vocab[index2word[i]].point, vocab[index2word[i]].code]).T for i in batch[0]]
      code_lengths = map(len, tmp)
      segment_ids = np.repeat(xrange(len(batch[0])), code_lengths).reshape((-1, 1))
      labels = np.hstack([np.vstack(tmp), segment_ids])
      contexts_repeated = np.repeat(batch[1], code_lengths)
      contexts_repeated_segment_ids = np.repeat(xrange(len(contexts_repeated)), map(len, contexts_repeated))
      inputs = np.array([np.concatenate(contexts_repeated), contexts_repeated_segment_ids]).T
      return inputs, labels

    def _yield_fn(batch):
      opts = self.opts
      if opts[0] and opts[2]:
        return _sg_ns(batch)
      elif opts[1] and opts[2]:
        return _cbow_ns(batch)
      elif opts[0] and opts[3]:
        return _sg_hs(batch)
      elif opts[1] and opts[3]:
        return _cbow_hs(batch)

    tarcon_generator = (tarcon for id_, sent in sents_iter for tarcon in self._tarcon_per_sent(sent, id_))

    batch = []
    for tarcon in tarcon_generator:
      if len(batch) < self.max_batch_size:
        batch.append(tarcon)
      else:
        batch = zip(*batch)
        yield _yield_fn(batch)
        batch = [tarcon]

    if batch: # last batch if not empty
      batch = zip(*batch)
      yield _yield_fn(batch)
      batch = []

  def _tarcon_per_target(self, sent_trimmed, word_index, id_):
    target = sent_trimmed[word_index]
    reduced_size = self._random_state.randint(self.window)
    before = map(lambda i: sent_trimmed[i],
              xrange(max(word_index - self.window + reduced_size, 0), word_index)) 
    after = map(lambda i: sent_trimmed[i],
              xrange(word_index + 1, min(word_index + 1 + self.window - reduced_size, len(sent_trimmed))))
    contexts = before + after

    if contexts:
      if self.opts[0]: # skip gram/dbow
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

  def initialize_variables(self, docvec_tags):
    def seeded_vector(seed_string):
      random = np.random.RandomState(hash(seed_string) & 0xffffffff)
      return (random.rand(self.embedding_size) - 0.5) / self.embedding_size

    syn0_val = np.empty((self.vocabulary_size+len(docvec_tags), self.embedding_size), dtype=np.float32)
    for i in xrange(self.vocabulary_size):
      syn0_val[i] = self._seeded_vector(self.index2word[i] + str(self.seed))
    for i in xrange(len(docvec_tags)):
      syn0_val[i+self.vocabulary_size] = self._seeded_vector("%d %s" % (self.seed, docvec_tags[i])) 

    self._syn0 = tf.Variable(syn0_val, dtype=tf.float32)
    self._syn1 = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                stddev=1.0/np.sqrt(self.embedding_size)), dtype=tf.float32)
    self._biases = tf.Variable(tf.zeros([self.vocabulary_size]), dtype=tf.float32)
    inputs = tf.placeholder(dtype=tf.int64, shape=[None] if self.opts[0] else [None, 2])
    labels = tf.placeholder(dtype=tf.int64, shape=[None] if self.opts[2] else [None, 3])
    
    return inputs, labels

  def train(self, sents, sess):
    self.build_vocab(sents)
    if self.opts[3]:
      self.create_binary_tree()

    sents_iter = itertools.chain(*itertools.tee(enumerate(sents, self.vocabulary_size), self.epochs))
    X_iter = self.generate_batch(sents_iter)

    progress = tf.placeholder(dtype=tf.float32, shape=[])
    lr = tf.maximum(self.start_alpha * (1 - progress) + self.end_alpha * progress, self.end_alpha)

    docvec_tags = ["TRAIN_%d" % i for i in xrange(len(sents))]
    inputs, labels = self.initialize_variables(docvec_tags)

    if self.opts[2]: # negative sampling
      loss = self.loss_ns(inputs, labels)
    else: # hierarchical softmax
      loss = self.loss_hs(inputs, labels)

    train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    sess.run(tf.global_variables_initializer())
    average_loss = 0.

    for step, batch in enumerate(X_iter):
      feed_dict = {inputs: batch[0], labels: batch[1]}
      feed_dict[progress] = self._progress

      _, loss_val, lr_val = sess.run([train_step, loss, lr], feed_dict)

      average_loss += loss_val.mean()
      if step % self.log_every_n_steps == 0:
        if step > 0:
          average_loss /= self.log_every_n_steps
        print "step =", step, "average_loss =", average_loss, "learning_rate =", lr_val
        average_loss = 0.

    syn0_final, syn1_final = self._syn0.eval(), self._syn1.eval()
    if self.norm_embeddings:
      norm =  np.sqrt(np.square(syn0_final).sum(axis=1, keepdims=True))
      syn0_final = syn0_final / norm

    return Embeddings(syn0_final, self.vocab, self.index2word)
