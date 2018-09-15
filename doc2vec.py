import tensorflow as tf

from word2vec.word2vec import Word2VecModel


class Doc2VecModel(Word2VecModel):
  """Doc2VecModel."""
  def __init__(self,
               arch='PV-DM',
               algm='negative_sampling',
               embed_size=100,
               batch_size=32,
               negatives=5,
               power=0.75,
               alpha=0.025,
               min_alpha=0.0001, 
               add_bias=True,
               random_seed=0,
               dm_concat=True,
               window_size=5):
    """Constructor.

    Args:
      arch: string scalar, architecture ('PV-DBOW' or 'PV-DM').
      algm: string scalar: training algorithm ('negative_sampling' or
        'hierarchical_softmax').
      embed_size: int scalar, length of word vector.
      batch_size: int scalar, batch size.
      negatives: int scalar, num of negative words to sample.
      power: float scalar, distortion for negative sampling. 
      alpha: float scalar, initial learning rate.
      min_alpha: float scalar, final learning rate.
      add_bias: bool scalar, whether to add bias term to dotproduct 
        between syn0 and syn1 vectors.
      random_seed: int scalar, random_seed.
      dm_concat: bool scalar, whether to concatenate word and document vectors
        instead of averaging them in dm architecture.
      window_size: int scalar, num of words on the left or right side of
        target word within a window.
    """
    super(Doc2VecModel, self).__init__(arch, algm, embed_size, batch_size,
        negatives, power, alpha, min_alpha, add_bias, random_seed)
    self._dm_concat = (arch == 'PV-DM') and dm_concat
    self._window_size = window_size

    self._syn0_w = None
    self._syn0_d = None

  @property
  def syn0_w(self):
    return self._syn0_w

  @property
  def syn0_d(self):
    return self._syn0_d

  def get_save_list(self):
    """Returns the list of variables to be saved by tf.train.Saver()."""
    return [w for w in tf.global_variables() if w != self._syn0_d] 

  def _build_loss(self, inputs, labels, unigram_counts, num_docs, scope=None):
    """Builds the graph that leads from data tensors (`inputs`, `labels`)
    to loss. Has the side effect of setting attribute `syn0_w`, `syn0_d`.

    Args:
      inputs: int tensor of shape [batch_size] (PV-DBOW) or 
        [batch_size, 2*window_size+2] (PV-DM) 
      labels: int tensor of shape [batch_size] (negative_sampling) or
        [batch_size, 2*max_depth+1] (hierarchical_softmax)
      unigram_count: list of int, holding word counts. Index of each entry
        is the same as the word index into the vocabulary.
      num_docs: int scalar, num of documents.
      scope: string scalar, scope name.

    Returns:
      loss: float tensor, cross entropy loss. 
    """
    syn0_w, syn0_d, syn1, biases = self._create_embeddings(
        len(unigram_counts), num_docs)
    syn0 = tf.concat([syn0_w, syn0_d], axis=0)
    self._syn0_w, self._syn0_d = syn0_w, syn0_d
    with tf.variable_scope(scope, 'Loss', [inputs, labels, syn0, syn1, biases]):
      if self._algm == 'negative_sampling':
        loss = self._negative_sampling_loss(
            unigram_counts, inputs, labels, syn0, syn1, biases)
      elif self._algm == 'hierarchical_softmax':
        loss = self._hierarchical_softmax_loss(
            inputs, labels, syn0, syn1, biases)
      return loss
    
  def _create_embeddings(self, vocab_size, num_docs, scope=None):
    """Creates initial word and document embedding variables.

    Args:
      vocab_size: int scalar, num of words in vocabulary.
      num_docs: int scalar, num of documents.
      scope: string scalar, scope name.

    Returns:
      syn0_w: float tensor of shape [vocab_size, embed_size], input word
        embeddings (i.e. weights of hidden layer).
      syn0_d: float tensor of shape [num_docs, embed_size], input doc
        embeddings (i.e. weights of hidden layer).
      syn1: float tensor of shape [syn1_rows, embed_size], output word
        embeddings (i.e. weights of output layer).
      biases: float tensor of shape [syn1_rows], biases added onto the logits.
    """   
    syn1_rows = (vocab_size if self._algm == 'negative_sampling'
                            else vocab_size - 1)
    syn1_cols = (self._embed_size*(2*self._window_size+1)
        if self._dm_concat else self._embed_size)
    with tf.variable_scope(scope, 'Embedding'):
      syn0_init = tf.random_uniform([vocab_size + num_docs, self._embed_size],
          -0.5/self._embed_size, 0.5/self._embed_size, seed=self._random_seed)

      syn0_w = tf.get_variable('syn0_w', initializer=syn0_init[:vocab_size])
      syn0_d = tf.get_variable('syn0_d', initializer=syn0_init[vocab_size:])
      syn1 = tf.get_variable('syn1', initializer=tf.random_uniform([
          syn1_rows, syn1_cols], -0.1, 0.1))
      biases = tf.get_variable('biases', initializer=tf.zeros([syn1_rows]))
      return syn0_w, syn0_d, syn1, biases

  def _get_inputs_syn0(self, syn0, inputs):
    """Builds the activations of hidden layer given input word and doc 
    embeddings `syn0` (concat of `syn0_w` and `syn0_d`) and input word and 
    doc indices.

    Args:
      syn0: float tensor of shape [vocab_size + num_docs, embed_size]
      inputs: int tensor of shape [batch_size] (PV-BOW) or 
        [batch_size, 2*window_size+2] (PV-DM)

    Returns:
      inputs_syn0: [batch_size, embed_size]
    """    
    if self._dm_concat:
      inputs_syn0 = tf.reshape(tf.nn.embedding_lookup(syn0, inputs[:, :-1]), 
          [-1, self._embed_size*(2*self._window_size+1)])
    elif self._arch == 'PV-DBOW':
      inputs_syn0 = tf.gather(syn0, inputs)
    else:
      inputs_syn0 = []
      contexts_list = tf.unstack(inputs)
      for contexts in contexts_list:
        context_words = contexts[:-1]
        true_size = contexts[-1]
        inputs_syn0.append(
            tf.reduce_mean(tf.gather(syn0, context_words[:true_size]), axis=0))
      inputs_syn0 = tf.stack(inputs_syn0)
    return inputs_syn0

  def _train_fn(self, dataset, filenames, is_inferring=False):
    """Adds training related ops to the graph. The `var_list` depends on whether
    `is_inferring` is True or False.

    Args:
      dataset: a `Doc2VecDataset` instance.
      filenames: a list of strings, holding names of text files.

    Returns: 
      to_be_run_dict: dict mapping from names to tensors/operations, holding
        the following entries:
        { 'grad_update_op': optimization ops,
          'loss': cross entropy loss,
          'learning_rate': float-scalar learning rate}
    """
    num_docs = sum([len(list(open(fn))) for fn in filenames])

    tensor_dict = dataset.get_tensor_dict(filenames)
    inputs, labels = tensor_dict['inputs'], tensor_dict['labels']

    loss = self._build_loss(inputs, labels, dataset.unigram_counts, num_docs)

    learning_rate = tf.maximum(self._alpha * (1 - tensor_dict['progress'][0]) +
         self._min_alpha * tensor_dict['progress'][0], self._min_alpha)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    var_list = [self._syn0_d] if is_inferring else tf.global_variables()
    grad_update_op = optimizer.minimize(loss,var_list=var_list)
    to_be_run_dict = {'grad_update_op': grad_update_op,
                      'loss': loss,
                      'learning_rate': learning_rate}
    return to_be_run_dict
    

class Doc2VecTrainer(Doc2VecModel):
  """Performs training of doc2vec model."""
  def train(self, dataset, filenames):
    """Adds training related ops to the graph. All variables (`syn0_w`,
    `syn0_d`, `syn1`, `biases`) will be updated. 

    Args:
      dataset: a `Doc2VecDataset` instance.
      filenames: a list of strings, holding names of text files.

    Returns: 
      to_be_run_dict: dict mapping from names to tensors/operations, holding
        the following entries:
        { 'grad_update_op': optimization ops,
          'loss': cross entropy loss,
          'learning_rate': float-scalar learning rate}
    """
    to_be_run_dict = self._train_fn(dataset, filenames, False)
    return to_be_run_dict


class Doc2VecInferencer(Doc2VecModel):
  """Performs inferences on vectors of unseen documents (not appearing in
   training set).
  """
  def infer(self, dataset, filenames):
    """Adds training related ops to the graph. Only document embeddings `syn0_d`
    will be updated.

    Args:
      dataset: a `Doc2VecDataset` instance.
      filenames: a list of strings, holding names of text files.

    Returns: 
      to_be_run_dict: dict mapping from names to tensors/operations, holding
        the following entries:
        { 'grad_update_op': optimization ops,
          'loss': cross entropy loss,
          'learning_rate': float-scalar learning rate}
    """   
    to_be_run_dict = self._train_fn(dataset, filenames, True)
    return to_be_run_dict
     
