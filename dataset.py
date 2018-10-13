import tensorflow as tf

from word2vec.dataset import Word2VecDataset
from word2vec.dataset import get_word_indices
from word2vec.dataset import subsample

OOV_ID = -1


class Doc2VecDataset(Word2VecDataset):
  """Dataset for generating matrices holding word indices to train Doc2Vec 
  models.
  """
  def __init__(self,
               arch='PV-DM',
               algm='negative_sampling',
               epochs=1,
               batch_size=32,
               max_vocab_size=0,
               min_count=10,
               sample=1e-3,
               window_size=5,
               dbow_train_words=False,
               dm_concat=True):
    """Constructor.

    Args:
      arch: string scalar, architecture ('PV-DBOW' or 'PV-DM').
      algm: string scalar: training algorithm ('negative_sampling' or
        'hierarchical_softmax').
      epochs: int scalar, num times the dataset is iterated.
      batch_size: int scalar, the returned tensors in `get_tensor_dict` have
        shapes [batch_size, :]. 
      max_vocab_size: int scalar, maximum vocabulary size. If > 0, the top 
        `max_vocab_size` most frequent words are kept in vocabulary.
      min_count: int scalar, words whose counts < `min_count` are not included
        in the vocabulary.
      sample: float scalar, subsampling rate.
      window_size: int scalar, num of words on the left or right side of
        target word within a window.
      dbow_train_words: bool scalar, whether to train word vectors in dbow
        architecture.
      dm_concat: bool scalar, whether to concatenate word and document vectors
        instead of averaging them in dm architecture.
    """
    super(Doc2VecDataset, self).__init__(
        arch=arch, 
        algm=algm, 
        epochs=epochs, 
        batch_size=batch_size, 
        max_vocab_size=max_vocab_size,
        min_count=min_count, 
        sample=sample, 
        window_size=window_size)
    self._dbow_train_words = dbow_train_words
    self._dm_concat = (arch == 'PV-DM') and dm_concat

  def build_vocab(self, filenames):
    """Builds vocabulary. Adds the dummy word for padding purpose if concat
    mode is enabled (`dm_concat`) for dm architecture.

    Has the side effect of setting the following attributes:   
    - table_words: list of string, holding the list of vocabulary words. Index
        of each entry is the same as the word index into the vocabulary.
    - unigram_counts: list of int, holding word counts. Index of each entry
        is the same as the word index into the vocabulary.
    - keep_probs: list of float, holding words' keep prob for subsampling. 
        Index of each entry is the same as the word index into the vocabulary.
    - corpus_size: int scalar, effective corpus size.

    Args:
      filenames: list of strings, holding names of text files.
    """
    super(Doc2VecDataset, self).build_vocab(filenames)
    if self._dm_concat:
      self._table_words.append('\0')
      self._unigram_counts.append(1)
      self._keep_probs.append(0.)

  def _prepare_inputs_labels(self, tensor):
    """Set shape of `tensor` according to architecture and training algorithm,
    and split `tensor` into `inputs` and `labels`.

    Args:
      tensor: rank-2 int tensor, holding word indices for prediction inputs
        and prediction labels, returned by `generate_instances`.

    Returns:
      inputs: rank-2 int tensor, holding word indices for prediction inputs. 
      labels: rank-2 int tensor, holding word indices for prediction labels.
    """
    if self._arch == 'PV-DBOW':
      if self._algm == 'negative_sampling':
        tensor.set_shape([self._batch_size, 2])
      else:
        tensor.set_shape([self._batch_size, 2*self._max_depth+2])
      inputs = tensor[:, :1]
      labels = tensor[:, 1:]
    else:
      if self._algm == 'negative_sampling':
        tensor.set_shape([self._batch_size, 2*self._window_size+3])
      else:
        tensor.set_shape([self._batch_size,
            2*self._window_size+2*self._max_depth+3])
      inputs = tensor[:, :2*self._window_size+2]
      labels = tensor[:, 2*self._window_size+2:]
    return inputs, labels
    
  def get_tensor_dict(self, filenames):
    """Generates tensor dict mapping from tensor names to tensors.

    Args:
      filenames: list of strings, holding names of text files.
      
    Returns:
      tensor_dict: a dict mapping from tensor names to tensors with shape being:
        when arch=='PV-DBOW', algm=='negative_sampling'
          inputs: [N],                    labels: [N]
        when arch=='PV-DM', algm=='negative_sampling'
          inputs: [N, 2*window_size+2],   labels: [N]
        when arch=='PV-DBOW', algm=='hierarchical_softmax'
          inputs: [N],                    labels: [N, 2*max_depth+1]
        when arch=='PV-DM', algm=='hierarchical_softmax'
          inputs: [N, 2*window_size+2],   labels: [N, 2*max_depth+1]
        progress: [N], the percentage of sentences covered so far. Used to 
          compute learning rate.
    """
    table_words = self._table_words
    unigram_counts = self._unigram_counts
    keep_probs = self._keep_probs
    if not table_words or not unigram_counts or not keep_probs:
      raise ValueError('`table_words`, `unigram_counts`, and `keep_probs` must',
          'be set by calling `build_vocab()`')

    if self._algm == 'hierarchical_softmax':
      codes_points = tf.constant(self._build_binary_tree(unigram_counts))
    elif self._algm == 'negative_sampling':
      codes_points = None
    else:
      raise ValueError('algm must be hierarchical_softmax or negative_sampling')

    table_words = tf.contrib.lookup.index_table_from_tensor(
        tf.constant(table_words), default_value=OOV_ID)
    keep_probs = tf.constant(keep_probs)

    num_docs = sum([len(list(open(fn))) for fn in filenames])

    vocab_size = len(self._unigram_counts)

    dataset = tf.data.Dataset.zip((tf.data.TextLineDataset(filenames),
        tf.data.Dataset.from_tensor_slices(
            tf.range(vocab_size, num_docs + vocab_size)))).repeat(self._epochs)
    dataset = tf.data.Dataset.zip((dataset, tf.data.Dataset.from_tensor_slices(
        tf.range(num_docs * self._epochs) / (num_docs * self._epochs))))
 
    dataset = dataset.map(lambda doc, progress:
        ((get_word_indices(doc[0], table_words), doc[1]), progress))
    dataset = dataset.map(lambda doc, progress:
        ((subsample(doc[0], keep_probs), doc[1]), progress))
    dataset = dataset.filter(lambda doc, progress:
        tf.greater(tf.size(doc[0]), 1))
    dataset = dataset.map(lambda doc, progress: (generate_instances(
        indices=doc[0], 
        arch=self._arch, 
        doc_id=doc[1], 
        window_size=self._window_size, 
        null_word_index=vocab_size - 1,
        dm_concat=self._dm_concat,
        dbow_train_words=self._dbow_train_words,
        codes_points=codes_points), progress))
    dataset = dataset.map(lambda instances, progress: (
        instances, tf.fill(tf.shape(instances)[:1], progress)))

    dataset = dataset.flat_map(lambda instances, progress:
        tf.data.Dataset.from_tensor_slices((instances, progress)))
    dataset = dataset.batch(self._batch_size, drop_remainder=True)

    iterator = dataset.make_initializable_iterator()
    self._iterator_initializer = iterator.initializer
    tensor, progress = iterator.get_next()

    progress.set_shape([self._batch_size])

    inputs, labels = self._prepare_inputs_labels(tensor)
    if self._arch == 'PV-DBOW':
      inputs = tf.squeeze(inputs, axis=1)
    if self._algm == 'negative_sampling':
      labels = tf.squeeze(labels, axis=1)
    return {'inputs': inputs, 'labels': labels, 'progress': progress}


def generate_instances(
    indices, 
    arch, 
    doc_id, 
    window_size, 
    null_word_index=0, 
    dm_concat=True, 
    dbow_train_words=False, 
    codes_points=None):
  """Generates matrices holding word indices to be passed to Doc2Vec models 
  for each sentence.

  Args:
    indices: rank-1 int tensor, the word indices within a sentence after
      subsampling.
    arch: scalar string, architecture ('PV-DBOW' or 'PV-DM').
    doc_id: int scalar, the unique ID assigned to a document. Starting from
      `vocab_size`.
    window_size: int scalar, num of words on the left or right side of
      target word within a window.
    null_word_index: int scalar, the ID (index) of the dummy word if 
      `dm_concat` is True.
    dm_concat: bool scalar, whether to concat word and doc vectors for 
      dm architecture.
    dbow_train_words: bool scalar, whether to add instances to train word 
      vectors in dbow architecture.
    codes_points: None, or an int tensor of shape [vocab_size, 2*max_depth+1] 
      where each row holds the codes (0-1 binary values) padded to `max_depth`, 
      and points (non-leaf node indices) padded to `max_depth`, of each 
      vocabulary word. The last entry is the true length of code and point 
      (<= `max_depth`).

  Returns: 
    instances: an int tensor holding word indices, with shape being
      when arch=='PV-DBOW', algm=='negative_sampling'
        shape: [N, 2]
      when arch=='PV-DM', algm=='negative_sampling'
        shape: [N, 2*window_size+3]
      when arch=='PV-DBOW', algm=='hierarchical_softmax'
        shape: [N, 2*max_depth+2]
      when arch=='PV-DM', algm='hierarchical_softmax'
        shape: [N, 2*window_size+2*max_depth+3]
  """
  def per_target_fn(index, init_array):
    target_index = index + window_size if dm_concat else index

    reduced_size = (0 if dm_concat else tf.random_uniform(
        [], maxval=window_size, dtype=tf.int32))
    left = tf.range(
        tf.maximum(target_index - window_size + reduced_size, 0), target_index)
    right = tf.range(target_index + 1, tf.minimum(
        target_index + 1 + window_size - reduced_size, tf.size(indices)))
    context = tf.concat([left, right], axis=0)
    context = tf.gather(indices, context)

    target = indices[target_index]
    if arch == 'PV-DBOW': # skip_gram
      window = tf.convert_to_tensor([[doc_id, target]])
      if dbow_train_words:
        word_instances = tf.stack(
            [tf.fill(tf.shape(context), target), context], axis=1)
        window = tf.concat([window, word_instances], axis=0)

    elif arch == 'PV-DM': # cbow
      context = tf.concat([context, [doc_id]], axis=0)
      true_size = tf.size(context)
      window = tf.concat([tf.pad(context, [[0, 2*window_size+1-true_size]]),
                          [true_size, target]], axis=0)
      window = tf.expand_dims(window, axis=0)
    else:
      raise ValueError('architecture must be PV-DBOW or PV-DM')

    if codes_points is not None:
      window = tf.concat([window[:, :-1],
                          tf.gather(codes_points, window[:, -1])], axis=1)
    return index + 1, init_array.write(index, window)

  size = tf.size(indices)
  init_array = tf.TensorArray(tf.int32, size=size, infer_shape=False)

  if dm_concat:
    indices = tf.concat([tf.fill([window_size], null_word_index),
        indices, tf.fill([window_size], null_word_index)], axis=0)
  _, result_array = tf.while_loop(lambda i, ta: i < size,
                                  per_target_fn,
                                  [0, init_array],
                                      back_prop=False)
  instances = tf.to_int64(result_array.concat())
  return instances

