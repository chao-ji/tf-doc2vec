r"""Executable for learning the embeddings of documents not present in
training set. 

Example:
  python run_inference.py \
    --filenames=/PATH/TO/FILE/file1.txt,/PATH/TO/FILE/file2.txt \
    --filenames_infer=/PATH/TO/FILE/infer_file.txt \
    --out_dir=/PATH/TO/OUT_DIR/ \
    --arch=PV-DM

NOTE: you must use the same parameters that were used in `run_training.py`.
The learned embeddings will be saved to /PATH/TO/OUT_DIR/infer_doc_embed.npy.
"""
import os

import tensorflow as tf
import numpy as np

from doc2vec import Doc2VecInferencer
from dataset import Doc2VecDataset

flags = tf.app.flags

flags.DEFINE_string('arch', 'PV-DBOW', 'Architecture (DBOW or DM).')
flags.DEFINE_string('algm', 'negative_sampling', 'Training algorithm '
    '(negative_sampling or hierarchical_softmax).')
flags.DEFINE_integer('epochs', 1, 'Num of epochs to iterate training data.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('max_vocab_size', 0, 'Maximum vocabulary size. If > 0, '
    'the top `max_vocab_size` most frequent words are kept in vocabulary.')
flags.DEFINE_integer('min_count', 10, 'Words whose counts < `min_count` are not'
    ' included in the vocabulary.')
flags.DEFINE_float('sample', 1e-3, 'Subsampling rate.')
flags.DEFINE_integer('window_size', 5, 'Num of words on the left or right side'
    ' of target word within a window.')
flags.DEFINE_boolean('dbow_train_words', False, 'Whether to train the word '
    'vectors in DBOW architecture.')
flags.DEFINE_boolean('dm_concat', True, 'Whether to concatenate word and '
    'document vectors or compute their mean in DM architecture.')

flags.DEFINE_integer('embed_size', 300, 'Length of word vector.')
flags.DEFINE_integer('negatives', 5, 'Num of negative words to sample.')
flags.DEFINE_float('power', 0.75, 'Distortion for negative sampling.')
flags.DEFINE_float('alpha', 0.025, 'Initial learning rate.')
flags.DEFINE_float('min_alpha', 0.0001, 'Final learning rate.')
flags.DEFINE_boolean('add_bias', True, 'Whether to add bias term to dotproduct '
    'between syn0 and syn1 vectors.')

flags.DEFINE_list('filenames', None, 'Names of comma-separated input text files'
    ' for training.')
flags.DEFINE_list('filenames_infer', None, 'Names of comma-separated input text' 
    'files for inference.')
flags.DEFINE_string('out_dir', '/tmp/doc2vec', 'Output directory.')

FLAGS = flags.FLAGS


def main(_):
  dataset = Doc2VecDataset(arch=FLAGS.arch,
                           algm=FLAGS.algm,
                           epochs=FLAGS.epochs,
                           batch_size=FLAGS.batch_size,
                           max_vocab_size=FLAGS.max_vocab_size,
                           min_count=FLAGS.min_count,
                           sample=FLAGS.sample,
                           window_size=FLAGS.window_size,
                           dbow_train_words=FLAGS.dbow_train_words,
                           dm_concat=FLAGS.dm_concat)
  dataset.build_vocab(FLAGS.filenames)

  doc2vec = Doc2VecInferencer(arch=FLAGS.arch,
                              algm=FLAGS.algm,
                              embed_size=FLAGS.embed_size,
                              batch_size=FLAGS.batch_size,
                              negatives=FLAGS.negatives,
                              power=FLAGS.power,
                              alpha=FLAGS.alpha,
                              min_alpha=FLAGS.min_alpha,
                              add_bias=FLAGS.add_bias,
                              random_seed=0,
                              dm_concat=FLAGS.dm_concat,
                              window_size=FLAGS.window_size)

  to_be_run_dict = doc2vec.infer(dataset, FLAGS.filenames_infer)
  save_list = doc2vec.get_save_list()

  sess = tf.Session()
  sess.run(dataset.iterator_initializer)
  sess.run(tf.tables_initializer())

  saver = tf.train.Saver(var_list=save_list)
  saver.restore(sess, os.path.join(FLAGS.out_dir, 'doc2vec.ckpt'))
  sess.run(doc2vec.syn0_d.initializer)

  average_loss = 0.
  step = 0
  while True:
    try:
      result_dict = sess.run(to_be_run_dict)
    except tf.errors.OutOfRangeError:
      break
    average_loss += result_dict['loss'].mean()
    if step % 10000 == 0:
      if step > 0:
        average_loss /= 10000
      print('step', step, 'average_loss', average_loss, 'learning_rate', 
          result_dict['learning_rate'])
      average_loss = 0.
    step += 1

  syn0_d_final = sess.run(doc2vec.syn0_d)
  np.save(os.path.join(FLAGS.out_dir, 'infer_doc_embed'), syn0_d_final)

  print('Infer doc embeddings saved to', 
      os.path.join(FLAGS.out_dir, 'infer_doc_embed.npy'))

if __name__ == '__main__':
  tf.flags.mark_flag_as_required('filenames')
  tf.flags.mark_flag_as_required('filenames_infer')

  tf.app.run()

