import tensorflow as tf
import os

tf.app.flags.DEFINE_string('mode', 'train', 'pretrain / train / val / val_pre / test')
#tf.app.flags.DEFINE_string('data_name', 'NLPCC', 'data name')
tf.app.flags.DEFINE_string('data_name', 'BG', 'data name')
tf.app.flags.DEFINE_string('model_dir', 'model', 'output model weight dir')
tf.app.flags.DEFINE_string('data_dir', 'data', 'data dir')
tf.app.flags.DEFINE_string('load', '', 'loaded model name')
tf.app.flags.DEFINE_string('output', 'output', 'output dir')
tf.app.flags.DEFINE_string('id_loss', False, 'id loss or not')
tf.app.flags.DEFINE_integer('batch_size', 36, 'batch size')
#tf.app.flags.DEFINE_integer('latent_dim', 100, 'laten size')

"""
tf.app.flags.DEFINE_integer('printing_step', 1, 'printing step')
tf.app.flags.DEFINE_integer('saving_step', 3, 'saving step')
tf.app.flags.DEFINE_integer('num_steps', 6, 'number of steps')
"""
tf.app.flags.DEFINE_integer('printing_step', 1000, 'printing step')
tf.app.flags.DEFINE_integer('saving_step', 10000, 'saving step')
tf.app.flags.DEFINE_integer('num_steps', 100000, 'number of steps')

tf.app.flags.DEFINE_integer('vocab_size', 50000, 'vocab size')
#tf.app.flags.DEFINE_integer('sequence_length', 15, 'sentence length')
tf.app.flags.DEFINE_integer('sequence_length', 10, 'sentence length')
tf.app.flags.DEFINE_integer('pre_dis', 0, 'pretrain discriminator iterations')
tf.app.flags.DEFINE_integer('dis_it', 1, 'discriminator iterations')
tf.app.flags.DEFINE_integer('gen_it', 1, 'generator iterations')

FLAGS = tf.app.flags.FLAGS

FLAGS.data_dir = os.path.join(FLAGS.data_dir, 'data_{}'.format(FLAGS.data_name))
FLAGS.model_dir = os.path.join(FLAGS.model_dir, 'model_{}'.format(FLAGS.data_name))
FLAGS.output = os.path.join(FLAGS.output, 'output_{}'.format(FLAGS.data_name))

for i in [FLAGS.model_dir, FLAGS.output]:
  if not os.path.exists(i):
    os.mkdir(i)
    print ('Create model dir : {}'.format(i))

FLAGS.output = os.path.join(FLAGS.output, 'output_{}_{}_{}{}{}'.format(FLAGS.data_name, FLAGS.dis_it, FLAGS.gen_it, '_id' if FLAGS.id_loss else '', '_'+FLAGS.load if FLAGS.load != '' else ''))

FLAGS.saving_step = [10000*i for i in [1, 2, 5, 10]]
FLAGS.num_steps = FLAGS.saving_step[-1]

