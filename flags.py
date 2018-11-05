import tensorflow as tf

tf.app.flags.DEFINE_string('mode', 'train', 'train / test / test_f')
tf.app.flags.DEFINE_string('mode_train', 'all', 'type pretrain or all')
tf.app.flags.DEFINE_string('model_dir', 'model/model_NLPCC', 'output model weight dir')
tf.app.flags.DEFINE_string('data_dir', 'data/data_NLPCC', 'data dir')
tf.app.flags.DEFINE_string('load', '', 'loaded model name')
tf.app.flags.DEFINE_string('id_loss', False, 'loaded model name')
tf.app.flags.DEFINE_integer('batch_size', 36, 'batch size')
#tf.app.flags.DEFINE_integer('latent_dim', 100, 'laten size')

"""
tf.app.flags.DEFINE_integer('printing_step', 1, 'printing step')
tf.app.flags.DEFINE_integer('saving_step', 3, 'saving step')
tf.app.flags.DEFINE_integer('num_steps', 6, 'number of steps')
"""
tf.app.flags.DEFINE_integer('printing_step', 1000, 'printing step')
tf.app.flags.DEFINE_integer('saving_step', 20000, 'saving step')
tf.app.flags.DEFINE_integer('num_steps', 100000, 'number of steps')

tf.app.flags.DEFINE_integer('vocab_size', 50000, 'vocab size')
tf.app.flags.DEFINE_integer('sequence_length', 15, 'sentence length')
tf.app.flags.DEFINE_integer('pre_dis', 0, 'pretrain discriminator iterations')
tf.app.flags.DEFINE_integer('dis_it', 3, 'discriminator iterations')
tf.app.flags.DEFINE_integer('gen_it', 1, 'generator iterations')

FLAGS = tf.app.flags.FLAGS

