import tensorflow as tf

tf.app.flags.DEFINE_string('mode', 'train', 'train / test / file_test')
tf.app.flags.DEFINE_string('mode_train', 'all', 'type pretrain or all')
tf.app.flags.DEFINE_string('model_dir', 'model', 'output model weight dir')
#tf.app.flags.DEFINE_string('model_path','', 'latest model path')
tf.app.flags.DEFINE_string('data_dir', 'data', 'data dir')
tf.app.flags.DEFINE_integer('batch_size', 36, 'batch size')
tf.app.flags.DEFINE_integer('latent_dim', 100, 'laten size')
tf.app.flags.DEFINE_integer('saving_step', 20000, 'saving step')
tf.app.flags.DEFINE_integer('printing_step', 1000, 'printing step')
tf.app.flags.DEFINE_integer('num_steps', 100000, 'number of steps')
tf.app.flags.DEFINE_integer('sequence_length', 10, 'sentence length')
tf.app.flags.DEFINE_integer('dis_iter', 3, 'discriminator iterations')

FLAGS = tf.app.flags.FLAGS

