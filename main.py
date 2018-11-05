import argparse
from cycle_gan import cycle_gan
import tensorflow as tf
from flags import FLAGS

def run():
    sess = tf.Session()
    model = cycle_gan(FLAGS, sess)
    if FLAGS.mode == 'train' and FLAGS.mode_train=='all':
        model.train()
    if FLAGS.mode == 'train' and FLAGS.mode_train=='pretrain':
        model.pretrain()
    if FLAGS.mode == 'test_f':
        model.file_test()
    if FLAGS.mode == 'test':
        model.test()

if __name__ == '__main__':
    #args = parse()
    run()

