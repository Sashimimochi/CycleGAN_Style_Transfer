import argparse
from cycle_gan import cycle_gan
import tensorflow as tf
from flags import FLAGS

def run():
    sess = tf.Session()
    model = cycle_gan(FLAGS, sess)
    if FLAGS.mode == 'train':
        model.train()
    elif FLAGS.mode == 'pretrain':
        model.pretrain()
    elif FLAGS.mode == 'val':
        model.val()
    elif FLAGS.mode == 'val_pre':
        model.val_pre()
    elif FLAGS.mode == 'test':
        model.test()

if __name__ == '__main__':
    run()

