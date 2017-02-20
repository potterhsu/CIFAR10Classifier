import os
from datetime import datetime
import time
import tensorflow as tf
from meta import Meta
from donkey import Donkey
from model import Model
from evaluator import Evaluator

tf.app.flags.DEFINE_string('data_dir', './data', 'Directory to read TFRecords files')
tf.app.flags.DEFINE_string('train_logdir', './logs/train', 'Directory to write training logs')
tf.app.flags.DEFINE_string('restore_checkpoint', None,
                           'Path to restore checkpoint (without postfix), e.g. ./logs/train/model.ckpt-100')
FLAGS = tf.app.flags.FLAGS


def _train(path_to_train_tfrecords_file, num_train_examples, path_to_val_tfrecords_file, num_val_examples,
           path_to_train_log_dir, path_to_restore_checkpoint_file):
    batch_size = 128
    initial_patience = 20
    num_steps_to_show_loss = 10
    num_steps_to_check = 100

    with tf.Graph().as_default():
        images, labels = Donkey.build_batch(path_to_train_tfrecords_file, num_examples=num_train_examples,
                                            batch_size=batch_size, shuffled=True)
        logits = Model.inference(images)
        loss = Model.loss(logits, labels)

        global_step = tf.Variable(0, name='global_step', trainable=False)
        learning_rate = tf.train.exponential_decay(0.1, global_step=global_step, decay_steps=10000, decay_rate=0.1,
                                                   staircase=True)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)

        tf.summary.image('image', images)
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('learning_rate', learning_rate)
        summary = tf.summary.merge_all()

        with tf.Session() as sess:
            summary_writer = tf.summary.FileWriter(path_to_train_log_dir, sess.graph)
            evaluator = Evaluator(os.path.join(path_to_train_log_dir, 'eval/val'))

            sess.run(tf.global_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            saver = tf.train.Saver()
            if path_to_restore_checkpoint_file is not None:
                assert tf.train.checkpoint_exists(path_to_restore_checkpoint_file), \
                    '%s not found' % path_to_restore_checkpoint_file
                saver.restore(sess, path_to_restore_checkpoint_file)
                print 'Model restored from file: %s' % path_to_restore_checkpoint_file

            print 'Start training'
            patience = initial_patience
            best_accuracy = 0.0
            duration = 0.0

            while True:
                start_time = time.time()
                _, loss_val, summary_val, global_step_val = sess.run([train_op, loss, summary, global_step])
                duration += time.time() - start_time

                if global_step_val % num_steps_to_show_loss == 0:
                    examples_per_sec = batch_size * num_steps_to_show_loss / duration
                    duration = 0.0
                    print '=> %s: step %d, loss = %f (%.1f examples/sec)' % (
                        datetime.now(), global_step_val, loss_val, examples_per_sec)

                if global_step_val % num_steps_to_check != 0:
                    continue

                summary_writer.add_summary(summary_val, global_step=global_step_val)

                print '=> Evaluating on validation dataset...'
                path_to_latest_checkpoint_file = saver.save(sess, os.path.join(path_to_train_log_dir, 'latest.ckpt'))
                accuracy = evaluator.evaluate(path_to_latest_checkpoint_file, path_to_val_tfrecords_file,
                                              num_val_examples,
                                              global_step_val)
                print '==> accuracy = %f, best accuracy %f' % (accuracy, best_accuracy)

                if accuracy > best_accuracy:
                    path_to_checkpoint_file = saver.save(sess, os.path.join(path_to_train_log_dir, 'model.ckpt'),
                                                         global_step=global_step_val)
                    print '=> Model saved to file: %s' % path_to_checkpoint_file
                    patience = initial_patience
                    best_accuracy = accuracy
                else:
                    patience -= 1

                print '=> patience = %d' % patience
                if patience == 0:
                    break

            coord.request_stop()
            coord.join(threads)
            print 'Finished'


def main(_):
    path_to_train_tfrecords_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
    path_to_val_tfrecords_file = os.path.join(FLAGS.data_dir, 'val.tfrecords')
    path_to_tfrecords_meta_file = os.path.join(FLAGS.data_dir, 'tfrecords_meta.json')
    path_to_train_log_dir = FLAGS.train_logdir
    path_to_restore_checkpoint_file = FLAGS.restore_checkpoint

    meta = Meta()
    meta.load(path_to_tfrecords_meta_file)

    _train(path_to_train_tfrecords_file, meta.num_train_examples,
           path_to_val_tfrecords_file, meta.num_val_examples,
           path_to_train_log_dir, path_to_restore_checkpoint_file)


if __name__ == '__main__':
    tf.app.run(main=main)
