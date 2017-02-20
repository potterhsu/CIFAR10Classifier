import os
import cPickle
import random
import tensorflow as tf
from meta import Meta

tf.app.flags.DEFINE_string('data_dir', './data',
                           'Directory to CIFAR-10 batches folder and write the converted files')
FLAGS = tf.app.flags.FLAGS


class ExampleReader(object):
    def __init__(self, path_to_batch_file):
        with open(path_to_batch_file, 'rb') as f:
            data_batch = cPickle.load(f)
        self._images = data_batch['data']
        self._labels = data_batch['labels']
        self._num_examples = self._images.shape[0]
        self._example_pointer = 0

    @staticmethod
    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def read_and_convert(self):
        """
        Read and convert to example, returns None if no data is available.
        """
        if self._example_pointer == self._num_examples:
            return None
        image = self._images[self._example_pointer].reshape([32, 32, 3], order='F').transpose(1, 0, 2).tostring()
        label = self._labels[self._example_pointer]
        self._example_pointer += 1

        example = tf.train.Example(features=tf.train.Features(feature={
            'image': ExampleReader._bytes_feature(image),
            'label': ExampleReader._int64_feature(label)
        }))
        return example


def convert_to_tfrecords(path_to_batch_files, path_to_tfrecords_files, choose_writer_callback):
    num_examples = []
    writers = []

    for path_to_tfrecords_file in path_to_tfrecords_files:
        num_examples.append(0)
        writers.append(tf.python_io.TFRecordWriter(path_to_tfrecords_file))

    for path_to_batch_file in path_to_batch_files:
        example_reader = ExampleReader(path_to_batch_file)
        while True:
            example = example_reader.read_and_convert()
            if example is None:
                break

            idx = choose_writer_callback(path_to_tfrecords_files)
            writers[idx].write(example.SerializeToString())
            num_examples[idx] += 1

    for writer in writers:
        writer.close()

    return num_examples


def create_tfrecords_meta_file(num_train_examples, num_val_examples, num_test_examples, path_to_batches_meta_file,
                               path_to_tfrecords_meta_file):
    print 'Saving meta file to %s...' % path_to_tfrecords_meta_file
    meta = Meta()
    meta.num_train_examples = num_train_examples
    meta.num_val_examples = num_val_examples
    meta.num_test_examples = num_test_examples
    with open(path_to_batches_meta_file, 'rb') as f:
        content = cPickle.load(f)
        meta.categories = content['label_names']
    meta.save(path_to_tfrecords_meta_file)


def main(_):
    path_to_batches_meta_file = os.path.join(FLAGS.data_dir, 'cifar-10-batches-py/batches.meta')
    path_to_batches = os.path.join(FLAGS.data_dir, 'cifar-10-batches-py')
    path_to_train_batch_files = [os.path.join(path_to_batches, 'data_batch_1'),
                                 os.path.join(path_to_batches, 'data_batch_2'),
                                 os.path.join(path_to_batches, 'data_batch_3'),
                                 os.path.join(path_to_batches, 'data_batch_4'),
                                 os.path.join(path_to_batches, 'data_batch_5')]
    path_to_test_batch_files = [os.path.join(path_to_batches, 'test_batch')]

    path_to_train_tfrecords_file = os.path.join(FLAGS.data_dir, 'train.tfrecords')
    path_to_val_tfrecords_file = os.path.join(FLAGS.data_dir, 'val.tfrecords')
    path_to_test_tfrecords_file = os.path.join(FLAGS.data_dir, 'test.tfrecords')
    path_to_tfrecords_meta_file = os.path.join(FLAGS.data_dir, 'tfrecords_meta.json')

    for path_to_file in [path_to_train_tfrecords_file, path_to_val_tfrecords_file, path_to_test_tfrecords_file]:
        assert not os.path.exists(path_to_file), 'The file %s already exists' % path_to_file

    print 'Processing training and validation data...'
    num_train_examples, num_val_examples = convert_to_tfrecords(path_to_train_batch_files,
                                                                [path_to_train_tfrecords_file, path_to_val_tfrecords_file],
                                                                lambda paths: 0 if random.random() > 0.1 else 1)
    print 'Processing test data...'
    num_test_examples = convert_to_tfrecords(path_to_test_batch_files,
                                             [path_to_test_tfrecords_file],
                                             lambda paths: 0)

    create_tfrecords_meta_file(num_train_examples, num_val_examples, num_test_examples, path_to_batches_meta_file,
                               path_to_tfrecords_meta_file)

    print 'Done'


if __name__ == '__main__':
    tf.app.run(main=main)
