import tensorflow as tf


class Model(object):
    @staticmethod
    def _variable(name, shape, initializer):
        var = tf.get_variable(name, shape=shape, dtype=tf.float32, initializer=initializer)
        return var

    @staticmethod
    def _variable_with_weight_decay(name, shape, stddev, wd):
        var = Model._variable(name,
                              shape=shape,
                              initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, 'weight_loss')
        tf.add_to_collection('losses', weight_decay)
        return var

    @staticmethod
    def inference(x):
        with tf.variable_scope('conv1') as scope:
            weights = Model._variable_with_weight_decay('weights', shape=[5, 5, 3, 64], stddev=5e-2, wd=0.0)
            biases = Model._variable('biases', [64], tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(x, weights, [1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv1 = tf.nn.relu(pre_activation, scope.name)

        with tf.name_scope('pool1'):
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        with tf.name_scope('norm1'):
            norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        with tf.variable_scope('conv2') as scope:
            weights = Model._variable_with_weight_decay('weights', shape=[5, 5, 64, 64], stddev=5e-2, wd=0.0)
            biases = Model._variable('biases', [64], tf.constant_initializer(0.1))
            conv = tf.nn.conv2d(norm1, weights, [1, 1, 1, 1], padding='SAME')
            pre_activation = tf.nn.bias_add(conv, biases)
            conv2 = tf.nn.relu(pre_activation, scope.name)

        with tf.name_scope('norm2'):
            norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

        with tf.name_scope('pool2'):
            pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        batch_size = x.get_shape()[0].value
        flatten = tf.reshape(pool2, [batch_size, -1])
        flatten_size = flatten.get_shape()[1].value

        with tf.variable_scope('fc3') as scope:
            weights = Model._variable_with_weight_decay('weights', shape=[flatten_size, 384],
                                                        stddev=0.04, wd=0.004)
            biases = Model._variable('biases', [384], tf.constant_initializer(0.1))
            fc3 = tf.nn.relu_layer(flatten, weights=weights, biases=biases, name=scope.name)

        with tf.variable_scope('fc4') as scope:
            weights = Model._variable_with_weight_decay('weights', shape=[384, 192],
                                                        stddev=0.04, wd=0.004)
            biases = Model._variable('biases', [192], tf.constant_initializer(0.1))
            fc4 = tf.nn.relu_layer(fc3, weights=weights, biases=biases, name=scope.name)

        with tf.variable_scope('fc5') as scope:
            weights = Model._variable_with_weight_decay('weights', shape=[192, 10],
                                                        stddev=1 / 192.0, wd=0.0)
            biases = Model._variable('biases', [10], tf.constant_initializer(0.0))
            fc5 = tf.nn.bias_add(tf.matmul(fc4, weights), biases, name=scope.name)

        logits = fc5
        return logits

    @staticmethod
    def loss(logits, labels):
        cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits))
        tf.add_to_collection('losses', cross_entropy)
        loss = tf.reduce_sum(tf.get_collection('losses'))
        return loss
