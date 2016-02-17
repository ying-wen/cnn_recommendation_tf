import tensorflow as tf
import numpy as np


class RecCNN(object):
    """
    A CNN for recommendation system.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, num_classes, user_size, item_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        self.batch_size = 16
        # Placeholders for input, output and dropout
        self.input_u = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name="input_u")
        self.input_i = tf.placeholder(tf.int32, shape=[self.batch_size, 1], name="input_i")
        self.input_y = tf.placeholder(tf.float32, [self.batch_size, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            U = tf.Variable(
                tf.random_uniform([user_size, embedding_size], -1.0, 1.0),
                name="U")
            I = tf.Variable(
                tf.random_uniform([item_size, embedding_size], -1.0, 1.0),
                name="I")
            self.embedded_user = tf.nn.embedding_lookup(U, self.input_u)
            self.embedded_item = tf.nn.embedding_lookup(I, self.input_i)

        with tf.name_scope("out_product"):
            self.embedded_outer_product = tf.batch_matmul(self.embedded_user, self.embedded_item, adj_x=True)
            self.embedded_outer_product_expanded = tf.expand_dims(self.embedded_outer_product, -1)


        # pooled_outputs = []
        # for i, filter_size in enumerate(filter_sizes):

        filter_size = 3
        # Create a convolution + maxpool layer for with filter size 3
        with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
            filter_shape = [filter_size, filter_size, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                self.embedded_outer_product_expanded,
                W,
                strides=[1, 1, 1,1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
            # Maxpooling over the outputs
            self.pooled = tf.nn.max_pool(
                h,
                ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1],
                padding='VALID',
                name="pool")
        # Combine all the pooled features
        # num_filters_total = num_filters * len(filter_sizes)
        # pool = pooled_outputs[0]
        # print "pool"
        # pool_shape =  self.pooled.get_shape()
        dim = 1
        for d in self.pooled.get_shape()[1:].as_list():
            dim *= d
        self.h_pool_flat = tf.reshape(self.pooled, [self.batch_size, dim])

                # self.h_pool = tf.concat(3, pooled_outputs)
        # self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])


        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            # print self.h_drop.get_shape()

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([dim, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # l2_loss += tf.nn.l2_loss(U)
            # l2_loss += tf.nn.l2_loss(I)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            print "shape for scores: " + str(self.scores.get_shape())
            # self.scores = tf.add(tf.matmul(local4, weights), biases, name=scope.name)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
