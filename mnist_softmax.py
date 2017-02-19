# Follows the tutorial https://www.tensorflow.org/get_started/mnist/beginners
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

class MnistSoftmax:

    def __init__(self):
        self.accuracy = None

    def find_accuracy(self):
        # Download input from tensorflow
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        # ==============================================================================================
        # Implementing the regression
        # 
        # The softmax regression has two steps: add up evnidence then convert them into probabilities.
        # The tally of evidence of all the pictures will result in positive and negative weights.
        #
        # See:
        #   https://www.tensorflow.org/images/softmax-weights.png
        #   https://www.tensorflow.org/images/softmax-regression-scalargraph.png
        #   https://www.tensorflow.org/images/softmax-regression-scalarequation.png
        #   https://www.tensorflow.org/images/softmax-regression-vectorequation.png
        #
        # Set up variables for softmax, 784 images with 10 labels (0 through 9)
        x = tf.placeholder(tf.float32, [None, 784])   # inputs
        W = tf.Variable(tf.zeros([784, 10]))          # weights of evidence
        b = tf.Variable(tf.zeros([10]))               # bias, additional evidence

        # x * W instead of W * x. Why? W * x would result in a single vector
        # and x * W would result in a matrix
        y = tf.nn.softmax(tf.matmul(x, W) + b)

        # ==============================================================================================
        # Training
        #
        # Determining the loss (or define what it means for a model to be bad), we determine the
        # "cross-entropy".
        #
        # Cross entropy:
        # The raw formulation of cross-entropy,
        #
        #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
        #                                 reduction_indices=[1]))
        #
        # can be numerically unstable.
        #
        # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
        y_ = tf.placeholder(tf.float32, [None, 10])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()

        # feed in data 100 at a time for a 1000 repetitions
        for _ in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        # ==============================================================================================
        # Evaluate the model
        #
        # Gives you the index of the highest entry in a tensor along some axis. The accuracy result
        # should be around 92%.
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        print sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        self.accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        return self.accuracy
