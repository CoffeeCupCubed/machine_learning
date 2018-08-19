import sys
import random

import numpy as np
import pandas as pa
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#Constants
MODEL_TYPE = 'ann_'
LOGFILE='./tensorboard/model_' + MODEL_TYPE
MODEL_SAVE='./model/model_' + MODEL_TYPE
ERRORS = 'Errors'

REC_INT = 50
SEED = 199832

CHANNELS = 1
IMG_WIDTH = 28
IMG_HEIGHT = 28
TOTAL_PIXELS = IMG_WIDTH * IMG_HEIGHT

COMPRESS_BY_7 = TOTAL_PIXELS / 7
COMPRESS_BY_49 = COMPRESS_BY_7 / 7

# Data Tuple Indices

"""
Loaded data is stored in a list. The below indices
give information about where training/test data/labels
are in the list.
"""
trd = 0     # Training Data
trl = 1     # Training Labels
ted = 2     # Test Data
tel = 3     # Test Labels


"""
The setup_model function creates several tensors
which are later evaluated. These ops are passed
back in a list. The indices tell us where each
op is located in the list.
"""
# Encode tf ops
tr = 0      # train step op
wr = 1      # file writer op
su = 2      # summary variables op
sv = 3      # model saver
ou = 4      # model output (y vals)
er = 5      # error calculation

########## Helpers ##########

### Data Helpers ###
"""
Displays Images (Original and Decoded Images side by side)
Takes a variable 'mod' s.t. every 'mod' images are displayed.
Logs files in the logfile dir with a specified extension.
"""

def display_images(images, labels, mod, ext, disp = True):
    writer = tf.summary.FileWriter(LOGFILE + 'img' + ext)
    for i,img in enumerate(images):
        if i % mod == 0:
            label = str(np.where(labels[i] == 1)[0][0]) + ext

            if disp:
                plain_img = tf.constant(img)
                square_img = tf.reshape(plain_img, [2, IMG_HEIGHT, IMG_WIDTH, 1])
                img_summ = tf.summary.image(label, square_img)
                writer.add_summary(img_summ.eval())
            else:
                print(label)
                print("Original")
                print(img[0])
                print("New")
                print(img[1])

"""
Helper to combine 2 image data sets into one, where
each image at index i in data_1/data_2 are placed
side by side, then placed into a larger list. The
final list of 2-tuples is returned.
"""
def combine_img_data(data_1, data_2):
    comb = []
    for i in range(len(data_1)):
        data = [list(data_1[i]), list(data_2[i])]
        comb.append(data)
    return comb

"""
Generates a random images and evaluates the output tensor.
"""
def run_rand_img(output):
    data = [[random.random() for _ in range(TOTAL_PIXELS)]]

    feed_dict = {'X:0':data}
    new_data = output.eval(feed_dict = feed_dict)

    print(data)
    print(new_data)

"""
Loads the train/test data set and returns them as
a 4-tuple. See encodings above for corresponding indices
for the returned tuple.
"""
def load_data():
    old_v = tf.logging.get_verbosity()
    tf.logging.set_verbosity(tf.logging.ERROR)

    mnist = tf.examples.tutorials.mnist.input_data.read_data_sets("../MNIST_data/", one_hot=True)

    train_data = mnist.train.images     # Returns train data array.
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

    eval_data = mnist.test.images       # Return test data array.
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    tf.logging.set_verbosity(old_v)
    return (train_data, train_labels, eval_data, eval_labels)


### Variable Initializers ###
"""
Initialized a weight variable with a truncated normal distribution
and places it in the global variables/regularization losses collection.
"""
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    collections = [tf.GraphKeys.REGULARIZATION_LOSSES, tf.GraphKeys.GLOBAL_VARIABLES]
    return tf.Variable(initial, collections=collections, name='weights')

"""
Initializes a bias variable as a constant.
"""
def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    collections = [tf.GraphKeys.REGULARIZATION_LOSSES, tf.GraphKeys.GLOBAL_VARIABLES]
    return tf.Variable(initial, collections=collections, name='bias')


### Neural Layer Helper ###
"""
Initializes a simple neural layer consisting of an input matrix,
and a new matrix of weights. We create a summary matrix for both
the weight/bias and take the sigmoid of the output (if hidden).
"""
def neural_layer(input_mat, input_neurons, output_neurons, hidden=True):
    weight = weight_variable([input_neurons, output_neurons])
    bias = bias_variable([output_neurons])

    tf.summary.histogram('Weight', weight)
    tf.summary.histogram('Bias', bias)

    output = tf.matmul(input_mat, weight) + bias
    if hidden:
        output = tf.sigmoid(output)
    return output

"""
Initializes a convolutional layer with a 5x5 kernel size.
The input layer and number of filters are passed as args.

The pooling layer reduces input size by 2 on each dim.
"""
def conv_pool_layer(input_layer, filters):
    # Conv Layer
    conv = tf.layers.conv2d(
                inputs=input_layer,
                filters=filters,
                kernel_size=[5, 5],
                padding="same",
                activation=tf.nn.relu)

    # Pooling Layer
    return tf.layers.max_pooling2d(inputs=conv, pool_size=[2, 2], strides=2)

"""
The dense layer reduces the input dim of a flat layer to
the number of units (passed). The activation function is relu
and dropout regularization is (optionally) used.
"""
def dense_layer(inputs, units, dropout=None):
    dense = tf.layers.dense(inputs=inputs, units=units, activation=tf.nn.relu)

    if dropout:
        return tf.nn.dropout(inputs, dropout)

    return dense


# Main Routines #
"""
The setup model routine takes a learning rate and
a hparam (pre-constructed to differentiate in tensorboard).

It creates:
    * An input placeholder for a flattened image.
    * Has to encoding steps (first we compress by a factor of 7
        then by a factor of 9).
    * Then we decode it back using 2 more layers.
    * We setup the loss tensor (MSE)
    * We set up the train step (AdamOptimizer minimizing
        loss tensor).
    * Finally, we set up accuracy (RMSE) and add it to
        the summary.

"""
def setup_model(alpha = 0.001, hparam=''):
    print('Setting Up Model ... ')
    writer = tf.summary.FileWriter(LOGFILE + hparam)

    # Input layer
    input_ph = tf.placeholder("float", [None, TOTAL_PIXELS], 'X')

    # Hidden Layer
    with tf.name_scope('compress_by_7'):
        encode_1 = neural_layer(input_ph, TOTAL_PIXELS, COMPRESS_BY_7)

    with tf.name_scope('compress_by_49'):
        latent = neural_layer(encode_1, COMPRESS_BY_7, COMPRESS_BY_49)

    with tf.name_scope('decompress_by_7'):
        decode_1 = neural_layer(latent, COMPRESS_BY_49, COMPRESS_BY_7)

    # Output
    with tf.name_scope('decompress_full'):
        output = neural_layer(decode_1, COMPRESS_BY_7, TOTAL_PIXELS)

    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.squared_difference(output, input_ph))

    # Train Step
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(alpha).minimize(loss)

    # Accuracy
    with tf.name_scope('accuracy'):
        error_ts = tf.reduce_mean(output - input_ph)
        error = tf.summary.scalar('error', error_ts)

    writer.add_graph(sess.graph)
    summ_vars = tf.summary.merge_all()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()

    return [train_step, writer, summ_vars, saver, output, error]

"""
Restores a saved model. If it fails, the user
has an option to quit the program.
"""
def restore_model(sess, saver):
    print('Restoring Model ... ')
    try:
        saver.restore(sess, MODEL_SAVE)
    except:
        ans = None
        while not(ans):
            ans = raw_input('Model Restore Failed. Do you want to create a new model (yes/no)? ')
            if ans == 'yes':
                return
            elif ans == 'no':
                raise SystemExit
            else:
                print('Please answer with a \'yes\' or \'no\'.')
                ans = None

"""
Trains the model and saves after every REC_INT steps.
Also updates the model summary on tensorboard after every
REC_INT steps.
"""
def train_model(sess, tf_ops, data, steps = 10000):
    print('Training Model ... ')

    train_step = tf_ops[tr]
    writer = tf_ops[wr]
    summ_vars = tf_ops[su]
    saver = tf_ops[sv]
    error = tf_ops[er]

    feed_dict = {'X:0':data[trd]}

    for i in range(steps):
        train_step.run(feed_dict = feed_dict)
        if i % REC_INT == 0:
            # Update Console
            print('Training Step', i)

            # Save Model
            path = saver.save(sess, MODEL_SAVE)
            print('Saved Iteration %d at path \'%s\'' % (i, path))

            # Write Summary
            summ = summ_vars.eval(feed_dict = feed_dict)
            writer.add_summary(summ, i)

            # Write test error
            test_feed_dict = {'X:0':data[ted]}
            test_summ = error.eval(feed_dict = test_feed_dict)

            test_writer = tf.summary.FileWriter(LOGFILE + 'test_error')
            test_writer.add_summary(test_summ, i)

"""
Runs the model on the test set. We write the encoded/decoded
images back out to tensorboard (using the 'display_images' func)
"""
def run_model(tf_ops, data):
    print('Running Model ... ')

    output = tf_ops[ou]
    writer = tf.summary.FileWriter(LOGFILE + 'test_set')

    feed_dict = {'X:0':data[ted]}
    new_data = output.eval(feed_dict = feed_dict)

    comb = combine_img_data(data[ted], new_data)
    display_images(comb, data[tel], 100, '_test')


#####   Main Routine    #####
sess = tf.InteractiveSession()
mnist = load_data()

tf.set_random_seed(SEED)

tf_ops = setup_model()
restore_model(sess, tf_ops[sv])

# If a 'train' arg was passed, we train.
if len(sys.argv) > 1 and sys.argv[1] == 'train':
    train_model(sess, tf_ops, mnist)

# Run the trained model
run_model(tf_ops, mnist)

