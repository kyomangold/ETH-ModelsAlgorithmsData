import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
import sys
from utils import *
from sklearn.utils import shuffle

#####################################################
#			MAD - Exercise set 7					#
#			Neural Networks         				#
#          		 Regression with NN                 #
#####################################################


# Creating the sinusoidal data for training
N = 200
noise = 0.0
data_input = np.linspace(0,2*np.pi, N) + noise * np.random.randn(N)
data_targets = np.sin(data_input) + noise * np.random.randn(data_input.shape[0])
plot(data_input, data_targets, 'train_data.pdf')


# divide data in training and validation
data_input, data_targets = shuffle(data_input, data_targets, random_state=0)

# First half of the data for training, second half for validation
train_input = data_input[:int(N/2)]
train_targets = data_targets[:int(N/2)]
val_input = data_input[int(N/2):]
val_targets = data_targets[int(N/2):]

# Testing data, the real sinusoidal signal
test_input = np.linspace(0,2*np.pi, 1000)
test_targets = np.sin(test_input)
plot(test_input, test_targets, 'sinus.pdf')

# Fixing the dimensions
train_input = np.reshape(train_input, (-1, 1))
train_targets = np.reshape(train_targets, (-1, 1))
val_input = np.reshape(val_input, (-1, 1))
val_targets = np.reshape(val_targets, (-1, 1))
test_input = np.reshape(test_input, (-1, 1))
test_targets = np.reshape(test_targets, (-1, 1))




# Building the model
input = tf.placeholder(tf.float32, [None, 1])
target = tf.placeholder(tf.float32, [None, 1])

# Define the variables (weights) of the model
num_hidden = 10
W1 = tf.Variable(tf.random_normal([1, num_hidden]))
b1 = tf.Variable(tf.random_normal([num_hidden]))
W2 = tf.Variable(tf.random_normal([num_hidden, 1]))
b2 = tf.Variable(tf.random_normal([1]))

layer = tf.nn.tanh(tf.einsum('ki,ih->kh', input, W1) + b1)
output = tf.einsum('kh,hc->kc', layer, W2) + b2

rmse_loss = tf.reduce_mean(tf.square(tf.subtract(output, target)))

trainer_rmse = tf.train.GradientDescentOptimizer(0.01).minimize(rmse_loss)


n_samples = train_input.shape[0]
batch_size = 10
number_of_batches = n_samples//batch_size
num_epochs = 100

train_loss = []
val_loss = []
print('Number of batches: {:d}'.format(number_of_batches))
with tf.Session() as sess:
    # Initialize all variables
    tf.global_variables_initializer().run()
    test_rmse = sess.run(rmse_loss, feed_dict={input:test_input, target:test_targets})
    print('Before training, Test RMSE {:.5f}'.format(test_rmse))
    for epoch in range(num_epochs):
        for i in range(number_of_batches):
            print('Batch {:d}/{:d}'.format(i, number_of_batches))
            sys.stdout.write("\033[F")
            batch_input = getBatch(train_input, i, batch_size)
            batch_target = getBatch(train_targets, i, batch_size)
            _, batch_loss = sess.run([trainer_rmse, rmse_loss], feed_dict={input:batch_input, target: batch_target})

        train_rmse = sess.run(rmse_loss, feed_dict={input:train_input, target:train_targets})
        val_rmse = sess.run(rmse_loss, feed_dict={input:val_input, target:val_targets})

        train_loss.append(train_rmse)
        val_loss.append(val_rmse)

        print('Epoch {:d}/{:d}, RMSE on Train data set: {:.5f}%, RMSE on Validation data set: {:.5f}%'.format(epoch, num_epochs, train_rmse, val_rmse))

    output_test, rmse_loss = sess.run([output, rmse_loss], feed_dict={input:test_input, target:test_targets})

    print('RMSE on TEST data set: {:.5f}%'.format(rmse_loss))


    plotSinusTest(test_input, output_test, test_targets, 'nn_sin.pdf')
    plotLosses(train_loss, val_loss, 'losses.pdf')









