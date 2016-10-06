#!/usr/bin/python

# Copyright 2016 deepPay

from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

from numpy import genfromtxt
import sys
import sklearn.preprocessing
import time

#
# TODO :
#        * there are varying gaps between events inside the sequences. This is not taken into account.
#			maybe it would be a good idea to take that into account.
#        * there is a warning from sklearn.preprocessing.scale that says that there could be numerical
#          instability from some columns having a stddev very close to 0. Maybe remove these columns.
#	     * From here, we can try neural turing machines. (point : cross-seed the accounts.)


harddrive = genfromtxt('data/harddrive.csv', delimiter=',', skip_header 
	= 3, dtype = np.float32)



# renormalize hardrive (mean = 0 and stddev = 1)
# Column 0, 1 and last columns are not renormalized.
hardrive_scaled = np.concatenate(
		(harddrive[:, 0:2],
		 sklearn.preprocessing.scale(harddrive[:, 2:-1], axis=1), 
		 harddrive[:, -1:]), 
		 axis=1)

# 
# This function create a 3D tensor from the 2D tensor.
# Input : 2D-Tensor <event timestep in sequence>  x <seqnum + timestep + input + target>
# Output: 3D-Tensor <Sequence> x <event timestep> x <seqnum + timestep + input + target>
#         1D-Tensor <Sequence>
#
def rotateAndPadZeros(harddrive):
	
	vector_size = harddrive.shape[1]
	
	# Split the 'harddrive' tensor along the sequence.
	split = np.split(harddrive, np.where(np.diff(harddrive[:,0]))[0]+1)
	mx = 0
	for s in split:
		length = len(s)
		if (length > mx):
			mx = length
	maximum_sequence = mx

	def topad(x):  # pad with zeros to the maximum sequence length any passed tensor.
		return np.concatenate( (x, np.zeros( (maximum_sequence - len(x), vector_size), dtype = np.float32)), axis=0)


	final  = np.reshape(np.concatenate(list(map(topad, split))), (-1, maximum_sequence, vector_size))
	final_seqlen = np.array(list(map(lambda x: len(x), split)))

	return (final, final_seqlen)

final, final_seqlen = rotateAndPadZeros(hardrive_scaled)
def shuffle_in_unison_scary(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

shuffle_in_unison_scary(final, final_seqlen)



train_set = final[0:300,:,:]
train_seqlen = final_seqlen[0:300]


test_set  = final[300:-1,:,:]
test_seqlen = final_seqlen[300:-1]

def buildparts(set, seqlen):
	inputs  = set[:,:,2:-1]  # Suppress the first two columns and the last column.
	pre_target = np.expand_dims(np.squeeze(set[:,:,-1:].transpose((1,2,0))[0]), 1)

	# From a single column target, join with its complement to have two exclusives classes.
	target = np.concatenate((pre_target, (np.logical_not(pre_target)).astype(np.float)), axis=1)

	return (inputs, target, seqlen)

train_inputs, train_target, train_length = buildparts(train_set, train_seqlen)

test_inputs, test_target, test_length = buildparts(test_set, test_seqlen)

# Parameters
learning_rate = 0.001
training_iters = 30000
batch_size = np.shape(train_inputs)[0] # a mini batch could be smaller than 
display_step = 5


# Network Parameters
n_input = np.shape(train_inputs)[2] # event vector size.
n_steps = np.shape(train_inputs)[1] # timesteps (== number of events in sequence).
n_hidden = 128 # hidden layer num of features

# Two classes : 1st class : Fraud. 2nd class : Not Fraud. 
#				 both class are exclusive.

n_classes = 2


# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}

biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}



def RNN(x, weights, biases, seqlen):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)

    # outputs is a tensor of sequence length. We need to get the last in sequence (at seqlen).

    outputs = tf.pack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    print(n_steps)
    index = tf.range(0, batch_size) * n_steps + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs, weights['out']) + biases['out']

pred = RNN(x, weights, biases, seqlen)




# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
print("start training")

with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep train_inputsing until reach max iterations
    while step * batch_size < training_iters:
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: train_inputs, y: train_target, seqlen: train_length})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: train_inputs, y: train_target, seqlen: train_length})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: train_inputs, y: train_target, seqlen: train_length})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    
    test_data = test_inputs
    test_label = test_target

    clock = time.clock()
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label, seqlen: test_length}))
    
    print("Execution time :%s"% (time.clock() - clock))
