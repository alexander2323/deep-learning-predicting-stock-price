import tensorflow as tf
import numpy as np
from MyLayers import *

epochs = 10 #Number of training epochs
logging_step = 50

#Hyperparameters
days = 30
lr = 0.0001
batch_size = 32
keep_prob = 0.6
test_split = 0.8
N_channels = 32
filter_size = 3

#Path to training and validation data
data_train = np.load("./data/dataset_30M_with_indicators_fixed_train.npy")
data_test = np.load("./data/dataset_30M_with_indicators_fixed_validation.npy")
data_train_labels = np.load("./data/dataset_30M_with_indicators_fixed_train_labels.npy")
data_test_labels = np.load("./data/dataset_30M_with_indicators_fixed_validation_labels.npy")
print(data_test_labels[:20])

input_data = tf.placeholder(tf.float32, shape=[None, days, 9])  #Open, High, Low, Close, Vol
drop_prob = tf.placeholder(tf.float32)
labels = tf.placeholder(tf.float32, shape=[None, 3])

#Network definition
#Convolutional layer 1
conv1_1 = tf.layers.conv1d(inputs=input_data, filters=N_channels, kernel_size=filter_size, padding = 'same')
conv1_1 = prelu(conv1_1)
norm1_1 = tf.contrib.layers.batch_norm(inputs = conv1_1)
conv1_2 = tf.layers.conv1d(inputs=norm1_1, filters=2*N_channels, kernel_size=filter_size, padding = 'same')
conv1_2 = prelu(conv1_2)
norm1_2 = tf.contrib.layers.batch_norm(inputs = conv1_2)
pool1 = tf.layers.max_pooling1d(inputs=norm1_2, pool_size=2, strides=2)

#Fully-connected layer
flatten = tf.reshape(pool1, [-1, (days//2)*2*N_channels])
flatten = tf.nn.dropout(flatten, drop_prob)
fc1 = tf.layers.dense(inputs=flatten, units=512, activation=None)
fc1 = prelu(fc1)
fc1 = tf.nn.dropout(fc1, drop_prob)
fc2 = tf.layers.dense(inputs=fc1, units=512, activation=None)
fc2 = prelu(fc2)
fc2 = tf.nn.dropout(fc2, drop_prob)
logits = tf.layers.dense(inputs=fc2, units=3, activation = None)
out = tf.nn.softmax(logits, name="out")

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, axis=1), tf.argmax(labels, axis=1)), tf.float32))
train_step = tf.train.AdamOptimizer(lr).minimize(loss)
saver = tf.train.Saver()
with tf.Session() as sess:
    max_ = 0
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess, 'Saves/net')
    for e in range(epochs):
        data_train, data_train_labels = shuffle_in_unison(data_train, data_train_labels)
        #Training
        for i in range(data_train.shape[0] // batch_size):
            batch = data_train[i*batch_size:(i+1)*batch_size]
            y_ = data_train_labels[i*batch_size:(i+1)*batch_size]
            train_step.run(feed_dict={input_data: batch, labels: y_, drop_prob: keep_prob})
            if i % logging_step == 0:
                print("Training on batch %d out of %d"%(i+1, data_train.shape[0] // batch_size))
        #Validation
        cross_entropy = 0
        average_error = 0
        for i in range(data_test.shape[0] // batch_size):
            batch = data_test[i*batch_size:(i+1)*batch_size]
            y_ = data_test_labels[i*batch_size:(i+1)*batch_size]
            cross_entropy += loss.eval(feed_dict={input_data: batch, labels: y_, drop_prob: 1}) / (data_test.shape[0] // batch_size)
            average_error += accuracy.eval(feed_dict={input_data: batch, labels: y_, drop_prob: 1}) / (data_test.shape[0] // batch_size)
        print('Epoch %d is finished! Results:\nCross-entropy: %f \nAccuracy: %f \nCurrent best accuracy: %f\n'%(e, cross_entropy, average_error, max_))
        if average_error > max_:
            saver.save(sess, 'Saves/net')
            max_ = average_error
    sess.close()