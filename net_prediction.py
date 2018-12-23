import tensorflow as tf
import pandas as pd
import numpy as np
import tech_analysis as tech
from MyLayers import *
from sklearn.preprocessing import MinMaxScaler

isReversed = False #Set it to True if your rows go from the present to past(i.e. 1st row - 9 am, 2nd row - 8 am, 3rd row - 7 am, and so on)
path = "DAT_MT_EURGBP_M1_201810.csv"

#A function which loads csv table and drops unused columns
def load_and_drop(path):
    current = pd.read_csv(path, sep=",", header=None)
    drop_labels = []
    for i in range(current.shape[1]):
        if (current[i][0] == " IGNORE") or (current[i][0] == "timestamp"):
            drop_labels.append(i)
    print("Dropping the following columns: ", drop_labels)
    current.drop(labels = drop_labels, axis=1, inplace = True) #dropping the unused columns
    current.drop(labels = 0, axis=0, inplace = True) #dropping the first row (we don't need the description of columns, we need only numbers)
    current = current.as_matrix().astype(np.float32)
    return current

data = load_and_drop(path)
if isReversed:
    data = np.flip(data, axis=0)
#Hyperparameters
days = 30
N_channels = 32
filter_size = 3

input_data = tf.placeholder(tf.float32, shape=[None, days, 9])  #Open, High, Low, Close, Vol
drop_prob = tf.placeholder(tf.float32)

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
out = tf.nn.sigmoid(logits)

scaler = MinMaxScaler(feature_range = (-1, 1), copy = True)
scaler_vol = MinMaxScaler(feature_range = (-1, 1), copy = True)
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'Saves/net')
    prediction = 0
    RSI = tech.compute_RSI(data[:, 3])
    RSI = scaler.fit_transform(RSI.reshape(data.shape[0], 1)).reshape(data.shape[0], 1) #Scaling the RSI independently from N-days scope as its absolute value is important
    MACD = tech.compute_MACD(data[:, 3]).reshape(data.shape[0], 1)
    EMA_slow = tech.compute_EMA(data[:, 3], T=30)
    EMA_fast = tech.compute_EMA(data[:, 3], T=10)
    example = data[-days:]
    example_RSI = RSI[-days:]
    example_MACD = MACD[-days:]
    example_EMA_slow = EMA_slow[-days:]
    example_EMA_fast = EMA_fast[-days:]
    prices = np.append(example[:, 0:4], example_EMA_slow.reshape(days, 1), axis=1)
    prices = np.append(prices, example_EMA_fast.reshape(days, 1), axis=1)
    prices = scaler.fit_transform(prices.reshape(days*6, 1)).reshape(days, 6)
    vol = scaler.fit_transform(example[:, 4].reshape(days, 1))
    scaled_MACD = scaler.fit_transform(example_MACD.reshape(days, 1)).reshape(days, 1)
    example = np.append(prices[:, 0:4], vol, axis=1) #Merging quotes and volumes
    example = np.append(example, prices[:, 4:], axis=1) #Appending EMAs
    example = np.append(example, scaled_MACD, axis=1) #Appending MACD
    example = np.append(example, example_RSI, axis=1) #Appending RSI
    net_out = out.eval(feed_dict={input_data: example.reshape(1, days, 9), drop_prob: 1}).reshape(3)
    prediction = np.argmax(net_out)
    print(prediction)
    if (prediction == 0):
        print('Prediction: Up')
    elif (prediction == 1):
        print('Prediction: No action')
    else:
        print('Prediction: Down')

