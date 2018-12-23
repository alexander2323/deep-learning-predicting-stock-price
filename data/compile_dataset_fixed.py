import numpy as np
import pandas as pd
import math
import tech_analysis as tech
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import MinMaxScaler

isReversed = False #Set it to True if your rows go from the present to past(i.e. 1 row - 9 am, 2 row - 8 am, 3 row - 7 am, and so on)

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

days = 30 #Number of days within 1 sample
shift = 1 #Interval between different samples of the same share

X_train = np.zeros((0, days, 9), dtype=np.float64)
X_validation = np.zeros((0, days, 9), dtype=np.float64)
X_train_labels = np.zeros((0, 3), dtype=np.float64)
X_validation_labels = np.zeros((0, 3), dtype=np.float64)
c = 0
val = 0
validation_split = 0.2
#Getting list of all csv filenames
mypath = "./"
files = [f for f in listdir(mypath) if (isfile(join(mypath, f)) and (f[-1]=="v"))]
#Format: OPEN, HIGH, LOW, CLOSE, VOL, slow EMA, fast EMA, MACD, RSI
scaler = MinMaxScaler(feature_range = (-1, 1), copy = True)
for i in range(len(files)):
    current = load_and_drop(join(mypath, files[i]))
    if isReversed:
        current = np.flip(current, axis=0)
    validation_days = math.floor(validation_split*current.shape[0])
    RSI = tech.compute_RSI(current[:, 3])
    RSI = scaler.fit_transform(RSI.reshape(current.shape[0], 1)).reshape(current.shape[0], 1) #Scaling the RSI independently from N-days scope as its absolute value is important
    MACD = tech.compute_MACD(current[:, 3]).reshape(current.shape[0], 1)
    EMA_slow = tech.compute_EMA(current[:, 3], T=30)
    EMA_fast = tech.compute_EMA(current[:, 3], T=10)
    print("Processing file %d out of %d"%(i+1, len(files)))
    print("Number of candles in the current file: ", current.shape[0])
    j = 0
    flag = True
    while True:
        if j+days+1 > current.shape[0]: #Preventing running out of array range
            break
        else:
            example = current[j:(j + days)]
            pricechange = (current[j+days, 3] - current[j+days-1, 3])*1000/current[j+days, 3]
            print( current[j+days, 3] )
            if current[j+days-1, 3] < current[j+days, 3] and pricechange > 0.1:
                label = np.array([[1, 0, 0]])
            elif current[j+days-1, 3] > current[j+days, 3] and pricechange < -0.1:
                label = np.array([[0, 0, 1]])
            else:
                label = np.array([[0, 1, 0]])
            example_RSI = RSI[j:(j + days)]
            example_MACD = MACD[j:(j + days)]
            example_EMA_slow = EMA_slow[j:(j + days)]
            example_EMA_fast = EMA_fast[j:(j + days)]
            #print(example)
            min_ = np.amin(example[:, 0:4])
            max_ = np.amax(example[:, 0:4])
            avg = np.average(example[:, 0:4])
            val += (max_-min_) / avg
            prices = np.append(example[:, 0:4], example_EMA_slow.reshape(days, 1), axis=1)
            prices = np.append(prices, example_EMA_fast.reshape(days, 1), axis=1)
            prices = scaler.fit_transform(prices.reshape(days*6, 1)).reshape(days, 6)
            vol = scaler.fit_transform(example[:, 4].reshape(days, 1))
            scaled_MACD = scaler.fit_transform(example_MACD.reshape(days, 1)).reshape(days, 1)
            example = np.append(prices[:, 0:4], vol, axis=1) #Merging quotes and volumes
            example = np.append(example, prices[:, 4:], axis=1) #Appending EMAs
            example = np.append(example, scaled_MACD, axis=1) #Appending MACD
            example = np.append(example, example_RSI, axis=1) #Appending RSI
            if j < (current.shape[0]-validation_days): 
                X_train = np.append(X_train, example.reshape((1, days, 9)), axis=0)
                X_train_labels = np.append(X_train_labels, label, axis=0)
            else:
                if flag:
                    j += days
                    flag = False
                X_validation = np.append(X_validation, example.reshape((1, days, 9)), axis=0)
                X_validation_labels = np.append(X_validation_labels, label, axis=0)
            c += 1
        j += shift
number_of_no_action = 0

#Calculating the number of "no action" samples
for i in range(X_train_labels.shape[0]):
    if X_train_labels[i][1] == 1:
        number_of_no_action += 1
percentage = number_of_no_action / X_train_labels.shape[0]
#Calculating the number of "no action" samples to be removed
if percentage > 0.33:
    to_remove = math.ceil(X_train_labels.shape[0] * (percentage - 0.33) / 0.66)
    #Removing the required number of "no action" samples
    i = 0
    while i < to_remove:
        if X_train_labels[i][1] == 0:
            X_train = np.delete(X_train, i, 0)
            X_train_labels = np.delete(X_train_labels, i, 0)
            i -= 1
        i += 1
print("The dataset was split in %d training samples and %d validation samples"%(X_train.shape[0], X_validation.shape[0]))
print("Preprocessing complete; Saving the dataset...")
np.save("dataset_30M_with_indicators_fixed_train.npy", X_train)
np.save("dataset_30M_with_indicators_fixed_validation.npy", X_validation)
np.save("dataset_30M_with_indicators_fixed_train_labels.npy", X_train_labels)
np.save("dataset_30M_with_indicators_fixed_validation_labels.npy", X_validation_labels)
print("The training dataset has been successfully saved!")