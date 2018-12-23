import numpy as np

eps = 1e-9 #A small value to prevent division by zero

def compute_EMA(signal, T):
    alpha = 2 / (T + 1)
    length = signal.shape[0]
    EMA = np.zeros(length, dtype=np.float64)
    EMA[0] = signal[0] #Setting the first EMA value as a first signal value
    for i in range(1, length):
        EMA[i] = alpha * signal[i] + (1 - alpha) * EMA[i-1]
    return EMA

def compute_MACD(signal, T_slow=26, T_fast=12):
    EMA_slow = compute_EMA(signal, T_slow)
    EMA_fast = compute_EMA(signal, T_fast)
    MACD = EMA_fast - EMA_slow
    return MACD

def compute_RSI(signal, T=14):
    length = signal.shape[0]
    signal = np.insert(signal, 0, signal[0]) #Padding the signal to compute the difference on the first step
    Up = np.zeros(length, dtype=np.float64)
    Down = np.zeros(length, dtype=np.float64)
    for i in range(1, length+1):
        if signal[i] >= signal[i-1]:
            Up[i-1] = signal[i] - signal[i-1]
        else:
            Down[i-1] = signal[i-1] - signal[i]
    RS = compute_EMA(Up, T) / (compute_EMA(Down, T)+eps)
    RSI = 100 - 100 / (1 + RS)
    return RSI

#A bunch of code that tests if indicators work properly
"""
data = np.load('Finam_data/dataset_32days.npy')
fig = plt.figure(figsize = (20, 10))
fig2 = plt.figure(figsize = (20, 10))
ax1 = fig.add_subplot(111)
ax2 = fig2.add_subplot(111)
ax1.plot(data[9, :, 3])
ax2.plot(compute_RSI(data[9, :, 3]))
plt.show()
"""

