# deep-learning-predicting-stock-price

1)Load all your csv files you want to train the network on into the "data" folder

2)Run compile_dataset.py, this will create a training dataset which will be used by network during training. 

3)Now you have everything set up to train the neural net. Run net_training.py to train the network. You may change the number of training epochs by changing the "epochs" variable in the beginning of the code of net_training.py. I recommend you to first leave everything untouched (number of epochs is 10 by default) and check how long does it take to train the network on your computer, and only then to increase the number of epochs depending on your machine's performance.
During the training process, the best configuration of the network will be saved in "Saves" folder to be later used by net_prediction.py

4)Now, when the network is trained, it's time to apply it! Load the csv file with data you want to predict on to the folder where net_prediction.py lies and rename it to "data.csv". Then, launch net_prediction.py. It should output one line: the prediction.
