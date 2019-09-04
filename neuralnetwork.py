import numpy as np
import pandas as pd
import csv
import math
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import Dense



#Neural Network CODE START
#This is a simple version of what we used in Project 1.1
#I have a Model with Epochs 250, batch size 15
#I am using 3 Dense layers. In the first two input layers i am using Relu Activation function
#In the Output Layer, i am using Sigmoid just because it is a classification problem with binary classifiers.
def runNeuralNetwork(inputDataSet,outPutDataSet,inputDimension):
	XTemp=np.array(inputDataSet)
	XNew=XTemp[0:,0:inputDimension]
	YNew=outPutDataSet
	
	model = Sequential()
	model.add(Dense(300, input_dim=inputDimension,init='uniform', activation='relu'))
	model.add(Dense(inputDimension,init='uniform', activation='relu'))
	model.add(Dense(1, init='uniform', activation='sigmoid'))
    # Compile model
	model.compile(loss='binary_crossentropy', optimizer='SGD', metrics=['accuracy'])
    # Fit the model
	model.fit(XNew, YNew, epochs=250, batch_size=15)
    # calculate predictions
	predictions = model.predict(XNew) 
    # round predictions
	rounded = [round(x[0]) for x in predictions]
	
	
	matches=0
	for i in range(len(rounded)):
		if rounded[i]==YNew[i]:
			matches+=1
	
	accuracy=(matches/np.size(YNew))*100
	print("----------Accuracy of Neural Network is---------:"+str(accuracy))