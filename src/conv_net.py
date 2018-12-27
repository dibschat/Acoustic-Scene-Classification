#----convolutional neural network for classification------


#importing required libraries and modules
import os
import numpy as np
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

class CNN():
	datasize = 'full'

	def __init__(self):	#constructor
		pass



	def create_1ConvModel(self):
		#creating the CNN model as per the architecture followed with 4-conv and pooling layers
		self.convnet = input_data(shape=[None, 128, 431, 1], name='input')

		self.convnet = conv_2d(self.convnet, 32, 5, activation='relu')
		self.convnet = max_pool_2d(self.convnet, 3)

		self.convnet = conv_2d(self.convnet, 64, 5, activation='relu')
		self.convnet = max_pool_2d(self.convnet, 3)

		self.convnet = conv_2d(self.convnet, 128, 5, activation='relu')
		self.convnet = max_pool_2d(self.convnet, 3)

		self.convnet = conv_2d(self.convnet, 256, 5, activation='relu')
		self.convnet = max_pool_2d(self.convnet, 3)

		self.convnet = tflearn.layers.conv.global_avg_pool(self.convnet)


		self.convnet = fully_connected(self.convnet, 1024, activation='relu')
		#self.convnet = dropout(self.convnet, 0.8)		can be used to avoid overfitting

		self.convnet = fully_connected(self.convnet, 15, activation='softmax')
		self.convnet = regression(self.convnet, optimizer='adam', learning_rate=0.01, 
					loss='categorical_crossentropy', name='targets')
			
		self.model = tflearn.DNN(self.convnet)

		return self.model



	def train_1ConvModel(self, arg, model, X_train, Y_train, X_val, Y_val):
		#training the created model with data from the user
		#here stochastic learning is deployed since the input data is not too high; minibatch_size=1
		
		self.epoch = 10		#set the number of epochs
		
		model.fit({'input' : X_train}, {'targets' : Y_train}, n_epoch=self.epoch, 
				validation_set=({'input' : X_val}, {'targets' : Y_val}),
						show_metric=True, run_id='DCNet')

		model.save('DNN/'+CNN.data_size+'/'+arg+'.model')	#saving the model in the DNN/full folder, 3 files will be created for each model 

		return model		



	def predict_test_data(self, arg, model, X_test, Y_test):
		self.ans = []
		count = 0
		for i in range(len(X_test)):
			self.pr = model.predict([X_test[i]])[0]
			self.ans.append(self.pr)	 
			if(np.array_equal((np.round(self.pr)).astype(int), Y_test[i])):
				count+=1
		
		print(arg, "Test Accuracy = ", (count/len(X_test))*100, "%")	#calculating test accuracy for each classifier

		#saving the softmax outputs for using them later for calculating the ensemble accuracy
		np.save('test_prediction/full/'+arg+'.npy', np.array(self.ans))