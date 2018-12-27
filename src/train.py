#---------------train the CNN or the ensemble-----------------


#importing required libraries and modules
import os
import sys
import cv2
import numpy as np
import tflearn
from preprocess import Preprocess
from data_split import Load
from conv_net import CNN
from ensemble import Ensemble



def load_numpy_data(arg, folder):
	#loading the numpy data (.npy files) from the required directory
	X_train = list(np.load('bin/'+folder+'/'+arg+'/X_train.npy'))
	X_val = list(np.load('bin/'+folder+'/'+arg+'/X_val.npy'))
	X_test = list(np.load('bin/'+folder+'/'+arg+'/X_test.npy'))

	Y_train = list(np.load('bin/'+folder+'/'+arg+'/Y_train.npy'))
	Y_val = list(np.load('bin/'+folder+'/'+arg+'/Y_val.npy'))
	Y_test = list(np.load('bin/'+folder+'/'+arg+'/Y_test.npy'))

	return X_train, X_val, X_test, Y_train, Y_val, Y_test



def train_CNN(arg, X_train, X_val, X_test, Y_train, Y_val, Y_test):
	#training the CNN model
	neural_net = CNN()
	model = neural_net.create_1ConvModel()
	model = neural_net.train_1ConvModel(arg, model, X_train[0], Y_train[0], X_val[0], Y_val[0])

	#predicting the test data and saving it to 'test_prediction/full' to avoid same computation every time
	neural_net.predict_test_data(arg, model, X_test[0], Y_test[0])


def train_Ensemble(arg, X_train, X_val, X_test, Y_train, Y_val, Y_test):
	#loading the model and training its corresponding SVR classifier
	data_size = 'full'
	neural_net = CNN()
	#creating the model structure and loading the saved trained model
	model = neural_net.create_1ConvModel()
	model.load('DNN/'+data_size+'/'+arg+'.model')
	
	#defining an ensemble class and training the SVR for the particular classifier
	en = Ensemble()
	en.regressor(arg, model, X_val[0], Y_val[0])



if __name__ == '__main__':
	#take two arguments from the terminal
	folder = sys.argv[1]	#the folder where all the data are saved, for 15 class classification, it is 'full'
	arg = sys.argv[2]	#the type of feature on which the classifier(CNN) will be trained

	X_train, X_val, X_test, Y_train, Y_val, Y_test = load_numpy_data(arg, folder)	#loading numpy data form the saved files


	#uncomment train_CNN() or train_Ensemble() which ever you want to train
	#for the first time, you need to train the CNN first, then the Ensemble

	train_CNN(arg, X_train, X_val, X_test, Y_train, Y_val, Y_test)	#training the CNN
	
	#train_Ensemble(arg, X_train, X_val, X_test, Y_train, Y_val, Y_test)	#training the Ensemble