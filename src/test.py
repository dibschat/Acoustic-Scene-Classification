#------testing the trained model and ensemble weights on the test data to get the final accuracy


#importing required libraries and modules
import os
import sys
import cv2
import numpy as np
from preprocess import Preprocess
from data_split import Load
from conv_net import CNN
from ensemble import Ensemble



def load_numpy_data(arg, folder):
	#loading the numpy data (.npy files) from the required directory
	X_test = list(np.load('bin/'+folder+'/'+arg+'/X_test.npy'))
	Y_test = list(np.load('bin/'+folder+'/'+arg+'/Y_test.npy'))

	X_test = list(np.array(X_test).reshape(-1, 128, 431))
	Y_test = list(np.array(Y_test).reshape(-1, 15))

	return X_test, Y_test


def predict_test(arg, X_train, X_val, X_test, Y_train, Y_val, Y_test):
	#loading the model and training its corresponding SVR classifier
	data_size = 'full'
	neural_net = CNN()
	model = neural_net.create_1ConvModel()
	model.load('DNN/'+data_size+'/'+arg+'.model')
	
	#defining an ensemble class and training the SVR for the particular classifier
	en = Ensemble()
	en.regressor(arg, model, X_val[0], Y_val[0])
	neural_net.predict_test_data(arg, model, X_test[0], Y_test[0])




if __name__ == '__main__':
	feature = ['mono', 'left', 'right', 'mid', 'side', 'harmonic', 'percussive', 'mfcc']	#all the features used in the architecture
	
	X_test = [0 for i in range(len(feature))]
	Y_test = [0 for i in range(len(feature))]

	for i in range(8):
		X_test[i], Y_test[i] = load_numpy_data(feature[i], 'full')
	
	en = Ensemble()	

	#uncomment whichever method you want to use in your ensemble(SVR or majority voting)
	acc = en.result_SVR(X_test, Y_test)
	#acc = en.result_majority_voting(X_test, Y_test)
	print("Ensemble Test Accuracy =", acc, '%')
	