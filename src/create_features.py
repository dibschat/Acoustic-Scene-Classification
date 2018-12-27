#------create features i.e. MEL and MFCC and put them in a temporary folder called bin/full for further access------

#including required libraries and modules
import sys
import os
import cv2
import numpy as np
import tflearn
from preprocess import Preprocess
from data_split import Load
from conv_net import CNN
from ensemble import Ensemble

if __name__ == '__main__':
	#classified folders within the dataset from ehetre the as
	classes = ['beach', 'bus', 'cafe_restaurant', 'car', 'city_center', 'forest_path' ,'grocery_store',
				'home', 'library', 'metro_station', 'office', 'park', 'residential_area', 
								'train', 'tram']
	
	feature_type = sys.argv[1]
	feature = sys.argv[2]

	dataset = Preprocess(classes, feature_type, feature)	


	X, Y = dataset.get_data()
	
	#intiliasing the training, validation and testing as list of elements
	#corresponding to the respective mel features
	X_train = [0 for i in range(len(X))]
	Y_train = [0 for i in range(len(X))]
	X_val = [0 for i in range(len(X))]
	Y_val = [0 for i in range(len(X))]
	X_test = [0 for i in range(len(X))]
	Y_test = [0 for i in range(len(X))]

	for i in range(len(X)):
		#create a Load object with required training and validation percentage
		dataset = Load(X[i], Y[i], 80, 20)
		#get the training, validation and test data for each feature
		X_train[i], Y_train[i], X_val[i], Y_val[i], X_test[i], Y_test[i] = dataset.split()
		
		#flatten the mel spectrograms
		for j in range(len(X_train[i])):
			X_train[i][j] = np.array(X_train[i][j]).flatten()
		for j in range(len(X_val[i])):
			X_val[i][j] = np.array(X_val[i][j]).flatten()
		for j in range(len(X_test[i])):
			X_test[i][j] = np.array(X_test[i][j]).flatten()
			

		#reshape the inputs as a 4 dimensional tensor
		X_train[i] = np.array(X_train[i]).reshape([-1, 128, 431, 1])
		X_val[i] = np.array(X_val[i]).reshape([-1, 128, 431, 1])
		X_test[i] = np.array(X_test[i]).reshape([-1, 128, 431, 1])
		#change the data type of the one hot encoded vectors as a 
		#numpy array for feeding it to the CNN
		Y_train[i] = np.array(Y_train[i])
		Y_val[i] = np.array(Y_val[i])
		Y_test[i] = np.array(Y_test[i])


	#saving the preprocessed data for the input feature to save time during training
	np.save('bin/full/'+feature+'/X_train.npy', np.array(X_train))
	np.save('bin/full/'+feature+'/X_val.npy', np.array(X_val))
	np.save('bin/full/'+feature+'/X_test.npy', np.array(X_test))

	np.save('bin/full/'+feature+'/Y_train.npy', np.array(Y_train))
	np.save('bin/full/'+feature+'/Y_val.npy', np.array(Y_val))
	np.save('bin/full/'+feature+'/Y_test.npy', np.array(Y_test))
