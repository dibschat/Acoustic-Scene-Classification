#-----spliting the total data into training, cross validation and testing seperately------


#importing required libraries and modules
import cv2
import numpy as np
import os
import random

class Load:
	#creating empty lists at a golbal scope for storing the splitted data
	X_train = []
	Y_train = []
	X_train_temp = []
	Y_train_temp = []
	X_test = []
	Y_test = []
	X_val = []
	Y_val = []


	def __init__(self, X, Y, tr_per, val_per):	#constructor
		#the constructor will itself split the input data into training, cross-validation and testing depending on the split percentage given by the user
		self.split_train_test(X, Y, tr_per)
		self.split_train_val(Load.X_train_temp, Load.Y_train_temp, val_per)



	def split(self):
		#return the global lists where the splitted data is stored
		return Load.X_train, Load.Y_train, Load.X_val, Load.Y_val, Load.X_test, Load.Y_test



	def split_train_test(self, X, Y, tr_per):
		#this method splits the input data into training and testing
		n = len(X)
		m = int((tr_per/100)*n)
		gen = random.sample(range(0, n), m)
		for i in range(n):
			if i in gen:
				Load.X_train_temp.append(X[i])
				Load.Y_train_temp.append(Y[i])
			else:
				Load.X_test.append(X[i])
				Load.Y_test.append(Y[i])



	def split_train_val(self, X, Y, val_per):
		#this method splits the input data into training and cross-validation
		n = len(X)
		m = int(((100-val_per)/100)*n)
		gen = random.sample(range(0, n), m)
		for i in range(n):
			if i in gen:
				Load.X_train.append(X[i])
				Load.Y_train.append(Y[i])
			else:
				Load.X_val.append(X[i])
				Load.Y_val.append(Y[i])
