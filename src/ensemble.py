#------Ensemble part of the architecture which assigns weights to the classifiers according to instances------
#------the weights also know as success prediction function(SPF) are predicted by training a regressor--------
#-------------------------------------on the 1 fold cross-validation set--------------------------------------


#including required libraries and modules
import os
import numpy as np
from sklearn.svm import SVR
from conv_net import CNN
import tensorflow as tf
import pickle


class Ensemble:
	#declaring global variables to be used by all the class methods
	path = './Ensemble_reg/full/'	#path where the trained SVRs will be stored for later use 
	spf = []	#success prediction function
	confusion_matrix = np.zeros((15, 15), dtype=int)	#initliasing the confusion-matrix
	features = {'mono':0, 'left':1, 'right':2, 'mid':3, 'side':4, 
						'harmonic':5, 'percussive':6, 'mfcc':7}


	def __init__(self):		#constructor
		pass



	def numpy_max(self, a):
		#returns the one hot encoded array of a sotmax output where 1 denotes the index of the maximum probablity
		self.index = np.argmax(a)
		self.b = np.zeros(len(a), dtype=int)
		self.b[self.index] = 1
		return self.b



	def regressor(self, arg, model, X_val, Y_val):
		#training SVRs for creating success-weighted-ensemble-classifiers
		self.Y = []
		self.validation_matrix = np.load('validation_matrix/mat.npy')

		for i in range(len(X_val)):
			self.pr = Ensemble.numpy_max(self, model.predict([X_val[i]])[0])
	
			self.cls = np.argmax(self.pr)
			self.lab = np.argmax(Y_val[i])
			self.validation_matrix[self.lab][self.cls]+=1

			#alternative for kronicker-delta of predicted and classified labels
			if(np.array_equal(self.pr, Y_val[i])):
				self.Y.append(1)
			else:
				self.Y.append(0)
		
		#saving the datas for validation matrix
		np.save('validation_matrix/mat.npy', self.validation_matrix)
		np.save('validation_matrix/'+arg+'.npy', np.array(self.Y))

		#printing the validation accuracy of each classifier
		print(arg, "Validation Accuracy = ", (self.Y.count(1)/len(self.Y))*100, "%")

		self.Y = np.array(self.Y)
		x = []
		for i in range(len(X_val)):
			x.append(X_val[i].flatten())
		x = np.array(x)

		self.reg = SVR()	#declaring the SVR regressor as reg
		self.reg.fit(x, self.Y)	#training the SVR regressor
		
		self.name = arg + '.pkl'
		self.output = os.path.join(Ensemble.path, self.name)
		with open(self.output, 'wb') as file:
			pickle.dump(self.reg, file)
		print(self.reg)




	def create_success_prediction_function(self):
		#loading the SPFs for each classifier and appending them to the global spf list
		for arg in Ensemble.features:
			self.name = arg + '.pkl'
			self. output = os.path.join(Ensemble.path, self.name)
			print(self.output)
			with open(self.output, 'rb') as file:
				Ensemble.spf.append(pickle.load(file))




	def SPF(self, R, X):
		#returns the weight or success prediction value for a particular and a given input instance
		return Ensemble.spf[R].predict(np.array(X).reshape(1, -1))




	def result_SVR(self, X_test, Y_test):
		#returns the result as the normalised weighted sum of all the classifier outputs
		self.ans = []

		Ensemble.create_success_prediction_function(self)		

		for arg in Ensemble.features:
			self.out_path = 'test_prediction/full/'+arg+'.npy'
			self.ans.append(list(np.load(self.out_path)))

		self.count = 0
		for j in range(len(self.ans[0])):
			self.gamma = 0
			self.w_sum = 0
			
			for i in range(len(self.ans)):
				self.s_p_f = Ensemble.SPF(self, i, X_test[i][j])
				self.gamma+=self.s_p_f
				self.clf_output = self.ans[i][j]
				self.w_sum += self.s_p_f*self.clf_output
			self.w_sum/=self.gamma

			self.w_sum = Ensemble.numpy_max(self, self.w_sum)
			self.cls = np.argmax(self.w_sum)
			self.lab = np.argmax(np.array(Y_test[i][j]))
			Ensemble.confusion_matrix[self.lab][self.cls]+=1
			if(np.array_equal(self.w_sum, np.array(Y_test[i][j]))):
				self.count+=1
			
			print(j+1, self.lab, self.cls, self.count)	#indexes test data and prints actual labels, classified label 
									#and number of correctly classified labels accordingly

		np.save('confusion_matrix/matrix.npy', Ensemble.confusion_matrix)
		#calculating the overall ensemble accuracy
		self.n = len(self.ans[0])
		self.accuracy = (self.count/self.n)*100
		return self.accuracy




	def result_majority_voting(self, X_test, Y_test):
		#returns the result as the majority predicted class of all the classifier outputs
		self.ans = []

		for arg in Ensemble.features:
			self.out_path = 'test_prediction/full/'+arg+'.npy'
			self.ans.append(list(np.load(self.out_path)))


		self.count = 0
		for j in range(len(self.ans[0])):			
			self.final = 0
			for i in range(len(self.ans)):
				self.clf_output = self.ans[i][j]
				self.final += self.clf_output

			self.final = Ensemble.numpy_max(self, self.final)
			self.cls = np.argmax(self.final)
			self.lab = np.argmax(np.array(Y_test[i][j]))
			Ensemble.confusion_matrix[self.lab][self.cls]+=1
			if(np.array_equal(self.final, np.array(Y_test[i][j]))):
				self.count+=1

			print(j+1, self.lab, self.cls, self.count)	#indexes test data and prints actual labels, classified label 
									#and number of correctly classified labels accordingly

		np.save('confusion_matrix/matrix_majority_voting.npy', Ensemble.confusion_matrix)
		#calculating overall ensemble accuracy
		self.n = len(self.ans[0])
		self.accuracy = (self.count/self.n)*100
		return self.accuracy	
