#------create 4 grayscale mel spectrogram features of the argument audio------
#------the created spectrograms are places in a folder called bin having------
#------the created features are left and right channel, mid-------------------
#------and side channel, harmonic and percussive channel and single-----------
#----------------------------mono channel-------------------------------------


#importing required libraries and modules
import os
import librosa
import librosa.display
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import pylab

class Preprocess():
	#creating empty lists to append data
	X = []
	Y = []

	#folder is a list containing all the folders under audio
	def __init__(self, folder, feature_type, feature):	#constructor
		if(feature_type=='mfcc'):
			self.feature_mfcc(folder)#compute mfcc features
		elif(feature_type=='mel'):
			self.ft = feature
			self.feature_mel(folder, self.ft)#compute mel features
		else:
			print("No such Feature as "+feature+".")	#if no such feature exists
			exit()	#forceful exit
			

	def get_data(self):			
		#returning the preprocessed data to the calling method
		return Preprocess.X, Preprocess.Y


	def feature_mel(self, folder, feature):
		#initiliasing one-hot encoded vectors corresponding to the folder labels
		idx = {'beach':[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'bus':[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
			'cafe_restaurant':[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'car':[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			'city_center':[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'forest_path':[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
			'grocery_store':[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'home':[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
			'library':[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'metro_station':[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
			'office':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'park':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
			'residential_area':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'train':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
			'tram':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}


		self.X_mel = []
		self.Y_mel = []

		#searching through all the folders and picking up audio
		for fold in folder:
			self.path = "./audio/"+fold 	#path where the audio is
			self.files = librosa.util.find_files(self.path, ext=["wav"])
			self.count = 1	#to see in the terminal how many files have been loaded
			print(fold)
			for audio in self.files:
				print(self.count)
				#according to the feature given by the user, its mel spectrogram is created
				if(feature=='mono'):
					self.X_mel.append(self.mono(audio, fold))
				elif(feature=='left'):
					self.X_mel.append(self.left_right(audio, fold)[0])
				elif(feature=='right'):
					self.X_mel.append(self.left_right(audio, fold)[1])
				elif(feature=='mid'):
					self.X_mel.append(self.mid_side(audio, fold)[0])
				elif(feature=='side'):				
					self.X_mel.append(self.mid_side(audio, fold)[1])
				elif(feature=='harmonic'):
					self.X_mel.append(self.HPSS(audio, fold)[0])
				elif(feature=='percussive'):				
					self.X_mel.append(self.HPSS(audio, fold)[1])
				else:
					print("No such Mel Feature as "+feature+".")	#if no such feature exists
					exit()	#forceful exit

				#appending the corresponding one-hot encoded output vectors
				self.Y_mel.append(np.array(idx[fold]))
				self.count+=1
		
		Preprocess.X = [self.X_mel]
		Preprocess.Y = [self.Y_mel]



	def feature_mfcc(self, folder):
		#initiliasing one-hot encoded vectors corresponding to the folder labels
		idx = {'beach':[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'bus':[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
			'cafe_restaurant':[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'car':[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			'city_center':[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'forest_path':[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
			'grocery_store':[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'home':[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
			'library':[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'metro_station':[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
			'office':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'park':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
			'residential_area':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'train':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
			'tram':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}
		

		self.X_mfcc = []
		self.Y_mfcc = []

		#searching through all the folders and picking up audio
		for fold in folder:
			self.path = "./audio/"+fold 	#path where the audio is
			self.files = librosa.util.find_files(self.path, ext=["wav"])
			self.count = 1	#to see in the terminal how many files have been loaded
			for audio in self.files:
				print(self.count)
				y, sr = librosa.core.load(audio, mono=True)#loading the audio

				#creating the corresponding mfcc numpy array
				Y = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=128, fmax=8000)
				self.X_mfcc.append(Y)#appending the mfcc feature 
				#appending the corresponding one-hot encoded output vectors
				self.Y_mfcc.append(np.array(idx[fold]))
				self.count+=1
		
		Preprocess.X = [self.X_mfcc]
		Preprocess.Y = [self.Y_mfcc]



	def feature_mel_ensemble(self, folder):
		#initiliasing one-hot encoded vectors corresponding to the folder labels
		idx = {'beach':[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'bus':[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
			'cafe_restaurant':[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'car':[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			'city_center':[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 'forest_path':[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
			'grocery_store':[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], 'home':[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], 
			'library':[0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], 'metro_station':[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
			'office':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], 'park':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
			'residential_area':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], 'train':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], 
			'tram':[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]}


		
		#initialising empty lists to append the mel features
		#features are classified as mono, left, right, mid, side, harmonic and percussive
		self.X_mono = []
		self.Y_mono = []

		self.X_left = []
		self.Y_left = []

		self.X_right = []
		self.Y_right = []

		self.X_mid = []
		self.Y_mid = []

		self.X_side = []
		self.Y_side = []

		self.X_harmonic = []
		self.Y_harmonic = []

		self.X_percussive = []
		self.Y_percussive = []

		#searching through all the folders and picking up audio
		for fold in folder:
			self.path = "./audio/"+fold 	#path where the audio is
			self.files = librosa.util.find_files(self.path, ext=["wav"])
			self.count = 1 	#to see in the terminal how many files have been loaded
			for audio in self.files:
				print(self.count)
				#monoaural audio
				self.X_mono.append(self.mono(audio, fold))

				#the left and right channels of stereo audio
				ret = self.left_right(audio, fold)
				self.X_left.append(ret[0])
				self.X_right.append(ret[1])

				#the mid and side channels of stereo audio to understand the 
				#gradual change of the audio nature over time
				ret = self.mid_side(audio, fold)
				self.X_mid.append(ret[0])
				self.X_side.append(ret[1])

				#seperating the harmonic and percussive part of the audio by
				#a technique similar to NMF
				ret = self.HPSS(audio, fold)
				self.X_harmonic.append(ret[0])
				self.X_percussive.append(ret[1])

				#appending the corresponding one-hot encoded output vectors
				self.Y_mono.append(np.array(idx[fold]))
				self.Y_left.append(np.array(idx[fold]))
				self.Y_right.append(np.array(idx[fold]))
				self.Y_mid.append(np.array(idx[fold]))
				self.Y_side.append(np.array(idx[fold]))
				self.Y_harmonic.append(np.array(idx[fold]))
				self.Y_percussive.append(np.array(idx[fold]))
				self.count+=1

		#create the final list consisting of all the features
		Preprocess.X = [self.X_mono, self.X_left, self.X_right, self.X_mid, self.X_side, self.X_harmonic, self.X_percussive]
		Preprocess.Y = [self.Y_mono, self.Y_left, self.Y_right, self.Y_mid, self.Y_side, self.Y_harmonic, self.Y_percussive]


	def mono(self, audio, fold):
		#this is for creating the mono channel
		y, sr = librosa.core.load(audio, mono=True)			
		#creating the corresponding mel spectrogram
		Y = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
		return Y



	def left_right(self, audio, fold):
		#this is for creating the left and right channels form the stereo audio
		y, sr = librosa.core.load(audio, mono=False)

		x = np.shape(y)
		
		#dividing the left and right channels
		l = y[0]
		r = y[1]

		#creating the corresponding mel spectrogram
		L = librosa.feature.melspectrogram(y=l, sr=sr, n_mels=128, fmax=8000)
		R = librosa.feature.melspectrogram(y=r, sr=sr, n_mels=128, fmax=8000)

		#returning two spectrograms as a list
		return [L, R]



	def mid_side(self, audio, fold):
		#this is for creating the mid and side channels form the left and right
		#channel of thestereo audio
		y, sr = librosa.core.load(audio, mono=False)

		x = np.shape(y)

		#dividing the left and right channels
		l = y[0]
		r = y[1]

		#mid channel created by taking the addition of left and right
		mid = np.zeros((x[1])) 
		for i in range(x[1]):
			mid[i] = l[i]+r[i]

		#side channel created by taking the difference of left and right
		side = np.zeros((x[1])) 
		for i in range(x[1]):
			side[i] = l[i]-r[i]

		#creating the corresponding mel spectrogram
		MID = librosa.feature.melspectrogram(y=mid, sr=sr, n_mels=128, fmax=8000)
		SIDE = librosa.feature.melspectrogram(y=side, sr=sr, n_mels=128, fmax=8000)

		#returning two spectrograms as a list
		return [MID, SIDE]
			


	def HPSS(self, audio, fold):
		#this is for creating the harmonic and percussive channels form the mono audio
		y, sr = librosa.core.load(audio, mono=True)
		Y = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
		
		#harmonic-percussive-source-seperation
		#different mel-spectrograms of h and p not required because the decompostion
		#was done on the mel-spectrogram of the mono
		h, p = librosa.decompose.hpss(Y) 

		#returning two spectrograms as a list
		return [h, p]
