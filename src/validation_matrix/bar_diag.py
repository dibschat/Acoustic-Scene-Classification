#------creating the bar chart for the cross-validation set------



#importing required libraries and modules
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image


directory = ['mono.npy', 'left.npy', 'right.npy', 'mfcc.npy', 'mid.npy',
				'side.npy', 'harmonic.npy', 'percussive.npy']


features = ['mono', 'left', 'right', 'mfcc', 'mid', 'side', 
						'harmonic', 'percussive']


image = []

for name in directory:
	image.append(np.load(name))

image = np.array(image)

s = []
for i in range(len(image)):
	nor = np.sum(len(image[i]))
	s.append(np.sum(image[i])/nor)


y_pos = np.arange(len(features))

plt.figure(figsize=(15, 15))
plt.rcParams['figure.figsize']=[30, 30]

plt.bar(y_pos, s, align='center', alpha=0.5)
plt.xticks(y_pos, features)
plt.ylabel('Number of correctly predicted data')
plt.title('Bar Chart')

plt.savefig("Bar Chart")
plt.show()