#------creating the confusion matrix for the test set------


#importing required libraries and modules
import numpy as np
import matplotlib.pyplot as plt
import itertools


classes = ['beach', 'bus', 'cafe_restaurant', 'car', 'city_center', 'forest_path' ,'grocery_store',
				'home', 'library', 'metro_station', 'office', 'park', 'residential_area', 
								'train', 'tram']


confusion_matrix = np.load('matrix_majority_voting.npy').astype(float)

no_labels = len(classes)

for j in range(no_labels):
	sum = 0
	for i in range(no_labels):
		sum+=confusion_matrix[i][j]
	for i in range(no_labels):
		confusion_matrix[i][j]/=sum


plt.figure(figsize=(15, 15))
plt.rcParams['figure.figsize']=[30, 30]

plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)

plt.title('Confusion Matrix')
plt.colorbar()

tick_marks = np.arange(no_labels)
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

fmt = '.2f'
thresh = confusion_matrix.max() / 2.

for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
    plt.text(j, i, format(confusion_matrix[i, j], fmt),
        horizontalalignment="center",
            color="white" if confusion_matrix[i, j] > thresh else "black")

plt.savefig('conf_matrix_mv.png')
plt.show()
