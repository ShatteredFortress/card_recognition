#!/usr/bin/env python

#https://scikit-learn.org/stable/modules/naive_bayes.html

from sklearn.naive_bayes import GaussianNB
# If we want to split the dataset sklearn's way
from sklearn.model_selection import train_test_split
import sys
import os
import numpy as np
from PIL import Image

gnb = GaussianNB()
'''
We can test all of these
- GaussianNB()
- CategoricalNB()
- BernoulliNB()
- MultinomialNB()
- ComplementNB()
'''
# For multi-class naive bayes we just have to run all the predictions, then argmax the positive probabilities.
# ACTUALLY WE DON'T HAVE TO DO THAT AT ALL IT ALREADY SUPPORTS MULTI-CLASSES!?

#Takes an greyscale image and returns a 1D array with the value of each pixel 
# scanned horizontally
def imageToArray(image):
    array3D = np.array(image) 
    array1D = array3D.flatten()
    #only need every 3rd value since RGB are the same in greyscale
    array = array1D[::3]
    return array

# Import DataSet
#x = features = pixel arrays
#y = classes = "KH"
#x and y must have the same number of indecies, one class per image
X=[]
y=[]
# command line exctracted Image folder
directoryPath = sys.argv[1]
print("Starting Directory: ", directoryPath)
for folder in os.listdir(directoryPath):
    print("Reading Folder: ", folder)
    folderPath = os.path.abspath(directoryPath) +"/"+ folder
    for file in os.listdir(folderPath):
        print("\t--> Reading File: ", file)
        filePath = os.path.abspath(folderPath) +"/"+ file
        cardPicture = Image.open(filePath)
        X.append(imageToArray(cardPicture))
        y.append(folder)
############################

#Split DataSet
print("Splitting DataSet")
#Probably want 80/20 split for test/train ?
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=1)
############################

# Train Data
print("Training")
y_pred = gnb.fit(X_train, y_train).predict(X_test)

print("Number of mislabeled points out of a total %d points : %d"% (np.shape(X_test)[0], (y_test != y_pred).sum()))
############################

#Confusion Matrix
#
############################

