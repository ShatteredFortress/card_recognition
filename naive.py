#!/usr/bin/env python

#https://scikit-learn.org/stable/modules/naive_bayes.html

from sklearn.naive_bayes import GaussianNB
# If we want to split the dataset sklearn's way
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
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

#Takes an image and returns a 1D greyscaled and Normalized array  
# scanned horizontally
def imageToArray(image):
    #skips pixel by skip size effectivly decreasing resolution by a factor of skip
    skip = 10

    #GreyScale and flatten the Image
    greyscaleArray=[]
    array3D = np.array(image) 
    for vertical in range(0, np.shape(array3D)[0] , skip):
        for horizontal in range(0, np.shape(array3D)[1], skip):
            greyscaleValue = (sum(array3D[vertical][horizontal])) / 3
            greyscaleArray.append(greyscaleValue)

    greyscaleArray = np.array(greyscaleArray, dtype=float)
    normalizedArray = preprocessing.normalize([greyscaleArray])
    return normalizedArray[0]

# Import DataSet
#x = features = pixel arrays
#y = classes = "KH"
#x and y must have the same number of indecies, one class per image
X=[]
y=[]
# if the dataSet isn't formatted then format it and save it as raw binary
if not (os.path.exists("./features.npy") and os.path.exists("./classes.npy")):
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
        
    # np.save does not keep the dimensions of secondary dimensions
    # therefore I have to save features and classes individually
    # https://stackoverflow.com/questions/51040059/numpy-saving-an-object-with-arrays-of-different-shape-but-same-leading-dimension
    print("Saving features")
    np.save('./features.npy',X)
    print("Saving classes")
    np.save('./classes.npy',y)

# load the saved .npy DataSet
else:
    print("Loading features")
    X = np.load('features.npy', allow_pickle=True)
    print("Loading classes")
    y = np.load('classes.npy', allow_pickle=True)
############################

#Split DataSet
print("Splitting DataSet")
#Probably want 80/20 split for test/train ?
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
############################

# Train Data
print("Training")
y_pred = gnb.fit(X_train, y_train).predict(X_test)

print("Number of mislabeled points out of a total %d points : %d"% (np.shape(X_test)[0], (y_test != y_pred).sum()))
############################

#Confusion Matrix
#
############################

