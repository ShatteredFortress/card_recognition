#!/usr/bin/env python

#https://scikit-learn.org/stable/modules/naive_bayes.html

from sklearn.naive_bayes import GaussianNB,ComplementNB,CategoricalNB,BernoulliNB,MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sys
import os
import numpy as np
from PIL import Image

# for conmat export
import seaborn as sn
import matplotlib.pyplot as plt

gnb = [GaussianNB(),ComplementNB(),CategoricalNB(),MultinomialNB(),BernoulliNB()]

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

# Simple function for preparing and exporting a conmat to a png
# Prints in a heatmap format, and includes the training type in the title
def exportConMat(test, pred, dist_type):
    # find all classes (sorted)
    classes = np.unique(test)
    conmat = np.zeros((np.shape(classes)[0], np.shape(classes)[0]))
    # for each input, mark where in the confusion matrix the input lies (x = test, y = prediction)
    for (t, p) in zip(test, pred):
        t_idx = np.argwhere(classes==t)[0][0]
        p_idx = np.argwhere(classes==p)[0][0]
        conmat[t_idx][p_idx] += 1
    # exporting conmat
    fname = f'{dist_type}_conmat.png'
    sn.set(color_codes=False)
    plt.figure(1, figsize=np.shape(conmat))
    plt.title(f'Confusion Matrix: {dist_type} Distribution')
    sn.set(font_scale=0.6)
    mp = sn.heatmap(conmat, annot=True, cbar=False, cmap='Blues')
    mp.set_xticklabels(classes)
    mp.set_yticklabels(classes)
    mp.set(xlabel="Test class", ylabel="Predicted class")
    plt.tight_layout()
    plt.savefig(fname)

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

# Train Data on all distrubution types offered by sklearn
for i in range(len(gnb)):
    print("Training: ",gnb[i])
    y_pred = gnb[i].fit(X_train, y_train).predict(X_test)
    print("Number of mislabeled points out of a total %d points : %d"% (np.shape(X_test)[0], (y_test != y_pred).sum()))

    #Confusion Matrix
    #
    ############################
###########################
    exportConMat(y_test, y_pred, gnb[i])
