import os
import numpy as np

# leave this seeded for reproducible test/training splits
np.random.seed(10)

# path to images directory (probably use absolute path for your system to avoid errors)
path = 'D:/Homework/CS 445 Machine Learning/Project - Card Recognition/Images'

# creates list of the file names in the path directory
fileNames = os.listdir(path)
# os.chdir(path)
os.mkdir(path + '/test')
os.mkdir(path + '/training')
numImages = len(fileNames)
fileNames.sort()
truncatedNames = []
for i in range(numImages):
    if fileNames[i][:2] == "10": truncatedNames.append(fileNames[i][:3])
    else: truncatedNames.append(fileNames[i][:2])
uniqueIDs = np.unique(truncatedNames)

separatedMatrix = []
for i in uniqueIDs:
    tempArray = []
    for j in fileNames:
        x = j[:2]
        if x == "10": x = j[:3]
        if x == i: tempArray.append(j)
    np.random.shuffle(tempArray)
    separatedMatrix.append(tempArray)

tempPath = ""
for i in separatedMatrix:
    for j in range(len(i)):
        if j < 41: tempPath = "/training/"
        else: tempPath = "/test/"
        os.rename(path + "/" + i[j], path + tempPath + i[j])