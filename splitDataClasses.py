import os
import numpy as np

# leave this seeded for reproducible test/training splits
np.random.seed(10)

# path to images directory (probably use absolute path for your system to avoid errors)
path = 'D:/Homework/CS 445 Machine Learning/Project - Card Recognition/Images'

# creates list of the file names in the path directory
pathNames = os.listdir(path)
fileNames = []
for i in pathNames:
    if i[-4:] == ".jpg":
        fileNames.append(i)
numImages = len(fileNames)
fileNames.sort()
truncatedNames = []
for i in range(numImages):
    if fileNames[i][:2] == "10": truncatedNames.append(fileNames[i][:3])
    else: truncatedNames.append(fileNames[i][:2])
uniqueIDs = np.unique(truncatedNames)

for i in uniqueIDs:
    tempPath = path + "/" + i
    # print(tempPath)
    if not os.path.exists(tempPath):
        os.mkdir(tempPath)

for i in fileNames:
    classID = i[:2]
    if classID == "10": classID = i[:3]
    # print(path + "/" + i)
    # print(path + "/" + classID + "/" + i)
    os.rename(path + "/" + i, path + "/" + classID + "/" + i)