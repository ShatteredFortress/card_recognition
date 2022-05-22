from PIL import Image, ImageOps
import numpy as np
import os




#path to photos
path = '../../photos/archive/Images/Images/.'

#stores the image name for collection (just saves file names, not contents of the file)
files = os.listdir(path)

photoCount = len(files)

#Array for the dataset
#864 x 1152 is the compressed size of the photos (can check with np.shape())
cards = np.zeros((photoCount, 864, 1152))

#function: convertToGray
#parameters: card image
# -> convets the image to gray scale and returns it
def convertToGray(image):
    gray_image = ImageOps.grayscale(image)
    #uncomment to see the photo in gray
    #gray_image.show()
    return gray_image


#function: normalizer
#parameters: gray scale image
# -> Divides the pixle value by 255 into range of [0-1]
def normalizer(grey_scale):
    normalized = np.asarray(grey_scale)/255
    #uncomment to print shape of normalized array
    #print(np.shape(normalized))
    #print(normalized)
    return normalized

#opens and saves an image (path will be different for each person likely)
#image_x = Image.open('../../photos/archive/Images/Images/AD15.jpg')

#for each file name in the directory
#the card is loaded with Image.open()
#the card object is passed into the composition of functions
#the normalized array is stored in the dataset
for x in range(0, photoCount):
    card_photo_path = '../../photos/archive/Images/Images/' + files[x]
    image_x = Image.open(card_photo_path)
    cards[x] = normalizer(convertToGray(image_x))
