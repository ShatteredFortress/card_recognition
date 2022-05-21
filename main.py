from PIL import Image, ImageOps
import numpy as np


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
    return normalized

#opens and saves an image (path will be different for each person likely)
image_x = Image.open('../photos/archive/Images/Images/2H30.jpg')

#prints the normalized array of pixels 
print(normalizer(convertToGray(image_x)))
