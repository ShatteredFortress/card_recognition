from PIL import Image, ImageOps
import numpy as np

def convertToGray(image):
    gray_image = ImageOps.grayscale(image)

    gray_image.show()
    return gray_image

def normalizer(grey_scale):
    normalized = np.asarray(grey_scale)/255
    print(np.shape(normalized))
    return normalized
  
image_x = Image.open('../photos/archive/Images/Images/2H30.jpg')


print(normalizer(convertToGray(image_x)))