from PIL import Image

import os

#path to photos
path = '../../photos/archive/Images/Images/.'

#stores the image name for collection (just saves file names, not contents of the file)
files = os.listdir(path)

#this function compresses the photo down to 1/4 the size in order to work for our dataset
def compress(file, verbose = False):
    filepath = file
    picture = Image.open(filepath)
    img = picture.resize((1152, 864), Image.ANTIALIAS)
    picture = img
    picture.save(filepath, "JPEG", optimize = True, quality = 10)
    return


for file in files:
    card_photo_path = '../../photos/archive/Images/Images/' + file
    compress(card_photo_path)



