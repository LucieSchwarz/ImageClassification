# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:41:13 2017

@author: schwarzl
"""

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model



# dimensions of our images.
img_width, img_height = 150, 150

# used to rescale the pixel values from [0, 255] to [0, 1] interval
datagen = ImageDataGenerator(rescale=1./255)

test_data_dir = 'data/test'


test_generator = datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=32,
        class_mode='binary')

from PIL import Image
import numpy as np

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray(img, dtype="int32" )
    return data


#ima = load_image("cni.jpg")


model = load_model('model1.h5')

evaluation = model.evaluate_generator(test_generator, 5)

print(evaluation)

#prediction = model.predict(ima, batch_size=32, verbose=0)

#print(prediction)

