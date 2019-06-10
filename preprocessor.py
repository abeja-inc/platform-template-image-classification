# coding: utf-8


from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import img_to_array


def preprocessor(img):
    img = img_to_array(img)
    img = preprocess_input(img, mode='tf')
    return img
