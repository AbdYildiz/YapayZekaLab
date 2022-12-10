from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, SpatialDropout2D, Flatten, Dropout, Dense
from keras.preprocessing import image
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os

train = ImageDataGenerator(rescale=1/255)
test = ImageDataGenerator(rescale=1/255)

traindataset = train.flow_from_directory('')