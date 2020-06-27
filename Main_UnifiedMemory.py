#!/usr/bin/env python

import tifffile
import numpy as np
from functions import unet_model_3d, data_gen

import tensorflow as tf
from tensorflow.python.keras import backend as K
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 4
config.gpu_options.allow_growth = True
K.set_session(tf.Session(config=config))

if __name__ == "__main__":
    image = tifffile.imread('traindata/training_input.tif')
    label = tifffile.imread('traindata/training_groundtruth.tif')
    res = 88  # 8*n
    window_size = (res, res, res)
    input_data = data_gen(image, window_size)
    label_data = data_gen(label, window_size)
    print('image size:', image.shape)
    print('data size:', input_data.shape)
    model = unet_model_3d((1,) + window_size)
    model.summary()
    batch_size = 8
    no_epochs = 10
    model.fit(input_data, label_data, batch_size=batch_size, epochs=no_epochs, verbose=1)
    model.save_weights('./3d_unet.h5')
