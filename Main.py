#!/usr/bin/env python

import tensorflow as tf
import tifffile
import numpy as np
from functions import unet_model_3d, data_gen

if __name__ == "__main__":
    image = tifffile.imread('traindata/training_input.tif')
    label = tifffile.imread('traindata/training_groundtruth.tif')
    res = 48  # 8*n
    window_size = (res, res, res)
    input_data = data_gen(image, window_size)
    label_data = data_gen(label, window_size)
    print('image size:', image.shape)
    print('data size:', input_data.shape)
    model = unet_model_3d((1,) + window_size)
    model.summary()
    batch_size = 8
    no_epochs = 1
    model.fit(input_data, label_data, batch_size=batch_size, epochs=no_epochs, verbose=1)
    pred = model.predict(input_data)
    np.save('prediction', pred)

