#!/usr/bin/env python

import tifffile
import numpy as np
from unet import unet_model_3d

def data_gen(image, window_size):
    size = image.shape
    mod0 = size[0] % window_size[0]
    mod1 = size[1] % window_size[1]
    mod2 = size[2] % window_size[2]
    tmp0 = np.delete(image, list(range(image.shape[0] - mod0,image.shape[0])), axis=0)
    tmp1 = np.delete(tmp0, list(range(image.shape[1] - mod1,image.shape[1])), axis=1)
    tmp2 = np.delete(tmp1, list(range(image.shape[2] - mod2,image.shape[2])), axis=2)
    data = tmp2.reshape((-1,1)+window_size)
    return data

if __name__ == "__main__":
    image = tifffile.imread('traindata/training_input.tif')
    label = tifffile.imread('traindata/training_groundtruth.tif')
    res = 48
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
    pred = model.predict(input_data)
    #iou = mean_iou(label_data, (pred > 0.5).int())
    #print('IOU:', iou)
    np.save('prediction', pred)

