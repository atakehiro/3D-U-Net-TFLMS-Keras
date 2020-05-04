#!/usr/bin/env python

import tensorflow as tf
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.keras import backend as K
config = tf.ConfigProto()
config.graph_options.rewrite_options.memory_optimization = rewriter_config_pb2.RewriterConfig.SCHEDULING_HEURISTICS
config.graph_options.rewrite_options.dependency_optimization = rewriter_config_pb2.RewriterConfig.OFF
K.set_session(tf.Session(config=config))

from tensorflow_large_model_support import LMSKerasCallback
lms_callback = LMSKerasCallback()

import tifffile
import numpy as np
from functions import unet_model_3d, data_gen

if __name__ == "__main__":
    image = tifffile.imread('traindata/training_input.tif')
    label = tifffile.imread('traindata/training_groundtruth.tif')
    res = 56  # 8*n
    window_size = (res, res, res)
    input_data = data_gen(image, window_size)
    label_data = data_gen(label, window_size)
    print('image size:', image.shape)
    print('data size:', input_data.shape)
    model = unet_model_3d((1,) + window_size)
    model.summary()
    batch_size = 8
    no_epochs = 10
    model.fit(input_data, label_data, batch_size=batch_size, epochs=no_epochs, verbose=1, callbacks=[lms_callback])
    model.save_weights('./3d_unet.h5')
