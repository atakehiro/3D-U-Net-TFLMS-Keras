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

"""
from tensorflow.python.keras.callbacks import Callback
class CudaProfileCallback(Callback):
    def __init__(self, profile_epoch, profile_batch_start, profile_batch_end):
        self._epoch = profile_epoch - 1
        self._start = profile_batch_start
        self._end = profile_batch_end
        self.epoch_keeper = 0
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_keeper = epoch
    def on_batch_begin(self, batch, logs=None):
        if batch == self._start and self.epoch_keeper == self._epoch:
            print('Starting cuda profiler')
            _cudart.cudaProfilerStart()
        if batch == self._end and self.epoch_keeper == self._epoch:
            print('Stopping cuda profiler')
            _cudart.cudaProfilerStop()
callbacks = []
callbacks.append(CudaProfileCallback(1, 4, 9))
from tensorflow_large_model_support import LMSKerasCallback
starting_names = ['up_sampling3d_2/concat_2']  #['bn_conv1/cond/pred_id']
lms = LMSKerasCallback(n_tensors=-1, lb=1, starting_op_names=starting_names)
callbacks.append(lms)
"""

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
    pred = model.predict(input_data)
    np.save('prediction', pred)
