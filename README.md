# 3D-U-Net-TFLMS-Keras

このコードはhttps://github.com/ellisdg/3DUnetCNN を参考に3D U-NetをUnifuedMemoryで出来るように改変したものです。

Code is adapted from https://github.com/ellisdg/3DUnetCNN for Unified Memory.

UnifuedMemoryはIBMのTensorFlow Large Model Support version 1を使用しています。

詳細はhttps://github.com/IBM/tensorflow-large-model-support/tree/tflmsv1

参考：https://qiita.com/takeajioka/items/22b3a6d1a2b72b649ce7


## Usage

```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
python Main_LMS.py
```


## Dependencies
Anaconda

Python 3.6

tensorflow-gpu==1.10.0

tensorfow-lms== 0.1.0

keras==2.3.1

tifffile==2020.2.16

## Author
Takehiro Ajioka

E-mail:1790651m@stu.kobe-u.ac.jp
