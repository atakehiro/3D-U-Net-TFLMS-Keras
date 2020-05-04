# 3D-U-Net-TFLMS-Keras

このコードはhttps://github.com/ellisdg/3DUnetCNN の3D U-NetをUnifuedMemoryで出来るように改変したものです。

Code is adapted from https://github.com/ellisdg/3DUnetCNN for Unified Memory.

＊元々のKerasのコードをtf.kerasでの実装に変更しました。

＊The original Keras code is changed to a tf.keras implementation.

UnifuedMemoryはIBMのTensorFlow Large Model Support version 1を使用しています。

詳細はhttps://github.com/IBM/tensorflow-large-model-support/tree/tflmsv1

参考：https://qiita.com/takeajioka/items/22b3a6d1a2b72b649ce7


## Usage
Main_LMS.pyで画像のパスやwindow sizeなどのパラメータを調整して実行

```bash
export TF_FORCE_GPU_ALLOW_GROWTH=true
python Main_LMS.py
```
Main.pyはUnifiedMemoryを使わないスクリプト(比較のために用いる)

## Benefit of UnifuedMemory
RTX2080(8GB)のGPU + メインメモリ32GB

UnifuedMemoryなし(Main.py)ではwindow sizeは(48, 48, 48)→UnifuedMemoryあり(Main_LMS.py)では、(56, 56, 56)まで学習が可能になった。(batch size = 8)

## Environment
Anaconda

Python 3.6

tensorflow-gpu==1.10.0

tensorfow-lms== 0.1.0

keras==2.3.1

tifffile==2020.2.16

## Author
Takehiro Ajioka

E-mail:1790651m@stu.kobe-u.ac.jp
