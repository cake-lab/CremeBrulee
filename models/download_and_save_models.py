#!env python

import sys
import logging

import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input


from tensorflow.python.keras.applications import densenet
from tensorflow.python.keras.applications import efficientnet
from tensorflow.python.keras.applications import inception_resnet_v2
from tensorflow.python.keras.applications import inception_v3
from tensorflow.python.keras.applications import mobilenet
from tensorflow.python.keras.applications import mobilenet_v2
from tensorflow.python.keras.applications import mobilenet_v3
from tensorflow.python.keras.applications import nasnet
from tensorflow.python.keras.applications import resnet
from tensorflow.python.keras.applications import resnet_v2
from tensorflow.python.keras.applications import vgg16
from tensorflow.python.keras.applications import vgg19
from tensorflow.python.keras.applications import xception
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.platform import test


ARG_TO_MODEL = {
    'resnet': (resnet, [resnet.ResNet50, resnet.ResNet101, resnet.ResNet152]),
    'resnet_v2': (resnet_v2, [resnet_v2.ResNet50V2, resnet_v2.ResNet101V2,
                              resnet_v2.ResNet152V2]),
    'vgg16': (vgg16, [vgg16.VGG16]),
    'vgg19': (vgg19, [vgg19.VGG19]),
    'xception': (xception, [xception.Xception]),
    'inception_v3': (inception_v3, [inception_v3.InceptionV3]),
    'inception_resnet_v2': (inception_resnet_v2,
                            [inception_resnet_v2.InceptionResNetV2]),
    'mobilenet': (mobilenet, [mobilenet.MobileNet]),
    'mobilenet_v2': (mobilenet_v2, [mobilenet_v2.MobileNetV2]),
    'mobilenet_v3_small': (mobilenet_v3, [mobilenet_v3.MobileNetV3Small]),
    'mobilenet_v3_large': (mobilenet_v3, [mobilenet_v3.MobileNetV3Large]),
    'densenet': (densenet, [densenet.DenseNet121,
                            densenet.DenseNet169, densenet.DenseNet201]),
    'nasnet_mobile': (nasnet, [nasnet.NASNetMobile]),
    'nasnet_large': (nasnet, [nasnet.NASNetLarge]),
    'efficientnet': (efficientnet,
                     [efficientnet.EfficientNetB0, efficientnet.EfficientNetB1,
                      efficientnet.EfficientNetB2, efficientnet.EfficientNetB3,
                      efficientnet.EfficientNetB4, efficientnet.EfficientNetB5,
                      efficientnet.EfficientNetB6, efficientnet.EfficientNetB7])
}

ARG_TO_MODEL = [
    resnet.ResNet50, 
    resnet.ResNet101, 
    resnet.ResNet152,
    resnet_v2.ResNet50V2, 
    resnet_v2.ResNet101V2, 
    resnet_v2.ResNet152V2,
    vgg16.VGG16,
    vgg19.VGG19,
    xception.Xception,
    inception_v3.InceptionV3,
    inception_resnet_v2.InceptionResNetV2,
    mobilenet.MobileNet,
    mobilenet_v2.MobileNetV2,
    mobilenet_v3.MobileNetV3Small,
    mobilenet_v3.MobileNetV3Large,
    densenet.DenseNet121, 
    densenet.DenseNet169, 
    densenet.DenseNet201,
    nasnet.NASNetMobile,
    nasnet.NASNetLarge,
    efficientnet.EfficientNetB0, 
    efficientnet.EfficientNetB1,
    efficientnet.EfficientNetB2,
    efficientnet.EfficientNetB3,
    efficientnet.EfficientNetB4, 
    efficientnet.EfficientNetB5,
    efficientnet.EfficientNetB6, 
    efficientnet.EfficientNetB7
]

 
for app in ARG_TO_MODEL:
  print(app.__name__)
  model = app(weights='imagenet')
  model.save(f"models/{app.__name__}/1/")

  

