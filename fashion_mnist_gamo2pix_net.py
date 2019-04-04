# For running in python 2.x
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

from keras import backend as K
from keras.layers import Input, Dense, RepeatVector, Conv2D, Reshape, Lambda
from keras.layers import BatchNormalization, AveragePooling2D, Flatten, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model, load_model
from keras import regularizers

import numpy as np
import fashion_mnist_net as nt

def encoderCreate(convPath):

    convCopy=nt.fMnistGamoConvCreate()
    convCopy.load_weights(convPath)
    convCopy.trainable=False

    return convCopy

def sampling(args):
    z_mean, z_sigma=args
    batch_size=K.shape(z_mean)[0]
    latDim=K.int_shape(z_mean)[1]
    epsilon=K.random_normal(shape=(batch_size, latDim))
    return z_mean+K.exp(z_sigma)*epsilon

def vaeEncoderCreate(latDim):

    ip1=Input(shape=(512,))

    z_mean=Dense(latDim, name='z_mean')(ip1)
    z_sigma=Dense(latDim, name='z_sigma')(ip1)
    z=Lambda(sampling)([z_mean, z_sigma])

    vaeEncoder=Model(ip1, [z, z_mean, z_sigma])

    return vaeEncoder

def decoderCreate(latDim):

    ip1=Input(shape=(latDim,))
    x=Dense(7*7*32)(ip1)
    x=LeakyReLU(0.1)(x)
    x=Reshape((7, 7, 32))(x)

    x=Conv2DTranspose(32, kernel_size=4, strides=2, padding='same')(x)
    x=LeakyReLU(0.1)(x)

    x=Conv2DTranspose(32, kernel_size=4, strides=2, padding='same')(x)
    x=LeakyReLU(0.1)(x)

    decodeOut=Conv2D(1, kernel_size=4, strides=1, padding='same', activation='tanh')(x)

    decoder=Model(ip1, decodeOut)

    decoder.summary()

    return decoder

def gamoExtractAlphas(numMinor):
    ip1=Input(shape=(64, ))

    x=Dense(numMinor, activation='softmax')(ip1)
    op1=RepeatVector(512)(x)

    extAlphas=Model(ip1, op1)

    return extAlphas
