# For running in python 2.x
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

from keras import backend as K
from keras.layers import Input, Dense, RepeatVector, Lambda, Multiply, Conv2D, Reshape
from keras.layers import BatchNormalization, Concatenate, AveragePooling2D, Flatten, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

def fMnistGamoGenCreate(latDim):
    noise=Input(shape=(latDim,))
    labels=Input(shape=(10,))
    gamoGenInput=Concatenate()([noise, labels])

    x=Dense(256, activation='relu')(gamoGenInput)
    x=BatchNormalization(momentum=0.9)(x)

    x=Dense(64, activation='relu')(x)
    gamoGenFinal=BatchNormalization(momentum=0.9)(x)

    gamoGen=Model([noise, labels], gamoGenFinal)
    gamoGen.summary()
    return gamoGen

def fMnistGenProcessCreate(numMinor):
    ip1=Input(shape=(64, ))
    ip2=Input(shape=(512, numMinor))

    x=Dense(numMinor, activation='softmax')(ip1)
    x=RepeatVector(512)(x)

    x=Multiply()([x, ip2])
    genProcFinal=Lambda(lambda x: K.sum(x, axis=-1))(x)

    genProcess=Model([ip1, ip2], genProcFinal)
    return genProcess

def fMnistGamoConvCreate():
    ip1=Input(shape=(28, 28, 1))

    x=Conv2D(32, kernel_size=5, padding='same')(ip1)
    x=LeakyReLU(0.1)(x)
    x=AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x=Conv2D(32, kernel_size=5, padding='same')(x)
    x=LeakyReLU(0.1)(x)
    x=AveragePooling2D(pool_size=(2, 2), strides=2, padding='same')(x)

    x=Flatten()(x)
    gamoConvFinal=Dense(512, activation='tanh')(x)

    gamoConv=Model(ip1, gamoConvFinal)
    gamoConv.summary()
    return gamoConv

def fMnistGamoDisCreate():
    imIn=Input(shape=(512,))
    labels=Input(shape=(10,))
    disInput=Concatenate()([imIn, labels])

    x=Dense(256)(disInput)
    x=LeakyReLU(alpha=0.1)(x)

    x=Dense(128)(x)
    x=LeakyReLU(alpha=0.1)(x)

    gamoDisFinal=Dense(1, activation='sigmoid')(x)

    gamoDis=Model([imIn, labels], gamoDisFinal)
    gamoDis.summary()
    return gamoDis

def fMnistGamoMlpCreate():
    ip1=Input(shape=(512,))

    x=Dense(256)(ip1)
    x=LeakyReLU(alpha=0.1)(x)

    x=Dense(128)(x)
    x=LeakyReLU(alpha=0.1)(x)

    gamoMlpFinal=Dense(10, activation='softmax')(x)

    gamoMlp=Model(ip1, gamoMlpFinal)
    gamoMlp.summary()
    return gamoMlp

