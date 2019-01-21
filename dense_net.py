# For running in python 2.x
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

from keras import backend as K
from keras.layers import Input, Dense, RepeatVector, Lambda
from keras.layers import BatchNormalization, Concatenate, Multiply
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

def denseGamoGenCreate(latDim):
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

def denseGenProcessCreate(numMinor, dataMinor):
    ip1=Input(shape=(64,))

    x=Dense(numMinor, activation='softmax')(ip1)
    x=RepeatVector(784)(x)
    genProcessFinal=Lambda(lambda x: K.sum(x*K.transpose(K.constant(dataMinor)), axis=2))(x)

    genProcess=Model(ip1, genProcessFinal)
    return genProcess

def denseDisCreate():
    imIn=Input(shape=(784,))
    labels=Input(shape=(10,))
    disInput=Concatenate()([imIn, labels])

    x=Dense(256)(disInput)
    x=LeakyReLU(alpha=0.1)(x)

    x=Dense(128)(x)
    x=LeakyReLU(alpha=0.1)(x)

    disFinal=Dense(1, activation='sigmoid')(x)

    dis=Model([imIn, labels], disFinal)
    dis.summary()
    return dis

def denseMlpCreate():

    imIn=Input(shape=(784,))

    x=Dense(256)(imIn)
    x=LeakyReLU(alpha=0.1)(x)

    x=Dense(128)(x)
    x=LeakyReLU(alpha=0.1)(x)

    mlpFinal=Dense(10, activation='softmax')(x)

    mlp=Model(imIn, mlpFinal)
    mlp.summary()
    return mlp

