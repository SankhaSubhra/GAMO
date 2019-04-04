# For running in python 2.x
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

import os
import numpy as np
import matplotlib.pyplot as plt
import fashion_mnist_gamo2pix_net as nt

import fashion_mnist_net as ntv
import fashion_mnist_suppli as spp

import keras.backend as K
from keras.layers import Input
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.losses import mean_squared_error

def vae_loss(y_true, y_pred):
    mse_loss=28*28*mean_squared_error(K.flatten(y_true), K.flatten(y_pred))
    kl_loss=-0.5*K.sum(1+z_sigma-K.square(z_mean)-K.exp(z_sigma), axis=-1)
    return K.mean(mse_loss+kl_loss)

# For selecting a GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

fileName=['fMnist_100_trainData.csv', 'fMnist_100_testData.csv']
folderStart='fMnist_100_Gamo/gamo_models_50000/'
imgFolderStart='fMnist_100_ImgGen'

genPath=folderStart+'GEN_50000_Model.h5'
genPPath=['GenP_0_50000_Model.h5', 'GenP_1_50000_Model.h5', 'GenP_2_50000_Model.h5', 'GenP_3_50000_Model.h5', 'GenP_4_50000_Model.h5', 'GenP_5_50000_Model.h5', 'GenP_6_50000_Model.h5', 'GenP_7_50000_Model.h5', 'GenP_8_50000_Model.h5', 'GenP_9_50000_Model.h5']
convPath=folderStart+'Conv_50000_Model.h5'

fileEnd, savePath='_Model.h5', imgFolderStart+'/'

plt.ion()
adamOpt=Adam(0.0002, 0.5)
latDim, modelSamplePd, resSamplePd=100, 2000, 500

batchSize, max_step=32, 25000

trainS, labelTr=spp.fileRead(fileName[0])
testS, labelTs=spp.fileRead(fileName[1])

n, m=trainS.shape[0], testS.shape[0]
trainS, testS=(trainS-127.5)/127.5, (testS-127.5)/127.5
trainS, testS=np.reshape(trainS, (n, 28, 28, 1)), np.reshape(testS, (m, 28, 28, 1))

labelTr, labelTs, c, pInClass, _=spp.relabel(labelTr, labelTs)
imbalancedCls, toBalance, imbClsNum, ir=spp.irFind(pInClass, c)

labelsCat=to_categorical(labelTr)

shuffleIndex=np.random.choice(np.arange(n), size=(n,), replace=False)
trainS=trainS[shuffleIndex]
labelTr=labelTr[shuffleIndex]
labelsCat=labelsCat[shuffleIndex]
classMap=list()
for i in range(c):
    classMap.append(np.where(labelTr==i)[0])

if not os.path.exists(imgFolderStart):
    os.makedirs(imgFolderStart)

for i in range(imbClsNum):

    encoder=nt.encoderCreate(convPath)
    encoder.trainable=False
    vaeEncoder=nt.vaeEncoderCreate(latDim)
    decoder=nt.decoderCreate(latDim)

    ip1=Input(shape=(28, 28, 1))
    op1=encoder(ip1)
    [op2, z_mean, z_sigma]=vaeEncoder(op1)
    op3=decoder(op2)
    autoencoder=Model(inputs=ip1, outputs=op3)
    autoencoder.compile(loss=vae_loss, optimizer=adamOpt)

    dataMinor=trainS[classMap[i], :]
    dataMinorFt=encoder.predict(dataMinor)
    numMinor=dataMinorFt.shape[0]
    tempData=np.copy(np.expand_dims(np.transpose(dataMinorFt), axis=0))

    gamoGen=load_model(genPath)
    gamoGenP=ntv.fMnistGenProcessCreate(numMinor)
    gamoGenP.load_weights(folderStart+genPPath[i])
    ea=nt.gamoExtractAlphas(numMinor)
    ea.set_weights(gamoGenP.get_weights())

    batchDiv, numBatches, bSStore=spp.batchDivision(numMinor, batchSize)
    fig1, axs1=plt.subplots(3, 2)
    fig2, axs2=plt.subplots(3, 3)

    picPath=savePath+'Pictures_'+str(i)
    if not os.path.exists(picPath):
        os.makedirs(picPath)

    direcPath=savePath+'models_'+str(i)
    if not os.path.exists(direcPath):
        os.makedirs(direcPath)

    step=0
    while step<max_step:
        for j in range(numBatches):

            repData=np.repeat(tempData, bSStore[j, 0], axis=0)

            x1, x2=batchDiv[j, 0], batchDiv[j+1, 0]
            autoencoder.train_on_batch(dataMinor[x1:x2], dataMinor[x1:x2])

            if step%resSamplePd==0:
                randInput=np.random.choice(numMinor, 3, replace=False)
                genImage=autoencoder.predict(dataMinor[randInput])
                for i1 in range(3):
                    realImageShow=image.array_to_img(dataMinor[randInput[i1]], scale=True)
                    genImageShow=image.array_to_img(genImage[i1], scale=True)
                    axs1[i1, 0].imshow(realImageShow)
                    axs1[i1, 1].imshow(genImageShow)
                    axs1[i1, 0].axis('off')
                    axs1[i1, 1].axis('off')
                plt.show()
                plt.pause(5)

                print('Train_Class: ', i, 'Step: ', step, ' completed')
                figFileName=picPath+'/Train_'+str(step)+'.png'
                fig1.savefig(figFileName, bbox_inches='tight')

                testNoise=np.random.normal(0, 1, (9, latDim))
                testLabel=np.zeros((9, c))
                testLabel[:, i]=1
                alphas=ea.predict(gamoGen.predict([testNoise, testLabel]))
                repData=np.repeat(tempData, 9, axis=0)
                gamoGenPData=np.sum(alphas*repData, axis=-1)
                [encoded, t1, t2]=vaeEncoder.predict(gamoGenPData)
                genImages=decoder.predict(encoded)
                for i1 in range(3):
                    for i2 in range(3):
                        img=image.array_to_img(genImages[(i1*3)+i2], scale=True)
                        axs2[i1,i2].imshow(img)
                        axs2[i1,i2].axis('off')
                plt.show()
                plt.pause(5)

                print('Test_Class: ', i, 'Step: ', step, ' completed')
                figFileName=picPath+'/Test_'+str(step)+'.png'
                fig2.savefig(figFileName, bbox_inches='tight')

            if step%modelSamplePd==0 and step!=0:
                vaeEncoder.save(direcPath+'/vaeEncoder_'+str(step)+fileEnd)
                decoder.save(direcPath+'/Decoder_'+str(step)+fileEnd)

            step=step+1
            if step>=max_step: break

    randInput=np.random.choice(numMinor, 3, replace=False)
    genImage=autoencoder.predict(dataMinor[randInput])
    for i1 in range(3):
        realImageShow=image.array_to_img(dataMinor[randInput[i1]], scale=True)
        genImageShow=image.array_to_img(genImage[i1], scale=True)
        axs1[i1, 0].imshow(realImageShow)
        axs1[i1, 1].imshow(genImageShow)
        axs1[i1, 0].axis('off')
        axs1[i1, 1].axis('off')
    plt.show()
    plt.pause(5)

    print('Train_Class: ', i, 'Step: ', step, ' completed')
    figFileName=picPath+'/Train_'+str(step)+'.png'
    fig1.savefig(figFileName, bbox_inches='tight')

    testNoise=np.random.normal(0, 1, (9, latDim))
    testLabel=np.zeros((9, c))
    testLabel[:, i]=1
    alphas=ea.predict(gamoGen.predict([testNoise, testLabel]))
    repData=np.repeat(tempData, 9, axis=0)
    gamoGenPData=np.sum(alphas*repData, axis=-1)
    [encoded, t1, t2]=vaeEncoder.predict(gamoGenPData)
    genImages=decoder.predict(encoded)
    for i1 in range(3):
        for i2 in range(3):
            img=image.array_to_img(genImages[(i1*3)+i2], scale=True)
            axs2[i1,i2].imshow(img)
            axs2[i1,i2].axis('off')
    plt.show()
    plt.pause(5)

    print('Test_Class: ', i, 'Step: ', step, ' completed')
    figFileName=picPath+'/Test_'+str(step)+'.png'
    fig2.savefig(figFileName, bbox_inches='tight')

    vaeEncoder.save(direcPath+'/vaeEncoder_'+str(step)+fileEnd)
    decoder.save(direcPath+'/Decoder_'+str(step)+fileEnd)


