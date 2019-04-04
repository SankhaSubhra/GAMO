# For running in python 2.x
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division

import os
import numpy as np
import dense_suppli as spp
import dense_net as nt
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical

# For selecting a GPU
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Ground works
fileName=['Mnist_100_trainData.csv', 'Mnist_100_testData.csv']
fileStart='Mnist_100_Gamo'
fileEnd, savePath='_Model.h5', fileStart+'/'
adamOpt=Adam(0.0002, 0.5)
latDim, modelSamplePd, resSamplePd=100, 5000, 500
plt.ion()

batchSize, max_step=32, 50000

trainS, labelTr=spp.fileRead(fileName[0])
testS, labelTs=spp.fileRead(fileName[1])

n, m=trainS.shape[0], testS.shape[0]
trainS, testS=(trainS-127.5)/127.5, (testS-127.5)/127.5

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

# model initialization
mlp=nt.denseMlpCreate()
mlp.compile(loss='mean_squared_error', optimizer=adamOpt)
mlp.trainable=False

dis=nt.denseDisCreate()
dis.compile(loss='mean_squared_error', optimizer=adamOpt)
dis.trainable=False

gen=nt.denseGamoGenCreate(latDim)

gen_processed, genP_mlp, genP_dis=list(), list(), list()
for i in range(imbClsNum):
    dataMinor=trainS[classMap[i], :]
    numMinor=dataMinor.shape[0]
    gen_processed.append(nt.denseGenProcessCreate(numMinor, dataMinor))

    ip1=Input(shape=(latDim,))
    ip2=Input(shape=(c,))
    op1=gen([ip1, ip2])
    op2=gen_processed[i](op1)
    op3=mlp(op2)
    genP_mlp.append(Model(inputs=[ip1, ip2], outputs=op3))
    genP_mlp[i].compile(loss='mean_squared_error', optimizer=adamOpt)

    ip1=Input(shape=(latDim,))
    ip2=Input(shape=(c,))
    ip3=Input(shape=(c,))
    op1=gen([ip1, ip2])
    op2=gen_processed[i](op1)
    op3=dis([op2, ip3])
    genP_dis.append(Model(inputs=[ip1, ip2, ip3], outputs=op3))
    genP_dis[i].compile(loss='mean_squared_error', optimizer=adamOpt)

# for record saving
batchDiv, numBatches, bSStore=spp.batchDivision(n, batchSize)
genClassPoints=int(np.ceil(batchSize/c))
fig, axs=plt.subplots(imbClsNum, 3)

if not os.path.exists(fileStart):
    os.makedirs(fileStart)
picPath=savePath+'Pictures'
if not os.path.exists(picPath):
    os.makedirs(picPath)

iter=np.int(np.ceil(max_step/resSamplePd)+1)
acsaSaveTr, gmSaveTr, accSaveTr=np.zeros((iter,)), np.zeros((iter,)), np.zeros((iter,))
acsaSaveTs, gmSaveTs, accSaveTs=np.zeros((iter,)), np.zeros((iter,)), np.zeros((iter,))
confMatSaveTr, confMatSaveTs=np.zeros((iter, c, c)), np.zeros((iter, c, c))
tprSaveTr, tprSaveTs=np.zeros((iter, c)), np.zeros((iter, c))

# training
step=0
while step<max_step:
    for j in range(numBatches):
        x1, x2=batchDiv[j, 0], batchDiv[j+1, 0]
        validR=np.ones((bSStore[j, 0],1))-np.random.uniform(0,0.1, size=(bSStore[j, 0], 1))
        mlp.train_on_batch(trainS[x1:x2], labelsCat[x1:x2])
        dis.train_on_batch([trainS[x1:x2], labelsCat[x1:x2]], validR)

        invalid=np.zeros((bSStore[j, 0], 1))+np.random.uniform(0, 0.1, size=(bSStore[j, 0], 1))
        randNoise=np.random.normal(0, 1, (bSStore[j, 0], latDim))
        fakeLabel=spp.randomLabelGen(toBalance, bSStore[j, 0], c)
        rLPerClass=spp.rearrange(fakeLabel, imbClsNum)
        fakePoints=np.zeros((bSStore[j, 0], 784))
        genFinal=gen.predict([randNoise, fakeLabel])
        for i1 in range(imbClsNum):
            if rLPerClass[i1].shape[0]!=0:
                temp=genFinal[rLPerClass[i1]]
                fakePoints[rLPerClass[i1]]=gen_processed[i1].predict(temp)

        mlp.train_on_batch(fakePoints, fakeLabel)
        dis.train_on_batch([fakePoints, fakeLabel], invalid)

        for i1 in range(imbClsNum):
            validA=np.ones((genClassPoints, 1))
            randomLabel=np.zeros((genClassPoints, c))
            randomLabel[:, i1]=1
            randNoise=np.random.normal(0, 1, (genClassPoints, latDim))
            oppositeLabel=np.ones((genClassPoints, c))-randomLabel
            genP_mlp[i1].train_on_batch([randNoise, randomLabel], oppositeLabel)
            genP_dis[i1].train_on_batch([randNoise, randomLabel, randomLabel], validA)

        if step%resSamplePd==0:
            saveStep=int(step//resSamplePd)

            pLabel=np.argmax(mlp.predict(trainS), axis=1)
            acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTr)
            print('Train: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
            print('TPR: ', np.round(tpr, 2))
            acsaSaveTr[saveStep], gmSaveTr[saveStep], accSaveTr[saveStep]=acsa, gm, acc
            confMatSaveTr[saveStep]=confMat
            tprSaveTr[saveStep]=tpr

            pLabel=np.argmax(mlp.predict(testS), axis=1)
            acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTs)
            print('Test: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
            print('TPR: ', np.round(tpr, 2))
            acsaSaveTs[saveStep], gmSaveTs[saveStep], accSaveTs[saveStep]=acsa, gm, acc
            confMatSaveTs[saveStep]=confMat
            tprSaveTs[saveStep]=tpr

            for i1 in range(imbClsNum):
                testNoise=np.random.normal(0, 1, (3, latDim))
                testLabel=np.zeros((3, c))
                testLabel[:, i1]=1
                genFinal=gen.predict([testNoise, testLabel])
                genImages=gen_processed[i1].predict(genFinal)
                genImages=np.reshape(genImages, (3, 28, 28))
                for i2 in range(3):
                    img=image.array_to_img(np.expand_dims(genImages[i2], axis=-1), scale=True)
                    axs[i1,i2].imshow(img, cmap='gray')
                    axs[i1,i2].axis('off')
            plt.show()
            plt.pause(5)

            figFileName=picPath+'/'+fileStart+'_'+str(step)+'.png'
            plt.savefig(figFileName, bbox_inches='tight')

        if step%modelSamplePd==0 and step!=0:
            direcPath=savePath+'gamo_models_'+str(step)
            if not os.path.exists(direcPath):
                os.makedirs(direcPath)
            gen.save(direcPath+'/GEN_'+str(step)+fileEnd)
            mlp.save(direcPath+'/MLP_'+str(step)+fileEnd)
            dis.save(direcPath+'/DIS_'+str(step)+fileEnd)
            for i in range(imbClsNum):
                gen_processed[i].save(direcPath+'/GenP_'+str(i)+'_'+str(step)+fileEnd)

        step=step+2
        if step>=max_step: break

figFileName=picPath+'/'+fileStart+'_'+str(step)+'.png'
plt.savefig(figFileName, bbox_inches='tight')

pLabel=np.argmax(mlp.predict(trainS), axis=1)
acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTr)
print('Performance on Train Set: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
print('TPR: ', np.round(tpr, 2))
acsaSaveTr[-1], gmSaveTr[-1], accSaveTr[-1]=acsa, gm, acc
confMatSaveTr[-1]=confMat
tprSaveTr[-1]=tpr

pLabel=np.argmax(mlp.predict(testS), axis=1)
acsa, gm, tpr, confMat, acc=spp.indices(pLabel, labelTs)
print('Performance on Test Set: Step: ', step, 'ACSA: ', np.round(acsa, 4), 'GM: ', np.round(gm, 4))
print('TPR: ', np.round(tpr, 2))
acsaSaveTs[-1], gmSaveTs[-1], accSaveTs[-1]=acsa, gm, acc
confMatSaveTs[-1]=confMat
tprSaveTs[-1]=tpr

direcPath=savePath+'gamo_models_'+str(step)
if not os.path.exists(direcPath):
    os.makedirs(direcPath)
gen.save(direcPath+'/GEN_'+str(step)+fileEnd)
mlp.save(direcPath+'/MLP_'+str(step)+fileEnd)
dis.save(direcPath+'/DIS_'+str(step)+fileEnd)
for i in range(imbClsNum):
    gen_processed[i].save(direcPath+'/GenP_'+str(i)+'_'+str(step)+fileEnd)

resSave=savePath+'Results'
np.savez(resSave, acsa=acsa, gm=gm, tpr=tpr, confMat=confMat, acc=acc)
recordSave=savePath+'Record'
np.savez(recordSave, acsaSaveTr=acsaSaveTr, gmSaveTr=gmSaveTr, accSaveTr=accSaveTr, acsaSaveTs=acsaSaveTs, gmSaveTs=gmSaveTs, accSaveTs=accSaveTs, confMatSaveTr=confMatSaveTr, confMatSaveTs=confMatSaveTs, tprSaveTr=tprSaveTr, tprSaveTs=tprSaveTs)
