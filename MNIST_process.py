# Can be used for both mnist and fashion mnist datasets.
import numpy as np
import pickle as pk

# For mnist dataset
trainDataOri=np.loadtxt('mnist_train.csv', delimiter= ',')
testDataOri=np.loadtxt('mnist_test.csv', delimiter= ',')

trainSetOri, trainLabOri=trainDataOri[:, 1:], trainDataOri[:, 0]
testSetOri, testLabOri=testDataOri[:, 1:], testDataOri[:, 0]

pointsInTrClass=((4000, 2000, 1000, 750, 500, 350, 200, 100, 60, 40))

numClass=10
pointsInTsClass=100
maxPTrClass, maxPTsClass=4000, 800

classLocTr=np.insert(np.cumsum(pointsInTrClass), 0, 0)
classMapTr, classMapTs, trainPoints, testPoints=list(), list(), list(), list()
for i in range(numClass):
    classMapTr.append(np.where(trainLabOri==i)[0])
    classMapTs.append(np.where(testLabOri==i)[0])
trainS=np.zeros((np.sum(pointsInTrClass), trainSetOri.shape[1]))
trainL=np.zeros((np.sum(pointsInTrClass),1))

for i in range(numClass):
    randIdxTr=np.random.randint(0, maxPTrClass, pointsInTrClass[i])
    trainPoints.append(classMapTr[i][randIdxTr])
    trainS[classLocTr[i]:classLocTr[i+1], :]=trainSetOri[trainPoints[i], :]
    trainL[classLocTr[i]:classLocTr[i+1], 0]=trainLabOri[trainPoints[i]]
trainDataFinal=np.hstack((trainS, trainL))

testS=np.zeros((int(numClass*pointsInTsClass), testSetOri.shape[1]))
testL=np.zeros((int(numClass*pointsInTsClass),1))
classLocTs=np.arange(0, (numClass+1)*pointsInTsClass, pointsInTsClass)
for i in range(numClass):
    randIdxTs=np.random.randint(0, maxPTsClass, pointsInTsClass)
    testPoints.append(classMapTs[i][randIdxTs])
    testS[classLocTs[i]:classLocTs[i+1], :]=testSetOri[testPoints[i], :]
    testL[classLocTs[i]:classLocTs[i+1], 0]=testLabOri[testPoints[i]]
testDataFinal=np.hstack((testS, testL))

sampledPoints={'Mnist_100_trainSamples':trainPoints, 'Mnist_100_testSamples':testPoints}
pk.dump(sampledPoints, open( 'Mnist_100_sampledPoints.pkl', 'wb' ))
np.savetxt('Mnist_100_trainData.csv', trainDataFinal, delimiter=",")
np.savetxt('Mnist_100_testData.csv', testDataFinal, delimiter=",")


