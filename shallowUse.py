from shallowTorchTest import *
import numpy as np
import pandas as pd

#early_stopping = callbacks.EarlyStopping(
#    monitor='val_prc', 
#    verbose=1,
#    patience=10,
#    mode='max',
#    restore_best_weights=True)


# load data from LogScene+Cols.csv
dfKnownOverlap=pd.read_csv('LogScene+Cols.csv')

# initialize yMatNew

# possible values:


# convert string to numeric code
def str2Num(inStr) :
   possibleVals=['tv', 'chair', 'person', 'potted plant', 'boat', 'bird', 'car',
       'bus', 'cat', 'airplane', 'dining table', 'couch', 'bottle',
       'sheep', 'train', 'horse', 'dog', 'motorcycle', 'bicycle', 'cow']
   for i in range(len(possibleVals)):
        if inStr==possibleVals[i]:
            return i

# initialize xMat and yMat
# in xMat:
#   first  20 columns are the FCN score for each object label
#   second 20 columns are the object-scene relation (based on wordnet)
xMat=dfKnownOverlap.iloc[:,-63:-21].to_numpy()
xMat=np.delete(xMat,20,axis=1)
xMat=np.delete(xMat,20,axis=1)

xMat[:,:20]=np.log10(xMat[:,:20]+3)

trueLabs=np.zeros(dfKnownOverlap.shape[0])

for i in range(trueLabs.shape[0]):
   trueLabs[i]=str2Num(dfKnownOverlap['OBJECT (PAPER)'].iloc[i])

trueY = np.zeros((trueLabs.shape[0],20))

for i in range(trueLabs.shape[0]):
   trueY[i,int(trueLabs[i])]=1

# weight classes based on frequency
wtVec=np.array([186,943,7375,203,240,223,1176,212,150,98,353,175,538,250,121,203,162,249,184,270])
total=np.sum(wtVec)
dictWt={}
for objInd in range(len(wtVec)):
    dictWt[objInd]=1/wtVec[objInd] * total/2.0

# or consider trueY instead of trueMat
#model.fit(xMat[:10000,:],trueY[:10000,:],batch_size=128,epochs=100,verbose=1):s
[histTrain,histVal]=fit(model,xMat[3000:13000,:],trueLabs[3000:13000],epochs=4000,shuffle=False,valRat=.5,patience=10) #,mustPrune=True,smartInit=True)
#[histTrain,histVal]=fit(model,xMat[:10000,:],trueLabs[:10000],batch_size=128,epochs=800,shuffle=True,patience=10),mustPrune=True,smartInit=True)
# add smartInit above
#batch_size=128,epochs=300,verbose=1,shuffle=True,validation_split=0.3,class_weight=dictWt, callbacks=[early_stopping])

#oldResults=model(xMat[:10000,:])
#oldLabels=np.argmax(oldResults,axis=1)

newResults=model(torch.Tensor(xMat[:3000,:])).detach().numpy()
newLabels=np.argmax(newResults,axis=1)

yLabels=np.argmax(trueY[:3000,:],axis=1)


# consider trueLabs
np.where(newLabels[:1000]-yLabels[:1000]!=0)[0].shape



