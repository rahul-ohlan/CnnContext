from tensorflow.keras import models, callbacks
from keras.layers import Dense, Dropout
import numpy as np
import pandas as pd

# initialize network
model=models.Sequential()
#model.add(Dense(20,activation='sigmoid',input_shape=(20,)))
model.add(Dense(20,activation='sigmoid',input_shape=(40,)))
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

early_stopping = callbacks.EarlyStopping(
    monitor='val_prc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)


from misclass import dfKnown

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
#dfKnownOverlap=dfKnown[dfKnown['percent_overlap']>.7]
#objList=dfKnownOverlap['OBJECT (PAPER)']

#xMat=dfKnownOverlap.iloc[:,-22:-2].to_numpy()
#xMat=dfKnownOverlap.iloc[:,-21:-1].to_numpy()
#xMat=dfKnownOverlap.iloc[:,-41:-1].to_numpy()
xMat=dfKnownOverlap.iloc[:,-63:-21].to_numpy()
xMat=np.delete(xMat,20,axis=1)
xMat=np.delete(xMat,20,axis=1)

#yLabs=np.zeros((dfKnownOverlap.shape[0],20))
#
#for i in range(yLabs.shape[0]):
#   yMat[i,int(str2Num(yLabs[i]))]=1
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
#model.fit(xMat[:10000,:],trueY[:10000,:],batch_size=128,epochs=100,verbose=1)
history=model.fit(xMat[:10000,:],trueY[:10000,:],batch_size=128,epochs=300,verbose=1,shuffle=True,validation_split=0.3,class_weight=dictWt, callbacks=[early_stopping])

oldResults=model(xMat[:10000,:])
oldLabels=np.argmax(oldResults,axis=1)

newResults=model(xMat[10000:11000,:])
newLabels=np.argmax(newResults,axis=1)

yLabels=np.argmax(trueY[10000:11000,:],axis=1)


# consider trueLabs
np.where(newLabels[:1000]-yLabels[:1000]!=0)[0].shape



