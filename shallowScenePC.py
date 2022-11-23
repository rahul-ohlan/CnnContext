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
#dfKnownOverlap=pd.read_csv('LogScene+Cols.csv')
dfKnownOverlap=pd.read_csv('LogRewtEK_Oct4.csv')
#dfKnownOverlap=pd.read_csv('rebalLogScene+Cols.csv')

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

# generate indices to extract train and test data
def trainTestInds(dataSize,kSplits,numSplit):
   trainInd=list(range(dataSize))
   splitSize=int(np.round(dataSize/kSplits))
   testInd =list(range(splitSize*numSplit,splitSize*(numSplit+1)))
   for num in testInd:
     trainInd=np.delete(trainInd,np.where(trainInd==num)[0])
   return trainInd, testInd

# drop columns where no object-scene info is available
dfScenes=dfKnownOverlap['Places365 Resnet50 Image Classification']
dfKnownOverlap=dfKnownOverlap[dfKnownOverlap.iloc[:,-41:-21].sum(axis=1)!=0]
dfPCA=pd.read_csv('ScenePrincComp.csv',index_col=0)

# initialize xMat and yMat
# in xMat:
#   first  20 columns are the FCN score for each object label
#   second 20 columns are the object-scene relation (based on wordnet)

xMat=np.zeros((dfKnownOverlap.shape[0],20))
xMat[:,:20]=dfKnownOverlap.iloc[:,-63:-43].to_numpy()

rInd=0
for row in dfScenes:
  try:
    xMat[rInd,20:]=dfPCA[dfPCA.index==row].to_numpy()
  except:
    print('errors '+row)
  rInd+=1

dfCols=dfKnownOverlap.columns[-63:-43]


xMat[:,:10]=np.log10(xMat[:,:10]+3)

trueLabs=np.zeros(dfKnownOverlap.shape[0])

for i in range(trueLabs.shape[0]):
   trueLabs[i]=str2Num(dfKnownOverlap['OBJECT (PAPER)'].iloc[i])

permScenes=True
# rand permute scenes
if permScenes:
  picNums=dfKnownOverlap['image_id_x'].to_numpy()
  picNumsUnique=np.unique(picNums)
  indPerm=np.random.permutation(picNumsUnique.shape[0])
  picNumsPerm=picNumsUnique[indPerm]
  xMatCopy=xMat.copy()
  trueLabsCopy=trueLabs.copy()
  k=0
  for sceneInd in range(len(picNumsPerm)):
    currPicRows=np.where(picNums==picNumsPerm[sceneInd])[0]
    xMatCopy[list(range(k,k+len(currPicRows))),:]=xMat[currPicRows,:]
    trueLabsCopy[list(range(k,k+len(currPicRows)))]=trueLabs[currPicRows]
    k=k+len(currPicRows)
    #print([sceneInd,k])
  
  xMat=xMatCopy
  trueLabs=trueLabsCopy

trueY = np.zeros((trueLabs.shape[0],20))

for i in range(trueLabs.shape[0]):
   trueY[i,int(trueLabs[i])]=1

# weight classes based on frequency
wtVec=np.array([186,943,7375,203,240,223,1176,212,150,98,353,175,538,250,121,203,162,249,184,270])
total=np.sum(wtVec)
dictWt={}
for objInd in range(len(wtVec)):
    dictWt[objInd]=1/wtVec[objInd] * total/2.0

trainInd, testInd = trainTestInds(xMat.shape[0],10,3)


initLearning(learnRate=.1,rebalance=True)

global model
#print('here is the use model:')
#print(model)

# or consider trueY instead of trueMat
#model.fit(xMat[:10000,:],trueY[:10000,:],batch_size=128,epochs=100,verbose=1):s
[histTrain,histVal]=fit(model,xMat[trainInd,:],trueLabs[trainInd],epochs=4000,shuffle=False,valRat=.75,patience=30) #,mustPrune=True,smartInit=True)
#[histTrain,histVal]=fit(model,xMat[:10000,:],trueLabs[:10000],batch_size=128,epochs=800,shuffle=True,patience=10),mustPrune=True,smartInit=True)
# add smartInit above
#batch_size=128,epochs=300,verbose=1,shuffle=True,validation_split=0.3,class_weight=dictWt, callbacks=[early_stopping])
#oldResults=model(xMat[:10000,:])
#oldLabels=np.argmax(oldResults,axis=1)

newResults=fwdPass(torch.Tensor(xMat[testInd,:])).detach().numpy()
newLabels=np.argmax(newResults,axis=1)

yLabels=np.argmax(trueY[testInd,:],axis=1)


# consider trueLabs
accuracy=1-np.where(newLabels-yLabels!=0)[0].shape[0]/yLabels.shape[0]

