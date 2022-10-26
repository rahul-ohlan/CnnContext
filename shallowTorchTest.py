import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import numpy as np
from sklearn.utils import resample

# possible prunings:
# prune.random_unstructured(model.fc1,name="weight",amount=.3)
# prune.ln_structured(model.fc1,name="weight",amount=.5,n=2,dim=1)
# prune.ln_unstructured(model.fc1,name="weight",amount=.5)

device = torch.device("cpu")


# pruning guidance from: https://pytorch.org/tutorials/intermediate/pruning_tutorial.html

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(40,20)

    def forward(self, x):
        x = self.fc1(x)
        return x

model=1
criterion=2
optimizer=3
rebalanceBool=False

def initLearning(learnRate=1, reweight=False, rebalance=False):
  global model
  model = MyNet().to(device=device)
  
  global criterion
  #criterion = nn.CrossEntropyLoss()
  wtVec=np.array([186,943,7375,203,240,223,1176,212,150,98,353,175,538,250,121,203,162,249,184,270])
  total=np.sum(wtVec)
  if reweight :
    criterion = nn.CrossEntropyLoss(weight=torch.Tensor(0.5*total/np.array(wtVec)))
  else:
    criterion = nn.CrossEntropyLoss()
  
  global rebalanceBool
  rebalanceBool = rebalance
  
  global optimizer
  optimizer = torch.optim.SGD(model.parameters(), lr = learnRate) #lr=1 was good unweighted # lr was 0.01
  
  print('model and optimizer initialized')


# seems like better results with lr=1, valRat=.5, shuffle=False, epochs=4000
# results are best on training data and validation data, quite poor on 
# remaining testing data (10000:13000); performance even better on validation
# if allow shuffle, but that mixes scenes between train and validation
# learned weights make a LOT of sense for FCN score, pretty much ignore
# scene (except a little bit human and plant ... odd); some negative
# FCN effects too (which I kinda predicted) and different weights for
# different objects
#
# note: no normalization before learning, mean scene relat values are 
# half the magnitude of mean fcn score vals
# results much stranger when train on later data and test on earlier 
# data

# assuming following row labels:
#['tv', 'chair', 'person', 'potted plant', 'boat', 'bird', 'car',
#       'bus', 'cat', 'airplane', 'dining table', 'couch', 'bottle',
#       'sheep', 'train', 'horse', 'dog', 'motorcycle', 'bicycle', 'cow']

# and following col labels:
# ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
#      'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person',
#       'potted plant', 'sheep', 'sofa', 'train', 'tvmonitor', 'Max Value',
#       'Max Column', 'personScRelate', 'plantScRelate', 'chairScRelate',
#       'tvScRelate', 'dining_tableScRelate', 'birdScRelate', 'carScRelate',
#       'catScRelate', 'busScRelate', 'airplaneScRelate', 'couchScRelate',
#       'bottleScRelate', 'sheepScRelate', 'cowScRelate', 'trainScRelate',
#       'horseScRelate', 'dogScRelate', 'motorcycleScRelate', 'bicycleScRelate',
#       'boatScRelate'],

def upsample(feats,labs):
   #classCount=df_LS['OBJECT (PAPER)'].value_counts()
   classCount=np.histogram(labs,len(np.unique(labs)))[0]
   maxClass=classCount.argmax()
   boostNum=np.zeros(classCount.shape[0])
   subClassDf={}
   subClassDf_upsamp={}
   featsList=feats.T.tolist()
   featsList.append(labs.tolist())
   feats=np.array(featsList).T
   for keyNum in range(len(classCount)):
     #boostNum[key]=classCount['person']/classCount[key]
     boostNum[keyNum]=classCount[maxClass]/classCount[keyNum]
     subClassDf[keyNum]=feats[np.where(labs==keyNum)[0],:]
     subClassDf_upsamp[keyNum]=resample(subClassDf[keyNum],n_samples=int(np.round(classCount[keyNum]*boostNum[keyNum])),replace=True)
   allUpSamp=subClassDf_upsamp[0]
   for keyNum in range(1, len(classCount)):
     allUpSamp=np.hstack((allUpSamp.T,subClassDf_upsamp[keyNum].T)).T
   
   return allUpSamp[:,:-1], allUpSamp[:,-1]


def fit(net,feats,labs,batch_size=np.Inf, epochs=20, shuffle=True, valRat=.8,patience=5,mustPrune=False,smartInit=False):
    ##train = datasets.MNIST('', train = True, transform = transforms, download = True)
    #train, valid = random_split(train,[50000,10000])
    #trainloader = DataLoader(train, batch_size=32)
    #validloader = DataLoader(valid, batch_size=32)
    #print('here is our model:')
    #print(model)
    if shuffle:
        permVals=np.random.permutation(feats.shape[0])
        feats=feats[permVals,:]
        labs=labs[permVals]
    loss_hist=[]
    lossVal_hist=[]
    valPartit=int(np.floor(valRat*feats.shape[0]))
    featsTrain=feats[:valPartit,:]
    labsTrain=labs[:valPartit]
    featsVal=feats[valPartit:,:]
    labsVal=labs[valPartit:]
    if rebalanceBool: # up-sample under-represented data
      featsTrain,labsTrain = upsample(featsTrain, labsTrain)
      featsVal, labsVal    = upsample(featsVal,   labsVal)
    
    dataFull=featsTrain
    labelsFull=labsTrain
    if smartInit:
        model.fc1.weight.data.zero_()
        model.fc1.weight.data[0,19]=.1
        model.fc1.weight.data[0,23]=.1
        model.fc1.weight.data[1,8]=.1
        model.fc1.weight.data[1,22]=.1
        model.fc1.weight.data[2,14]=.1
        model.fc1.weight.data[2,20]=.1
        model.fc1.weight.data[3,15]=.1
        model.fc1.weight.data[3,21]=.1
        model.fc1.weight.data[4,3]=.1
        model.fc1.weight.data[4,39]=.1
        model.fc1.weight.data[5,2]=.1
        model.fc1.weight.data[5,25]=.1
        model.fc1.weight.data[6,6]=.1
        model.fc1.weight.data[6,26]=.1
        model.fc1.weight.data[7,5]=.1
        model.fc1.weight.data[7,28]=.1
        model.fc1.weight.data[8,7]=.1
        model.fc1.weight.data[8,27]=.1
        model.fc1.weight.data[9,0]=.1
        model.fc1.weight.data[9,29]=.1
        model.fc1.weight.data[10,10]=.1
        model.fc1.weight.data[10,24]=.1
        model.fc1.weight.data[11,17]=.1
        model.fc1.weight.data[11,30]=.1
        model.fc1.weight.data[12,4]=.1
        model.fc1.weight.data[12,31]=.1
        model.fc1.weight.data[13,16]=.1
        model.fc1.weight.data[13,32]=.1
        model.fc1.weight.data[14,18]=.1
        model.fc1.weight.data[14,34]=.1
        model.fc1.weight.data[15,12]=.1
        model.fc1.weight.data[15,35]=.1
        model.fc1.weight.data[16,11]=.1
        model.fc1.weight.data[16,36]=.1
        model.fc1.weight.data[17,13]=.1
        model.fc1.weight.data[17,37]=.1
        model.fc1.weight.data[18,1]=.1
        model.fc1.weight.data[18,38]=.1
        model.fc1.weight.data[19,9]=.1
        model.fc1.weight.data[19,33]=.1
    if batch_size<feats.shape[0]:
        totalBatches=np.floor(dataFull.shape[0]/batch_size)
        batchInd=0
    for e in range(epochs):
        train_loss = 0.0
        #for data, labels in tqdm(trainloader):
        # Clear the gradients
        optimizer.zero_grad()
        if batch_size>=feats.shape[0]:
            data=dataFull
            labels=labelsFull
        else:
            data=dataFull[batchInd*batch_size:(batchInd*(batch_size+1)-1),:]
            labels=labelsFull[batchInd*batch_size:(batchInd*(batch_size+1)-1)]
        # Forward Pass
        target = model(torch.Tensor(data))
        targetVal = model(torch.Tensor(featsVal))
        # Find the Loss
        loss = criterion(target,torch.LongTensor(labels))
        lossVal = criterion(targetVal,torch.LongTensor(labsVal))
        loss_hist.append(loss.item())
        lossVal_hist.append(lossVal.item())
        if len(lossVal_hist)>patience:
            if np.min(lossVal_hist[-patience:])>lossVal_hist[-(patience+1)]:
                return loss_hist, lossVal_hist
        # Calculate gradients 
        loss.backward()
        # Update Weights
        optimizer.step()
        # Add a pruning section
        if mustPrune:
            prune.ln_unstructured(model.fc1,name="weight",amount=.5,n=2,dim=1)
        # Calculate Loss
        train_loss += loss.item()
        if batch_size<feats.shape[0]:
            batchInd+=1
            if batchInd==totalBatches:
                batchInd=0
    return loss_hist, lossVal_hist
#        valid_loss = 0.0
#        model.eval()     # Optional when not using Model Specific layer
#        for data, labels in validloader:
#            target = model(data)
#            loss = criterion(target,labels)
#            valid_loss = loss.item() * data.size(0)
#
#        print(f'Epoch {e+1} \t\t Training Loss: {train_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(validloader)}')
#        if min_valid_loss > valid_loss:
#            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
#            min_valid_loss = valid_loss
#            # Saving State Dict
#            torch.save(model.state_dict(), 'saved_model.pth')

def fwdPass(inTensor):
  return model(inTensor)

def get_model() :
  return model
