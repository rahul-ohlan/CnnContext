import numpy  as np
import pandas as pd

# load Eric's baseline results (before adjusting for
# object-scene relations)
df=pd.read_csv('all_combined_table4.csv')

# load pre-computed object-place wordnet similarities
dfCocoPlace=pd.read_csv('COCO-PlacesWordnetSimilarities2_DDL.csv')

# define some variables for long label names
fcnLabel='Best FCN-Resnet101 Guess from 20 COCO Objects'

# only include objects whose ground-truth identity
# is one of the 20 FCN pre-trained COCO identities
fcnObjs=df[fcnLabel].unique().tolist()
dfKnown=df[df['OBJECT (PAPER)'].isin(fcnObjs)]

# only look at subset of data where objects were
# MISclassified by FCN-Resnet
dfMisClass=dfKnown[dfKnown['OBJECT (PAPER)']!=dfKnown[fcnLabel]]

# add a column specifying the pair of true label,
# FCN-predicted label
dfMisClass.insert(52,"ObjTruePred",(dfMisClass['OBJECT (PAPER)']+'-'+dfMisClass[fcnLabel]),True)

misclassCounts=dfMisClass['ObjTruePred'].value_counts()


# convert list of strings to include [' '] around each string
def bracketList(strListIn):
    strListOut=[]
    for word in strListIn :
        strListOut.append('[\''+word+'\']')
    return strListOut

placeLabel='Places365 Resnet50 Image Classification'

def sceneMisses(errorPair):
    return dfMisClass[dfMisClass['ObjTruePred']==errorPair][placeLabel].value_counts()

def objSceneSims(obj,sceneList):
    objWord='[\''+obj+'\']'
    sceneBrack = bracketList(sceneList)
    simList=dfCocoPlace[(dfCocoPlace['COCO Word']==objWord)&(dfCocoPlace['Places365 Label'].isin(sceneBrack))][['Places365 Label','Wordnet Similarity']]
    return simList

