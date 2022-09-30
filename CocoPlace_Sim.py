import pandas as pd
import numpy as np

pathStr='/u/erdos/cnslab/'

table = pd.read_csv(pathStr+'YOLO-COCO-Imagenet-Places365.csv', encoding='utf-8', engine='python')
table['Coco Object']

coco_categories_subsample = [
    'airplane', 
    'bicycle', 
    'bird', 
    'boat', 
    'bottle', 
    'bus', 
    'car', 
    'cat', 
    'chair', 
    'cow', 
    'dining table', 
    'dog', 
    'horse', 
    'motorcycle', 
    'person', 
    'potted plant', 
    'sheep', 
    'couch', 
    'train', 
    'tv'
]

indexes = []
for i in range(len(table['Coco Object'])):
    if table['Coco Object'][i] in coco_categories_subsample: indexes.append(i)

filtered_table = table[table.index.isin(indexes)]

for category in coco_categories_subsample:
    print(category, ':', len(table[table['Coco Object'] == category]))

subList = pd.read_csv(pathStr+'Places365-Wordnet Alternatives.csv')


import nltk
from nltk.corpus import wordnet as wn

column_names = ['COCO Word',
                'COCO Wordnet Label',
                'Places365 Label',
                'Places365 Wordnet Label',
                'Wordnet Similarity'
               ]
sList = dict(zip(subList['Places365 Label'], subList['Wordnet noun']))

df = pd.DataFrame(columns=column_names)

coco = pd.DataFrame(coco_categories_subsample)
coco[0][10] = 'dining_table'
coco[0][15] = 'plant'
# coco

# DDL: Correction, synset for plant and tv really should be:
#   plant.n.02 ,  television_receiver

print('Beginning paired combinations...')
coco_places_combos = []
labelPairs = {}
coco_output = coco[0]
places365 = pd.DataFrame(filtered_table['Places365 Image Classification'].drop_duplicates())
# print(list(places365['Places365 Image Classification']))
places_output = list(places365['Places365 Image Classification'])
for i in range(len(coco_output)):
    for j in range(len(places_output)):
        tempComboList = []
        pair = coco_output[i] + '-' + places_output[j]            
        if pair in labelPairs:
            pass
        else:
            l1 = [coco_output[i]]
            if places_output[j] in sList:
                print(j)
                #if not np.isnan(sList[places_output[j]]):
                if type(sList[places_output[j]])==str:
                  print(sList[places_output[j]])
                  l2 = [sList[places_output[j]]][0].strip()
                  sl1 = wn.synsets(l1[0])[0]
                  sl2 = wn.synset(l2)
                  rows = pd.DataFrame([[l1, sl1, l2, sl2, sl1.wup_similarity(sl2)]])
                else:
                   print('nan skipped')
            else:
                l2 = places_output[j].split(" ")
#                 print(l1[0])
                sl1 = wn.synsets(l1[0])[0]
                sl2 = wn.synsets(l2[0])[0]
#                 print(sl1, sl2)
                rows = pd.DataFrame([[l1, sl1, l2, sl2, sl1.wup_similarity(sl2)]])
            df = pd.concat([df, rows])


colList=df.columns[-5:]
df=df.iloc[:,:5]
df.columns=colList

df.to_csv('COCO-PlacesWordnetSimilarities2_DDL.csv')



label_similarities = pd.read_csv('COCO-PlacesWordnetSimilarities2_DDL.csv', engine='python')
#label_similarities=df
label_similarities['NamePair'] = label_similarities['Places365 Label'].str.cat(label_similarities['COCO Word'], sep='-')
label_similarities




subDict = dict(zip(subList['Wordnet noun'], subList['Places365 Label']))

placesCoco_Wordsim_dict = dict(zip(label_similarities['NamePair'], label_similarities['Wordnet Similarity']))
table = pd.read_csv('all_combined_table4.csv', encoding='utf-8', engine='python')

pANDco = []

for i in range(len(table)):
    P365 = table.iloc[i]['Places365 Resnet50 Image Classification']
    CO20 = table.iloc[i]['Best FCN-Resnet101 Guess from 20 COCO Objects']
    if CO20 == 'potted plant':
        CO20 = 'plant'
    if P365 in subDict:
        P365 = subDict[P365]
        print(P365)
    pair = str(P365) + '-' + str(CO20)
    try:
        print(pair)
        print(placesCoco_Wordsim_dict[pair])
        pANDco.append(placesCoco_Wordsim_dict[pair])
    except:
        pANDco.append('NA')

table['Places365-YOLOCOCO Wordnet Similarity'] = pANDco
#table
