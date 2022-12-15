import os

os.chdir('../cb_work')
from word2vec_distance_analysis_ahc import *

import numpy as np
import pandas as pd

nClust=40
C=Cluster(dfC, num_clusters=nClust)

dfLabel=pd.DataFrame(np.array(C.label))
dfLabel.index=dfC.index

dfLabel.to_csv('clustW2V'+str(nClust)+'.csv')

onesVec=np.zeros((dfC.shape[0],nClust))

for k in range(dfC.shape[0]):
  onesVec[k,C.label]=1

dfVec=pd.DataFrame(np.array(onesVec))
dfVec.index=dfC.index
dfVec.to_csv('clustW2V'+str(nClust)+'Vec.csv')

