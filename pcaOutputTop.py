import pandas as pd
from sklearn.decomposition import PCA

for layIter in range(3):
  realLay = (layIter+1)*3
  df=pd.read_csv('~gkalaitzis/val2017stuff/Val2017netResponse'+'.csv',index_col='col1')
  pca=PCA(n_components=10)
  pca.fit(df.to_numpy())
  out=pca.transform(df.to_numpy())
  outDf=df[['0','1','2','3','4','5','6','7','8','9']].copy()
  outDf.iloc[:,:]=out
  outDf.to_csv('/u/erdos/cnslab/cnnContext/CnnContext/gkPC_Lay'+'Top'+'.csv')
