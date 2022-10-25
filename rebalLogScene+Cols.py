import pandas as pd

df_LS = pd.read_csv('LogScene+Cols.csv')

classCount=df_LS['OBJECT (PAPER)'].count_values()

boostNum={}
for key in classCount.keys():
  boostNum[key]=classCounts['person']/classCounts[key]

# iteratively remove most common classes from df_LS_temp
# add more copies of df_LS_temp to df_LS_temp
# problem: later rows lack more common classes
# how to randomize while preventing same data from being 
# train and test?
#   - pre-split train and test data and THEN re-balance
#   - incorporate sklearn.resample to upsample: example:
#           df1 = df[df['store'] == 1]
#           other_df = df[df['store'] != 1]
#           df1_upsamp = resample(df1,random_state=42,n_samples=2,replace=True)
#           df_upsampled = pd.concat([df1_upsampled,other_df])
#        see: https://towardsdatascience.com/heres-what-i-ve-learnt-about-sklearn-resample-ab735ae1abc4

df_LS_temp = df_LS.copy()
df_LS_temp.drop(labels=list(df_LS_temp[df_LS_temp['OBJECT (PAPER)']==currKey].index),axis=0,inplace=True)

df_LS_more = df_LS.copy()

df_LS_more = df_LS_more.append(df_LS_temp)
