# from copyreg import pickle
import pickle
from math import floor, sqrt
import os
import scipy.stats as stats
import numpy as np
import pandas

# basePath='/u/erdos/cnslab/cnnContext/pretrainStudy/'  # let's keep it local for now
basePath = r'C:\Users\rahul__ohlan\OneDrive\Desktop\coding\FordhamDS\Research\ComputerVision\code\pretrainStudy'

layerData = {
    'Alexnet_RP': list(range(11)),
    'Alexnet_Places': list(range(11)),
    'Googlenet': list(range(0,17)),
    'Resnet18_PR365': list(range(7)),
    'Resnet50Places': list(range(7)),
    'Resnet18_RP': list(range(7)),
    'Resnet50_RP': list(range(7)),
    'Resnet101_RP': list(range(7)),
    'Resnet152_RP': list(range(7)),
    'Resnext50_32x4d': list(range(7)),
    'Vgg16': list(range(31)),    # changed it from 28 to 31
    'Vgg19': list(range(35))
}

# layerData = {
#     'Alexnet-R': [0,3,6,8,10],
#     'Googlenet': list(range(0,17)),
#     'P365-Alexnet': [0,3,6,8,10],
#     'P365-Resnet18': [0,4,5,6,7],
#     'P365-Resnet50': [0,4,5,6,7],
#     'Resnet18-R': [0,4,5,6,7],
#     'Resnet50-R': [0,4,5,6,7],
#     'Resnet101-R': [0,4,5,6,7],
#     'Resnet152-R': [0,4,5,6,7],
#     'Resnext50_32x4d': [0,4,5,6,7],
#     'Vgg16-R': [0,4,6,8,9,11,13,16,18,20,23,25,27],
#     'Vgg19-R': [0,2,5,7,10,12,14,16,19,21,23,25,28,30,32,34]
# }

#### Confound Expansion function creates a new matrix matching the dimensions of the data matrix
def confoundExpansion(confoundMatrix, dimNumber, expansionNumber):
    expansionMatrix = np.empty(((dimNumber*expansionNumber) - 1, (dimNumber*expansionNumber) - 1))
    confoundMatrix = confoundMatrix[:dimNumber-1].T
    # This for loop is missing not covering the last few elements, so they are being labeled 'True' when they should be
    # False along the diagonal
    for i in range((dimNumber*expansionNumber) - expansionNumber):
        for j in range((dimNumber*expansionNumber) - expansionNumber):
            expansionMatrix[i,j] = confoundMatrix[floor(i/expansionNumber), floor(j/expansionNumber)]
    
    # Copy lower triangle confounds to upper triangle such that the matrices are symmetric along the diagonal,
    # then convert into a boolean matrix
    expansionMatrix = expansionMatrix + expansionMatrix.T - np.diag(np.diag(expansionMatrix))
    confoundMatrix = expansionMatrix == 0
    return confoundMatrix
#####

def computeRatios(contexts=73, useConfounds=False):
   database = pandas.DataFrame(columns=['network', 'layer', 'ratioCon', 'pCon1', 'pConRel', 'conErrBars', 'ratioCat', 'pCat1', 'pCatRel', 'catErrBars', 'pConVCat'])
   files = os.listdir()

   # This for loop goes through all available models and networks folders containing the results of interest (Pearson's correlation matrices)
   for modelName in range(len(files)):
       #if files[modelName] == "71-confounds" or files[modelName] =="__pycache__" or files[modelName]=="data" : continue
       if not os.path.isdir(basePath+"\\"+files[modelName]) or files[modelName] == '__pycache__' or files[modelName] == 'data' or files[modelName]=="NeuralLayersDict" or files[modelName] == "NeuralLayersDict_stimuli_Data": continue
       
       subFiles = os.listdir("./" + files[modelName] + "/")
       for x in range(len(subFiles)):
           print(str(x) + ": " + subFiles[x])
       #pathToFile = "Pearson\'s Correlations/"
       pathToFile = "Correl Similarity/"
       matrixType = "correl_sim_"
   
       # if subFiles[pathNumber] == "Pearson's Correlations":
       #     matrixType = "pearson_matrix_"
   
       # if subFiles[pathNumber] == "Cosine Similarity":
       #     matrixType = "cosine_sim_"
   
       print("Type the directory number you're intereted in analyzing in ", files[modelName])
       # numbers = input("Type in layer numbers of interest in " + files[modelName] + " , separated by spaces: ")
       layVec = layerData[files[modelName]]
       # layVex = [int(i) for i in layVec]
   
       ## Input info
       #contexts = 2 #71
       categories = contexts * 2
       matrixName = files[modelName] + "/" + pathToFile + matrixType + files[modelName] + 'Layer'
       print(matrixName)
       # layVec=[0,1,2,3,4,5,6]
       index = 0
   
       if useConfounds:
          # Import confound matrices; ..Confounds2 has been modified to exclude the "Burger" and "FarmAnimals". If a Heatmap is of interest, 
          # then the original .npy files need to be changed to have those contexts and their categories removed
          contextConfounds = np.loadtxt('71-confounds/contextConfounds.txt', dtype=int, usecols=range(contexts - 1))
          categoryConfounds = np.loadtxt('71-confounds/categoryConfounds.txt', dtype=int, usecols=range(categories - 1))
   
          # Expand confound matrix to match data dimensions
          contextConfounds = confoundExpansion(contextConfounds, contexts, 10)
          categoryConfounds = confoundExpansion(categoryConfounds, categories, 5)
   
       # Create empty lists for storing future values for t-tests
       inContextValues = []
       outContextValues = []
       ratioCon = []
   
       inCategoryValues = []
       outCategoryValues = []
       ratioCat = []
   
       #file=open(files[modelName] + "/" + pathToFile + 'contextRatio.txt', 'w')
       #file=open(files[modelName] + "/" + pathToFile + 'categoryRatio.txt', 'w')
   
       # Context/Category Ratio analysis
       for layer in range(1, len(layVec)+1):
           # Load in layer data
           layerNumber = None
           if (layer-1) != layVec[index]:
               continue
           else:
               if (layer < 10):
                   layerNumber = '0' + str(layer)
                   myData = np.load(matrixName + layerNumber + '.npy')
                   index += 1
               else:
                   layerNumber = str(layer)
                   myData = np.load(matrixName + layerNumber + '.npy')
                   index += 1
   
           # Context: outContext, inContext, and contextRatio calculations
   
           # Context
           for k in range(contexts):
               # outContext
               subMatData=np.hstack((myData[(10*k):(10*(k+1)),:(10*k)],myData[(10*k):(10*(k+1)),(10*(k+1)):]))
               if useConfounds:
                  subMatConfounds=np.hstack((contextConfounds[(10*k):(10*(k+1)),:(10*k)],contextConfounds[(10*k):(10*(k+1)),(10*(k+1)):]))
                  subMat = np.extract(subMatConfounds, subMatData) # remove confounds
               else:
                  subMat = subMatData
               outVal=subMat.mean()
               outContextValues.append(outVal)
               
               # inContext
               inVal=(myData[(10*k):(10*(k+1)),(10*k):(10*(k+1))].sum()-10)/90
               # inVal=(myData[(10*k):(10*k+5),(10*k+5):(10*(k+1))].sum())/25 # BETTER
               inContextValues.append(inVal)
               
               # contextRatio
               contextRatio = inVal/outVal
               ratioCon.append(contextRatio)
               # print(layer, ':', k, ':', subMat.T.shape, '\t:outVal', outVal, '\t:inVal', inVal, '\t:ratioCon', contextRatio)
               print(layerNumber + "\t" + str(inVal) + "\t" + str(outVal) + "\t" + str(contextRatio), file=open(files[modelName] + "/" + pathToFile + 'contextRatio.txt', 'a'))
           
           # Category
           for k in range(categories):
               # outCategory
               subMatData = np.hstack((myData[(5*k):(5*(k+1)),:(5*k)],myData[(5*k):(5*(k+1)),(5*(k+1)):]))
               if useConfounds:
                  subMatConfounds = np.hstack((categoryConfounds[(5*k):(5*(k+1)),:(5*k)],categoryConfounds[(5*k):(5*(k+1)),(5*(k+1)):]))
                  subMat = np.extract(subMatConfounds, subMatData) # remove confounds
               else:
                  subMat = subMatData
               outVal= subMat.mean()
               outCategoryValues.append(outVal)
               
               # inCategory
               inVal=(myData[(5*k):(5*(k+1)),(5*k):(5*(k+1))].sum()-5)/20
               inCategoryValues.append(inVal)
               
               # categoryRatio
               categoryRatio = inVal/outVal
               ratioCat.append(categoryRatio)
               print(layerNumber + "\t" + str(inVal) + "\t" + str(outVal) + "\t" + str(categoryRatio), file=open(files[modelName] + "/" + pathToFile + 'categoryRatio.txt', 'a'))
   
       print("ratioCon: ", ratioCon)
       print("ratioCat: ", ratioCat)
       # rCon = pd.DataFrame()
       # Calculate context ratios and p-values
       inConRatios = []
       outConRatios = []
       pVecRcon=[]
       pVec1con=[]
       mnVecCon=[]
       conErrBars = []

       # dictionary to capture in/out context values from each layer
       layer_context = dict()
   
       print('contextValues length: ', len(ratioCon))
       for lay in range(len(layVec)):
           # In/Out-context ratios for each layer of interest with confounds removed
        
           layerNumber = None
           if lay < 9:
            layerNumber = "layer" + "0" + str(lay+1)
           else:
            layerNumber = "layer" + str(lay+1)
           inConValues = inContextValues[(contexts*lay):(contexts*lay)+contexts]  # 73 values in each layer ideally
           outConValues = outContextValues[(contexts*lay):(contexts*lay)+contexts]
           
           layer_context[layerNumber+"_inCon"] = inConValues
           layer_context[layerNumber+"_outCon"] = inConValues
           layer_context[layerNumber+"_inOutRatio"] = np.array(inConValues)/np.array(outConValues)
           
           # T-tests for contexts
           out=stats.ttest_rel(inConValues, outConValues)
           pVecRcon.append(out.pvalue)
           out=stats.ttest_1samp(ratioCon[(contexts*lay):((contexts*lay)+contexts)],1)
           pVec1con.append(out.pvalue)
           mnVecCon.append(np.array(ratioCon[(contexts*lay):(contexts*lay)+contexts]).mean())
           conErrBars.append(np.std(ratioCon[(contexts*lay):(contexts*lay)+contexts])/sqrt(contexts))
       layer_dataframe_context = pandas.DataFrame(layer_context)
       print('\nGood, contexts complete.\n')
   
       # Calculate category ratios and p-values
       networkName = []
       inCatRatios = []
       inCatRatios1 = []
       inCatRatios2 = []
       outCatRatios = []
       pVecRcat=[]
       pVec1cat=[]
       mnVecCat=[]
       catErrBars = []
   
       pVecRconVcat=[] # Holds the p-values for pairwise t-tests between category and context for each layer
   
       print('categoryValues: ', len(ratioCat))
       # dictionary to capture inOut Category value for each layer:
       layer_category = dict()
       for lay in range(len(layVec)):
           print("testin")
           networkName.append(files[modelName])
           print('layer:' , layVec[lay])
           print('categories*lay = ', categories*lay)
           print('(categories*lay)+categories) = ', (categories*lay)+categories)
           
           # # In/Out-category ratios for each layer of interest with confounds removed
           layerNumber = None
           if lay < 9:
            layerNumber = "layer" + "0" + str(lay+1)
           else:
            layerNumber = "layer" +  str(lay+1)

           inCatValues = inCategoryValues[(categories*lay):(categories*lay)+categories]
           outCatValues = outCategoryValues[(categories*lay):((categories*lay)+categories)]

           layer_category[layerNumber+"_inCat"] = inCatValues
           layer_category[layerNumber+"_outCat"] = outCatValues
           layer_category[layerNumber+"_inOutRatio"] = np.array(inCatValues)/np.array(outCatValues)
   
           # T-tests for categories
           out=stats.ttest_rel(inCatValues, outCatValues)
           pVecRcat.append(out.pvalue)
           out=stats.ttest_1samp(ratioCat[(categories*lay):(categories*(lay+1))],1)
           pVec1cat.append(out.pvalue)
           mnVecCat.append(np.array(ratioCat[(categories*lay):(categories*lay)+categories]).mean())
           catErrBars.append(np.std(ratioCat[(categories*lay):(categories*lay)+categories])/sqrt((contexts)*2))
           
           # Calculate pairwise t-test for categories and context
           out=stats.ttest_rel(ratioCat[(categories*lay):(categories*(lay+1)):2],ratioCon[(contexts*lay):(contexts*(lay+1))])
           pVecRconVcat.append(out.pvalue)
       layer_dataframe_category = pandas.DataFrame(layer_category)
       print('\nGood, categories complete.\n')

       # save the two dictionarie in a pickle file
       with open("layer_context_data.pkl",'wb') as layCon:
        pickle.dump(layer_dataframe_context,layCon)

       with open("layer_category_data.pkl",'wb') as layCat:
        pickle.dump(layer_dataframe_category,layCat)
       # Create and save context/categories ratios and p-values, concatonate with previous results
       dataMat=[networkName, layVec, mnVecCon, pVec1con, pVecRcon, conErrBars, mnVecCat, pVec1cat, pVecRcat, catErrBars, pVecRconVcat]
       df=pandas.DataFrame(np.array(dataMat).T,columns=['network', 'layer', 'ratioCon', 'pCon1', 'pConRel', 'conErrBars', 'ratioCat', 'pCat1', 'pCatRel', 'catErrBars', 'pConVCat'])
       database = database.append(df)


   
   
   # Save all dataframe results to a single .csv file
   database.to_csv("all_con_cat_ratios.csv")
   
   print('Done.')
