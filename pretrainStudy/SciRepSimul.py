from netResponses import *
from ConCat import *
import pickle
# code is broken up into different functions

def runNetResponses():

    # loading animal pictures in data/ directory
    # assuming two categories per context, only two contexts
    # contexts : pets and nonpets
    # categories in pets : cats and dogs (5 images in each category)
    # categories in nonpets: spider and turtle ( 5 images in each category)
    # and five pictures per category
    # you can change this code to reflect your actual data and analysis
    contexts = listdir('data/STIMULI-Shira-F73')
    for i in range(len(contexts)):
        contexts[i] = "data/STIMULI-Shira-F73/" + contexts[i]
    directoriesForAnalysis = contexts
    # now we don't want to put a limit on number of file to load for now
    # startFileNumber = 1 
    # endFileNumber = 10
    # filePaths = organize_paths_for(directoriesForAnalysis, endFileNumber)
    filePaths = organize_paths_for(directoriesForAnalysis)
    cnnModel = 'Vgg16'

    outputName = cnnModel

    # get layer responses for all pictures
    dictionary = find_max_neurons_and_layers_for(outputName, directoriesForAnalysis, filePaths, cnnModel)

    # store this dictionary data in  a pickle file
    if os.path.exists("NeuralLayersDict_stimuli_Data") == False:
        os.mkdir("NeuralLayersDict_stimuli_Data")
    print("creating a pickle file ...")
    pickle_file = open("NeuralLayersDict_stimuli_Data/NeuralLayersDictionary.pkl", "wb")
    pickle.dump(dictionary,pickle_file)
    pickle_file.close()
    print("done!")
    # set up the variable numberOfDataPoints
    numberOfDataPoints = number_of_scatterplot_dots(directoriesForAnalysis)

    # save inter-image distances
    run_analytics_suite(dictionary, outputName, filePaths, numberOfDataPoints)


def get_layer_ratios():
    contexts = listdir('data/STIMULI-Shira-F73')
    for i in range(len(contexts)):
        contexts[i] = "data/STIMULI-Shira-F73/" + contexts[i]
    directoriesForAnalysis = contexts

    filePaths = organize_paths_for(directoriesForAnalysis)

    with open("layer_category_data.pkl",'rb') as layCat:
        layCat_df = pickle.load(layCat)

    with open("layer_context_data.pkl",'rb') as layCon:
        layCon_df = pickle.load(layCon)

    layCon_df.index = os.listdir("./data/STIMULI-Shira-F73/")

    ratioCols = list()
    c = 3
    while c <= (len(layCon_df.columns)+2):
        ratioCols.append(layCon_df.columns[c-1])
        c+=3
    print(ratioCols)

    topFiveCon = dict()
    bottomFiveCon = dict()
    for layer in ratioCols:
        t5 = layCon_df[layer].sort_values(ascending = False).iloc[0:5]
        topFiveCon[layer+"_top5"] = tuple(zip(t5.index,t5.values))
        b5 = layCon_df[layer].sort_values().iloc[0:5]
        bottomFiveCon[layer+"_least5"] = tuple(zip(b5.index,b5.values))

    pandas.DataFrame(topFiveCon).to_csv("topFiveContexts.csv")
    pandas.DataFrame(bottomFiveCon).to_csv("bottomFiveContexts.csv")


    categories = list()
    for i in range(1,len(filePaths),5):
        categories.append(filePaths[i])
    print(len(categories))
    print(categories)

    layCat_df.index = categories

    topFiveCat = dict()
    bottomFiveCat = dict()
    for layer in ratioCols:
        t5 = layCon_df[layer].sort_values(ascending = False).iloc[0:5]
        topFiveCat[layer+"_top5"] = tuple(zip(t5.index,t5.values))
        b5 = layCon_df[layer].sort_values().iloc[0:5]
        bottomFiveCat[layer+"_least5"] = tuple(zip(b5.index,b5.values))

    pandas.DataFrame(topFiveCat).to_csv("topFiveCategories.csv")
    pandas.DataFrame(bottomFiveCat).to_csv("bottomFiveCategories.csv")
    
    
# compute ratio of in-category/out-category and in-context/out-context
def runConCat():
    computeRatios()

# Currently we run nothing, but can un-comment one line to either 
#   1) compute outputs of network for particular data (runNetResponses)
#   2) compute in/out ratios at each layer (runConCat)
# runNetResponses()
# runConCat()
get_layer_ratios()

