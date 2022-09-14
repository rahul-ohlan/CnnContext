from netResponses import *
from ConCat import *

# code is broken up into different functions

def runNetResponses():

    # loading animal pictures in data/ directory
    # assuming two categories per context, only two contexts
    # and five pictures per category
    # you can change this code to reflect your actual data and analysis
    directoriesForAnalysis = ['data/1pet','data/2nonpet']
    startFileNumber = 1 
    endFileNumber = 10 
    filePaths = organize_paths_for(directoriesForAnalysis, endFileNumber)
    cnnModel = 'Vgg16'

    outputName = cnnModel

    # get layer responses for all pictures
    dictionary = find_max_neurons_and_layers_for(outputName, directoriesForAnalysis, startFileNumber, endFileNumber, filePaths, cnnModel)

    # set up the variable numberOfDataPoints
    numberOfDataPoints = number_of_scatterplot_dots(directoriesForAnalysis, endFileNumber)

    # save inter-image distances
    run_analytics_suite(dictionary, outputName, filePaths, numberOfDataPoints)


# compute ratio of in-category/out-category and in-context/out-context
def runConCat():
    computeRatios()

# Currently we run nothing, but can un-comment one line to either 
#   1) compute outputs of network for particular data (runNetResponses)
#   2) compute in/out ratios at each layer (runConCat)
#runNetResponses()
#runConCat()
