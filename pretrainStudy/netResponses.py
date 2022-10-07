from copyreg import pickle
import pickle as pkl          # added pickle lib as pkl
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import numpy
import pandas
import os, glob
import math
from os import listdir
from os.path import isfile, join
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE, MDS
from scipy.io import savemat
import random
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import csv
from sklearn.metrics.pairwise import cosine_similarity

failedImages = []

# adapted from eric_roginek/archives/User_Specified...13.py

# Set the image transformation parameters to standardize all input images
transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )])

# Save the pretrained Vgg16 model
original_model = models.vgg16(pretrained=True)
resnet152_model = models.resnet152(pretrained=True)

for param in original_model.parameters():
        param.requires_grad = False

# Artificial neural network model that accepts a number representing the intermediary layer of interest
# The class treats the passed number as the neural network's topmost layer
class Vgg16Conv(nn.Module):
        def __init__(self, layer = None):
                super(Vgg16Conv, self).__init__()
                if layer != None:
                        self.features = nn.Sequential(*list(original_model.features.children())[:layer])
                else:
                        self.features = nn.Sequential(*list(original_model.features.children())[:])
        def forward(self, x):
                x = self.features(x)
                return x
        
# Extract highest weighted neuron for a file path image in each intermediary neural layer of interest
def extract_max_neurons(filePath, model):
        try:
                print("Beginning transformation of image ", filePath, " ...")
                # Open and transform the image
                img = Image.open(filePath)
                img_t = transform(img)
                batch_t = torch.unsqueeze(img_t, 0)
                print(model)
                print("Organizing neural layers...")
                numOfLayers = 0
                if model == "Vgg16":
                        # networkLayers[0-4] are conv1-conv5, while [5-7] are maxpool1-maxpool3
                        #networkLayers = [2, 5, 8, 10, 12, 3, 6, None]
                        networkLayers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
                        numOfLayers = len(networkLayers)

                        # Instantiate neural networks up to specified layer and pass the transformed image into each subnetwork, saving the results of each top layer in a list
                        layer = [None] * numOfLayers
                        for number in range(len(networkLayers)):
                                conv = Vgg16Conv(networkLayers[number])
                                layer[number] = conv(batch_t)
                if model == "Resnet152":
                        numOfLayers = 7
                        layer = [None] * numOfLayers
                        for number in range(len(layer)):
                                subNet = nn.Sequential(*list(resnet152_model.children()))[0:number]
                                try:
                                        fullLayer = subNet[-1]
                                        numberOfSubLayers = len(fullLayer)
                                        for k in range(numberOfSubLayers):
                                                fullLayerCopy = fullLayer
                                                fullLayerCopy = fullLayerCopy[0:k]
                                                subNet[-1] = fullLayerCopy
                                                layer[number] = subNet(batch_t)
                                except:
                                        layer[number] = subNet(batch_t)
                
                print("Extracting maximum neuron from each layer...")
                # Save maximum neuron from each layer in a list
                maxNeurons = []
                for neurons in range(len(layer)):
                        listLayer = layer[neurons].detach().numpy()
                        neuronNumber = listLayer.shape[1]
                        highestFired = numpy.zeros(neuronNumber)
                        for neuron in range(neuronNumber):
                                highestFired[neuron] = numpy.max(listLayer[0][neuron])
                        maxNeurons.append(highestFired)
                print("Done!\n")
                return [maxNeurons, numOfLayers]
        except:
                print("Extraction of ", filePath, " failed.\n")
                failedImages.append(filePath)
                return None

# This function returns an ordered list of paths when all filenames are numbers
def sort_numbered_file_names(listOfPaths, numOfFiles):
        pathsForAnalysis = [fileName for fileName in listOfPaths if not(fileName.lower().endswith(('.tar')))]
        imageNames = pathsForAnalysis[:numOfFiles]
        imageNames.sort()
        return imageNames

# This function returns a list that contains all the filepaths containing images of interest for analysis
# def organize_paths_for(directories, maxFiles):
def organize_paths_for(directories):
        onlyfiles = []
        for path in directories:
                print("For images in path \'" + path + "\':\n")
                sortedPicList = sort_numbered_file_names(
                        [fileName for fileName in listdir(path)], len(listdir(path))
                ) #since the picture names are string numbers, they must be properly sorted before continuing
                onlyfiles.extend(sortedPicList)
        print(onlyfiles)
        return onlyfiles

# This function analyzes a specified number of file images within a directory, extracting maximum neurons from each layer and printing a matrix showing Cosine Similarity between images within the same layers of the neural network

def find_max_neurons_and_layers_for(picturesOf, directories, onlyfiles, usingModel):
        print("\n")
        print("************************************")
        
        # Include files with specified extensions in the path directory for analysis and save to a data object for Cosine Similarity
        # currentFile = minFiles - 1
        currentFile = 0
        data = dict()
        directoryNumber = 0
        counter = 0
        while(currentFile < len(onlyfiles)):
                imgName = onlyfiles[currentFile]
                path = directories[directoryNumber] # path for current directory
                print("Finding neurons for image ", imgName, "...")
                imgList, numOfLayers = extract_max_neurons(path + "/" + imgName, usingModel)
                counter += 1
                if counter == len(listdir(path)):   # maxFiles in the eachdirectory i.e 10
                        directoryNumber += 1
                        counter = 0
                # If the extraction fails to return anything, continue to next image
                if(imgList == None):
                        # numberOfDataPoints[directories[directoryNumber]] -= 1
                        currentFile += 1
                        continue
                print(len(imgList))
                image = 'Img' + imgName + "_no" + str(currentFile + 1)
                data[image] = imgList
                print(len(data[image]))
                currentFile += 1
        
        print("Extracting neuron data from image layers...")
        # This for loop extracts and saves the neuron data for each image layer in two dictionaries for subsequent analysis.
        neuralLayers = dict()
        for layer in range(numOfLayers):
                if layer < 9:
                        layerNumber = picturesOf + "Layer0" + str(layer + 1)
                else:
                        layerNumber = picturesOf + "Layer" + str(layer + 1)

                print(layerNumber)
                neurons = {}
                for image, allNeurons in data.items():
                      neurons[image] = data[image][layer]
                df = pandas.DataFrame(neurons)
                
                # Save neurons in current layer for TSNE and MDS analyses
                neuralLayers[layerNumber] = df #df shape is in the format: (#of neurons, #of images)
        
        print("************************************")
        print("Done!\n")
        return neuralLayers


def correlSim(neuralLayersDictionary, placeInPath):
        print("************************************")
        print("Calculating Correl Similarity...\n")
        print("testing september 3-")
        inFilePath = placeInPath + "/" + "Correl Similarity"
        if os.path.exists(inFilePath) == False: os.mkdir(inFilePath)
        # pickle_file = open("NeuralLayersDict_stimuli_Data/NeuralLayersDictionary.pkl","rb")
        # neuralLayersDictionary = pkl.load(pickle_file)
        for key in neuralLayersDictionary:
                correlMatrix = np.corrcoef(neuralLayersDictionary[key].T)
                # was correl_similarit
                numpy.save(inFilePath + "/correl_sim_" + key + ".npy", correlMatrix)
                # matrixData = numpy.load(inFilePath + "/correl_sim_" + key + ".npy")
                # plt.imsave(inFilePath + "/correl_sim_" + key + ".png", matrixData)
                plt.imsave(inFilePath + "/correl_sim_" + key + ".png", correlMatrix)
                print("Correl Similarity: ", "\n", correlMatrix, "\n")
        # pickle_file.close()
        print("************************************")
        print("Done!\n")

# This function performs a cluster analysis on each layer stored in a dictionary produced by image_layer_analysis()
def cluster_analysis(neuralLayersDictionary, placeInPath):
        print("************************************")
        print("Beginning Hierarchical Clustering Analysis\n")
        inFilePath = placeInPath + "/" + "Ward Linkage Analysis"
        if os.path.exists(inFilePath) == False: os.mkdir(inFilePath)
        for key in neuralLayersDictionary:
                linkMatrix = linkage(neuralLayersDictionary[key].T, method='ward', metric = 'euclidean', optimal_ordering=False)
                numpy.save(inFilePath + "/linkage_" + key + ".npy", linkMatrix)
                fig = plt.figure(figsize=(25, 10))
                dn = dendrogram(linkMatrix)
                plt.savefig(inFilePath + "/" + key + ".jpg") # Save the image
                plt.clf()
                print(key, ": ", linkMatrix)
        print("************************************")
        print("Done!\n\n")

def pyplot_scatterplot_colors(numberOfDataPoints, subImages = False):
        numberOfColors = len(numberOfDataPoints)
        colorDictionary = dict()
        for x in range(numberOfColors):
                r = lambda: random.randint(0,255)
                color = '#%02X%02X%02X' % (r(),r(),r())
                if color in colorDictionary:
                        x -= 1
                else:
                        colorDictionary[color] = color
        colorOptions = list(colorDictionary.keys())
        colors = []
        index = 0
        print("Number of Data Points: ", numberOfDataPoints)
        if subImages == False:
                for path in numberOfDataPoints:
                        colorDuplicates = numberOfDataPoints[path] * [colorOptions[index]]
                        colors.extend(colorDuplicates)
                        index += 1
        else: #this else statement is hardcoded and specific to Shira's object-file organization
                for path in numberOfDataPoints:
                        subColor1 = (int(numberOfDataPoints[path]/2)) * [colorOptions[index]]
                        r = lambda: random.randint(0,255)
                        anotherColor = '#%02X%02X%02X' % (r(),r(),r())
                        subColor2 = (int(numberOfDataPoints[path]/2)) * [anotherColor]
                        colors.extend(subColor1)
                        colors.extend(subColor2)
                        index += 1
        return colors
        
# This function performs either a TSNE or MDS analysis using all neurons and images in each layer being compared
def manifold_analysis(neuralLayersDictionary, analysisType, placeInPath, picDirectory, numberOfDataPoints):
        print("************************************")
        print("Initializing ", analysisType, " analysis \n")
        inFilePath = placeInPath + "/" + analysisType + "/"
        if os.path.exists(inFilePath) == False: os.mkdir(inFilePath)
        for key in neuralLayersDictionary:
                print("Current layer: ", key, "...\n")
                if analysisType == "TSNE":
                        layerEmbedded = TSNE(n_components = 2).fit_transform(neuralLayersDictionary[key].T)
                elif analysisType == "MDS":
                        layerEmbedded = MDS(n_components = 2).fit_transform(neuralLayersDictionary[key].T)
                        
                # Create Scatterplot and loop through each coordinate in the scatterplot and label it with the x, y coordinates before saving the figure and clearing the pyplot
                colorDots = pyplot_scatterplot_colors(numberOfDataPoints)
                currentPic = 0
                plt.figure(figsize=(30, 15))
                for color in colorDots:
                        n = 1
                        x = layerEmbedded[:,0][currentPic]
                        y = layerEmbedded[:,1][currentPic]
                        xyCoordinates = str(x) + ", " + str(y)
                        plt.scatter(x, y, c = color)
                        plt.text(x + .2, y + .2, currentPic)
                        currentPic += 1
                plt.savefig(inFilePath + analysisType + "_" + key + '.jpg')
                plt.clf()
                
                #Create Legend List for the images and their colors/coordinates
                #idList = []
                #for index in colorDots:
                        #idList.append(index)
                #np.savetxt("Legend_" + inFilePath + key + '.csv', np.c_[idList, colorDots, layerEmbedded[:,0], layerEmbedded[:,1]], delimiter=',', fmt='%d')
                
                #KMeans clustering analysis
                clusters = [len(numberOfDataPoints), (len(numberOfDataPoints) * 2)] #this line is purely for the benefit of analyzing Shira's data, since she includes two sets of objects in each directory 
                secondConfusionMatrix = False
                print("Determining K-means clusters...\n")
                plt.figure(figsize=(30, 15))
                # Create an x,y coordinates dictionary for scatterplot use
                xyCoordinates = {
                        'x': layerEmbedded[:,0],
                        'y': layerEmbedded[:,1]
                }
                xy = pandas.DataFrame(xyCoordinates, columns=['x','y'])
                for clustering in clusters: # this for loop produces two clusters, again based on Shira's organization methodology
                        print(str(clustering) + " clusters\n")
                        kmeans = KMeans(n_clusters=clustering)
                        kmeans.fit(xy)
                        centroids = kmeans.cluster_centers_
                        print(centroids)
                        currentPic = 0
                        plt.scatter(xyCoordinates['x'], xyCoordinates['y'], c= kmeans.labels_.astype(float), alpha=0.5)
                        plt.scatter(centroids[:, 0], centroids[:, 1], c='red')
                        for color in colorDots:
                                plt.text(xyCoordinates['x'][currentPic] + .2, xyCoordinates['y'][currentPic] + .2, currentPic)
                                currentPic += 1
                        kPath = inFilePath + "/Kmeans/"
                        if os.path.exists(kPath) == False: os.mkdir(kPath)
                        plt.savefig(kPath + str(clustering) + "-KClusters" + "_" + key + '.jpg')
                        plt.clf()
                        
                        # Confusion Matrix prep step
                        trueValues = []
                        colorSet = []
                        if secondConfusionMatrix == True: #like the clustering/clusters loop, this if/else is just for Shira's files
                                colorSet = pyplot_scatterplot_colors(numberOfDataPoints, secondConfusionMatrix)
                        else:
                                colorSet = colorDots
                        trueValues = np.empty(len(colorSet))
                        trueValues[0] = 0
                        groupID = 0
                        for color in range(1, len(colorSet)):
                                if colorSet[color - 1] == colorSet[color]:
                                        trueValues[color] = groupID
                                else:
                                        groupID += 1
                                        trueValues[color] = groupID
                        k_labels = kmeans.labels_
                        k_labels_matched = np.empty_like(k_labels)
                        for k in np.unique(k_labels):
                                # ...find and assign the best-matching truth label
                                match_nums = [np.sum((k_labels==k)*(trueValues==t)) for t in np.unique(trueValues)]
                                k_labels_matched[k_labels==k] = np.unique(trueValues)[np.argmax(match_nums)]
                        # Confusion Matrix and Classification Report
                        print("TrueValues: ", trueValues)
                        print("k_labels_matched: ", k_labels_matched)
                        cm = confusion_matrix(trueValues, k_labels_matched)
                        report = classification_report(trueValues, k_labels_matched)
                        print(report, file=open(kPath + str(clustering) + "-ConfusionM" + "_" + key + ".txt", "a"))
                        
                        # Plot confusion matrix
                        plt.imshow(cm,interpolation='none',cmap='Blues')
                        for (i, j), z in np.ndenumerate(cm):
                                plt.text(j, i, z, ha='center', va='center')
                                plt.xlabel("kmeans label")
                                plt.ylabel("truth label")
                        plt.savefig(kPath + str(clustering) + "-ConfusionM" + "_" + key + '.jpg')
                        plt.clf()
                        secondConfusionMatrix = True #Also for Shira's files
                plt.close(fig='all')
                
                
                
                # Save the image as a numpy file
                numpy.save(inFilePath + key + ".npy", tuple(zip(layerEmbedded[:,0], layerEmbedded[:,1])))
                
        print("************************************")
        print("Done! Analysis complete. \n")

def run_analytics_suite(neuralLayersDictionary, placeInPath, picDirectory, numberOfDataPoints):

        # These next few lines of code runs hierarchical cluster, TSNE, and MDS analyses on the data produced by the last for loop
        print("Begin analytics suite:\n")
        if os.path.exists(placeInPath + "/") == False: os.mkdir(placeInPath + "/")
        correlSim(neuralLayersDictionary, placeInPath)
        #cluster_analysis(neuralLayersDictionary, placeInPath)
        #manifold_analysis(neuralLayersDictionary, "MDS", placeInPath, picDirectory, numberOfDataPoints)
        #manifold_analysis(neuralLayersDictionary, "TSNE", placeInPath, picDirectory, numberOfDataPoints)
        print("PearsonCorr, H-Cluster, MDS, and TSNE analyses complete.\n")

def number_of_scatterplot_dots(directoriesForAnalysis):
        categoryList = dict()
        for categories in directoriesForAnalysis:
                categoryList[categories] = len(listdir(categories))
        return categoryList
        

#Not implemented yet
####
analysesAvailable = {
        'PearsonCorrelation': True,
        'Heirarchical Clustering': True,
        'MDS': True,
        'tSNE': True,
        'KMeans': True
}
#print("These are the analysesAvailable:\n")
#for analysis in analysesAvailable:
        #print(analysis + "\n")
#print("Include all?\n")
#includeAll = input("(Type 'Y' or 'Yes'\n")
####
#print(directories)
#filePaths = organize_paths_for(directoriesForAnalysis, endFileNumber)
#numberOfDataPoints = number_of_scatterplot_dots(directoriesForAnalysis, endFileNumber)
#dictionary = find_max_neurons_and_layers_for(imageAbstraction, directoriesForAnalysis, startFileNumber, endFileNumber, filePaths, cnnModel)
#neurons = dictionary.to_numpy()
#numpy.save('neuralLayersDictionary.npy', dictionary)
#run_analytics_suite(dictionary, imageAbstraction, filePaths, numberOfDataPoints)
#print("Replace these files: ", failedImages)
