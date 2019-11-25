# SMRobot project
# Ignacio Garrido Botella
# Code with the needed functions to create the dataset of features of each image.
import scipy.io
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from skimage.feature import local_binary_pattern
from scipy.stats import entropy
from sklearn import preprocessing
from sklearn.cluster import KMeans
#from sklearn.cluster import AgglomerativeClustering
from scipy.stats import wasserstein_distance
import math
import pickle

os.chdir("/Users/ignacio/Documents/Universidad/Master/Segundo/SegundoSemestre/MasterThesisMoAI/Project/smrobot_py_knowledgeGraph")
import kGraph_functions as kgf
# %% Links
# lbp: https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html
# lbp: https://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.local_binary_pattern
# canny: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_canny/py_canny.html
# clustering: https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering
# Agglomerative clustering: https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/
# See best kMeans -> https://www.aprendemachinelearning.com/k-means-en-python-paso-a-paso/
# Line detection: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
# Hough: https://docs.opencv.org/3.3.1/d9/db0/tutorial_hough_lines.html
# Increase saturation of image: https://docs.opencv.org/3.4/d3/dc1/tutorial_basic_linear_transform.html
# Laplacian: https://docs.opencv.org/3.4.3/d5/db5/tutorial_laplace_operator.html
# Edge detection: https://medium.com/@nikatsanka/comparing-edge-detection-methods-638a2919476e
# F. segmentation: https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.felzenszwalb

#%% Load datasets & create merged dataframe

df_1_gt = scipy.io.loadmat('/Users/ignacio/Documents/Universidad/Master/Segundo/SegundoSemestre/MasterThesisMoAI/Project/smrobot_py_knowledgeGraph/datasets/dataset_1_gt.mat')['data_gt']
df_2_gt = scipy.io.loadmat('/Users/ignacio/Documents/Universidad/Master/Segundo/SegundoSemestre/MasterThesisMoAI/Project/smrobot_py_knowledgeGraph/datasets/dataset_2_gt.mat')['data_gt']
df_3_gt = scipy.io.loadmat('/Users/ignacio/Documents/Universidad/Master/Segundo/SegundoSemestre/MasterThesisMoAI/Project/smrobot_py_knowledgeGraph/datasets/dataset_3_gt.mat')['data_gt']
df_4_gt = scipy.io.loadmat('/Users/ignacio/Documents/Universidad/Master/Segundo/SegundoSemestre/MasterThesisMoAI/Project/smrobot_py_knowledgeGraph/datasets/dataset_4_gt.mat')['data_gt']
df_1_im = scipy.io.loadmat('/Users/ignacio/Documents/Universidad/Master/Segundo/SegundoSemestre/MasterThesisMoAI/Project/smrobot_py_knowledgeGraph/datasets/dataset_1_im.mat')['data_pic']
df_2_im = scipy.io.loadmat('/Users/ignacio/Documents/Universidad/Master/Segundo/SegundoSemestre/MasterThesisMoAI/Project/smrobot_py_knowledgeGraph/datasets/dataset_2_im.mat')['data_pic']
df_3_im = scipy.io.loadmat('/Users/ignacio/Documents/Universidad/Master/Segundo/SegundoSemestre/MasterThesisMoAI/Project/smrobot_py_knowledgeGraph/datasets/dataset_3_im.mat')['data_pic']
df_4_im = scipy.io.loadmat('/Users/ignacio/Documents/Universidad/Master/Segundo/SegundoSemestre/MasterThesisMoAI/Project/smrobot_py_knowledgeGraph/datasets/dataset_4_im.mat')['data_pic']

#Merged df with color images
df_merged_colorImages = np.concatenate((df_1_im, df_2_im, df_3_im, df_4_im), axis=0)

#Get images in grayscale:
df_1_im_gray = np.zeros([df_1_im.shape[0], df_1_im.shape[1], df_1_im.shape[2]], dtype=np.uint8)
for i in range(len(df_1_im)):
    df_1_im_gray[i,:,:] = cv2.cvtColor(df_1_im[i,:,:,:], cv2.COLOR_RGB2GRAY)
    
df_2_im_gray = np.zeros([df_2_im.shape[0], df_2_im.shape[1], df_2_im.shape[2]], dtype=np.uint8)
for i in range(len(df_2_im)):
    df_2_im_gray[i,:,:] = cv2.cvtColor(df_2_im[i,:,:,:], cv2.COLOR_RGB2GRAY)
    
df_3_im_gray = np.zeros([df_3_im.shape[0], df_3_im.shape[1], df_3_im.shape[2]], dtype=np.uint8)
for i in range(len(df_3_im)):
    df_3_im_gray[i,:,:] = cv2.cvtColor(df_3_im[i,:,:,:], cv2.COLOR_RGB2GRAY)

df_4_im_gray = np.zeros([df_4_im.shape[0], df_4_im.shape[1], df_4_im.shape[2]], dtype=np.uint8)
for i in range(len(df_4_im)):
    df_4_im_gray[i,:,:] = cv2.cvtColor(df_4_im[i,:,:,:], cv2.COLOR_RGB2GRAY)  
 
#Merged df with grayscale images
df_merged_grayImages = np.concatenate((df_1_im_gray, df_2_im_gray, df_3_im_gray, df_4_im_gray), axis=0)

#Merged df with masks -> ground truth
df_merged_gt = np.concatenate((df_1_gt, df_2_gt, df_3_gt, df_4_gt), axis=0)


infile = open('datasets/df_Prediction.pickle','rb')
df_Prediction = pickle.load(infile)
infile.close()

del  df_1_gt, df_2_gt, df_3_gt, df_4_gt, df_1_im, df_2_im, df_3_im, df_4_im, df_1_im_gray, df_2_im_gray, df_3_im_gray, df_4_im_gray, i
#%% Create dataframes

#GETCLASS
#Description: This function receives an image and returns an array with the 
#   classes of each tile -> If a tile contains ones and zeros, then it is an
#   edge tile. If a tile contains only ones, then it is a floor tile and if a 
#   tile contains only z
#   @Inputs:
#       mask: mask with zeros on bakcground and ones in floor.
#       sizeX: size (number of pixels) of the hyperpixels in X axis. By default
#           it is 18 (image of size 400x702 & 780 sections).
#       sizeY: size (number of pixels) of the hyperpixels in Y axis. By default
#           it is 20 (image of size 400x702 & 780 sections).
#       num: Number of hyperpixels in Y axis (vertical). The number of 
#           hyperpixels in the horizontal axis will be num*2-1. By default it 
#           is 20 (image of size 400x702 & 780 sections). 
#   @Outputs:
#       listClasses: Array with two columns, the first is the ID of the 
#           tile and the second is the class of the tile. 
#           (0 -> background, 1 -> floor, 2 -> edge)
def getClass(mask, sizeX = 18, sizeY = 20, num = 20):
    listHyperpixels = kgf.getHyperpixels(mask, sizeX, sizeY, num) #Get the pixels in each hyperpixel
    listHyperpixels = listHyperpixels[:,9:] #Remove all metadata
    listClasses = np.zeros([num*(2*num-1),2]) 
    for i in range(num*(2*num-1)): #Set the ID numbers (first column)
        listClasses[i,0] = i
    for i in range(num*(2*num-1)):
        if (0 in listHyperpixels[i,:]) and (1 in listHyperpixels[i,:]): #Edge
            listClasses[i,1] = 2
        elif (0 in listHyperpixels[i,:]) and (1 not in listHyperpixels[i,:]): #Background
            listClasses[i,1] = 0
        else: #Floor
            listClasses[i,1] = 1
    return listClasses


#LOCALFEATURESIMAGE
#Description: This function receives an image and a mask and 
#   returns a dataframe with all the local features for each section for each 
#   image.
#   @Inputs:
#       image: image in GRAYSCALE!!!
#       mask: mask of image (it is as 
#           df_merged_gt).
#       sizeX: size (number of pixels) of the hyperpixels in X axis. By default
#           it is 18 (image of size 400x702 & 780 sections).
#       sizeY: size (number of pixels) of the hyperpixels in Y axis. By default
#           it is 20 (image of size 400x702 & 780 sections).
#       num: Number of hyperpixels in Y axis (vertical). The number of 
#           hyperpixels in the horizontal axis will be num*2-1. By default it 
#           is 20 (image of size 400x702 & 780 sections). 
#   @Outputs:
#       df_features_cluster: Dataset with all the features of the hyperpixels:
#           - 0: ID of the hyperpixel referenced to the image (in each image it 
#               starts from 0). 
#           - 1: class of the tile (0 -> background, 1 -> floor, 2 -> edge). 
#           - 2: Position in the X axis relative to the center of the image.
#           - 3: Position in the Y axis relative.
#           - 4: entropy of grayscale distribution.
#           - 5: entropy of lbp distribiution.
#           - 6: Mean value of the grayscale image relative to the mean color of the whole image.
#           - 7: Mean value of the lbp relative to the mean lbp of the whole image.
#           - 8: Proportion of pixels that are edges
#           - 9: Edge detected
def localFeaturesImage(image, mask, sizeX = 18, sizeY = 20, num = 20):
    num_hyperpixels = num*(2*num-1) #Number of hyperpixels per image
    df_features_cluster = np.zeros([num_hyperpixels, 10])        
    hyperpixels_grayImage = kgf.getHyperpixels(image, sizeX, sizeY, num) #Get the dataframe of hyperpixels of the gray image
    classes_mask = getClass(mask, sizeX, sizeY, num) #Get classes mask        
    # Settings for LBP -> code from: https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html
    radius = 3
    n_points = 8 * radius  
    METHOD = 'uniform'
    lbp_image = local_binary_pattern(image, n_points, radius, METHOD)  
    lbp_image = lbp_image.astype(np.uint8)
    hyperpixels_lbp = kgf.getHyperpixels(lbp_image, sizeX, sizeY, num) #Get the dataframe of hyperpixels of the lbp
    edges_list = kgf.getEdges(image, sizeX, sizeY, num)     
    mean_color_image = np.mean(image)
    mean_lbp_image = np.mean(lbp_image)
    for j in range(num_hyperpixels):
        df_features_cluster[j, 0] = j #ID of the hyperpixel
        df_features_cluster[j, 1] = classes_mask[j,1] #Class detected in the tile  
        df_features_cluster[j, 2] = hyperpixels_grayImage[j,5] #Position X
        df_features_cluster[j, 3] = hyperpixels_grayImage[j,6] #Position Y
        gray_hist = np.bincount(hyperpixels_grayImage[j,9:].astype(np.uint8), minlength = 256)
        df_features_cluster[j, 4] = entropy(gray_hist) #Entropy of the grayscale distribution
        lbp_hist = np.bincount(hyperpixels_lbp[j,9:].astype(np.uint8), minlength = 26)
        df_features_cluster[j, 5] = entropy(lbp_hist) #Entropy of the lbp distribution
        df_features_cluster[j, 6] = np.mean(hyperpixels_grayImage[j,9:])/mean_color_image #Mean grayscale color relative to the mean color of the whole image
        df_features_cluster[j, 7] = np.mean(hyperpixels_lbp[j,9:])/mean_lbp_image #Mean lbp value relative to the mean lbp value of the whole image
        df_features_cluster[j, 8] = edges_list[j,2] #proportion of pixels that belong to the edge
        df_features_cluster[j, 9] = edges_list[j,1] #Edge detected
    return df_features_cluster   


#%% Clustering


#CREATEDATASETFORCLUSTERING
#Description: This function receives a set of images and a set of masks and 
#   returns a dataframe with all the parameters for each section for each 
#   image.
#   @Inputs:
#       dfImages: dataframe with all the images to be analysed in grayscale (it
#           is as df_merged_grayImages). GRAYSCALE!!!
#       dfMask: dataframe with all the images to be analysed (it is as 
#           df_merged_gt).
#       sizeX: size (number of pixels) of the hyperpixels in X axis. By default
#           it is 18 (image of size 400x702 & 780 sections).
#       sizeY: size (number of pixels) of the hyperpixels in Y axis. By default
#           it is 20 (image of size 400x702 & 780 sections).
#       num: Number of hyperpixels in Y axis (vertical). The number of 
#           hyperpixels in the horizontal axis will be num*2-1. By default it 
#           is 20 (image of size 400x702 & 780 sections). 
#   @Outputs:
#       df_features_cluster: Dataset with all the features of the hyperpixels:
#           - 0: ID of the image
#           - 1: ID of the hyperpixel referenced to the image (in each image it 
#               starts from 0). 
#           - 2: class of the tile (0 -> background, 1 -> floor, 2 -> edge). 
#           - 3: Position in the X axis relative to the center of the image.
#           - 4: Position in the Y axis relative.
#           - 5: entropy of grayscale distribution.
#           - 6: entropy of lbp distribiution.
#           - 7: Mean value of the grayscale image relative to the mean color of the whole image.
#           - 8: Mean value of the lbp relative to the mean lbp of the whole image.
#           - 9: Edge detected
#           - 10: Proportion of pixels that are edges    
def createDatasetForClustering(dfImages, dfMask, sizeX = 18, sizeY = 20, num = 20):
    num_hyperpixels = num*(2*num-1) #Number of hyperpixels per image
    df_features_cluster = np.zeros([len(dfImages)*num_hyperpixels, 11])
    for i in range(len(dfImages)): #Analysis per image
        hyperpixels_grayImage = kgf.getHyperpixels(dfImages[i,:,:], sizeX, sizeY, num) #Get the dataframe of hyperpixels of the gray image
        classes_mask = getClass(dfMask[i,:,:], sizeX, sizeY, num) #Get classes mask        
        # Settings for LBP -> code from: https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html
        radius = 3
        n_points = 8 * radius  
        METHOD = 'uniform'
        lbp_image = local_binary_pattern(dfImages[i,:,:], n_points, radius, METHOD)  
        lbp_image = lbp_image.astype(np.uint8)
        hyperpixels_lbp = kgf.getHyperpixels(lbp_image, sizeX, sizeY, num) #Get the dataframe of hyperpixels of the lbp
        edges_list = kgf.getEdges(dfImages[i,:,:], sizeX, sizeY, num)     
        mean_color_image = np.mean(dfImages[i,:,:])
        mean_lbp_image = np.mean(lbp_image)
        for j in range(num_hyperpixels):
            df_features_cluster[j+(i*num_hyperpixels), 0] = i #ID of the image
            df_features_cluster[j+(i*num_hyperpixels), 1] = j #ID of the hyperpixel
            df_features_cluster[j+(i*num_hyperpixels), 2] = classes_mask[j,1] #Class detected in the tile  
            df_features_cluster[j+(i*num_hyperpixels), 3] = hyperpixels_grayImage[j,5] #Position X
            df_features_cluster[j+(i*num_hyperpixels), 4] = hyperpixels_grayImage[j,6] #Position Y
            gray_hist = np.bincount(hyperpixels_grayImage[j,9:].astype(np.uint8), minlength = 256)
            df_features_cluster[j+(i*num_hyperpixels), 5] = entropy(gray_hist) #Entropy of the grayscale distribution
            lbp_hist = np.bincount(hyperpixels_lbp[j,9:].astype(np.uint8), minlength = 26)
            df_features_cluster[j+(i*num_hyperpixels), 6] = entropy(lbp_hist) #Entropy of the lbp distribution
            df_features_cluster[j+(i*num_hyperpixels), 7] = np.mean(hyperpixels_grayImage[j,9:])/mean_color_image #Mean grayscale color relative to the mean color of the whole image
            df_features_cluster[j+(i*num_hyperpixels), 8] = np.mean(hyperpixels_lbp[j,9:])/mean_lbp_image #Mean lbp value relative to the mean lbp value of the whole image
            df_features_cluster[j+(i*num_hyperpixels), 9] = edges_list[j,1] #Edge detected
            df_features_cluster[j+(i*num_hyperpixels), 10] = edges_list[j,2] #Proportion of pixels that belong to the edge
    return df_features_cluster    


       
#Create the numpy dataset for clustering
df_clustering = createDatasetForClustering(df_merged_grayImages, df_merged_gt, sizeX = 18, sizeY = 20, num = 20)

#Create the pandas df
df_clustering = pd.DataFrame(data=df_clustering[0:,0:], index=range(len(df_clustering)), columns=['Image_id','Tile_id','Class', 'X', 'Y', 'Ent_gray', 'Ent_lbp', 'Mean_gray', 'Mean_lbp', 'Edge_detected', 'Proportion_edges'])  #Create pandas df.

#Divide between floor, background and edges
df_clustering_background = df_clustering.loc[df_clustering['Class'] == 0] #Background
df_clustering_floor = df_clustering.loc[df_clustering['Class'] == 1] #Floor
df_clustering_edge = df_clustering.loc[df_clustering['Class'] == 2] #Edges

#Select the features for doing clustering of each of the 3 classes
clustering_features = ['X', 'Y', 'Ent_gray', 'Ent_lbp', 'Proportion_edges'] # , 'Mean_gray', 'Mean_lbp'] #Features used for clustering

#Prepare the background points for clustering and normalize them
background_points = df_clustering_background.loc[ : , clustering_features]
background_index = background_points.index
scaler_background = preprocessing.MinMaxScaler()
background_points = scaler_background.fit_transform(background_points)
filename = 'Models/Scalers/scaler_background_clustering.sav'
pickle.dump(scaler_background, open(filename, 'wb'))
#background_points[:,0] = background_points[:,0]/2
#background_points[:,1] = background_points[:,1]/2

#Prepare the floor points for clustering and normalize them
floor_points = df_clustering_floor.loc[ : , clustering_features]
floor_index = floor_points.index
scaler_floor = preprocessing.MinMaxScaler()
floor_points = scaler_floor.fit_transform(floor_points)
filename = 'Models/Scalers/scaler_floor_clustering.sav'
pickle.dump(scaler_floor, open(filename, 'wb'))
#floor_points[:,0] = floor_points[:,0]/2
#floor_points[:,1] = floor_points[:,1]/2

#Prepare the edge points for clustering and normalize them
edge_points = df_clustering_edge.loc[ : , clustering_features]
edge_index = edge_points.index
scaler_edge = preprocessing.MinMaxScaler()
edge_points = scaler_edge.fit_transform(edge_points)
filename = 'Models/Scalers/scaler_edge_clustering.sav'
pickle.dump(scaler_edge, open(filename, 'wb'))
#edge_points[:,0] = edge_points[:,0]/2
#edge_points[:,1] = edge_points[:,1]/2

#KMEANS:

#Background
kmeans_background = KMeans(n_clusters=10)
#kmeans_background.fit(background_points)
clustered_background_kM = kmeans_background.fit_predict(background_points)
filename = 'Models/Clustering/kmeans_background.sav'
pickle.dump(kmeans_background, open(filename, 'wb'))

#Floor
kmeans_floor = KMeans(n_clusters=10)
#kmeans_floor.fit(floor_points)
clustered_floor_kM = kmeans_floor.fit_predict(floor_points)
filename = 'Models/Clustering/kmeans_floor.sav'
pickle.dump(kmeans_floor, open(filename, 'wb'))

#Edges
kmeans_edge = KMeans(n_clusters=5)
#kmeans_edge.fit(floor_points)
clustered_edge_kM = kmeans_edge.fit_predict(edge_points)
filename = 'Models/Clustering/kmeans_edge.sav'
pickle.dump(kmeans_edge, open(filename, 'wb'))

##See best number of clusters:
#Nc = range(1, 20)
#kmeans = [KMeans(n_clusters=i) for i in Nc]
#kmeans
#score = [kmeans[i].fit(background_points).score(background_points) for i in range(len(kmeans))]
#score
#plt.plot(Nc,score)
#plt.xlabel('Number of Clusters')
#plt.ylabel('Score')
#plt.title('Elbow Curve')
#plt.show()


#Add clusters to the complete dataframe
df_clustering['K_cluster'] = -1
df_clustering.loc[background_index, 'K_cluster'] = clustered_background_kM
df_clustering.loc[floor_index, 'K_cluster'] = clustered_floor_kM
df_clustering.loc[edge_index, 'K_cluster'] = clustered_edge_kM


##Test paintCluster:
Image_id = 70
image = df_merged_colorImages[Image_id,:,:,:]
clusters = np.zeros([len(df_clustering.loc[df_clustering['Image_id'] == Image_id]),3])
clusters[:,0:3] = df_clustering.loc[df_clustering['Image_id'] == Image_id].loc[:, ['Tile_id', 'K_cluster', 'Class']]
new_image = kgf.paintClusters(image, clusters,classToPaint = 0)
plt.imshow(new_image)
plt.savefig('figures/clustering/cluster_.png')
del Image_id, clusters, image, new_image

##AGGLOMERATIVE:
#
##Background
#agglomerative_background = AgglomerativeClustering(n_clusters=10, affinity = 'euclidean', linkage = 'ward')
#clustered_background_Agg = agglomerative_background.fit_predict(background_points)
##Floor
#agglomerative_floor = AgglomerativeClustering(n_clusters=10, affinity = 'euclidean', linkage = 'ward')
#clustered_floor_Agg = agglomerative_floor.fit_predict(floor_points)
##Edges
#agglomerative_edge= AgglomerativeClustering(n_clusters=5, affinity = 'euclidean', linkage = 'ward')
#clustered_edge_Agg = agglomerative_edge.fit_predict(edge_points)


del clustered_background_kM, clustered_floor_kM, clustered_edge_kM, df_clustering_background, df_clustering_floor, df_clustering_edge, background_points, floor_points, edge_points


#%% Prediction -> Create dataset for prediction


#CREATEDATASETFORPREDICTION
#Description: This function receives a set of images and a set of masks and 
#   returns a dataframe with all the parameters for each section for each 
#   image.
#   @Inputs:
#       dfImages: dataframe with all the images to be analysed in grayscale (it
#           is as df_merged_grayImages). GRAYSCALE!!!
#       dfMask: dataframe with all the images to be analysed (it is as 
#           df_merged_gt).
#       sizeX: size (number of pixels) of the hyperpixels in X axis. By default
#           it is 18 (image of size 400x702 & 780 sections).
#       sizeY: size (number of pixels) of the hyperpixels in Y axis. By default
#           it is 20 (image of size 400x702 & 780 sections).
#       num: Number of hyperpixels in Y axis (vertical). The number of 
#           hyperpixels in the horizontal axis will be num*2-1. By default it 
#           is 20 (image of size 400x702 & 780 sections). 
#   @Outputs:
#       df_features: Pandas df with all the features of the hyperpixels:
#           - 0: ID of the image.
#           - 1: ID of the tile referenced to the image (in each image it 
#               starts from 0). 
#           - 2: Class of the tile (0 -> background, 1 -> floor, 2 -> edge).
#           - 3: Position X of the tile.
#           - 4: Position Y of the tile.
#           - 5: Edge detected in tile.
#           - 6: Proportion of pixels that are edges in tile.
#           - 7: entropy of grayscale distribution.
#           - 8: entropy of lbp distribiution.
#           - 9: Mean value of the grayscale image relative to the mean color of the whole image.
#           - 10: Mean value of the lbp relative to the mean lbp of the whole image.
#           - 11: Section of analysis of the tile ID1 (1), ID2 (2) or ID3 (3)
#           - 12: Class of previous tile.
#           - 13: If line detected -> angle of the line.
#           - 14: If line detected -> length of line.
#           - 15: If line detected -> number of lines over this line with the 
#              same slope (non-vertical lines).
#           - 16: If line detected -> most similar slope of those lines that are 
#              over it.
#           - 17: Biggest gradient change (Laplacian absolute value). 
#           - 18: Smallest area after Felzenszwalb segmentation to which the 
#               hyperpixel belongs.
#           - 19: Grayscale intensity relative to the neightbour intensity
#           - 20: max gradient change over the hyperpixel.
#           - 21: max gradient change under the hyperpixel.
#           - 22: max gradient change at right of the hyperpixel.
#           - 23: max gradient change at left of the hyperpixel.
#           - 24: ID of the hyperpixel A1 (see notebook)
#           - 25: ID of the hyperpixel A2 (see notebook)
#           - 26: ID of the hyperpixel A3 (see notebook)
#           - 27: ID of the hyperpixel A4 (see notebook)
#           - 28: ID of the hyperpixel A5 (see notebook)
#           - 29: ID of the hyperpixel A9 (see notebook)
#           - 30: ID of the hyperpixel A6 (see notebook)
#           - 31: ID of the hyperpixel A8 (see notebook)
#           - 32: ID of the hyperpixel B1 (see notebook)
#           - 33: ID of the hyperpixel B2 (see notebook)
#           - 34: ID of the hyperpixel B3 (see notebook)
#           - 35: ID of the hyperpixel B4 (see notebook)
#           - 36: ID of the hyperpixel B5 (see notebook)
#           - 37: ID of the hyperpixel B9 (see notebook)
#           - 38: Class of A1.
#           - 39: Wasserstein distance with grayscale distribution of hyperpixel A1.
#           - 40: Wasserstein distance with lbp distribution of hyperpixel A1.
#           - 41: Proportion of pixels that are edges in hyperpixel A1.
#           - 42: Class of A2.
#           - 43: Wasserstein distance with grayscale distribution of hyperpixel A2.
#           - 44: Wasserstein distance with lbp distribution of hyperpixel A2.
#           - 45: Proportion of pixels that are edges in hyperpixel A2.
#           - 46: Class of A3.
#           - 47: Wasserstein distance with grayscale distribution of hyperpixel A3.
#           - 48: Wasserstein distance with lbp distribution of hyperpixel A3.
#           - 49: Proportion of pixels that are edges in hyperpixel A3.
#           - 50: Class of A4.
#           - 51: Wasserstein distance with grayscale distribution of hyperpixel A4.
#           - 52: Wasserstein distance with lbp distribution of hyperpixel A4.
#           - 53: Proportion of pixels that are edges in hyperpixel A4.
#           - 54: Class of A5.
#           - 55: Wasserstein distance with grayscale distribution of hyperpixel A5.
#           - 56: Wasserstein distance with lbp distribution of hyperpixel A5.
#           - 57: Proportion of pixels that are edges in hyperpixel A5.
#           - 58: Class of A9.
#           - 59: Wasserstein distance with grayscale distribution of hyperpixel A9.
#           - 60: Wasserstein distance with lbp distribution of hyperpixel A9.
#           - 61: Proportion of pixels that are edges in hyperpixel A9.
#           - 62: Class of A6.
#           - 63: Class of A8.
#           - 64: Wasserstein distance with grayscale distribution of hyperpixel B1.
#           - 65: Wasserstein distance with lbp distribution of hyperpixel B1.
#           - 66: Proportion of pixels that are edges in hyperpixel B1.
#           - 67: Wasserstein distance with grayscale distribution of hyperpixel B2.
#           - 68: Wasserstein distance with lbp distribution of hyperpixel B2.
#           - 69: Proportion of pixels that are edges in hyperpixel B2.
#           - 70: Wasserstein distance with grayscale distribution of hyperpixel B3.
#           - 71: Wasserstein distance with lbp distribution of hyperpixel B3.
#           - 72: Proportion of pixels that are edges in hyperpixel B3.
#           - 73: Wasserstein distance with grayscale distribution of hyperpixel B4.
#           - 74: Wasserstein distance with lbp distribution of hyperpixel B4.
#           - 75: Proportion of pixels that are edges in hyperpixel B4.
#           - 76: Wasserstein distance with grayscale distribution of hyperpixel B5.
#           - 77: Wasserstein distance with lbp distribution of hyperpixel B5.
#           - 78: Proportion of pixels that are edges in hyperpixel B5.
#           - 79: Wasserstein distance with grayscale distribution of hyperpixel B9.
#           - 80: Wasserstein distance with lbp distribution of hyperpixel B9.
#           - 81: Proportion of pixels that are edges in hyperpixel B9.
#   @Call:
#       createDatasetForPrediction(df_merged_grayImages, df_merged_gt, kmeans_floor, scaler_floor, kmeans_background, scaler_background, kmeans_edge, scaler_edge, sizeX = 18, sizeY = 20, num = 20, clustering_features = clustering_features)
def createDatasetForPrediction(dfImages, dfMask, sizeX = 18, sizeY = 20, num = 20):
    #clustering_features = ['X', 'Y', 'Ent_gray', 'Ent_lbp', 'Proportion_edges']# , 'Mean_gray', 'Mean_lbp'] #Features used for clustering
    neighbors_ID = kgf.getNeighborsId(dfImages[0]) #First get the ID of all the neighbors. As all the images are equaly segmented, it is only needed to get the neighbors of one image. Then it is applicable to the rest of the images.
    num_hyperpixels = num*(2*num-1) #Number of hyperpixels per image
    df_features = np.zeros([len(dfImages)*num_hyperpixels,82])
    for i in range(len(dfImages)): #Analysis per image
        print('Image:', i)
        dim = (int(sizeX*(2*num-1)), int(sizeY*num))
        classes = getClass(dfMask[i,:,:], sizeX, sizeY, num) #Get classes   
        image = cv2.resize(dfImages[i,:,:], dim, interpolation = cv2.INTER_AREA) #Reshape the image with the used standards
        #image = kgf.increaseContrast(image, alpha = 2, beta = 0)
        hyperpixels_grayImage = kgf.getHyperpixels(image, sizeX, sizeY, num) #Get the dataframe of hyperpixels of the gray image
        local_features = localFeaturesImage(image, dfMask[i,:,:], sizeX, sizeY, num) #Get dataset for clustering
        # Settings for LBP -> code from: https://scikit-image.org/docs/dev/auto_examples/features_detection/plot_local_binary_pattern.html
        radius = 3
        n_points = 8 * radius  
        METHOD = 'uniform'
        lbp_image = local_binary_pattern(image, n_points, radius, METHOD)  
        lbp_image = lbp_image.astype(np.uint8)
        hyperpixels_lbp = kgf.getHyperpixels(lbp_image, sizeX, sizeY, num) #Get the dataframe of hyperpixels of the lbp
        lines, mask_lines = kgf.getLines(image, sizeX, sizeY, num)
        hyperpixels_lines = kgf.getHyperpixels(mask_lines, sizeX, sizeY, num) #Get the dataframe of hyperpixels of the lines
        gradient_image = abs(kgf.getGradientMagnitude(image,cv2.CV_64F))
        hyperpixels_grad = kgf.getHyperpixels(gradient_image, sizeX, sizeY, num) #Get the dataframe of hyperpixels of the laplacian
        f_segmentation, mask_f = kgf.getFelzenszwalb(image, sizeX, sizeY, num)
        mask_f = mask_f.astype(int)
        hyperpixels_f_segmentation = kgf.getHyperpixels(mask_f, sizeX, sizeY, num) #Get the dataframe of hyperpixels of the F. segmentation
        for j in range(num_hyperpixels):
            df_features[j+(i*num_hyperpixels), 0] = i #Image ID
            df_features[j+(i*num_hyperpixels), 1] = j #Tile Id
            df_features[j+(i*num_hyperpixels), 2] = classes[j,1] #Class of the tile
            df_features[j+(i*num_hyperpixels), 3] = hyperpixels_grayImage[j,5] #Position X of the tile
            df_features[j+(i*num_hyperpixels), 4] = hyperpixels_grayImage[j,6] #Position Y of the tile
            df_features[j+(i*num_hyperpixels), 5] = local_features[j,9] #Edges detected
            df_features[j+(i*num_hyperpixels), 6] = local_features[j,8] #Proportion of edges
            df_features[j+(i*num_hyperpixels), 7] = local_features[j,4] #Entropy of gray image
            df_features[j+(i*num_hyperpixels), 8] = local_features[j,5] #Entropy of lbp       
            df_features[j+(i*num_hyperpixels), 9] = local_features[j,6] #Mean gray value relative to the image's values
            df_features[j+(i*num_hyperpixels), 10] = local_features[j,7] #Mean lbp value relative to the image's values
            df_features[j+(i*num_hyperpixels), 11] = hyperpixels_grayImage[j,8] #Section of analysis of the tile
            if not math.isnan(hyperpixels_grayImage[j,7]): #Not 0
                ID_previousTile = int(hyperpixels_grayImage[j,7]) #ID of the previous tile
                df_features[j+(i*num_hyperpixels), 12] = classes[ID_previousTile,1] #Class of the previous tile               
            else:
                df_features[j+(i*num_hyperpixels), 12] = float('nan') #Class of the previous tile       
            if np.count_nonzero(hyperpixels_lines[j,9:]) != 0: #If a line is detected
                ID_line = max(hyperpixels_lines[j,9:]) #Take the maximum length line
                df_features[j+(i*num_hyperpixels), 13] = lines.loc[lines['id_line'] == ID_line, 'degrees'] #Degrees -> Angle of the line in degrees
                df_features[j+(i*num_hyperpixels), 14] = lines.loc[lines['id_line'] == ID_line, 'length'] #length
                df_features[j+(i*num_hyperpixels), 15] = lines.loc[lines['id_line'] == ID_line, 'number_lines_over'] #number_lines_over
                df_features[j+(i*num_hyperpixels), 16] = lines.loc[lines['id_line'] == ID_line, 'closest_slope_over'] #closest_slope_over
            else:
                df_features[j+(i*num_hyperpixels), 13] = float('nan') #Degrees
                df_features[j+(i*num_hyperpixels), 14] = float('nan') #Lenght
                df_features[j+(i*num_hyperpixels), 15] = float('nan') #Number of lines over
                df_features[j+(i*num_hyperpixels), 16] = float('nan') #Closest slope over
            df_features[j+(i*num_hyperpixels), 17] = np.max(hyperpixels_grad[j,9:]) #Maximum gradient change
            f_ids = np.unique(hyperpixels_f_segmentation[j,9:]) #Get all the unique segmentations
            f_ids = f_ids.astype(int)
            df_features[j+(i*num_hyperpixels), 18] = np.min(f_segmentation[f_ids,1]) #Get the minimum size of the F segmentation.
            IDs_neighbour = np.array([(neighbors_ID.loc[j,'A1']), (neighbors_ID.loc[j,'A2']), (neighbors_ID.loc[j,'A3']), (neighbors_ID.loc[j,'A4']), (neighbors_ID.loc[j,'A5']), (neighbors_ID.loc[j,'A9']), (neighbors_ID.loc[j,'B1']), (neighbors_ID.loc[j,'B2']), (neighbors_ID.loc[j,'B3']), (neighbors_ID.loc[j,'B4']), (neighbors_ID.loc[j,'B5']), (neighbors_ID.loc[j,'B9'])])
            IDs_neighbour = IDs_neighbour[~np.isnan(IDs_neighbour)]
            IDs_neighbour = IDs_neighbour.astype(int)            
            df_features[j+(i*num_hyperpixels), 19] = local_features[j,6]/np.mean(local_features[IDs_neighbour,6]) #Mean grayscale value relative to the neighbour
            max_grad_up, max_grad_down, max_grad_right, max_grad_left = kgf.getSurroundingGradient(hyperpixels_grad, j, sizeX = 18, sizeY = 20, num = 20)
            df_features[j+(i*num_hyperpixels), 20] = max_grad_up #Maximum gradient change over the hyperpixel
            df_features[j+(i*num_hyperpixels), 21] = max_grad_down #Maximum gradient change under the hyperpixel
            df_features[j+(i*num_hyperpixels), 22] = max_grad_right #Maximum gradient change at right of the hyperpixel
            df_features[j+(i*num_hyperpixels), 23] = max_grad_left #Maximum gradient change at left of the hyperpixel           
            if not math.isnan(neighbors_ID.loc[j,'A1']):
                df_features[j+(i*num_hyperpixels), 24] = int(neighbors_ID.loc[j,'A1']) #A1 Id
                df_features[j+(i*num_hyperpixels), 38] = classes[int(neighbors_ID.loc[j,'A1']),1] #A1 class
                df_features[j+(i*num_hyperpixels), 39] = wasserstein_distance(np.bincount(hyperpixels_grayImage[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_grayImage[int(neighbors_ID.loc[j,'A1']),9:].astype(np.uint8), minlength = 256)) #A1 W gray
                df_features[j+(i*num_hyperpixels), 40] = wasserstein_distance(np.bincount(hyperpixels_lbp[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_lbp[int(neighbors_ID.loc[j,'A1']),9:].astype(np.uint8), minlength = 256)) #A1 W lbp
                df_features[j+(i*num_hyperpixels), 41] = local_features[int(neighbors_ID.loc[j,'A1']),8] #A1 proportion of edges
            else:
                df_features[j+(i*num_hyperpixels), 24] = float('nan') #A1 Id
                df_features[j+(i*num_hyperpixels), 38] = float('nan') #A1 class
                df_features[j+(i*num_hyperpixels), 39] = float('nan') #A1 W gray
                df_features[j+(i*num_hyperpixels), 40] = float('nan') #A1 W lbp
                df_features[j+(i*num_hyperpixels), 41] = float('nan') #A1 proportion of edges
            if not math.isnan(neighbors_ID.loc[j,'A2']):
                df_features[j+(i*num_hyperpixels), 25] = int(neighbors_ID.loc[j,'A2']) #A2 Id
                df_features[j+(i*num_hyperpixels), 42] = classes[int(neighbors_ID.loc[j,'A2']),1] #A2 class
                df_features[j+(i*num_hyperpixels), 43] = wasserstein_distance(np.bincount(hyperpixels_grayImage[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_grayImage[int(neighbors_ID.loc[j,'A2']),9:].astype(np.uint8), minlength = 256)) #A2 W gray
                df_features[j+(i*num_hyperpixels), 44] = wasserstein_distance(np.bincount(hyperpixels_lbp[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_lbp[int(neighbors_ID.loc[j,'A2']),9:].astype(np.uint8), minlength = 256)) #A2 W lbp
                df_features[j+(i*num_hyperpixels), 45] = local_features[int(neighbors_ID.loc[j,'A2']),8] #A2 proportion of edges
            else:
                df_features[j+(i*num_hyperpixels), 25] = float('nan') #A2 Id
                df_features[j+(i*num_hyperpixels), 42] = float('nan') #A2 class
                df_features[j+(i*num_hyperpixels), 43] = float('nan') #A2 W gray
                df_features[j+(i*num_hyperpixels), 44] = float('nan') #A2 W lbp
                df_features[j+(i*num_hyperpixels), 45] = float('nan') #A2 proportion of edges
            if not math.isnan(neighbors_ID.loc[j,'A3']):
                df_features[j+(i*num_hyperpixels), 26] = int(neighbors_ID.loc[j,'A3']) #A3 Id
                df_features[j+(i*num_hyperpixels), 46] = classes[int(neighbors_ID.loc[j,'A3']),1] #A3 class
                df_features[j+(i*num_hyperpixels), 47] = wasserstein_distance(np.bincount(hyperpixels_grayImage[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_grayImage[int(neighbors_ID.loc[j,'A3']),9:].astype(np.uint8), minlength = 256)) #A3 W gray
                df_features[j+(i*num_hyperpixels), 48] = wasserstein_distance(np.bincount(hyperpixels_lbp[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_lbp[int(neighbors_ID.loc[j,'A3']),9:].astype(np.uint8), minlength = 256)) #A3 W lbp
                df_features[j+(i*num_hyperpixels), 49] = local_features[int(neighbors_ID.loc[j,'A3']),8] #A3 proportion of edges
            else:
                df_features[j+(i*num_hyperpixels), 26] = float('nan') #A3 Id
                df_features[j+(i*num_hyperpixels), 46] = float('nan') #A3 class
                df_features[j+(i*num_hyperpixels), 47] = float('nan') #A3 W gray
                df_features[j+(i*num_hyperpixels), 48] = float('nan') #A3 W lbp
                df_features[j+(i*num_hyperpixels), 49] = float('nan') #A3 proportion of edges  
            if not math.isnan(neighbors_ID.loc[j,'A4']):
                df_features[j+(i*num_hyperpixels), 27] = int(neighbors_ID.loc[j,'A4']) #A4 Id
                df_features[j+(i*num_hyperpixels), 50] = classes[int(neighbors_ID.loc[j,'A4']),1] #A4 class
                df_features[j+(i*num_hyperpixels), 51] = wasserstein_distance(np.bincount(hyperpixels_grayImage[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_grayImage[int(neighbors_ID.loc[j,'A4']),9:].astype(np.uint8), minlength = 256)) #A4 W gray
                df_features[j+(i*num_hyperpixels), 52] = wasserstein_distance(np.bincount(hyperpixels_lbp[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_lbp[int(neighbors_ID.loc[j,'A4']),9:].astype(np.uint8), minlength = 256)) #A4 W lbp
                df_features[j+(i*num_hyperpixels), 53] = local_features[int(neighbors_ID.loc[j,'A4']),8] #A4 proportion of edges
            else:
                df_features[j+(i*num_hyperpixels), 27] = float('nan') #A4 ID
                df_features[j+(i*num_hyperpixels), 50] = float('nan') #A4 class
                df_features[j+(i*num_hyperpixels), 51] = float('nan') #A4 W gray
                df_features[j+(i*num_hyperpixels), 52] = float('nan') #A4 W lbp
                df_features[j+(i*num_hyperpixels), 53] = float('nan') #A4 proportion of edges  
            if not math.isnan(neighbors_ID.loc[j,'A5']):
                df_features[j+(i*num_hyperpixels), 28] = int(neighbors_ID.loc[j,'A5']) #A5 Id
                df_features[j+(i*num_hyperpixels), 54] = classes[int(neighbors_ID.loc[j,'A5']),1] #A5 class
                df_features[j+(i*num_hyperpixels), 55] = wasserstein_distance(np.bincount(hyperpixels_grayImage[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_grayImage[int(neighbors_ID.loc[j,'A5']),9:].astype(np.uint8), minlength = 256)) #A5 W gray
                df_features[j+(i*num_hyperpixels), 56] = wasserstein_distance(np.bincount(hyperpixels_lbp[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_lbp[int(neighbors_ID.loc[j,'A5']),9:].astype(np.uint8), minlength = 256)) #A5 W lbp
                df_features[j+(i*num_hyperpixels), 57] = local_features[int(neighbors_ID.loc[j,'A5']),8] #A5 proportion of edges
            else:
                df_features[j+(i*num_hyperpixels), 28] = float('nan') #A5 Id
                df_features[j+(i*num_hyperpixels), 54] = float('nan') #A5 class
                df_features[j+(i*num_hyperpixels), 55] = float('nan') #A5 W gray
                df_features[j+(i*num_hyperpixels), 56] = float('nan') #A5 W lbp
                df_features[j+(i*num_hyperpixels), 57] = float('nan') #A5 W edges
            if not math.isnan(neighbors_ID.loc[j,'A9']):
                df_features[j+(i*num_hyperpixels), 29] = int(neighbors_ID.loc[j,'A9']) #A9 Id
                df_features[j+(i*num_hyperpixels), 58] = classes[int(neighbors_ID.loc[j,'A9']),1] #A9 class
                df_features[j+(i*num_hyperpixels), 59] = wasserstein_distance(np.bincount(hyperpixels_grayImage[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_grayImage[int(neighbors_ID.loc[j,'A9']),9:].astype(np.uint8), minlength = 256)) #A9 W gray
                df_features[j+(i*num_hyperpixels), 60] = wasserstein_distance(np.bincount(hyperpixels_lbp[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_lbp[int(neighbors_ID.loc[j,'A9']),9:].astype(np.uint8), minlength = 256)) #A9 W lbp
                df_features[j+(i*num_hyperpixels), 61] = local_features[int(neighbors_ID.loc[j,'A9']),8] #A9 proportion of edges
            else:
                df_features[j+(i*num_hyperpixels), 29] = float('nan') #A9 Id
                df_features[j+(i*num_hyperpixels), 58] = float('nan') #A9 class
                df_features[j+(i*num_hyperpixels), 59] = float('nan') #A9 W gray
                df_features[j+(i*num_hyperpixels), 60] = float('nan') #A9 W lbp
                df_features[j+(i*num_hyperpixels), 61] = float('nan') #A9 proportion of edges
            if not math.isnan(neighbors_ID.loc[j,'A6']):
                df_features[j+(i*num_hyperpixels), 30] = int(neighbors_ID.loc[j,'A6']) #A6 Id
                df_features[j+(i*num_hyperpixels), 62] = classes[int(neighbors_ID.loc[j,'A6']),1] #A6 class
            else:
                df_features[j+(i*num_hyperpixels), 30] = float('nan') #A6 Id
                df_features[j+(i*num_hyperpixels), 62] = float('nan') #A6 class
            if not math.isnan(neighbors_ID.loc[j,'A8']):
                df_features[j+(i*num_hyperpixels), 31] = int(neighbors_ID.loc[j,'A8']) #A8 Id
                df_features[j+(i*num_hyperpixels), 63] = classes[int(neighbors_ID.loc[j,'A8']),1] #A8 class
            else:
                df_features[j+(i*num_hyperpixels), 31] = float('nan') #A8 Id
                df_features[j+(i*num_hyperpixels), 63] = float('nan') #A8 class
            if not math.isnan(neighbors_ID.loc[j,'B1']):
                df_features[j+(i*num_hyperpixels), 32] = int(neighbors_ID.loc[j,'B1']) #B1 Id
                df_features[j+(i*num_hyperpixels), 64] = wasserstein_distance(np.bincount(hyperpixels_grayImage[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_grayImage[int(neighbors_ID.loc[j,'B1']),9:].astype(np.uint8), minlength = 256)) #B1 W gray
                df_features[j+(i*num_hyperpixels), 65] = wasserstein_distance(np.bincount(hyperpixels_lbp[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_lbp[int(neighbors_ID.loc[j,'B1']),9:].astype(np.uint8), minlength = 256)) #B1 W lbp
                df_features[j+(i*num_hyperpixels), 66] = local_features[int(neighbors_ID.loc[j,'B1']),8] #B1 proportion of edges
            else:
                df_features[j+(i*num_hyperpixels), 32] = float('nan') #B1 Id
                df_features[j+(i*num_hyperpixels), 64] = float('nan') #B1 W gray
                df_features[j+(i*num_hyperpixels), 65] = float('nan') #B1 W lbp
                df_features[j+(i*num_hyperpixels), 66] = float('nan') #B1 proportion of edges
            if not math.isnan(neighbors_ID.loc[j,'B2']):
                df_features[j+(i*num_hyperpixels), 33] = int(neighbors_ID.loc[j,'B2']) #B2 Id
                df_features[j+(i*num_hyperpixels), 67] = wasserstein_distance(np.bincount(hyperpixels_grayImage[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_grayImage[int(neighbors_ID.loc[j,'B2']),9:].astype(np.uint8), minlength = 256)) #B2 W gray
                df_features[j+(i*num_hyperpixels), 68] = wasserstein_distance(np.bincount(hyperpixels_lbp[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_lbp[int(neighbors_ID.loc[j,'B2']),9:].astype(np.uint8), minlength = 256)) #B2 W lbp
                df_features[j+(i*num_hyperpixels), 69] = local_features[int(neighbors_ID.loc[j,'B2']),8] #B2 porportion of edges
            else:
                df_features[j+(i*num_hyperpixels), 33] = float('nan') #B2 Id
                df_features[j+(i*num_hyperpixels), 67] = float('nan') #B2 W gray
                df_features[j+(i*num_hyperpixels), 68] = float('nan') #B2 W lbp
                df_features[j+(i*num_hyperpixels), 69] = float('nan') #B2 proportion of edges
            if not math.isnan(neighbors_ID.loc[j,'B3']):
                df_features[j+(i*num_hyperpixels), 34] = int(neighbors_ID.loc[j,'B3']) #B3 Id
                df_features[j+(i*num_hyperpixels), 70] = wasserstein_distance(np.bincount(hyperpixels_grayImage[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_grayImage[int(neighbors_ID.loc[j,'B3']),9:].astype(np.uint8), minlength = 256)) #B3 W gray
                df_features[j+(i*num_hyperpixels), 71] = wasserstein_distance(np.bincount(hyperpixels_lbp[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_lbp[int(neighbors_ID.loc[j,'B3']),9:].astype(np.uint8), minlength = 256)) #B3 W lbp
                df_features[j+(i*num_hyperpixels), 72] = local_features[int(neighbors_ID.loc[j,'B3']),8] #B3 porportion of edges
            else:
                df_features[j+(i*num_hyperpixels), 34] = float('nan') #B3 Id
                df_features[j+(i*num_hyperpixels), 70] = float('nan') #B3 W gray
                df_features[j+(i*num_hyperpixels), 71] = float('nan') #B3 W lbp
                df_features[j+(i*num_hyperpixels), 72] = float('nan') #B3 proportion of edges
            if not math.isnan(neighbors_ID.loc[j,'B4']):
                df_features[j+(i*num_hyperpixels), 35] = int(neighbors_ID.loc[j,'B4']) #B4 Id
                df_features[j+(i*num_hyperpixels), 73] = wasserstein_distance(np.bincount(hyperpixels_grayImage[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_grayImage[int(neighbors_ID.loc[j,'B4']),9:].astype(np.uint8), minlength = 256)) #B4 W gray
                df_features[j+(i*num_hyperpixels), 74] = wasserstein_distance(np.bincount(hyperpixels_lbp[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_lbp[int(neighbors_ID.loc[j,'B4']),9:].astype(np.uint8), minlength = 256)) #B4 W lbp
                df_features[j+(i*num_hyperpixels), 75] = local_features[int(neighbors_ID.loc[j,'B4']),8] #B4 proportion of edges
            else:
                df_features[j+(i*num_hyperpixels), 35] = float('nan') #B4 Id
                df_features[j+(i*num_hyperpixels), 73] = float('nan') #B4_W_gray
                df_features[j+(i*num_hyperpixels), 74] = float('nan') #B4 W lbp
                df_features[j+(i*num_hyperpixels), 75] = float('nan') #B4 proportion of edges
            if not math.isnan(neighbors_ID.loc[j,'B5']):
                df_features[j+(i*num_hyperpixels), 36] = int(neighbors_ID.loc[j,'B5']) #B5 Id
                df_features[j+(i*num_hyperpixels), 76] = wasserstein_distance(np.bincount(hyperpixels_grayImage[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_grayImage[int(neighbors_ID.loc[j,'B5']),9:].astype(np.uint8), minlength = 256)) #B5 W gray
                df_features[j+(i*num_hyperpixels), 77] = wasserstein_distance(np.bincount(hyperpixels_lbp[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_lbp[int(neighbors_ID.loc[j,'B5']),9:].astype(np.uint8), minlength = 256)) #B5 W lbp
                df_features[j+(i*num_hyperpixels), 78] = local_features[int(neighbors_ID.loc[j,'B5']),8] #B5 proportion of edges
            else:
                df_features[j+(i*num_hyperpixels), 36] = float('nan') #B5 Id
                df_features[j+(i*num_hyperpixels), 76] = float('nan') #B5 W gray
                df_features[j+(i*num_hyperpixels), 77] = float('nan') #B5 W lbp
                df_features[j+(i*num_hyperpixels), 78] = float('nan') #B5 propotion of edges
            if not math.isnan(neighbors_ID.loc[j,'B9']):
                df_features[j+(i*num_hyperpixels), 37] = int(neighbors_ID.loc[j,'B9']) #B9 Id
                df_features[j+(i*num_hyperpixels), 79] = wasserstein_distance(np.bincount(hyperpixels_grayImage[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_grayImage[int(neighbors_ID.loc[j,'B9']),9:].astype(np.uint8), minlength = 256)) #B9 W gray
                df_features[j+(i*num_hyperpixels), 80] = wasserstein_distance(np.bincount(hyperpixels_lbp[j,9:].astype(np.uint8), minlength = 256), np.bincount(hyperpixels_lbp[int(neighbors_ID.loc[j,'B9']),9:].astype(np.uint8), minlength = 256)) #B9 W lbp
                df_features[j+(i*num_hyperpixels), 81] = local_features[int(neighbors_ID.loc[j,'B9']),8] #B9 proportion of edges
            else:
                df_features[j+(i*num_hyperpixels), 37] = float('nan') #B9 Id
                df_features[j+(i*num_hyperpixels), 79] = float('nan') #B9 W gray
                df_features[j+(i*num_hyperpixels), 80] = float('nan') #B9 W lbp
                df_features[j+(i*num_hyperpixels), 81] = float('nan') #B9 proportion of edges  
    df_features = pd.DataFrame(data = df_features,index = range(len(dfImages)*num_hyperpixels) ,columns=['image_ID', 'tile_ID', 'tile_class', 'tile_X', 'tile_Y', 'tile_edge_detected', 'tile_proportion_edges', 'tile_entropy_gray', 'tile_entropy_lbp', 'tile_mean_gray', 'tile_mean_lbp', 'tile_section_of_analysis', 'class_previous_tile', 'degrees_line', 'length_line', 'num_lines_over', 'similar_slope', 'gray_gradient', 'f_area_size', 'gray_intensity_neighbor', 'max_grad_up', 'max_grad_down', 'max_grad_right', 'max_grad_left', 'A1_ID', 'A2_ID', 'A3_ID', 'A4_ID', 'A5_ID', 'A9_ID', 'A6_ID', 'A8_ID', 'B1_ID', 'B2_ID', 'B3_ID', 'B4_ID', 'B5_ID', 'B9_ID', 'A1_class', 'A1_W_gray', 'A1_W_lbp', 'A1_proportion_edges','A2_class', 'A2_W_gray', 'A2_W_lbp', 'A2_proportion_edges', 'A3_class', 'A3_W_gray', 'A3_W_lbp', 'A3_proportion_edges', 'A4_class', 'A4_W_gray', 'A4_W_lbp', 'A4_proportion_edges', 'A5_class', 'A5_W_gray', 'A5_W_lbp', 'A5_proportion_edges', 'A9_class', 'A9_W_gray', 'A9_W_lbp', 'A9_proportion_edges', 'A6_class', 'A8_class', 'B1_W_gray', 'B1_W_lbp', 'B1_proportion_edges', 'B2_W_gray', 'B2_W_lbp', 'B2_proportion_edges', 'B3_W_gray', 'B3_W_lbp', 'B3_proportion_edges', 'B4_W_gray', 'B4_W_lbp', 'B4_proportion_edges', 'B5_W_gray', 'B5_W_lbp', 'B5_proportion_edges', 'B9_W_gray', 'B9_W_lbp', 'B9_proportion_edges'])                 
    
    return df_features   

df_Prediction = createDatasetForPrediction(df_merged_grayImages, df_merged_gt, sizeX = 18, sizeY = 20, num = 20)

df_Prediction.to_pickle('datasets/df_Prediction_global.pickle')

