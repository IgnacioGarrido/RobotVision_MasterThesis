# SMRobot project
# Ignacio Garrido Botella
# DEPRECATED

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn import preprocessing
import pickle
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
import scipy.io
import cv2
import pydotplus
import random
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import math
from sklearn.utils import shuffle

os.chdir("/Users/ignacio/Documents/Universidad/Master/Segundo/SegundoSemestre/MasterThesisMoAI/Project/smrobot_py_knowledgeGraph")
import kGraph_functions as kgf
# %% Links

# Decision tree: https://www.datacamp.com/community/tutorials/decision-tree-classification-python
# Feature selection: https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
# Decision trees on time series: https://codeburst.io/machine-learning-basic-understand-the-limit-of-trees-with-time-series-data-53875b1f1ef8
# Feature importance: https://medium.com/@srnghn/the-mathematics-of-decision-trees-random-forest-and-feature-importance-in-scikit-learn-and-spark-f2861df67e3
# Missing data: https://machinelearningmastery.com/handle-missing-data-python/

#%% Load Dataset

infile = open('datasets/df_Prediction.pickle','rb')
df_Prediction = pickle.load(infile)
infile.close()

#Load images
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


infile = open('datasets/df_Prediction_global.pickle','rb')
df_Prediction = pickle.load(infile)
infile.close()

del  df_1_gt, df_2_gt, df_3_gt, df_4_gt, df_1_im, df_2_im, df_3_im, df_4_im, df_1_im_gray, df_2_im_gray, df_3_im_gray, df_4_im_gray, i

#%% Select features

#!!!!!!!!!!
#ATTENTION:
#   In this version the feature similar_slope have been eliminated in all the cases.
#   TODO: Check features tile_mean_gray, tile_mean_lbp
#   I have eliminated 'degrees_line', 'length_line', 'num_lines_over'

#SELECTFEATURES
#Description: This function receives the id of the wanted features and returns 
#   features for the training.
#   @Inputs:
#       feat_id: 
#           0: Returns only local knowledge features.
#           1: Returns only global knowledge features.
#           2: Returns all features.
#           3: Only classes of previous tile, A1, A2, A5, A6, A8  - local 
#               knowledge -> For correct circular prediction.
#           4: Only classes of previous tile, A1, A2, A5, A6, A8  - All -> For 
#               correct circular prediction.
#       data: dataset.
#       
#   @Outputs:
#       features_X: Those features for X_train/X_test
#       features_Y: Those features for y_train/y_test
#       columns_to_scale: Those columns' names that have to be scaled.
#       columns_to_categorical: Those columns' names that are categorical 
#           variables.
def selectFeatures(feat_id, data = df_Prediction):
    if feat_id == 0: # Only local knowledge
        features_X = ['tile_X', 'tile_Y', 'tile_proportion_edges', 'tile_entropy_gray', 'gray_intensity_neighbor', 'tile_mean_lbp', 'class_previous_tile', 'A1_class', 'A1_W_gray', 'A1_W_lbp', 'A1_proportion_edges','A2_class', 'A2_W_gray', 'A2_W_lbp', 'A2_proportion_edges', 'A3_class', 'A3_W_gray', 'A3_W_lbp', 'A3_proportion_edges','A4_class', 'A4_W_gray', 'A4_W_lbp', 'A4_proportion_edges', 'A5_class', 'A5_W_gray', 'A5_W_lbp', 'A5_proportion_edges', 'A9_class', 'A9_W_gray', 'A9_W_lbp', 'A9_proportion_edges', 'A6_class', 'A8_class', 'B1_W_gray', 'B1_W_lbp', 'B1_proportion_edges', 'B2_W_gray', 'B2_W_lbp', 'B2_proportion_edges', 'B3_W_gray', 'B3_W_lbp', 'B3_proportion_edges', 'B4_W_gray', 'B4_W_lbp', 'B4_proportion_edges', 'B5_W_gray', 'B5_W_lbp', 'B5_proportion_edges', 'B9_W_gray', 'B9_W_lbp', 'B9_proportion_edges']                
        features_Y = ['tile_class']
        columns_to_scale = ['tile_X', 'tile_Y', 'tile_proportion_edges', 'tile_entropy_gray', 'tile_entropy_lbp', 'gray_intensity_neighbor', 'A1_W_gray', 'A1_W_lbp', 'A1_proportion_edges', 'A2_W_gray', 'A2_W_lbp', 'A2_proportion_edges', 'A3_W_gray', 'A3_W_lbp', 'A3_proportion_edges', 'A4_W_gray', 'A4_W_lbp', 'A4_proportion_edges', 'A5_W_gray', 'A5_W_lbp', 'A5_proportion_edges', 'A9_W_gray', 'A9_W_lbp', 'A9_proportion_edges', 'B1_W_gray', 'B1_W_lbp', 'B1_proportion_edges', 'B2_W_gray', 'B2_W_lbp', 'B2_proportion_edges', 'B3_W_gray', 'B3_W_lbp', 'B3_proportion_edges', 'B4_W_gray', 'B4_W_lbp', 'B4_proportion_edges', 'B5_W_gray', 'B5_W_lbp', 'B5_proportion_edges', 'B9_W_gray', 'B9_W_lbp', 'B9_proportion_edges']
        columns_to_categorical = ['class_previous_tile', 'A1_class', 'A2_class', 'A3_class', 'A4_class', 'A5_class', 'A9_class', 'A6_class', 'A8_class']
    elif feat_id == 1: #Only global knowledge
        features_X = ['tile_X', 'tile_Y', 'tile_mean_gray', 'tile_mean_lbp', 'f_area_size', 'max_grad_up', 'max_grad_down', 'max_grad_right', 'max_grad_left']                
        features_Y = ['tile_class']
        columns_to_scale = ['tile_X', 'tile_Y', 'tile_mean_gray', 'tile_mean_lbp', 'f_area_size', 'max_grad_up', 'max_grad_down', 'max_grad_right', 'max_grad_left']
        columns_to_categorical = []
    elif feat_id == 2: #All knowledge  
        features_X = ['tile_X', 'tile_Y', 'tile_proportion_edges', 'tile_entropy_gray', 'tile_entropy_lbp', 'tile_mean_gray', 'tile_mean_lbp', 'class_previous_tile', 'gray_gradient', 'f_area_size', 'gray_intensity_neighbor', 'max_grad_up', 'max_grad_down', 'max_grad_right', 'max_grad_left', 'A1_class', 'A1_W_gray', 'A1_W_lbp', 'A1_proportion_edges','A2_class', 'A2_W_gray', 'A2_W_lbp', 'A2_proportion_edges', 'A3_class', 'A3_W_gray', 'A3_W_lbp', 'A3_proportion_edges', 'A4_class', 'A4_W_gray', 'A4_W_lbp', 'A4_proportion_edges', 'A5_class', 'A5_W_gray', 'A5_W_lbp', 'A5_proportion_edges', 'A9_class', 'A9_W_gray', 'A9_W_lbp', 'A9_proportion_edges', 'A6_class', 'A8_class', 'B1_W_gray', 'B1_W_lbp', 'B1_proportion_edges', 'B2_W_gray', 'B2_W_lbp', 'B2_proportion_edges', 'B3_W_gray', 'B3_W_lbp', 'B3_proportion_edges', 'B4_W_gray', 'B4_W_lbp', 'B4_proportion_edges', 'B5_W_gray', 'B5_W_lbp', 'B5_proportion_edges', 'B9_W_gray', 'B9_W_lbp', 'B9_proportion_edges']                
        features_Y = ['tile_class']
        columns_to_scale = ['tile_X', 'tile_Y', 'tile_proportion_edges', 'tile_entropy_gray', 'tile_entropy_lbp', 'tile_mean_gray', 'tile_mean_lbp', 'gray_gradient', 'f_area_size', 'gray_intensity_neighbor', 'max_grad_up', 'max_grad_down', 'max_grad_right', 'max_grad_left', 'A1_W_gray', 'A1_W_lbp', 'A1_proportion_edges', 'A2_W_gray', 'A2_W_lbp', 'A2_proportion_edges', 'A3_W_gray', 'A3_W_lbp', 'A3_proportion_edges', 'A4_W_gray', 'A4_W_lbp', 'A4_proportion_edges', 'A5_W_gray', 'A5_W_lbp', 'A5_proportion_edges', 'A9_W_gray', 'A9_W_lbp', 'A9_proportion_edges', 'B1_W_gray', 'B1_W_lbp', 'B1_proportion_edges', 'B2_W_gray', 'B2_W_lbp', 'B2_proportion_edges', 'B3_W_gray', 'B3_W_lbp', 'B3_proportion_edges', 'B4_W_gray', 'B4_W_lbp', 'B4_proportion_edges', 'B5_W_gray', 'B5_W_lbp', 'B5_proportion_edges', 'B9_W_gray', 'B9_W_lbp', 'B9_proportion_edges']
        columns_to_categorical = ['class_previous_tile', 'A1_class', 'A2_class', 'A3_class', 'A4_class', 'A5_class', 'A9_class', 'A6_class', 'A8_class']
    elif feat_id == 3: #Only classes of previous tile, A1, A2, A5, A6, A8  - local knowledge
        features_X = ['tile_proportion_edges', 'tile_entropy_gray', 'tile_entropy_lbp', 'gray_gradient', 'gray_intensity_neighbor', 'class_previous_tile', 'A1_class', 'A1_W_gray', 'A1_W_lbp', 'A1_proportion_edges','A2_class', 'A2_W_gray', 'A2_W_lbp', 'A2_proportion_edges', 'A3_W_gray', 'A3_W_lbp', 'A3_proportion_edges', 'A4_W_gray', 'A4_W_lbp', 'A4_proportion_edges', 'A5_class', 'A5_W_gray', 'A5_W_lbp', 'A5_proportion_edges', 'A9_W_gray', 'A9_W_lbp', 'A9_proportion_edges', 'A6_class', 'A8_class', 'B1_W_gray', 'B1_W_lbp', 'B1_proportion_edges', 'B2_W_gray', 'B2_W_lbp', 'B2_proportion_edges', 'B3_W_gray', 'B3_W_lbp', 'B3_proportion_edges', 'B4_W_gray', 'B4_W_lbp', 'B4_proportion_edges', 'B5_W_gray', 'B5_W_lbp', 'B5_proportion_edges', 'B9_W_gray', 'B9_W_lbp', 'B9_proportion_edges']                
        features_Y = ['tile_class']
        columns_to_scale = ['tile_proportion_edges', 'tile_entropy_gray', 'tile_entropy_lbp', 'gray_gradient', 'gray_intensity_neighbor', 'A1_W_gray', 'A1_W_lbp', 'A1_proportion_edges', 'A2_W_gray', 'A2_W_lbp', 'A2_proportion_edges', 'A3_W_gray', 'A3_W_lbp', 'A3_proportion_edges', 'A4_W_gray', 'A4_W_lbp', 'A4_proportion_edges', 'A5_W_gray', 'A5_W_lbp', 'A5_proportion_edges', 'A9_W_gray', 'A9_W_lbp', 'A9_proportion_edges', 'B1_W_gray', 'B1_W_lbp', 'B1_proportion_edges', 'B2_W_gray', 'B2_W_lbp', 'B2_proportion_edges', 'B3_W_gray', 'B3_W_lbp', 'B3_proportion_edges', 'B4_W_gray', 'B4_W_lbp', 'B4_proportion_edges', 'B5_W_gray', 'B5_W_lbp', 'B5_proportion_edges', 'B9_W_gray', 'B9_W_lbp', 'B9_proportion_edges']
        columns_to_categorical = ['class_previous_tile', 'A1_class', 'A2_class', 'A5_class', 'A6_class', 'A8_class']
    elif feat_id == 4: #Only classes of previous tile, A1, A2, A5, A6, A8  - local knowledge + position + mean gray/lbp
        features_X = ['tile_X', 'tile_Y', 'tile_proportion_edges', 'tile_entropy_gray', 'gray_gradient', 'tile_entropy_lbp', 'tile_mean_gray', 'tile_mean_lbp', 'class_previous_tile', 'A1_class', 'A1_W_gray', 'A1_W_lbp', 'A1_proportion_edges','A2_class', 'A2_W_gray', 'A2_W_lbp', 'A2_proportion_edges', 'A3_W_gray', 'A3_W_lbp', 'A3_proportion_edges', 'A4_W_gray', 'A4_W_lbp', 'A4_proportion_edges', 'A5_class', 'A5_W_gray', 'A5_W_lbp', 'A5_proportion_edges', 'A9_W_gray', 'A9_W_lbp', 'A9_proportion_edges', 'A6_class', 'A8_class', 'B1_W_gray', 'B1_W_lbp', 'B1_proportion_edges', 'B2_W_gray', 'B2_W_lbp', 'B2_proportion_edges', 'B3_W_gray', 'B3_W_lbp', 'B3_proportion_edges', 'B4_W_gray', 'B4_W_lbp', 'B4_proportion_edges', 'B5_W_gray', 'B5_W_lbp', 'B5_proportion_edges', 'B9_W_gray', 'B9_W_lbp', 'B9_proportion_edges']                
        features_Y = ['tile_class']
        columns_to_scale = ['tile_X', 'tile_Y', 'tile_proportion_edges', 'tile_entropy_gray', 'gray_gradient', 'tile_entropy_lbp', 'tile_mean_gray', 'tile_mean_lbp', 'A1_W_gray', 'A1_W_lbp', 'A1_proportion_edges', 'A2_W_gray', 'A2_W_lbp', 'A2_proportion_edges', 'A3_W_gray', 'A3_W_lbp', 'A3_proportion_edges', 'A4_W_gray', 'A4_W_lbp', 'A4_proportion_edges', 'A5_W_gray', 'A5_W_lbp', 'A5_proportion_edges', 'A9_W_gray', 'A9_W_lbp', 'A9_proportion_edges', 'B1_W_gray', 'B1_W_lbp', 'B1_proportion_edges', 'B2_W_gray', 'B2_W_lbp', 'B2_proportion_edges', 'B3_W_gray', 'B3_W_lbp', 'B3_proportion_edges', 'B4_W_gray', 'B4_W_lbp', 'B4_proportion_edges', 'B5_W_gray', 'B5_W_lbp', 'B5_proportion_edges', 'B9_W_gray', 'B9_W_lbp', 'B9_proportion_edges']
        columns_to_categorical = ['class_previous_tile', 'A1_class', 'A2_class', 'A5_class', 'A6_class', 'A8_class']
    elif feat_id == 5: #Only classes of previous tile, A1, A2, A5, A6, A8  - All
        features_X = ['tile_X', 'tile_Y', 'tile_proportion_edges', 'tile_entropy_gray', 'tile_entropy_lbp', 'tile_mean_gray', 'tile_mean_lbp', 'class_previous_tile', 'gray_gradient', 'f_area_size', 'gray_intensity_neighbor', 'max_grad_up', 'max_grad_down', 'max_grad_right', 'max_grad_left', 'A1_class', 'A1_W_gray', 'A1_W_lbp', 'A1_proportion_edges','A2_class', 'A2_W_gray', 'A2_W_lbp', 'A2_proportion_edges', 'A3_W_gray', 'A3_W_lbp', 'A3_proportion_edges', 'A4_W_gray', 'A4_W_lbp', 'A4_proportion_edges', 'A5_class', 'A5_W_gray', 'A5_W_lbp', 'A5_proportion_edges', 'A9_W_gray', 'A9_W_lbp', 'A9_proportion_edges', 'A6_class', 'A8_class', 'B1_W_gray', 'B1_W_lbp', 'B1_proportion_edges', 'B2_W_gray', 'B2_W_lbp', 'B2_proportion_edges', 'B3_W_gray', 'B3_W_lbp', 'B3_proportion_edges', 'B4_W_gray', 'B4_W_lbp', 'B4_proportion_edges', 'B5_W_gray', 'B5_W_lbp', 'B5_proportion_edges', 'B9_W_gray', 'B9_W_lbp', 'B9_proportion_edges']                
        features_Y = ['tile_class']
        columns_to_scale = ['tile_X', 'tile_Y', 'tile_proportion_edges', 'tile_entropy_gray', 'tile_entropy_lbp', 'tile_mean_gray', 'tile_mean_lbp', 'gray_gradient', 'f_area_size', 'gray_intensity_neighbor', 'max_grad_up', 'max_grad_down', 'max_grad_right', 'max_grad_left', 'A1_W_gray', 'A1_W_lbp', 'A1_proportion_edges', 'A2_W_gray', 'A2_W_lbp', 'A2_proportion_edges', 'A3_W_gray', 'A3_W_lbp', 'A3_proportion_edges', 'A4_W_gray', 'A4_W_lbp', 'A4_proportion_edges', 'A5_W_gray', 'A5_W_lbp', 'A5_proportion_edges', 'A9_W_gray', 'A9_W_lbp', 'A9_proportion_edges', 'B1_W_gray', 'B1_W_lbp', 'B1_proportion_edges', 'B2_W_gray', 'B2_W_lbp', 'B2_proportion_edges', 'B3_W_gray', 'B3_W_lbp', 'B3_proportion_edges', 'B4_W_gray', 'B4_W_lbp', 'B4_proportion_edges', 'B5_W_gray', 'B5_W_lbp', 'B5_proportion_edges', 'B9_W_gray', 'B9_W_lbp', 'B9_proportion_edges']
        columns_to_categorical = ['class_previous_tile', 'A1_class', 'A2_class', 'A5_class', 'A6_class', 'A8_class']
    elif feat_id == 6: #No classes of previous tiles - All
        features_X = ['tile_X', 'tile_Y', 'tile_proportion_edges', 'tile_entropy_gray', 'tile_entropy_lbp', 'tile_mean_gray', 'tile_mean_lbp', 'gray_gradient', 'f_area_size', 'gray_intensity_neighbor', 'max_grad_up', 'max_grad_down', 'max_grad_right', 'max_grad_left', 'A1_W_gray', 'A1_W_lbp', 'A1_proportion_edges', 'A2_W_gray', 'A2_W_lbp', 'A2_proportion_edges', 'A3_W_gray', 'A3_W_lbp', 'A3_proportion_edges', 'A4_W_gray', 'A4_W_lbp', 'A4_proportion_edges', 'A5_W_gray', 'A5_W_lbp', 'A5_proportion_edges', 'A9_W_gray', 'A9_W_lbp', 'A9_proportion_edges', 'B1_W_gray', 'B1_W_lbp', 'B1_proportion_edges', 'B2_W_gray', 'B2_W_lbp', 'B2_proportion_edges', 'B3_W_gray', 'B3_W_lbp', 'B3_proportion_edges', 'B4_W_gray', 'B4_W_lbp', 'B4_proportion_edges', 'B5_W_gray', 'B5_W_lbp', 'B5_proportion_edges', 'B9_W_gray', 'B9_W_lbp', 'B9_proportion_edges']                
        features_Y = ['tile_class']
        columns_to_scale = ['tile_X', 'tile_Y', 'tile_proportion_edges', 'tile_entropy_gray', 'tile_entropy_lbp', 'tile_mean_gray', 'tile_mean_lbp', 'gray_gradient', 'f_area_size', 'gray_intensity_neighbor', 'max_grad_up', 'max_grad_down', 'max_grad_right', 'max_grad_left', 'A1_W_gray', 'A1_W_lbp', 'A1_proportion_edges', 'A2_W_gray', 'A2_W_lbp', 'A2_proportion_edges', 'A3_W_gray', 'A3_W_lbp', 'A3_proportion_edges', 'A4_W_gray', 'A4_W_lbp', 'A4_proportion_edges', 'A5_W_gray', 'A5_W_lbp', 'A5_proportion_edges', 'A9_W_gray', 'A9_W_lbp', 'A9_proportion_edges', 'B1_W_gray', 'B1_W_lbp', 'B1_proportion_edges', 'B2_W_gray', 'B2_W_lbp', 'B2_proportion_edges', 'B3_W_gray', 'B3_W_lbp', 'B3_proportion_edges', 'B4_W_gray', 'B4_W_lbp', 'B4_proportion_edges', 'B5_W_gray', 'B5_W_lbp', 'B5_proportion_edges', 'B9_W_gray', 'B9_W_lbp', 'B9_proportion_edges']
        columns_to_categorical = []
    return features_X, features_Y, columns_to_scale, columns_to_categorical


#%% Preprocessing: elect features and normalize:

#For now we drop na:
FEATURES_SELECTION = 1

df_Prediction = df_Prediction.fillna(int(-1))
df_Prediction = df_Prediction[~df_Prediction['tile_ID'].isin([0,1,2,3,4,5])]

df_Prediction['dist_new'] = 1/(df_Prediction['tile_X']+df_Prediction['tile_Y']+0.1)

#Save images for test
Images_ID = [0,70,100,130,150,160,178,198]
df_Test = df_Prediction.loc[df_Prediction['image_ID'].isin(Images_ID)]
df_Prediction = df_Prediction[~df_Prediction['image_ID'].isin(Images_ID)]
#df_Prediction = shuffle(df_Prediction)

#Select features:
features_X, features_Y, columns_to_scale, columns_to_categorical = selectFeatures(feat_id = FEATURES_SELECTION)

#Scaling and select features
df_Prediction_X = df_Prediction.loc[ : , features_X]
df_Prediction_Y = df_Prediction.loc[ : , features_Y]
scaler_df_Prediction = preprocessing.MinMaxScaler()
df_Prediction_X[columns_to_scale] = scaler_df_Prediction.fit_transform(df_Prediction_X[columns_to_scale])
if len(columns_to_categorical) > 0:
    df_Prediction_X[columns_to_categorical] = df_Prediction_X[columns_to_categorical].astype('category')
df_Prediction_Y[features_Y] = df_Prediction_Y[features_Y].astype('category')
#filename = 'Models/Scalers/scaler_df_Prediction.sav'
#pickle.dump(scaler_df_Prediction, open(filename, 'wb'))

#Change to categorical -> strings
listCat = ['class_previous_tile', 'A1_class', 'A2_class', 'A5_class', 'A6_class', 'A8_class']
for cat in listCat:
    if cat in df_Prediction_X.columns:
        df_Prediction_X[cat] = df_Prediction_X[cat].map({-1: 'out_of_bounds', 0:'back', 1:'floor', 2: 'edge'})
#One hot encoding:
for ohe in listCat: 
    if ohe in df_Prediction_X.columns:
        dfCat = pd.get_dummies(df_Prediction_X[ohe], prefix = ohe)
        df_Prediction_X = df_Prediction_X.drop([ohe], axis=1)
        df_Prediction_X = pd.concat([df_Prediction_X, dfCat], axis=1)

#Train-test split (on whole images)
test_size = 0.1
random.seed(42)
IDs = np.random.RandomState(seed=42).permutation(int(1+max(df_Prediction.loc[:,'image_ID'])))
for el in Images_ID:
    IDs = IDs[IDs != el]
N = int(test_size*(1+len(IDs)))
train_ids = IDs[N:]
test_ids = IDs[0:N]

X_train = df_Prediction_X.loc[df_Prediction.loc[:,'image_ID'].isin(train_ids),:]
X_test = df_Prediction_X.loc[df_Prediction.loc[:,'image_ID'].isin(test_ids),:]
y_train = df_Prediction_Y.loc[df_Prediction.loc[:,'image_ID'].isin(train_ids),:]
y_test = df_Prediction_Y.loc[df_Prediction.loc[:,'image_ID'].isin(test_ids),:]

#For the training
X = pd.concat([X_train, X_test])
X = X.reset_index(drop = True)
Y = pd.concat([y_train, y_test])
Y = Y.reset_index(drop = True)

#%% Decision tree

#Split data
# Create Decision Tree classifer object
decisionTree = DecisionTreeClassifier(max_depth = 10, random_state=42, criterion = 'gini', min_samples_split = 20, min_samples_leaf = 5)
# Train Decision Tree Classifer
decisionTree = decisionTree.fit(X,Y)
#Predict the response for test dataset
#y_pred = decisionTree.predict(X_test)

filename = 'Models/Prediction/decisionTree.sav'
#pickle.dump(decisionTree, open(filename, 'wb'))

#Accuracy:
#print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

#Importance:
a = dict(zip(X_train.columns, decisionTree.feature_importances_))
importances = pd.DataFrame(columns=['feature', 'DT'])
for key in a.keys():
    a0 = round(a[key],3)
    data = pd.DataFrame([[str(key), a0]])
    data.columns = importances.columns
    importances = importances.append(data)

importances = importances.sort_values(by=['DT'], ascending=False)
importances.to_csv('figures/decision_tree/test2_global_features/importances/imp.csv', sep=';', encoding='utf-8')

##Save the visualization of the decision tree:
#dot_data = StringIO()
#feat_names = df_Prediction_X.columns
#export_graphviz(decisionTree, out_file=dot_data, filled=True, rounded=True, special_characters=True,feature_names = feat_names, class_names=['0','1','2'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png('figures/decision_tree/test2_global_features/decisionTree.png')

   
#%% Plot image - prediction

#Images_IDs: 0,70,100,130,150,160,178,198
image_ID = 150

filename = 'figures/test/2_globalFeatures/im' + str(Images_ID.index(image_ID)+1) + '.png'

image = df_merged_grayImages[image_ID]
image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
X_pred = df_Test.loc[df_Test['image_ID']==image_ID, :]
X_pred.index = X_pred['tile_ID'].astype('int')
X_pred = X_pred.loc[:,features_X]

classes = np.zeros([len(X_pred),2])
classes[:,0]=range(len(X_pred))

X_pred[columns_to_scale] = scaler_df_Prediction.transform(X_pred[columns_to_scale])
if len(columns_to_categorical) > 0: 
    X_pred[columns_to_categorical] = X_pred[columns_to_categorical].astype('category')


#Change to categorical -> strings
listCat = ['class_previous_tile', 'A1_class', 'A2_class', 'A5_class', 'A6_class', 'A8_class']
for cat in listCat:
    if cat in X_pred.columns:
        X_pred[cat] = X_pred[cat].map({-1: 'out_of_bounds', 0:'back', 1:'floor', 2: 'edge'})
#One hot encoding:
for ohe in listCat:  
    if ohe in X_pred.columns:
        dfCat = pd.get_dummies(X_pred[ohe], prefix = ohe)
        X_pred = X_pred.drop([ohe], axis=1)
        X_pred = pd.concat([X_pred, dfCat], axis=1)

#classes[:,1] = decisionTree.predict(X_pred)
#columns_online = ['class_previous_tile_back','class_previous_tile_edge', 'class_previous_tile_floor', 'A1_class_back', 'A1_class_edge', 'A1_class_floor', 'A2_class_back', 'A2_class_edge', 'A2_class_floor', 'A5_class_back', 'A5_class_edge', 'A5_class_floor', 'A6_class_back', 'A6_class_edge', 'A6_class_floor', 'A8_class_back', 'A8_class_edge', 'A8_class_floor']
#classes_to_analyse= ['class_previous_tile','A1_class','A2_class','A5_class','A6_class', 'A8_class']
columns_online = [] #Global knowledge
classes_to_analyse = [] #Global knowledge
classes_pref_funct, X = kgf.predictTree(X_pred, decisionTree = decisionTree, image = df_merged_grayImages[0], num = 20, classes_names = classes_to_analyse, columns_online = columns_online)

a = kgf.paintHyperpixels(image, classes_pref_funct)

fig = plt.figure(frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)

ax.imshow(a)
plt.savefig(filename)


#%% Loop accuracies/depth

#Number of maximum depths tested
num_depths = 25

accuracy = np.zeros([num_depths-1])
for i in range(num_depths-1):
    decisionTree = DecisionTreeClassifier(max_depth = i+2, random_state=42, criterion = 'gini' , min_samples_split = 20, min_samples_leaf = 5)
    # Train Decision Tree Classifer
    decisionTree = decisionTree.fit(X_train,y_train)
    #Predict the response for test dataset
    y_pred = decisionTree.predict(X_test)
    #Accuracy:
    accuracy[i] = metrics.accuracy_score(y_test, y_pred)

#Plot
    
labels = list(range(2, num_depths+1))    
fig = plt.figure()
plt.plot(labels, accuracy)
fig.suptitle('Accuracy', fontsize=15)
plt.xlabel('Depth of the tree', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.xlim((2,num_depths))
plt.ylim((0.85,1.01))
plt.savefig('figures/decision_tree/test2_global_features/accuracies/test_depths.png')



