# SMRobot project
# Ignacio Garrido Botella
# Given the dataset of features this code makes a selection of them and the subsequent analysis (scaling + training).

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
from sklearn.feature_selection import RFE

os.chdir("/Users/ignacio/Documents/Universidad/Master/Segundo/SegundoSemestre/MasterThesisMoAI/Project/smrobot_py_knowledgeGraph")
import kGraph_functions as kgf

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

#%% Table keys
  
createTableKeys = {'tile_X': 'Position of the tile in the X-axis', 
    'tile_Y': 'Position of the tile in the Y-axis', 
    'tile_proportion_edges': 'Proportion of pixels detected as edge of tile T',
    'tile_entropy_gray': 'Entropy of the grayscale distribution of tile T', 
    'tile_entropy_lbp': 'Entropy of the LBP distribution of tile T',
    'tile_mean_gray': 'Mean grayscale value with respect to the whole image',
    'tile_mean_lbp': 'Mean LBP value with respect to the whole image',
    'gray_gradient': 'Max gradient of the grayscale pixels of tile T',
    'f_area_size': 'Minimum size of the Felzenswalb area',
    'gray_intensity_neighbor': 'Gray intensity relative to neighbourhood',
    'max_grad_up': 'Maximum gradient change over the tile T',
    'max_grad_down': 'Maximum gradient change under the tile T',
    'max_grad_right': 'Maximum gradient change at right of the tile T',
    'max_grad_left': 'Maximum gradient change at left of the tile T',
    'A1_W_gray': 'EMD with the grayscale distribution of tile A1',
    'A1_W_lbp': 'EMD with the LBP distribution of tile A1',
    'A1_proportion_edges': 'Proportion of pixels detected as edge of tile A1',
    'A2_W_gray': 'EMD with the grayscale distribution of tile A2',
    'A2_W_lbp': 'EMD with the LBP distribution of tile A2',
    'A2_proportion_edges': 'Proportion of pixels detected as edge of tile A2',
    'A3_W_gray': 'EMD with the grayscale distribution of tile A3',
    'A3_W_lbp': 'EMD with the LBP distribution of tile A3',
    'A3_proportion_edges': 'Proportion of pixels detected as edge of tile A3',
    'A4_W_gray': 'EMD with the grayscale distribution of tile A4',
    'A4_W_lbp': 'EMD with the LBP distribution of tile A4',
    'A4_proportion_edges': 'Proportion of pixels detected as edge of tile A4',
    'A5_W_gray': 'EMD with the grayscale distribution of tile A5',
    'A5_W_lbp': 'EMD with the LBP distribution of tile A5',
    'A5_proportion_edges': 'Proportion of pixels detected as edge of tile A5',
    'A9_W_gray': 'EMD with the grayscale distribution of tile A9',
    'A9_W_lbp': 'EMD with the LBP distribution of tile A9',
    'A9_proportion_edges': 'Proportion of pixels detected as edge of tile A9',
    'B1_W_gray': 'EMD with the grayscale distribution of tile B1',
    'B1_W_lbp': 'EMD with the LBP distribution of tile B1',
    'B1_proportion_edges': 'Proportion of pixels detected as edge of tile B1',
    'B2_W_gray': 'EMD with the grayscale distribution of tile B2',
    'B2_W_lbp': 'EMD with the LBP distribution of tile B2',
    'B2_proportion_edges': 'Proportion of pixels detected as edge of tile B2',
    'B3_W_gray': 'EMD with the grayscale distribution of tile B3',
    'B3_W_lbp': 'EMD with the LBP distribution of tile B3',
    'B3_proportion_edges': 'Proportion of pixels detected as edge of tile B3',
    'B4_W_gray': 'EMD with the grayscale distribution of tile B4',
    'B4_W_lbp': 'EMD with the LBP distribution of tile B4',
    'B4_proportion_edges': 'Proportion of pixels detected as edge of tile B4',
    'B5_W_gray': 'EMD with the grayscale distribution of tile B5',
    'B5_W_lbp': 'EMD with the LBP distribution of tile B5',
    'B5_proportion_edges': 'Proportion of pixels detected as edge of tile B5',
    'B9_W_gray': 'EMD with the grayscale distribution of tile B9',
    'B9_W_lbp': 'EMD with the LBP distribution of tile B9',
    'B9_proportion_edges': 'Proportion of pixels detected as edge of tile B9'
    } 


#%% Preprocessing: select feature and normalize:
  
FEATURES_SELECTION = 6

df_Prediction = df_Prediction.fillna(int(-1))
df_Prediction = df_Prediction[~df_Prediction['tile_ID'].isin([0,1,2,3,4,5])]
df_Prediction['dist_new'] = 1/(df_Prediction['tile_X']+df_Prediction['tile_Y']+0.5)

#Eliminate the superpixels that are not being classified -> Those that are at the extremes of the image.
num_hyperpixels = int(max(df_Prediction['tile_ID']) + 1)
left1, left2 = kgf.getLeftHyperpixels(num_hyperpixels)
right1, right2 = kgf.getRightHyperpixels(num_hyperpixels)
low1, low2 = kgf.getLowerHyperpixels(num_hyperpixels)
top1, top2 = kgf.getTopHyperpixels(num_hyperpixels)
drop_tiles = left1 + left2 + right1+ right2 + top1 + top2
df_Prediction = df_Prediction[~df_Prediction['tile_ID'].isin(drop_tiles)]    

#Divide in three datasets -> One for each superpixel
df_Prediction_main = df_Prediction[~df_Prediction['tile_ID'].isin(low1+low2)]  
df_Prediction_low1 = df_Prediction[df_Prediction['tile_ID'].isin(low1)]  
df_Prediction_low2 = df_Prediction[df_Prediction['tile_ID'].isin(low2)] 
df_Prediction_main_ID1ID3 = df_Prediction_main[df_Prediction_main['tile_section_of_analysis'] != 2] 
df_Prediction_main_ID2 = df_Prediction_main[df_Prediction_main['tile_section_of_analysis'] == 2]

#Save images for test
Images_ID = [0,70,100,130,150,160,178,198]
df_Test = df_Prediction.loc[df_Prediction['image_ID'].isin(Images_ID)] #Df with the images for the prediction
df_Prediction_main_ID1ID3 = df_Prediction_main_ID1ID3[~df_Prediction_main_ID1ID3['image_ID'].isin(Images_ID)]
df_Prediction_main_ID2 = df_Prediction_main_ID2[~df_Prediction_main_ID2['image_ID'].isin(Images_ID)]
df_Prediction_low1 = df_Prediction_low1[~df_Prediction_low1['image_ID'].isin(Images_ID)]
df_Prediction_low2 = df_Prediction_low2[~df_Prediction_low2['image_ID'].isin(Images_ID)]

#Select features
features_X_main, features_Y, columns_to_scale_main, columns_to_categorical_main = selectFeatures(feat_id = FEATURES_SELECTION)
list_notin_low1 = ['A1_class', 'A1_W_gray', 'A1_W_lbp', 'A1_proportion_edges','A2_class', 'A2_W_gray', 'A2_W_lbp', 'A2_proportion_edges', 'A5_class', 'A5_W_gray', 'A5_W_lbp', 'A5_proportion_edges', 'A6_class', 'A8_class', 'B4_W_gray', 'B4_W_lbp', 'B4_proportion_edges']
features_X_low1 = [x for x in features_X_main if x not in list_notin_low1]
columns_to_scale_low1 = [x for x in columns_to_scale_main if x not in list_notin_low1]
columns_to_categorical_low1 = [x for x in columns_to_categorical_main if x not in list_notin_low1]
list_notin_low2 = ['A5_class', 'A5_W_gray', 'A5_W_lbp', 'A5_proportion_edges']
features_X_low2 = [x for x in features_X_main if x not in list_notin_low2]
columns_to_scale_low2 = [x for x in columns_to_scale_main if x not in list_notin_low2]
columns_to_categorical_low2 = [x for x in columns_to_categorical_main if x not in list_notin_low2]

#Scaling and select features - main
df_Prediction_X_main_ID1ID3 = df_Prediction_main_ID1ID3.loc[ : , features_X_main]
df_Prediction_Y_main_ID1ID3 = df_Prediction_main_ID1ID3.loc[ : , features_Y]
df_Prediction_X_main_ID2 = df_Prediction_main_ID2.loc[ : , features_X_main]
df_Prediction_Y_main_ID2 = df_Prediction_main_ID2.loc[ : , features_Y]
scaler_df_Prediction_main_ID1ID3 = preprocessing.MinMaxScaler()
df_Prediction_X_main_ID1ID3[columns_to_scale_main] = scaler_df_Prediction_main_ID1ID3.fit_transform(df_Prediction_X_main_ID1ID3[columns_to_scale_main])
if len(columns_to_categorical_main) > 0:
    df_Prediction_X_main_ID1ID3[columns_to_categorical_main] = df_Prediction_X_main_ID1ID3[columns_to_categorical_main].astype('category')
df_Prediction_Y_main_ID1ID3[features_Y] = df_Prediction_Y_main_ID1ID3[features_Y].astype('category')
scaler_df_Prediction_main_ID2 = preprocessing.MinMaxScaler()
df_Prediction_X_main_ID2[columns_to_scale_main] = scaler_df_Prediction_main_ID2.fit_transform(df_Prediction_X_main_ID2[columns_to_scale_main])
if len(columns_to_categorical_main) > 0:
    df_Prediction_X_main_ID2[columns_to_categorical_main] = df_Prediction_X_main_ID2[columns_to_categorical_main].astype('category')
df_Prediction_Y_main_ID2[features_Y] = df_Prediction_Y_main_ID2[features_Y].astype('category')

#Scaling and select features - low1
df_Prediction_X_low1 = df_Prediction_low1.loc[ : , features_X_low1]
df_Prediction_Y_low1 = df_Prediction_low1.loc[ : , features_Y]
scaler_df_Prediction_low1= preprocessing.MinMaxScaler()
df_Prediction_X_low1[columns_to_scale_low1] = scaler_df_Prediction_low1.fit_transform(df_Prediction_X_low1[columns_to_scale_low1])
if len(columns_to_categorical_low1) > 0:
    df_Prediction_X_low1[columns_to_categorical_low1] = df_Prediction_X_low1[columns_to_categorical_low1].astype('category')
df_Prediction_Y_low1[features_Y] = df_Prediction_Y_low1[features_Y].astype('category')

#Scaling and select features - low2
df_Prediction_X_low2 = df_Prediction_low2.loc[ : , features_X_low2]
df_Prediction_Y_low2 = df_Prediction_low2.loc[ : , features_Y]
scaler_df_Prediction_low2 = preprocessing.MinMaxScaler()
df_Prediction_X_low2[columns_to_scale_low2] = scaler_df_Prediction_low2.fit_transform(df_Prediction_X_low2[columns_to_scale_low2])
if len(columns_to_categorical_low2) > 0:
    df_Prediction_X_low2[columns_to_categorical_low2] = df_Prediction_X_low2[columns_to_categorical_low2].astype('category')
df_Prediction_Y_low2[features_Y] = df_Prediction_Y_low2[features_Y].astype('category')
del left1, left2, right1, right2, top1, top2, drop_tiles, num_hyperpixels, list_notin_low1, list_notin_low2, df_Prediction_main

#Change to categorical -> strings
listCat = ['class_previous_tile', 'A1_class', 'A2_class', 'A5_class', 'A6_class', 'A8_class']
for cat in listCat:
    if cat in df_Prediction_X_main_ID1ID3.columns:
        df_Prediction_X_main_ID1ID3[cat] = df_Prediction_X_main_ID1ID3[cat].map({-1: 'out_of_bounds', 0:'back', 1:'floor', 2: 'edge'})
    if cat in df_Prediction_X_main_ID2.columns:
        df_Prediction_X_main_ID2[cat] = df_Prediction_X_main_ID2[cat].map({-1: 'out_of_bounds', 0:'back', 1:'floor', 2: 'edge'})
    if cat in df_Prediction_X_low1.columns:
        df_Prediction_X_low1[cat] = df_Prediction_X_low1[cat].map({-1: 'out_of_bounds', 0:'back', 1:'floor', 2: 'edge'})
    if cat in df_Prediction_X_low2.columns:
        df_Prediction_X_low2[cat] = df_Prediction_X_low2[cat].map({-1: 'out_of_bounds', 0:'back', 1:'floor', 2: 'edge'})
#One hot encoding:
for ohe in (listCat):
    if ohe in df_Prediction_X_main_ID1ID3.columns:
        dfCat = pd.get_dummies(df_Prediction_X_main_ID1ID3[ohe], prefix = ohe)
        df_Prediction_X_main_ID1ID3 = df_Prediction_X_main_ID1ID3.drop([ohe], axis=1)
        df_Prediction_X_main_ID1ID3 = pd.concat([df_Prediction_X_main_ID1ID3, dfCat], axis=1)
        if (ohe + '_back') not in df_Prediction_X_main_ID1ID3.columns:
            df_Prediction_X_main_ID1ID3[ohe + '_back'] = np.zeros([len(df_Prediction_X_main_ID1ID3),1])
        if (ohe + '_floor') not in df_Prediction_X_main_ID1ID3.columns:
            df_Prediction_X_main_ID1ID3[ohe + '_floor'] = np.zeros([len(df_Prediction_X_main_ID1ID3),1])   
        if (ohe + '_edge') not in df_Prediction_X_main_ID1ID3.columns:
            df_Prediction_X_main_ID1ID3[ohe + '_edge'] = np.zeros([len(df_Prediction_X_main_ID1ID3),1])  
    if ohe in df_Prediction_X_main_ID2.columns:
        dfCat = pd.get_dummies(df_Prediction_X_main_ID2[ohe], prefix = ohe)
        df_Prediction_X_main_ID2 = df_Prediction_X_main_ID2.drop([ohe], axis=1)
        df_Prediction_X_main_ID2 = pd.concat([df_Prediction_X_main_ID2, dfCat], axis=1)
        if (ohe + '_back') not in df_Prediction_X_main_ID2.columns:
            df_Prediction_X_main_ID2[ohe + '_back'] = np.zeros([len(df_Prediction_X_main_ID2),1])
        if (ohe + '_floor') not in df_Prediction_X_main_ID2.columns:
            df_Prediction_X_main_ID2[ohe + '_floor'] = np.zeros([len(df_Prediction_X_main_ID2),1])   
        if (ohe + '_edge') not in df_Prediction_X_main_ID2.columns:
            df_Prediction_X_main_ID2[ohe + '_edge'] = np.zeros([len(df_Prediction_X_main_ID2),1])   
    if ohe in df_Prediction_X_low1.columns:
        dfCat = pd.get_dummies(df_Prediction_X_low1[ohe], prefix = ohe)
        df_Prediction_X_low1 = df_Prediction_X_low1.drop([ohe], axis=1)
        df_Prediction_X_low1 = pd.concat([df_Prediction_X_low1, dfCat], axis=1)  
        if (ohe + '_out_of_bounds') not in df_Prediction_X_low1.columns:
            df_Prediction_X_low1[ohe + '_out_of_bounds'] = np.zeros([len(df_Prediction_X_low1),1])
        if (ohe + '_back') not in df_Prediction_X_low1.columns:
            df_Prediction_X_low1[ohe + '_back'] = np.zeros([len(df_Prediction_X_low1),1])
        if (ohe + '_floor') not in df_Prediction_X_low1.columns:
            df_Prediction_X_low1[ohe + '_floor'] = np.zeros([len(df_Prediction_X_low1),1])   
        if (ohe + '_edge') not in df_Prediction_X_low1.columns:
            df_Prediction_X_low1[ohe + '_edge'] = np.zeros([len(df_Prediction_X_low1),1])   
    if ohe in df_Prediction_X_low2.columns:
        dfCat = pd.get_dummies(df_Prediction_X_low2[ohe], prefix = ohe)
        df_Prediction_X_low2 = df_Prediction_X_low2.drop([ohe], axis=1)
        df_Prediction_X_low2 = pd.concat([df_Prediction_X_low2, dfCat], axis=1)
        if (ohe + '_out_of_bounds') not in df_Prediction_X_low2.columns:
            df_Prediction_X_low2[ohe + '_out_of_bounds'] = np.zeros([len(df_Prediction_X_low2),1])
        if (ohe + '_back') not in df_Prediction_X_low2.columns:
            df_Prediction_X_low2[ohe + '_back'] = np.zeros([len(df_Prediction_X_low2),1])
        if (ohe + '_floor') not in df_Prediction_X_low2.columns:
            df_Prediction_X_low2[ohe + '_floor'] = np.zeros([len(df_Prediction_X_low2),1])   
        if (ohe + '_edge') not in df_Prediction_X_low2.columns:
            df_Prediction_X_low2[ohe + '_edge'] = np.zeros([len(df_Prediction_X_low2),1])  

#Train-test split (on whole images)
test_size = 0.1
random.seed(42)
IDs = np.random.RandomState(seed=42).permutation(int(1+max(df_Prediction.loc[:,'image_ID'])))
for el in Images_ID:
    IDs = IDs[IDs != el]
N = int(test_size*(1+len(IDs)))
train_ids = IDs[N:]
test_ids = IDs[0:N]

X_train_main_ID1ID3 = df_Prediction_X_main_ID1ID3.loc[df_Prediction_main_ID1ID3.loc[:,'image_ID'].isin(train_ids),:]
X_test_main_ID1ID3 = df_Prediction_X_main_ID1ID3.loc[df_Prediction_main_ID1ID3.loc[:,'image_ID'].isin(test_ids),:]
y_train_main_ID1ID3 = df_Prediction_Y_main_ID1ID3.loc[df_Prediction_main_ID1ID3.loc[:,'image_ID'].isin(train_ids),:]
y_test_main_ID1ID3 = df_Prediction_Y_main_ID1ID3.loc[df_Prediction_main_ID1ID3.loc[:,'image_ID'].isin(test_ids),:]
#For the training
X_main_ID1ID3 = pd.concat([X_train_main_ID1ID3, X_test_main_ID1ID3])
X_main_ID1ID3 = X_main_ID1ID3.reset_index(drop = True)
Y_main_ID1ID3 = pd.concat([y_train_main_ID1ID3, y_test_main_ID1ID3])
Y_main_ID1ID3 = Y_main_ID1ID3.reset_index(drop = True)
Y_main_ID1ID3 = Y_main_ID1ID3.values.ravel()

X_train_main_ID2 = df_Prediction_X_main_ID2.loc[df_Prediction_main_ID2.loc[:,'image_ID'].isin(train_ids),:]
X_test_main_ID2 = df_Prediction_X_main_ID2.loc[df_Prediction_main_ID2.loc[:,'image_ID'].isin(test_ids),:]
y_train_main_ID2 = df_Prediction_Y_main_ID2.loc[df_Prediction_main_ID2.loc[:,'image_ID'].isin(train_ids),:]
y_test_main_ID2 = df_Prediction_Y_main_ID2.loc[df_Prediction_main_ID2.loc[:,'image_ID'].isin(test_ids),:]
#For the training
X_main_ID2 = pd.concat([X_train_main_ID2, X_test_main_ID2])
X_main_ID2 = X_main_ID2.reset_index(drop = True)
Y_main_ID2 = pd.concat([y_train_main_ID2, y_test_main_ID2])
Y_main_ID2 = Y_main_ID2.reset_index(drop = True)
Y_main_ID2 = Y_main_ID2.values.ravel()

X_train_low1 = df_Prediction_X_low1.loc[df_Prediction_low1.loc[:,'image_ID'].isin(train_ids),:]
X_test_low1 = df_Prediction_X_low1.loc[df_Prediction_low1.loc[:,'image_ID'].isin(test_ids),:]
y_train_low1 = df_Prediction_Y_low1.loc[df_Prediction_low1.loc[:,'image_ID'].isin(train_ids),:]
y_test_low1 = df_Prediction_Y_low1.loc[df_Prediction_low1.loc[:,'image_ID'].isin(test_ids),:]
#For the training
X_low1 = pd.concat([X_train_low1, X_test_low1])
X_low1 = X_low1.reset_index(drop = True)
Y_low1 = pd.concat([y_train_low1, y_test_low1])
Y_low1 = Y_low1.reset_index(drop = True)
Y_low1 = Y_low1.values.ravel()

X_train_low2 = df_Prediction_X_low2.loc[df_Prediction_low2.loc[:,'image_ID'].isin(train_ids),:]
X_test_low2 = df_Prediction_X_low2.loc[df_Prediction_low2.loc[:,'image_ID'].isin(test_ids),:]
y_train_low2 = df_Prediction_Y_low2.loc[df_Prediction_low2.loc[:,'image_ID'].isin(train_ids),:]
y_test_low2 = df_Prediction_Y_low2.loc[df_Prediction_low2.loc[:,'image_ID'].isin(test_ids),:]
#For the training
X_low2 = pd.concat([X_train_low2, X_test_low2])
X_low2 = X_low2.reset_index(drop = True)
Y_low2 = pd.concat([y_train_low2, y_test_low2])
Y_low2 = Y_low2.reset_index(drop = True)
Y_low2 = Y_low2.values.ravel()


del IDs, N, test_size, df_Prediction_low1, df_Prediction_low2, listCat, train_ids, test_ids, ohe, cat


#%% Decision trees -> Training

# Create Decision Tree classifer object
decisionTree_main_ID1ID3 = DecisionTreeClassifier(max_depth = 10, random_state=42, criterion = 'gini') # min_samples_split = 2, min_samples_leaf = 1)
decisionTree_main_ID2 = DecisionTreeClassifier(max_depth = 10, random_state=42, criterion = 'gini') #, min_samples_split = 2, min_samples_leaf = 1)
decisionTree_low1 = DecisionTreeClassifier(max_depth = 4, random_state=42, criterion = 'gini') #, min_samples_split = 2, min_samples_leaf = 1)
decisionTree_low2 = DecisionTreeClassifier(max_depth = 4, random_state=42, criterion = 'gini') #, min_samples_split = 2, min_samples_leaf = 1)
# Train Decision Tree Classifer
decisionTree_main_ID1ID3 = decisionTree_main_ID1ID3.fit(X_main_ID1ID3,Y_main_ID1ID3)
decisionTree_main_ID2 = decisionTree_main_ID2.fit(X_main_ID2,Y_main_ID2)
decisionTree_low1 = decisionTree_low1.fit(X_low1,Y_low1)
decisionTree_low2 = decisionTree_low2.fit(X_low2,Y_low2)

##Importance:
a0 = dict(zip(X_train_main_ID1ID3.columns, decisionTree_main_ID1ID3.feature_importances_))
a1 = dict(zip(X_train_main_ID2.columns, decisionTree_main_ID2.feature_importances_))
a2 = dict(zip(X_train_low1.columns, decisionTree_low1.feature_importances_))
a3 = dict(zip(X_train_low2.columns, decisionTree_low2.feature_importances_))


importances = pd.DataFrame(columns=['feature', 'DT1', 'DT2', 'DT3', 'DT4'])
for key in a0.keys():
    a0_f = round(a0[key],3)
    a1_f = round(a1[key],3)
    a2_f = '-'
    a3_f = '-'
    if key in a2:
        a2_f = round(a2[key],3)
    if key in a3:
        a3_f = round(a3[key],3)
    data = pd.DataFrame([[str(key), a2_f, a3_f, a0_f, a1_f]])
    data.columns = importances.columns
    importances = importances.append(data)


#%% RFE

maximum_depths = [8, 9, 10, 11, 12]
max_depth_DT1DT3 = [2, 3, 4, 5, 6]
cols_12 = ['d2', 'd3', 'd4', 'd5', 'd6']
cols_34 = ['d8', 'd9', 'd10', 'd11', 'd12']

rfe_main_ID1ID3 = pd.DataFrame(columns=['Feature', 'd8', 'd9', 'd10', 'd11', 'd12'])
rfe_main_ID1ID3['Feature'] = createTableKeys.keys()
rfe_main_ID2 = pd.DataFrame(columns=['Feature', 'd8', 'd9', 'd10', 'd11', 'd12'])
rfe_main_ID2['Feature'] = createTableKeys.keys()
rfe_low1 = pd.DataFrame(columns=['Feature', 'd2', 'd3', 'd4', 'd5', 'd6'])
rfe_low1['Feature'] = createTableKeys.keys()
rfe_low2 = pd.DataFrame(columns=['Feature', 'd2', 'd3', 'd4', 'd5', 'd6'])
rfe_low2['Feature'] = createTableKeys.keys()

for i in range(len(maximum_depths)):
    d = maximum_depths[i]
    d_ = max_depth_DT1DT3[i]
    #filename = 'figures/decision_tree/test3_all_features/rfe/rfe' + str(d) + '.csv'
    # Create Decision Tree classifer object
    decisionTree_main_ID1ID3 = DecisionTreeClassifier(max_depth = d, random_state=42, criterion = 'gini', min_samples_split = 15, min_samples_leaf = 5)
    decisionTree_main_ID2 = DecisionTreeClassifier(max_depth = d, random_state=42, criterion = 'gini', min_samples_split = 15, min_samples_leaf = 5)
    decisionTree_low1 = DecisionTreeClassifier(max_depth = d_, random_state=42, criterion = 'gini') #, min_samples_split = 2, min_samples_leaf = 1)
    decisionTree_low2 = DecisionTreeClassifier(max_depth = d_, random_state=42, criterion = 'gini') #, min_samples_split = 2, min_samples_leaf = 1)
    
    # create the RFE model and select 3 attributes
    rfe_main_ID1ID3_aux = RFE(decisionTree_main_ID1ID3, 1)
    rfe_main_ID1ID3_aux = rfe_main_ID1ID3_aux.fit(X_main_ID1ID3,Y_main_ID1ID3)
    rfe_main_ID2_aux = RFE(decisionTree_main_ID2, 1)
    rfe_main_ID2_aux = rfe_main_ID2_aux.fit(X_main_ID2,Y_main_ID2)
    rfe_low1_aux = RFE(decisionTree_low1, 1)
    rfe_low1_aux= rfe_low1_aux.fit(X_low1,Y_low1)
    rfe_low2_aux = RFE(decisionTree_low2, 1)
    rfe_low2_aux = rfe_low2_aux.fit(X_low2,Y_low2)
    
    #ranking of RFE:
    a0 = dict(zip(X_train_main_ID1ID3.columns, rfe_main_ID1ID3_aux.ranking_))
    a1 = dict(zip(X_train_main_ID2.columns, rfe_main_ID2_aux.ranking_))
    a2 = dict(zip(X_train_low1.columns, rfe_low1_aux.ranking_))
    a3 = dict(zip(X_train_low2.columns, rfe_low2_aux.ranking_))
    
    rfe = pd.DataFrame(columns=['Feature', 'DT1', 'DT2', 'DT3', 'DT4'])
    for key in a0.keys():
        a0_f = round(a0[key],3)
        a1_f = round(a1[key],3)
        a2_f = float('nan')
        a3_f = float('nan')
        if key in a2:
            a2_f = round(a2[key],3)
        if key in a3:
            a3_f = round(a3[key],3)
        data = pd.DataFrame([[str(key), a2_f, a3_f, a0_f, a1_f]])
        data.columns = rfe.columns
        rfe = rfe.append(data)
    
    #Add values to the previous df:
    for key in createTableKeys.keys():
        rfe_low1.loc[rfe_low1['Feature'] == key, cols_12[i]] = rfe.loc[rfe['Feature'] == key, 'DT1'][0]
        rfe_low2.loc[rfe_low2['Feature'] == key, cols_12[i]] = rfe.loc[rfe['Feature'] == key, 'DT2'][0]
        rfe_main_ID1ID3.loc[rfe_main_ID1ID3['Feature'] == key, cols_34[i]] = rfe.loc[rfe['Feature'] == key, 'DT3'][0]
        rfe_main_ID2.loc[rfe_main_ID2['Feature'] == key, cols_34[i]] = rfe.loc[rfe['Feature'] == key, 'DT4'][0]

#Substitute by values:
for key in createTableKeys.keys():
    rfe_low1.loc[rfe_low1['Feature'] == key,'Feature'] = createTableKeys[key]
    rfe_low2.loc[rfe_low2['Feature'] == key,'Feature'] = createTableKeys[key]
    rfe_main_ID1ID3.loc[rfe_main_ID1ID3['Feature'] == key,'Feature'] = createTableKeys[key]
    rfe_main_ID2.loc[rfe_main_ID2['Feature'] == key,'Feature'] = createTableKeys[key]
        
#Sort and save results   
rfe_low1 = rfe_low1.sort_values(by=['d2'], ascending=True)
rfe_low1.dropna()
rfe_low1.to_csv('figures/decision_tree/test3_all_features/rfe/rfe_DT1.csv', sep=';', encoding='utf-8', index=False)  
rfe_low2 = rfe_low2.sort_values(by=['d2'], ascending=True)
rfe_low2.dropna()
rfe_low2.to_csv('figures/decision_tree/test3_all_features/rfe/rfe_DT2.csv', sep=';', encoding='utf-8', index=False)  
rfe_main_ID1ID3 = rfe_main_ID1ID3.sort_values(by=['d8'], ascending=True)
rfe_main_ID1ID3.to_csv('figures/decision_tree/test3_all_features/rfe/rfe_DT3.csv', sep=';', encoding='utf-8', index=False)  
rfe_main_ID2 = rfe_main_ID2.sort_values(by=['d8'], ascending=True)
rfe_main_ID2.to_csv('figures/decision_tree/test3_all_features/rfe/rfe_DT4.csv', sep=';', encoding='utf-8', index=False)  


    
#rfe = rfe.sort_values(by=['DT3', 'DT1'], ascending=True)
#rfe.to_csv(filename, sep=';', encoding='utf-8', index=False)

#%%
    
a = dict()
dt = 'DT2'
for i in rfe['Feature']:
    if rfe.loc[rfe['Feature'] == i][dt][0] != "-":
        a.update({i : rfe.loc[rfe['Feature'] == i][dt][0]})
        
        
       


    





























