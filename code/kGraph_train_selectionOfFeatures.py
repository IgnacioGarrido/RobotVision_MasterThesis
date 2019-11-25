# SMRobot project
# Ignacio Garrido Botella
# Auxiliar file for testing the extraction of features

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


#%% Selection of features
features_X_low1 = ['gray_intensity_neighbor', 'B1_proportion_edges', 'tile_mean_gray', 'B3_W_gray', 'gray_gradient', 'B1_W_lbp', 'max_grad_right', 'max_grad_left', 'tile_entropy_lbp', 'tile_entropy_gray', 'tile_proportion_edges', 'tile_X']
features_X_low2 = ['gray_gradient', 'tile_mean_gray', 'max_grad_right', 'max_grad_left', 'A1_proportion_edges', 'B3_W_gray', 'B1_proportion_edges', 'tile_entropy_lbp', 'tile_entropy_gray', 'tile_proportion_edges', 'tile_X']
features_X_main_ID1ID3 = ['tile_X', 'tile_Y', 'gray_gradient', 'tile_proportion_edges', 'tile_mean_gray', 'max_grad_up', 'max_grad_down', 'max_grad_right', 'max_grad_left', 'B1_proportion_edges', 'A1_proportion_edges', 'gray_intensity_neighbor', 'tile_entropy_gray', 'B5_W_gray', 'A2_proportion_edges', 'B1_W_gray', 'tile_entropy_lbp', 'A4_W_gray', 'A5_proportion_edges', 'B4_W_gray', 'f_area_size', 'A5_W_gray', 'B3_W_gray', 'tile_mean_lbp', 'A3_proportion_edges']
features_X_main_ID2 = ['tile_Y','tile_X',  'max_grad_up', 'max_grad_down', 'max_grad_right', 'max_grad_left', 'tile_entropy_gray', 'tile_proportion_edges', 'gray_gradient', 'tile_mean_gray', 'A3_proportion_edges', 'B3_proportion_edges', 'gray_intensity_neighbor', 'B9_W_gray', 'A4_proportion_edges', 'B2_proportion_edges', 'A9_proportion_edges', 'f_area_size', 'B3_W_lbp', 'B3_W_gray', 'B4_proportion_edges', 'tile_entropy_lbp', 'tile_mean_lbp']
features_Y = ['tile_class']
   
#%% Preprocessing: select feature and normalize:

df_Prediction = df_Prediction.fillna(int(-1))
df_Prediction = df_Prediction[~df_Prediction['tile_ID'].isin([0,1,2,3,4,5])]

#Eliminate the superpixels that are not being classified -> Those that are at the extremes of the image.
num_hyperpixels = int(max(df_Prediction['tile_ID']) + 1)
left1, left2 = kgf.getLeftHyperpixels(num_hyperpixels)
right1, right2 = kgf.getRightHyperpixels(num_hyperpixels)
low1, low2 = kgf.getLowerHyperpixels(num_hyperpixels)
top1, top2 = kgf.getTopHyperpixels(num_hyperpixels)
drop_tiles = left1 + left2 + right1+ right2 + top1 + top2
df_Prediction = df_Prediction[~df_Prediction['tile_ID'].isin(drop_tiles)]    

#Divide in foue datasets -> One for each DT
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

#Scaling and select features - main
df_Prediction_X_main_ID1ID3 = df_Prediction_main_ID1ID3.loc[ : , features_X_main_ID1ID3]
df_Prediction_Y_main_ID1ID3 = df_Prediction_main_ID1ID3.loc[ : , features_Y]
df_Prediction_X_main_ID2 = df_Prediction_main_ID2.loc[ : , features_X_main_ID2]
df_Prediction_Y_main_ID2 = df_Prediction_main_ID2.loc[ : , features_Y]
scaler_df_Prediction_main_ID1ID3 = preprocessing.MinMaxScaler()
df_Prediction_X_main_ID1ID3[features_X_main_ID1ID3] = scaler_df_Prediction_main_ID1ID3.fit_transform(df_Prediction_X_main_ID1ID3[features_X_main_ID1ID3])
df_Prediction_Y_main_ID1ID3[features_Y] = df_Prediction_Y_main_ID1ID3[features_Y].astype('category')
scaler_df_Prediction_main_ID2 = preprocessing.MinMaxScaler()
df_Prediction_X_main_ID2[features_X_main_ID2] = scaler_df_Prediction_main_ID2.fit_transform(df_Prediction_X_main_ID2[features_X_main_ID2])
df_Prediction_Y_main_ID2[features_Y] = df_Prediction_Y_main_ID2[features_Y].astype('category')

#Scaling and select features - low1
df_Prediction_X_low1 = df_Prediction_low1.loc[ : , features_X_low1]
df_Prediction_Y_low1 = df_Prediction_low1.loc[ : , features_Y]
scaler_df_Prediction_low1= preprocessing.MinMaxScaler()
df_Prediction_X_low1[features_X_low1] = scaler_df_Prediction_low1.fit_transform(df_Prediction_X_low1[features_X_low1])
df_Prediction_Y_low1[features_Y] = df_Prediction_Y_low1[features_Y].astype('category')

#Scaling and select features - low2
df_Prediction_X_low2 = df_Prediction_low2.loc[ : , features_X_low2]
df_Prediction_Y_low2 = df_Prediction_low2.loc[ : , features_Y]
scaler_df_Prediction_low2 = preprocessing.MinMaxScaler()
df_Prediction_X_low2[features_X_low2] = scaler_df_Prediction_low2.fit_transform(df_Prediction_X_low2[features_X_low2])
df_Prediction_Y_low2[features_Y] = df_Prediction_Y_low2[features_Y].astype('category')
del left1, left2, right1, right2, top1, top2, drop_tiles, num_hyperpixels, df_Prediction_main

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

X_train_main_ID2 = df_Prediction_X_main_ID2.loc[df_Prediction_main_ID2.loc[:,'image_ID'].isin(train_ids),:]
X_test_main_ID2 = df_Prediction_X_main_ID2.loc[df_Prediction_main_ID2.loc[:,'image_ID'].isin(test_ids),:]
y_train_main_ID2 = df_Prediction_Y_main_ID2.loc[df_Prediction_main_ID2.loc[:,'image_ID'].isin(train_ids),:]
y_test_main_ID2 = df_Prediction_Y_main_ID2.loc[df_Prediction_main_ID2.loc[:,'image_ID'].isin(test_ids),:]
#For the training
X_main_ID2 = pd.concat([X_train_main_ID2, X_test_main_ID2])
X_main_ID2 = X_main_ID2.reset_index(drop = True)
Y_main_ID2 = pd.concat([y_train_main_ID2, y_test_main_ID2])
Y_main_ID2 = Y_main_ID2.reset_index(drop = True)

X_train_low1 = df_Prediction_X_low1.loc[df_Prediction_low1.loc[:,'image_ID'].isin(train_ids),:]
X_test_low1 = df_Prediction_X_low1.loc[df_Prediction_low1.loc[:,'image_ID'].isin(test_ids),:]
y_train_low1 = df_Prediction_Y_low1.loc[df_Prediction_low1.loc[:,'image_ID'].isin(train_ids),:]
y_test_low1 = df_Prediction_Y_low1.loc[df_Prediction_low1.loc[:,'image_ID'].isin(test_ids),:]
#For the training
X_low1 = pd.concat([X_train_low1, X_test_low1])
X_low1 = X_low1.reset_index(drop = True)
Y_low1 = pd.concat([y_train_low1, y_test_low1])
Y_low1 = Y_low1.reset_index(drop = True)

X_train_low2 = df_Prediction_X_low2.loc[df_Prediction_low2.loc[:,'image_ID'].isin(train_ids),:]
X_test_low2 = df_Prediction_X_low2.loc[df_Prediction_low2.loc[:,'image_ID'].isin(test_ids),:]
y_train_low2 = df_Prediction_Y_low2.loc[df_Prediction_low2.loc[:,'image_ID'].isin(train_ids),:]
y_test_low2 = df_Prediction_Y_low2.loc[df_Prediction_low2.loc[:,'image_ID'].isin(test_ids),:]
#For the training
X_low2 = pd.concat([X_train_low2, X_test_low2])
X_low2 = X_low2.reset_index(drop = True)
Y_low2 = pd.concat([y_train_low2, y_test_low2])
Y_low2 = Y_low2.reset_index(drop = True)

del IDs, N, test_size, df_Prediction_low1, df_Prediction_low2, train_ids, test_ids


#%% Decision trees -> Training

#Split data
# Create Decision Tree classifer object
decisionTree_main_ID1ID3 = DecisionTreeClassifier(max_depth = 10, random_state=42, criterion = 'gini') #, min_samples_split = 15, min_samples_leaf = 5)
decisionTree_main_ID2 = DecisionTreeClassifier(max_depth = 10, random_state=42, criterion = 'gini') #, min_samples_split = 15, min_samples_leaf = 5)
decisionTree_low1 = DecisionTreeClassifier(max_depth = 3, random_state=42, criterion = 'gini') #, min_samples_split = 2, min_samples_leaf = 1)
decisionTree_low2 = DecisionTreeClassifier(max_depth = 3, random_state=42, criterion = 'gini') #, min_samples_split = 2, min_samples_leaf = 1)
# Train Decision Tree Classifer
decisionTree_main_ID1ID3 = decisionTree_main_ID1ID3.fit(X_main_ID1ID3,Y_main_ID1ID3)
decisionTree_main_ID2 = decisionTree_main_ID2.fit(X_main_ID2,Y_main_ID2)
decisionTree_low1 = decisionTree_low1.fit(X_low1,Y_low1)
decisionTree_low2 = decisionTree_low2.fit(X_low2,Y_low2)

##Importance:
a0 = dict(zip(X_main_ID1ID3.columns, decisionTree_main_ID1ID3.feature_importances_))
a1 = dict(zip(X_main_ID2.columns, decisionTree_main_ID2.feature_importances_))
a2 = dict(zip(X_low1.columns, decisionTree_low1.feature_importances_))
a3 = dict(zip(X_low2.columns, decisionTree_low2.feature_importances_))

importances = pd.DataFrame(columns=['Feature', 'DT1', 'DT2', 'DT3', 'DT4'])

list_keys = list(a0.keys())
list_keys.extend(x for x in list(a1.keys()) if x not in list_keys)
list_keys.extend(x for x in list(a2.keys()) if x not in list_keys)
list_keys.extend(x for x in list(a3.keys()) if x not in list_keys)

for key in list_keys:
    a0_f = '-'
    a1_f = '-'
    a2_f = '-'
    a3_f = '-'
    if key in a0:
        a0_f = round(a0[key],3)
    if key in a1:
        a1_f = round(a1[key],3)
    if key in a2:
        a2_f = round(a2[key],3)
    if key in a3:
        a3_f = round(a3[key],3)
    data = pd.DataFrame([[str(key), a2_f, a3_f, a0_f, a1_f]])
    data.columns = importances.columns
    importances = importances.append(data)
    
#Substitute by values:
for key in createTableKeys.keys():
    importances.loc[importances['Feature'] == key,'Feature'] = createTableKeys[key]
    
importances = importances.sort_values(by=['DT3', 'DT1'], ascending=False)
#importances.to_csv('figures/decision_tree/test3_all_features/importances/imp.csv', sep=';', encoding='utf-8', index = False)

##Save the visualization of the decision tree:
#dot_data = StringIO()
#feat_names = df_Prediction_X_main_ID1ID3.columns
#export_graphviz(decisionTree_main_ID1ID3, out_file=dot_data, filled=True, rounded=True, special_characters=True,feature_names = feat_names, class_names=['0','1','2'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png('figures/decision_tree/test3_all_features/dtID1ID3.png')
#dot_data = StringIO()
#feat_names = df_Prediction_X_main_ID2.columns
#export_graphviz(decisionTree_main_ID2, out_file=dot_data, filled=True, rounded=True, special_characters=True,feature_names = feat_names, class_names=['0','1','2'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png('figures/decision_tree/test3_all_features/dtID2.png')
#dot_data = StringIO()
#feat_names = df_Prediction_X_low1.columns
#export_graphviz(decisionTree_low1, out_file=dot_data, filled=True, rounded=True, special_characters=True,feature_names = feat_names, class_names=['0','1','2'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png('figures/decision_tree/test3_all_features/dtlow1.png')
#dot_data = StringIO()
#feat_names = df_Prediction_X_low2.columns
#export_graphviz(decisionTree_low2, out_file=dot_data, filled=True, rounded=True, special_characters=True,feature_names = feat_names, class_names=['0','1','2'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#graph.write_png('figures/decision_tree/test3_all_features/dtlow2.png')

#%% Predict and show image

#Images_IDs: 0,70,100,130,150,160,178,198
image_ID = 198

filename = 'figures/test/3_allFeatures/im' + str(Images_ID.index(image_ID)+1) + '.png'

#Get the image
image = df_merged_grayImages[image_ID]
image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)

#Get and scale the datasets for scaling
X_pred = df_Test.loc[df_Test['image_ID']==image_ID, :]
X_pred.index = X_pred['tile_ID'].astype('int')
X_pred_main = X_pred[~X_pred['tile_ID'].isin(low1+low2)]  
X_pred_main_ID1ID3 = X_pred_main[X_pred_main['tile_section_of_analysis'] != 2]
X_pred_main_ID2 = X_pred_main[X_pred_main['tile_section_of_analysis'] == 2] 
del X_pred_main
X_pred_low1 = X_pred[X_pred['tile_ID'].isin(low1)]  
X_pred_low2 = X_pred[X_pred['tile_ID'].isin(low2)] 
X_pred_main_ID1ID3 = X_pred_main_ID1ID3.loc[ : , features_X_main_ID1ID3]
X_pred_main_ID2 = X_pred_main_ID2.loc[ : , features_X_main_ID2]
X_pred_low1 = X_pred_low1.loc[ : , features_X_low1]
X_pred_low2 = X_pred_low2.loc[ : , features_X_low2]

X_pred_main_ID1ID3[features_X_main_ID1ID3] = scaler_df_Prediction_main_ID1ID3.transform(X_pred_main_ID1ID3[features_X_main_ID1ID3])
X_pred_main_ID2[features_X_main_ID2] = scaler_df_Prediction_main_ID2.transform(X_pred_main_ID2[features_X_main_ID2])
X_pred_low1[features_X_low1] = scaler_df_Prediction_low1.transform(X_pred_low1[features_X_low1])
X_pred_low2[features_X_low2] = scaler_df_Prediction_low2.transform(X_pred_low2[features_X_low2])

#Ordering of the columns
X_pred_main_ID1ID3 = X_pred_main_ID1ID3[X_main_ID1ID3.columns]
X_pred_main_ID2 = X_pred_main_ID2[X_main_ID2.columns]
X_pred_low1 = X_pred_low1[X_low1.columns]
X_pred_low2 = X_pred_low2[X_low2.columns]
      
#columns_online = ['class_previous_tile_back','class_previous_tile_edge', 'class_previous_tile_floor', 'A1_class_back', 'A1_class_edge', 'A1_class_floor', 'A2_class_back', 'A2_class_edge', 'A2_class_floor', 'A5_class_back', 'A5_class_edge', 'A5_class_floor', 'A6_class_back', 'A6_class_edge', 'A6_class_floor', 'A8_class_back', 'A8_class_edge', 'A8_class_floor']
#classes_to_analyse= ['class_previous_tile','A1_class','A2_class','A5_class','A6_class', 'A8_class']
columns_online = [] #Global knowledge
classes_to_analyse = [] #Global knowledge
classes_pref_funct = kgf.predictTree4DT(X_pred_main_ID1ID3, X_pred_main_ID2, X_pred_low1, X_pred_low2, decisionTree_main_ID1ID3, decisionTree_main_ID2, decisionTree_low1, decisionTree_low2, image = df_merged_grayImages[0], num = 20, classes_names = classes_to_analyse, columns_online = columns_online)

#Plot the image
a = kgf.paintHyperpixelsNoBorder4DT(image, classes_pref_funct)
fig = plt.figure(frameon=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
ax.imshow(a)
#plt.savefig(filename)

del filename   

#%% Loop accuracies/depth

#Number of maximum depths tested
num_depths = 25
num_depths_low = 10

accuracy_ID1ID3 = np.zeros([num_depths-1])
accuracy_ID2 = np.zeros([num_depths-1])
for i in range(num_depths-1):
    decisionTree_main_ID1ID3 = DecisionTreeClassifier(max_depth = i+2, random_state=42, criterion = 'gini') #, min_samples_split =  15, min_samples_leaf = 5)
    decisionTree_main_ID2 = DecisionTreeClassifier(max_depth = i+2, random_state=42, criterion = 'gini') #, min_samples_split = 15, min_samples_leaf = 5)
    # Train Decision Tree Classifer
    decisionTree_main_ID1ID3 = decisionTree_main_ID1ID3.fit(X_train_main_ID1ID3,y_train_main_ID1ID3)
    decisionTree_main_ID2 = decisionTree_main_ID2.fit(X_train_main_ID2,y_train_main_ID2)
    #Predict the response for test dataset
    y_pred_main_ID1ID3 = decisionTree_main_ID1ID3.predict(X_test_main_ID1ID3)
    y_pred_main_ID2 = decisionTree_main_ID2.predict(X_test_main_ID2)
    #Accuracy:
    accuracy_ID1ID3[i] = metrics.accuracy_score(y_test_main_ID1ID3, y_pred_main_ID1ID3)
    accuracy_ID2[i] = metrics.accuracy_score(y_test_main_ID2, y_pred_main_ID2)
    
accuracy_low1 = np.zeros([num_depths_low-1])
accuracy_low2 = np.zeros([num_depths_low-1])
for i in range(num_depths_low-1):
    decisionTree_low1 = DecisionTreeClassifier(max_depth = i+2, random_state=42, criterion = 'gini') #, min_samples_split = 2, min_samples_leaf = 1)
    decisionTree_low2 = DecisionTreeClassifier(max_depth = i+2, random_state=42, criterion = 'gini') #, min_samples_split = 2, min_samples_leaf = 1)
    # Train Decision Tree Classifer
    decisionTree_low1 = decisionTree_low1.fit(X_train_low1,y_train_low1)
    decisionTree_low2 = decisionTree_low2.fit(X_train_low2,y_train_low2)
    #Predict the response for test dataset
    y_pred_low1 = decisionTree_low1.predict(X_test_low1)
    y_pred_low2 = decisionTree_low2.predict(X_test_low2)
    #Accuracy:
    accuracy_low1[i] = metrics.accuracy_score(y_test_low1, y_pred_low1)
    accuracy_low2[i] = metrics.accuracy_score(y_test_low2, y_pred_low2)


#Plot
labels_main = list(range(2, num_depths+1))    
    
fig = plt.figure()
plt.plot(labels_main, accuracy_ID1ID3)
fig.suptitle('Accuracy DT3', fontsize=15)
plt.xlabel('Depth of the tree', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.xlim((2,num_depths))
plt.ylim((0.8,1.01))
plt.savefig('figures/decision_tree/test3_all_features/accuracies/test_ID1ID3_depths.png')
fig = plt.figure()

plt.plot(labels_main, accuracy_ID2)
fig.suptitle('Accuracy DT4', fontsize=15)
plt.xlabel('Depth of the tree', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.xlim((2,num_depths))
plt.ylim((0.8,1.01))
plt.savefig('figures/decision_tree/test3_all_features/accuracies/test_ID2_depths.png')
fig = plt.figure()

labels_low = list(range(2, num_depths_low+1))

plt.plot(labels_low, accuracy_low1)
fig.suptitle('Accuracy of DT1', fontsize=15)
plt.xlabel('Depth of the tree', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.xlim((2,num_depths_low))
plt.ylim((0.99,1.01))
plt.savefig('figures/decision_tree/test3_all_features/accuracies/test_low1_depths.png')
fig = plt.figure()

plt.plot(labels_low, accuracy_low2)
fig.suptitle('Accuracy of DT2', fontsize=15)
plt.xlabel('Depth of the tree', fontsize=10)
plt.ylabel('Accuracy', fontsize=10)
plt.xlim((2,num_depths_low))
plt.ylim((0.98,1.01))
plt.savefig('figures/decision_tree/test3_all_features/accuracies/test_low2_depths.png')



