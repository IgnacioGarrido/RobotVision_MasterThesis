# SMRobot project
# Ignacio Garrido Botella
# DEPRECATED
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
import pickle


#My files:
#First have to change the working directory to where the files are:
os.chdir("/Users/ignacio/Documents/Universidad/Master/Segundo/SegundoSemestre/MasterThesisMoAI/Project/smrobot_py_knowledgeGraph")
import kGraph_functions as kgf

#%%

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

#%% Get the images of the corridors

df_id = 0
for df in [(df_1_im, df_1_gt), (df_2_im, df_2_gt), (df_3_im, df_3_gt), (df_4_im, df_4_gt)]:
    df_im = df[0]
    df_gt = df[1]
    ID_1 = 0
    ID_2 = int(len(df_im)/2)
    ID_3 = len(df_im) - 1
    df_id = df_id + 1
    im_id = 0
    for i in [ID_1,ID_2, ID_3]:
        filename_im = 'figures/images_corridor/raw_images_4_videos/df_' + str(df_id) + '_im_' + str(im_id) + '.png'
        filename_gt = 'figures/images_corridor/raw_images_4_videos/df_' + str(df_id) + '_gt_' + str(im_id) + '.png'
        im_id = im_id + 1
        plt.imshow(kgf.increaseContrast(df_im[i], alpha = 1.5))
        plt.axis('off')
        plt.savefig(filename_im)
        plt.imshow(df_gt[i])
        plt.axis('off')
        plt.savefig(filename_gt)
        

#%% Save 8 images tests
        
Images_ID = [0,70,100,130,150,160,178,198]

for i in Images_ID:
    filename = 'figures/test/0_rawImages/im' + str(Images_ID.index(i)+1) + '.png'
    image = df_merged_grayImages[i]
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, cmap = 'gray')
    plt.savefig(filename)





        









