#%%
# SMRobot project
# Ignacio Garrido Botella
# conda environment: SMRobot
import cv2
import numpy as np
import math
import pandas as pd
from skimage.segmentation import felzenszwalb


   
#GETHYPERPIXELS
#Description: This function receives an image and returns an array with the 
#   hyperpixels.
#
#   @Inputs:
#       image: image (grayscale!!!).
#       sizeX: size (number of pixels) of the hyperpixels in X axis. By default
#           it is 18 (image of size 400x702 & 780 sections).
#       sizeY: size (number of pixels) of the hyperpixels in Y axis. By default
#           it is 20 (image of size 400x702 & 780 sections).
#       num: Number of hyperpixels in Y axis (vertical). The number of 
#           hyperpixels in the horizontal axis will be num*2-1. By default it 
#           is 20 (image of size 400x702 & 780 sections). 
#   @Outputs:
#       hyperpixelsArray: Numpy array with the next columns ordered:
#           - 0: ID of the hyperpixel
#           - 1: ID of the superior hyperpixel
#           - 2: ID of the hyperpixel at right
#           - 3: ID of the lower hyperpixel
#           - 4: ID of the left hyperpixel
#           - 5: Position in the X axis relative to the center of the image
#           - 6: Position in the Y axis relative to the bottom of the image
#           - 7: ID of previous tile (see previous tile's class)
#           - 8: Section of analysis ID1 (1), ID2 (2) or ID3 (3)
#           - ...as many columns as pixels there are in an hyperpixel, with the
#             values of those pixels
def getHyperpixels(image, sizeX = 18, sizeY = 20, num = 20):
    #Normalize the number of pixels for doing the squared traverse
    numY = num
    numX = num*2-1
    dim = (sizeX*numX, sizeY*numY)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) 
    hyperpixelsArray = np.zeros([numX*numY,(9+sizeX*sizeY)])
    hyperpixelsArray[0,0] = 0 #Own ID
    hyperpixelsArray[0,1] = 3 #ID of the upper hyperpixel
    hyperpixelsArray[0,2] = 5 #ID of the right hyperpixel
    hyperpixelsArray[0,3] = float('nan') #ID of the lower hyperpixel  
    hyperpixelsArray[0,4] = 1 #ID of the left hyperpixel
    hyperpixelsArray[0,5] = 0 #Position X
    hyperpixelsArray[0,6] = 0 #Position Y
    hyperpixelsArray[0,7] = float('nan')  #ID of previous tile
    hyperpixelsArray[0,8] = 2 #Section of analysis
    hyperpixelsArray[0,9:(9+sizeX*sizeY)] = image[(numY-1)*sizeY: numY*sizeY, (int(numX/2)*sizeX):(int(numX/2 + 1)*sizeX)].flatten()
    ID = 1
    #Get hyperpixels in the correct order:
    for i in range(numY-1):
        #We traverse the square:
        #Go up -> En el cuaderno -> ID1
        for j in range(i+2):
            if j == 0: #Lower hyperpixel in ID1
                hyperpixelsArray[ID,0] = ID #Own ID
                hyperpixelsArray[ID,1] = ID+1 #ID of the upper hyperpixel
                hyperpixelsArray[ID,2] = ID-i*4-1 #ID of the right hyperpixel
                hyperpixelsArray[ID,3] = float('nan') #ID of the lower hyperpixel  
                if i == numY-2: #Border
                    hyperpixelsArray[ID,4] = float('nan') #ID of the left hyperpixel 
                else:                        
                    hyperpixelsArray[ID,4] = ID + 4*i + 5 #ID of the left hyperpixel
                hyperpixelsArray[ID,5] = i+1 #Position X
                hyperpixelsArray[ID,6] = 0 #Position Y
                hyperpixelsArray[ID,7] = ID-i*4-1  #ID of previous tile
                hyperpixelsArray[ID,8] = 1 #Section of analysis
                hyperpixelsArray[ID,9:(9+sizeX*sizeY)] = image[(numY-1)*sizeY: numY*sizeY, (int(numX/2-(i+1))*sizeX):(int(numX/2-i)*sizeX)].flatten()                              
            elif j == (i+1): #Higher hyperpixel in ID1
                hyperpixelsArray[ID,0] = ID #Own ID
                if i == numY-2: #Border
                    hyperpixelsArray[ID,1] = float('nan') #ID of the superior hyperpixel
                else:
                    hyperpixelsArray[ID,1] = ID+4*i+7 #ID of the upper hyperpixel
                hyperpixelsArray[ID,2] = ID+1 #ID of the right hyperpixel
                hyperpixelsArray[ID,3] = ID-1 #ID of the lower hyperpixel  
                if i == numY-2: #Border
                    hyperpixelsArray[ID,4] = float('nan') #ID of the left hyperpixel 
                else:                        
                    hyperpixelsArray[ID,4] = ID + 4*i + 5 #ID of the left hyperpixel                 
                hyperpixelsArray[ID,5] = i+1 #Position X
                hyperpixelsArray[ID,6] = i+1 #Position Y
                hyperpixelsArray[ID,7] = ID-i*4-2  #ID of previous tile
                hyperpixelsArray[ID,8] = 1 #Section of analysis
                hyperpixelsArray[ID,9:(9+sizeX*sizeY)] = image[(numY-1-j)*sizeY: (numY-j)*sizeY, (int(numX/2-(i+1))*sizeX):(int(numX/2-i)*sizeX)].flatten()                        
            else: #Intermediate hyperpixels in ID1
                hyperpixelsArray[ID,0] = ID #Own ID
                hyperpixelsArray[ID,1] = ID+1 #ID of the upper hyperpixel
                hyperpixelsArray[ID,2] = ID-i*4-1 #ID of the right hyperpixel
                hyperpixelsArray[ID,3] = ID-1 #ID of the lower hyperpixel  
                if i == numY-2: #Border
                    hyperpixelsArray[ID,4] = float('nan') #ID of the left hyperpixel 
                else:                        
                    hyperpixelsArray[ID,4] = ID + 4*i + 5 #ID of the left hyperpixel   
                hyperpixelsArray[ID,5] = i+1 #Position X
                hyperpixelsArray[ID,6] = j #Position Y
                hyperpixelsArray[ID,7] = ID-i*4-1  #ID of previous tile
                hyperpixelsArray[ID,8] = 1 #Section of analysis
                hyperpixelsArray[ID,9:(9+sizeX*sizeY)] = image[(numY-1-j)*sizeY: (numY-j)*sizeY, (int(numX/2-(i+1))*sizeX):(int(numX/2-i)*sizeX)].flatten()                                
            ID = ID+1                
        #Go up -> En el cuaderno -> ID2
        for j in range(i*2+1):
            hyperpixelsArray[ID,0] = ID #Own ID
            if i == numY-2: #Border
                hyperpixelsArray[ID,1] = float('nan') #ID of the superior hyperpixel
            else:
                hyperpixelsArray[ID,1] = ID+4*i+7 #ID of the upper hyperpixel
            hyperpixelsArray[ID,2] = ID+1 #ID of the right hyperpixel            
            hyperpixelsArray[ID,3] = ID-4*i-3 #ID of the lower hyperpixel  
            hyperpixelsArray[ID,4] = ID - 1 #ID of the left hyperpixel   
            hyperpixelsArray[ID,5] = abs(j-i) #Position X
            hyperpixelsArray[ID,6] = i+1 #Position Y
            hyperpixelsArray[ID,7] = ID-4*i-3  #ID of previous tile
            hyperpixelsArray[ID,8] = 2 #Section of analysis
            hyperpixelsArray[ID,9:(9+sizeX*sizeY)] = image[(numY-2-i)*sizeY: (numY-1-i)*sizeY, (int(numX/2+(j-i))*sizeX):(int(numX/2+(j-i)+1)*sizeX)].flatten()                      
            ID = ID+1                 
        #Go up -> En el cuaderno -> ID3
        for j in range(i+2):
            if j == 0: #Upper hyperpixel in ID3         
                hyperpixelsArray[ID,0] = ID #Own ID            
                if i == numY-2: #Border
                    hyperpixelsArray[ID,1] = float('nan') #ID of the upper hyperpixel
                else:
                    hyperpixelsArray[ID,1] = ID+4*i+7 #ID of the upper hyperpixel
                hyperpixelsArray[ID,2] = ID+1 #ID of the right hyperpixel
                if i == numY-2: #Border
                    hyperpixelsArray[ID,2] = float('nan') #ID of the right hyperpixel 
                else:                        
                    hyperpixelsArray[ID,2] = ID+4*i+9 #ID of the right hyperpixel 
                hyperpixelsArray[ID,3] = ID+1 #ID of the lower hyperpixel  
                hyperpixelsArray[ID,4] = ID-1 #ID of the left hyperpixel                 
                hyperpixelsArray[ID,5] = i+1 #Position X
                hyperpixelsArray[ID,6] = i+1 #Position Y
                hyperpixelsArray[ID,7] = ID-4*i-4  #ID of previous tile
                hyperpixelsArray[ID,8] = 3 #Section of analysis
                hyperpixelsArray[ID,9:(9+sizeX*sizeY)] = image[(numY-i-2)*sizeY: (numY-i-1)*sizeY, (int(numX/2+(i+1))*sizeX):(int(numX/2+(i+2))*sizeX)].flatten()                               
            elif j == (i+1): #Lower hyperpixel in ID3
                hyperpixelsArray[ID,0] = ID #Own ID
                hyperpixelsArray[ID,1] = ID-1 #ID of the upper hyperpixel
                if i == numY-2: #Border
                    hyperpixelsArray[ID,2] = float('nan') #ID of the right hyperpixel 
                else:                        
                    hyperpixelsArray[ID,2] = ID + 4*i + 9 #ID of the right hyperpixel
                hyperpixelsArray[ID,3] = float('nan') #ID of the lower hyperpixel  
                hyperpixelsArray[ID,4] = ID-4*i-5 #ID of the left hyperpixel
                hyperpixelsArray[ID,5] = i+1 #Position X
                hyperpixelsArray[ID,6] = 0 #Position Y
                hyperpixelsArray[ID,7] = ID-4*i-5  #ID of previous tile
                hyperpixelsArray[ID,8] = 3 #Section of analysis
                hyperpixelsArray[ID,9:(9+sizeX*sizeY)] = image[(numY-1)*sizeY: numY*sizeY, (int(numX/2+(i+1))*sizeX):(int(numX/2+(i+2))*sizeX)].flatten()                                
            else: #Intermediate hyperpixels in ID1
                hyperpixelsArray[ID,0] = ID #Own ID
                hyperpixelsArray[ID,1] = ID-1 #ID of the upper hyperpixel
                if i == numY-2: #Border
                    hyperpixelsArray[ID,2] = float('nan') #ID of the right hyperpixel 
                else:                        
                    hyperpixelsArray[ID,2] = ID+4*i+9 #ID of the right hyperpixel                 
                hyperpixelsArray[ID,3] = ID+1 #ID of the lower hyperpixel  
                hyperpixelsArray[ID,4] = ID-4*i-5 #ID of the left hyperpixel                
                hyperpixelsArray[ID,5] = i+1 #Position X
                hyperpixelsArray[ID,6] = i+1-j #Position Y   
                hyperpixelsArray[ID,7] = ID-4*i-5  #ID of previous tile
                hyperpixelsArray[ID,8] = 3 #Section of analysis
                hyperpixelsArray[ID,9:(9+sizeX*sizeY)] = image[(numY-2-i+j)*sizeY: (numY-1-i+j)*sizeY, (int(numX/2+(i+1))*sizeX):(int(numX/2+(i+2))*sizeX)].flatten()                                
            ID = ID+1                   
    return hyperpixelsArray



#PAINTHYPERPIXELS
#Description: This function receives an image and an array with the class of 
#   each hyperpixel, and returns the weighted image with each of the classes
#   overlaped with the original image.
#
#   @Inputs:
#       image: image in grayscale.
#       array_tilesClass: Array with two columns, the first is the ID of the 
#           tile and the second is the class of the tile. 
#           (0 -> background, 1 -> floor, 2 -> edge)
#   @Outputs:
#       overlapedImage: Image with the coloured tiles and the input overlapped.
#   @Call:
#       paintHyperpixels(grayImage, getClass(groundThruth))
def paintHyperpixels(image, array_tilesClass):
    rows = image.shape[0]
    cols = image.shape[1]
    num_exp = int(1+(-3+math.sqrt(9-8*(1-len(array_tilesClass))))/4) #We get the number of "expansions" of squares there are (It is the number of hyperpixels in the Y axis).
    sizeY = int(math.ceil(rows/num_exp)) #Size of the hyperpixel in Y axis
    sizeX = int(math.ceil(cols/(num_exp*2 - 1))) #Size of the hyperpixel in X axis
    numY = num_exp #Number of hyperpixels in Y axis
    numX = num_exp*2-1 #Number of hyperpixels in X axis
    dim = (int(sizeX*numX), int(sizeY*numY))
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) #Reshape the image with the used standards
    if len(image.shape) == 2:
        mask = np.zeros([int(sizeY*numY),int(sizeX*numX)], dtype=np.uint8) #mask where the hyperpixels are painted
    elif len(image.shape) == 3:
        mask = np.zeros([int(sizeY*numY),int(sizeX*numX),3], dtype=np.uint8) #mask where the hyperpixels are painted        
    count = 0 #To keep the count of the number of hyperpixels that have already been painted
    for i in range(num_exp):
        sizeUp = i+1 #Number of hiperpixels in ID1 per expansion (num_exp)
        sizeMid = 2*(i+1)-3 #Number of hiperpixels in ID2 per expansion (num_exp)
        sizeDown = i+1 #Number of hiperpixels in ID3 per expansion (num_exp)
        for j in range(sizeUp): #Paint hyperpixels of ID1
            #Notice, j=0 is lower hyperpixel of ID1
            if array_tilesClass[count,1] == 1: #If it is detected as floor
                mask[(numY-1-j)*sizeY: (numY-j)*sizeY, (int(numX/2-(i))*sizeX):(int(numX/2-(i-1))*sizeX)] = 20 if len(image.shape) == 2 else (0,255,0)
            elif array_tilesClass[count,1] == 0: #If it is detected as background
                mask[(numY-1-j)*sizeY: (numY-j)*sizeY, (int(numX/2-(i))*sizeX):(int(numX/2-(i-1))*sizeX)] = 160 if len(image.shape) == 2 else (51,255,255)
            else: #If it is detected as edge
                mask[(numY-1-j)*sizeY: (numY-j)*sizeY, (int(numX/2-(i))*sizeX):(int(numX/2-(i-1))*sizeX)] = 250 if len(image.shape) == 2 else (102,0,204)
            count = count + 1
        for j in range(sizeMid): #Paint hyperpixels of ID2
            #Notice, j=0 is the most left hyperpixel of ID2
            if array_tilesClass[count,1] == 1: #If it is detected as floor
                mask[(numY-2-(i-1))*sizeY: (numY-1-(i-1))*sizeY, (int(numX/2+(j-(i-1)))*sizeX):(int(numX/2+(j-(i-1))+1)*sizeX)] = 20 if len(image.shape) == 2 else (0,255,0)
            elif array_tilesClass[count,1] == 0: #If it is detected as background
                mask[(numY-2-(i-1))*sizeY: (numY-1-(i-1))*sizeY, (int(numX/2+(j-(i-1)))*sizeX):(int(numX/2+(j-(i-1))+1)*sizeX)] = 160 if len(image.shape) == 2 else (51,255,255)
            else: #If it is detected as edge
                mask[(numY-2-(i-1))*sizeY: (numY-1-(i-1))*sizeY, (int(numX/2+(j-(i-1)))*sizeX):(int(numX/2+(j-(i-1))+1)*sizeX)] = 250 if len(image.shape) == 2 else (102,0,204)
            count = count + 1
        for j in range(sizeDown): #Paint hyperpixels of ID3
            #Notice, j=0 is the most upper hyperpixel of ID3
            if array_tilesClass[count,1] == 1: #If it is detected as floor
                mask[(numY-2-(i-1)+j)*sizeY: (numY-1-(i-1)+j)*sizeY, (int(numX/2+(i))*sizeX):(int(numX/2+(i+1))*sizeX)] = 20 if len(image.shape) == 2 else (0,255,0)
            elif array_tilesClass[count,1] == 0: #If it is detected as background
                mask[(numY-2-(i-1)+j)*sizeY: (numY-1-(i-1)+j)*sizeY, (int(numX/2+(i))*sizeX):(int(numX/2+(i+1))*sizeX)] = 160 if len(image.shape) == 2 else (51,255,255)
            else: #If it is detected as edge
                mask[(numY-2-(i-1)+j)*sizeY: (numY-1-(i-1)+j)*sizeY, (int(numX/2+(i))*sizeX):(int(numX/2+(i+1))*sizeX)] = 250 if len(image.shape) == 2 else (102,0,204)
            count = count + 1
        if i == 0:
            count = count - 1
    overlapedImage = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
    return overlapedImage
            


#PAINTHYPERPIXELSNOBORDER4DT
#Description: This function receives an image and an array with the class of 
#   each hyperpixel, and returns the weighted image with each of the classes
#   overlaped with the original image. However, it eliminates the border from 
#   the image (low, right, left).
#
#   @Inputs:
#       image: image in grayscale.
#       array_tilesClass: Array with two columns, the first is the ID of the 
#           tile and the second is the class of the tile. 
#           (0 -> background, 1 -> floor, 2 -> edge)
#   @Outputs:
#       overlapedImage: Image with the coloured tiles and the input overlapped.
#   @Call:
#       paintHyperpixels(grayImage, getClass(groundThruth))
def paintHyperpixelsNoBorder4DT(image, array_tilesClass):
    rows = image.shape[0]
    cols = image.shape[1]
    num_exp = int(1+(-3+math.sqrt(9-8*(1-len(array_tilesClass))))/4) #We get the number of "expansions" of squares there are (It is the number of hyperpixels in the Y axis).
    sizeY = int(math.ceil(rows/num_exp)) #Size of the hyperpixel in Y axis
    sizeX = int(math.ceil(cols/(num_exp*2 - 1))) #Size of the hyperpixel in X axis
    numY = num_exp #Number of hyperpixels in Y axis
    numX = num_exp*2-1 #Number of hyperpixels in X axis
    dim = (int(sizeX*numX), int(sizeY*numY))
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) #Reshape the image with the used standards
    if len(image.shape) == 2:
        mask = np.zeros([int(sizeY*numY),int(sizeX*numX)], dtype=np.uint8) #mask where the hyperpixels are painted
    elif len(image.shape) == 3:
        mask = np.zeros([int(sizeY*numY),int(sizeX*numX),3], dtype=np.uint8) #mask where the hyperpixels are painted        
    count = 0 #To keep the count of the number of hyperpixels that have already been painted
    for i in range(num_exp-2):
        sizeUp = i+1 #Number of hiperpixels in ID1 per expansion (num_exp)
        sizeMid = 2*(i+1)-3 #Number of hiperpixels in ID2 per expansion (num_exp)
        sizeDown = i+1 #Number of hiperpixels in ID3 per expansion (num_exp)
        for j in range(sizeUp): #Paint hyperpixels of ID1
            #Notice, j=0 is lower hyperpixel of ID1
            if array_tilesClass[count,1] == 1: #If it is detected as floor
                mask[(numY-1-j)*sizeY: (numY-j)*sizeY, (int(numX/2-(i))*sizeX):(int(numX/2-(i-1))*sizeX)] = 20 if len(image.shape) == 2 else (0,255,0)
            elif array_tilesClass[count,1] == 0: #If it is detected as background
                mask[(numY-1-j)*sizeY: (numY-j)*sizeY, (int(numX/2-(i))*sizeX):(int(numX/2-(i-1))*sizeX)] = 160 if len(image.shape) == 2 else (51,255,255)
            else: #If it is detected as edge
                mask[(numY-1-j)*sizeY: (numY-j)*sizeY, (int(numX/2-(i))*sizeX):(int(numX/2-(i-1))*sizeX)] = 250 if len(image.shape) == 2 else (102,0,204)
            count = count + 1
        for j in range(sizeMid): #Paint hyperpixels of ID2
            #Notice, j=0 is the most left hyperpixel of ID2
            if array_tilesClass[count,1] == 1: #If it is detected as floor
                mask[(numY-2-(i-1))*sizeY: (numY-1-(i-1))*sizeY, (int(numX/2+(j-(i-1)))*sizeX):(int(numX/2+(j-(i-1))+1)*sizeX)] = 20 if len(image.shape) == 2 else (0,255,0)
            elif array_tilesClass[count,1] == 0: #If it is detected as background
                mask[(numY-2-(i-1))*sizeY: (numY-1-(i-1))*sizeY, (int(numX/2+(j-(i-1)))*sizeX):(int(numX/2+(j-(i-1))+1)*sizeX)] = 160 if len(image.shape) == 2 else (51,255,255)
            else: #If it is detected as edge
                mask[(numY-2-(i-1))*sizeY: (numY-1-(i-1))*sizeY, (int(numX/2+(j-(i-1)))*sizeX):(int(numX/2+(j-(i-1))+1)*sizeX)] = 250 if len(image.shape) == 2 else (102,0,204)
            count = count + 1
        for j in range(sizeDown): #Paint hyperpixels of ID3
            #Notice, j=0 is the most upper hyperpixel of ID3
            if array_tilesClass[count,1] == 1: #If it is detected as floor
                mask[(numY-2-(i-1)+j)*sizeY: (numY-1-(i-1)+j)*sizeY, (int(numX/2+(i))*sizeX):(int(numX/2+(i+1))*sizeX)] = 20 if len(image.shape) == 2 else (0,255,0)
            elif array_tilesClass[count,1] == 0: #If it is detected as background
                mask[(numY-2-(i-1)+j)*sizeY: (numY-1-(i-1)+j)*sizeY, (int(numX/2+(i))*sizeX):(int(numX/2+(i+1))*sizeX)] = 160 if len(image.shape) == 2 else (51,255,255)
            else: #If it is detected as edge
                mask[(numY-2-(i-1)+j)*sizeY: (numY-1-(i-1)+j)*sizeY, (int(numX/2+(i))*sizeX):(int(numX/2+(i+1))*sizeX)] = 250 if len(image.shape) == 2 else (102,0,204)
            count = count + 1
        if i == 0:
            count = count - 1
    overlapedImage = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
    overlapedImage = overlapedImage[(sizeY*2):,(sizeX*2):sizeX*(numX-2) , :] #Reduce the sice of the image
    return overlapedImage



#PAINTCLUSTERS
#Description: This function receives an image and an array with the cluster of 
#   each of the hyperpixels, and returns the overlapped image with the class of
#   each hyperpixel and the original image. Additionaly you can choose which 
#   class to paint.
#
#   @Inputs:
#       image: image in COLOR!!!
#       cluster_array: Array with three columns, the first is the ID of the 
#           tile and the second is the ID of the cluster of the tile. The third
#           one is the class of the hyperpixel. 
#           (0 -> background, 1 -> floor, 2 -> edge)
#       classToPaint: Class to paint. 
#           (0 -> background, 1 -> floor,  2 -> edge, 3 -> all)
#   @Outputs:
#       overlapedImage: Image with the coloured tiles and the input overlapped.
#   @Call:
#       paintHyperpixels(grayImage, getClass(groundThruth))
def paintClusters(image, cluster_array, classToPaint = 3):
    rows = image.shape[0]
    cols = image.shape[1]
    num_exp = int(1+(-3+math.sqrt(9-8*(1-len(cluster_array))))/4) #We get the number of "expansions" of squares there are (It is the number of hyperpixels in the Y axis).
    sizeY = int(math.ceil(rows/num_exp)) #Size of the hyperpixel in Y axis
    sizeX = int(math.ceil(cols/(num_exp*2 - 1))) #Size of the hyperpixel in X axis
    numY = num_exp #Number of hyperpixels in Y axis
    numX = num_exp*2-1 #Number of hyperpixels in X axis
    dim = (int(sizeX*numX), int(sizeY*numY))
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) #Reshape the image with the used standards
    mask = np.zeros([int(sizeY*numY),int(sizeX*numX), 3], dtype=np.uint8) #mask where the hyperpixels are painted
    count = 0 #To keep the count of the number of hyperpixels that have already been painted
    normClusters = np.zeros([len(cluster_array)])
    #Select clusters to paint:
    if classToPaint == 3: #Paint all
        maxClusterValue = np.max(cluster_array[:,1])
        normClusters = cluster_array[:,1]/maxClusterValue #Normalize the cluster's values
    elif classToPaint == 0: #Paint only background
        maxClusterValue = np.max(cluster_array[np.where(cluster_array[:,2] == 0), 1])
        for i in range(len(cluster_array)):
            if cluster_array[i,2] == 0: #Only give a value to those tiles of background. Rest are valued 0.
                normClusters[i] = cluster_array[i,1]/maxClusterValue #Normalize the cluster's values    
    elif classToPaint == 1: #Paint only floor
        maxClusterValue = np.max(cluster_array[np.where(cluster_array[:,2] == 1), 1])
        for i in range(len(cluster_array)):
            if cluster_array[i,2] == 1: #Only give a value to those tiles of floor. Rest are valued 0.
                normClusters[i] = cluster_array[i,1]/maxClusterValue #Normalize the cluster's values  
    else: #Paint only edges
        maxClusterValue = np.max(cluster_array[np.where(cluster_array[:,2] == 2), 1])
        for i in range(len(cluster_array)):
            if cluster_array[i,2] == 2: #Only give a value to those tiles of edges. Rest are valued 0.
                normClusters[i] = cluster_array[i,1]/maxClusterValue #Normalize the cluster's values  
    #Paint each of the clusters:
    for i in range(num_exp):
        sizeUp = i+1 #Number of hiperpixels in ID1 per expansion (num_exp)
        sizeMid = 2*(i+1)-3 #Number of hiperpixels in ID2 per expansion (num_exp)
        sizeDown = i+1 #Number of hiperpixels in ID3 per expansion (num_exp)
        for j in range(sizeUp): #Paint hyperpixels of ID1
            #Notice, j=0 is lower hyperpixel of ID1
            mask[(numY-1-j)*sizeY: (numY-j)*sizeY, (int(numX/2-(i))*sizeX):(int(numX/2-(i-1))*sizeX), :] = [int(normClusters[count]*255), int(255-normClusters[count]*255), int(abs(122.5 - normClusters[count]*255))]                   
            if classToPaint == 0 and cluster_array[count,2] != 0: #Paint only background
                mask[(numY-1-j)*sizeY: (numY-j)*sizeY, (int(numX/2-(i))*sizeX):(int(numX/2-(i-1))*sizeX), :] = image[(numY-1-j)*sizeY: (numY-j)*sizeY, (int(numX/2-(i))*sizeX):(int(numX/2-(i-1))*sizeX), :]
            elif classToPaint == 1 and cluster_array[count,2] != 1: #Paint only floor
                mask[(numY-1-j)*sizeY: (numY-j)*sizeY, (int(numX/2-(i))*sizeX):(int(numX/2-(i-1))*sizeX), :] = image[(numY-1-j)*sizeY: (numY-j)*sizeY, (int(numX/2-(i))*sizeX):(int(numX/2-(i-1))*sizeX), :]
            elif classToPaint == 2 and cluster_array[count,2] != 2:
                mask[(numY-1-j)*sizeY: (numY-j)*sizeY, (int(numX/2-(i))*sizeX):(int(numX/2-(i-1))*sizeX), :] = image[(numY-1-j)*sizeY: (numY-j)*sizeY, (int(numX/2-(i))*sizeX):(int(numX/2-(i-1))*sizeX), :]
            count = count + 1
        for j in range(sizeMid): #Paint hyperpixels of ID2
            #Notice, j=0 is the most left hyperpixel of ID2
            mask[(numY-2-(i-1))*sizeY: (numY-1-(i-1))*sizeY, (int(numX/2+(j-(i-1)))*sizeX):(int(numX/2+(j-(i-1))+1)*sizeX),:] = [int(normClusters[count]*255), int(255-normClusters[count]*255), int(abs(122.5 - normClusters[count]*255))] 
            if classToPaint == 0 and cluster_array[count,2] != 0: #Paint only background
                mask[(numY-2-(i-1))*sizeY: (numY-1-(i-1))*sizeY, (int(numX/2+(j-(i-1)))*sizeX):(int(numX/2+(j-(i-1))+1)*sizeX),:] = image[(numY-2-(i-1))*sizeY: (numY-1-(i-1))*sizeY, (int(numX/2+(j-(i-1)))*sizeX):(int(numX/2+(j-(i-1))+1)*sizeX),:]
            elif classToPaint == 1 and cluster_array[count,2] != 1: #Paint only floor
                mask[(numY-2-(i-1))*sizeY: (numY-1-(i-1))*sizeY, (int(numX/2+(j-(i-1)))*sizeX):(int(numX/2+(j-(i-1))+1)*sizeX),:] = image[(numY-2-(i-1))*sizeY: (numY-1-(i-1))*sizeY, (int(numX/2+(j-(i-1)))*sizeX):(int(numX/2+(j-(i-1))+1)*sizeX),:]
            elif classToPaint == 2 and cluster_array[count,2] != 2:
                mask[(numY-2-(i-1))*sizeY: (numY-1-(i-1))*sizeY, (int(numX/2+(j-(i-1)))*sizeX):(int(numX/2+(j-(i-1))+1)*sizeX),:] = image[(numY-2-(i-1))*sizeY: (numY-1-(i-1))*sizeY, (int(numX/2+(j-(i-1)))*sizeX):(int(numX/2+(j-(i-1))+1)*sizeX),:]
            count = count + 1
        for j in range(sizeDown): #Paint hyperpixels of ID3
            #Notice, j=0 is the most upper hyperpixel of ID3
            mask[(numY-2-(i-1)+j)*sizeY: (numY-1-(i-1)+j)*sizeY, (int(numX/2+(i))*sizeX):(int(numX/2+(i+1))*sizeX),:] = [int(normClusters[count]*255), int(255-normClusters[count]*255), int(abs(122.5 - normClusters[count]*255))]
            if classToPaint == 0 and cluster_array[count,2] != 0: #Paint only background
                mask[(numY-2-(i-1)+j)*sizeY: (numY-1-(i-1)+j)*sizeY, (int(numX/2+(i))*sizeX):(int(numX/2+(i+1))*sizeX),:] = image[(numY-2-(i-1)+j)*sizeY: (numY-1-(i-1)+j)*sizeY, (int(numX/2+(i))*sizeX):(int(numX/2+(i+1))*sizeX),:]
            elif classToPaint == 1 and cluster_array[count,2] != 1: #Paint only floor
                mask[(numY-2-(i-1)+j)*sizeY: (numY-1-(i-1)+j)*sizeY, (int(numX/2+(i))*sizeX):(int(numX/2+(i+1))*sizeX),:] = image[(numY-2-(i-1)+j)*sizeY: (numY-1-(i-1)+j)*sizeY, (int(numX/2+(i))*sizeX):(int(numX/2+(i+1))*sizeX),:]
            elif classToPaint == 2 and cluster_array[count,2] != 2:
                mask[(numY-2-(i-1)+j)*sizeY: (numY-1-(i-1)+j)*sizeY, (int(numX/2+(i))*sizeX):(int(numX/2+(i+1))*sizeX),:] = image[(numY-2-(i-1)+j)*sizeY: (numY-1-(i-1)+j)*sizeY, (int(numX/2+(i))*sizeX):(int(numX/2+(i+1))*sizeX),:]
            count = count + 1
        if i == 0: #Tile zero is counted 2 times
            count = count - 1
    overlapedImage = cv2.addWeighted(image, 0.6, mask, 0.4, 0)
    return overlapedImage
     
      

#GETLOWERHYPERPIXELS
#Description: This function receives the number of hyperpixels that an image 
#   contains and returns two lists with the IDs of those superpixels that are 
#   at the bottom of the image.
#
#   @Inputs:
#       numHyperpixels: Number of hyperpixels that an image contains
#   @Outputs:
#       list_Hyperpixels_LowRow1: List of the IDs of the first row of the lower
#           hyperpixels.
#       list_Hyperpixels_LowRow2: List of the IDs of the second row of the 
#           lower hyperpixels.
def getLowerHyperpixels(numHyperpixels):
    num_exp = int((-3+math.sqrt(9-8*(1-numHyperpixels)))/4)
    list_Hyperpixels_LowRow1 = list()
    list_Hyperpixels_LowRow2 = list()
    list_Hyperpixels_LowRow1.append(0)
    list_Hyperpixels_LowRow1.append(1)
    list_Hyperpixels_LowRow2.append(2)
    list_Hyperpixels_LowRow2.append(3)
    list_Hyperpixels_LowRow2.append(4)
    list_Hyperpixels_LowRow1.append(5)
    ID_1 = 1
    ID_2 = 5
    for i in range(num_exp-1):
        ID_1 = 4*i + 5 + ID_1
        ID_2 = 4*i + 9 + ID_2
        list_Hyperpixels_LowRow1.append(ID_1)
        list_Hyperpixels_LowRow1.append(ID_2)
        list_Hyperpixels_LowRow2.append(ID_1 + 1)
        list_Hyperpixels_LowRow2.append(ID_2 - 1)
    return list_Hyperpixels_LowRow1, list_Hyperpixels_LowRow2
 

  
#GETRIGHTHYPERPIXELS
#Description: This function receives the number of hyperpixels that an image 
#   contains and returns two lists with the IDs of those superpixels that are 
#   at the right part of the image.
#
#   @Inputs:
#       numHyperpixels: Number of hyperpixels that an image contains
#   @Outputs:
#       list_Hyperpixels_LowRow1: List of the IDS of the first row of the right
#           hyperpixels.
#       list_Hyperpixels_LowRow2: List of the IDs of the second row of the 
#           right hyperpixels.
def getRightHyperpixels(numHyperpixels):
    num_exp = int(1+(-3+math.sqrt(9-8*(1-numHyperpixels)))/4)
    list_Hyperpixels_RightRow1 = list(range(numHyperpixels-num_exp, numHyperpixels))
    list_Hyperpixels_RightRow2 = list(range(numHyperpixels-5*num_exp+4, numHyperpixels-4*num_exp+3))
    return list_Hyperpixels_RightRow1, list_Hyperpixels_RightRow2



#GETLEFTHYPERPIXELS
#Description: This function receives the number of hyperpixels that an image 
#   contains and returns two lists with the IDs of those superpixels that are 
#   at the left part of the image.
#
#   @Inputs:
#       numHyperpixels: Number of hyperpixels that an image contains
#   @Outputs:
#       list_Hyperpixels_LowRow1: List of the IDS of the first row of the left
#           hyperpixels.
#       list_Hyperpixels_LowRow2: List of the IDs of the second row of the 
#           left hyperpixels.
def getLeftHyperpixels(numHyperpixels):
    num_exp = int(1+(-3+math.sqrt(9-8*(1-numHyperpixels)))/4)
    list_Hyperpixels_LeftRow1 = list(range(numHyperpixels-4*num_exp+3, numHyperpixels-3*num_exp+3))
    list_Hyperpixels_LeftRow2 = list(range(numHyperpixels-8*num_exp+10, numHyperpixels-7*num_exp+9))
    return list_Hyperpixels_LeftRow1, list_Hyperpixels_LeftRow2



#GETTOPHYPERPIXELS
#Description: This function receives the number of hyperpixels that an image 
#   contains and returns two lists with the IDs of those superpixels that are 
#   at the top part of the image.
#
#   @Inputs:
#       numHyperpixels: Number of hyperpixels that an image contains
#   @Outputs:
#       list_Hyperpixels_TopRow1: List of the IDS of the first row of the top
#           hyperpixels.
#       list_Hyperpixels_TopRow2: List of the IDs of the second row of the 
#           top hyperpixels.
def getTopHyperpixels(numHyperpixels):
    num_exp = int(1+(-3+math.sqrt(9-8*(1-numHyperpixels)))/4)
    list_Hyperpixels_TopRow1 = list(range(numHyperpixels-3*num_exp+2, numHyperpixels-num_exp+1))
    list_Hyperpixels_TopRow2 = list(range(numHyperpixels-7*num_exp+8, numHyperpixels-5*num_exp+5))
    return list_Hyperpixels_TopRow1, list_Hyperpixels_TopRow2


     
#GETEDGES
#Description: This function receives an image and returns an array with the 
#   hyperpixels and a boolean indicating if they contain edges and the 
#   proportion of pixels detected as edges.
#
#   @Inputs:
#       image: image in grayscale.
#       sizeX: size (number of pixels) of the hyperpixels in X axis. By default
#           it is 18 (image of size 400x702 & 780 sections).
#       sizeY: size (number of pixels) of the hyperpixels in Y axis. By default
#           it is 20 (image of size 400x702 & 780 sections).
#       num: Number of hyperpixels in Y axis (vertical). The number of 
#           hyperpixels in the horizontal axis will be num*2-1. By default it 
#           is 20 (image of size 400x702 & 780 sections). 
#   @Outputs:
#       listEdges: Array with three columns, the first is the ID of the 
#           tile, the second is a boolean (1 -> the hyperpixel contains edges)
#           and the third is the proportion of pixels in the hyperpixel
#           detected as edges.
def getEdges(image, sizeX = 18, sizeY = 20, num = 20):
    dim = (int(sizeX*(2*num-1)), int(sizeY*num))
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) #Reshape the image with the used standards
    #Settings for canny edge detection
    lower_limit = 30 #45
    higher_limit = 70 #90
    edges_image = cv2.Canny(image,lower_limit,higher_limit)   
    listHyperpixels = getHyperpixels(edges_image, sizeX, sizeY, num)
    listHyperpixels = listHyperpixels[:,9:] #Remove all metadata
    listEdges = np.zeros([num*(2*num-1),3])
    for i in range(num*(2*num-1)): #Set the ID numbers (first column)
        listEdges[i,0] = i
    for i in range(num*(2*num-1)):
        if (255 in listHyperpixels[i,:]): #Edge
            listEdges[i,1] = 1
        else:
            listEdges[i,1] = 0
        listEdges[i,2] = np.sum(listHyperpixels[i,:])/(255*sizeX*sizeY)
    return listEdges
 


#GETORDEROFANALYSIS
#Description: This function receives the number of hyperpixels and returns 
#   an array with the ordered IDs for the analysis.
#
#   @Inputs:
#       numHyperpixels: Number of hyperpixels
#   @Outputs:
#       listOrderedID: Array with four columns, the first is the ID of the 
#           tile (they are ordered in the supposed order of analysis), the 
#           second indicates to which section of analysis it belongs
#           (ID1 -> 1, ID2 -> 2 or ID3 -> 3), the third one indicates if
#           there are at least two hyperpixels between the border of the image  
#           and the tile (0 -> zero hyp, 1 -> one hyp, 2 -> two or more hyp)
#           and the last one indicates the expansion number.
def getOrderOfAnalysis(numHyperpixels):
    listOrderedID = np.zeros([numHyperpixels, 4])
    num_exp = int((-3+math.sqrt(9-8*(1-numHyperpixels)))/4) #We get the number of "expansions" of squares there are (It is the number of hyperpixels in the Y axis).
    count = 1 #To keep the count of the number of hyperpixels that have already been painted
    listOrderedID[0,0] = 0 #ID of hyperpixel zero
    listOrderedID[0,1] = 2 #Section of analysis of hyperpixel zero
    listOrderedID[0,2] = 0 #Minimum number of hyperpixels to the border of the image 
    listOrderedID[0,3] = -1 #Expansion number
    for i in range(num_exp):
        sizeUp = i+2 #Number of hiperpixels in ID1 per expansion (num_exp)
        sizeMid = 2*(i+1)-1 #Number of hiperpixels in ID2 per expansion (num_exp)
        sizeDown = i+2 #Number of hiperpixels in ID3 per expansion (num_exp)
        numHyp = 0 #Number of hyperpixels between the border of the image and the hyperpixel
        for j in range(sizeUp): #Paint hyperpixels of ID1
            #Notice, j=0 is lower hyperpixel of ID1
            listOrderedID[count,0] = count #ID of hyperpixel
            listOrderedID[count,1] = 1 #Section of analysis of hyperpixel
            if sizeMid == (2*num_exp + 1 - 2): #Last expansion
                listOrderedID[count,2] = 0 #Minimum number of hyperpixels to the border of the image 
            elif sizeMid == (2*num_exp + 1 - 4): #Penultimate expansion
                listOrderedID[count,2] = 1 #Minimum number of hyperpixels to the border of the image
            else:
                listOrderedID[count,2] = numHyp #Minimum number of hyperpixels to the border of the image 
                if numHyp < 2:
                    numHyp  = numHyp + 1
            listOrderedID[count,3] = i #Expansion number
            count = count + 1
        for j in range(sizeMid): #Paint hyperpixels of ID2
            #Notice, j=0 is the most left hyperpixel of ID2
            listOrderedID[count,0] = count #ID of hyperpixel
            listOrderedID[count,1] = 2 #Section of analysis of hyperpixel
            if sizeMid == (2*num_exp + 1 - 2): #Last expansion
                listOrderedID[count,2] = 0 #Minimum number of hyperpixels to the border of the image 
            elif sizeMid == (2*num_exp + 1 - 4) or count == 3: #Penultimate expansion
                listOrderedID[count,2] = 1 #Minimum number of hyperpixels to the border of the image
            else:
                listOrderedID[count,2] = 2 #Minimum number of hyperpixels to the border of the image
            listOrderedID[count,3] = i #Expansion number
            count = count + 1
        prevCount = count #To check the ID
        numHyp = 0 #Number of hyperpixels between the border of the image and the hyperpixel
        for j in range(sizeDown): #Paint hyperpixels of ID3
            #Notice, j=0 is the most upper hyperpixel of ID3
            listOrderedID[count,0] = prevCount + sizeDown - j - 1 #ID of hyperpixel
            listOrderedID[count,1] = 3 #Section of analysis of hyperpixel
            if sizeMid == (2*num_exp + 1 - 2): #Last expansion
                listOrderedID[count,2] = 0 #Minimum number of hyperpixels to the border of the image 
            elif sizeMid == (2*num_exp + 1 - 4): #Penultimate expansion
                listOrderedID[count,2] = 1 #Minimum number of hyperpixels to the border of the image
            else:
                listOrderedID[count,2] = numHyp #Minimum number of hyperpixels to the border of the image 
                if numHyp < 2:
                    numHyp  = numHyp + 1
            listOrderedID[count,3] = i #Expansion number
            count = count + 1
    listOrderedID = pd.DataFrame(data=listOrderedID, index=range(numHyperpixels), columns = ['ID','Expansion','Num_hyperpixels_to_border', 'Expansion_number'])
    return listOrderedID



#GETID1ID2ID3HYPERPIXELS
#Description: This function receives the number of hyperpixels and returns two 
#   lists with the IDs of those superpixels that are at the the sections of 
#   analysis ID1ID3 and ID2
#  
#
#   @Inputs:
#       image: Example of an image.
#   @Outputs:
#       list_ID1ID3: List of the IDS of the section of analysis ID1ID3.
#       list_ID2: List of the IDS of the section of analysis ID2.
def getID1ID2ID3Hyperpixels(numHyperpixels):
    df_ordered = getOrderOfAnalysis(numHyperpixels)
    list_ID1ID3 = list(df_ordered[df_ordered['Expansion'] != 2]['ID'])
    list_ID2 = list(df_ordered[df_ordered['Expansion'] == 2]['ID'])
    return list_ID1ID3, list_ID2    



#GETNEIGHBORSID
#Description: This function receives an image and returns an array with the 
#   neighbors of each hyperpixel.
#
#   @Inputs:
#       image: image (grayscale!!!).
#       sizeX: size (number of pixels) of the hyperpixels in X axis. By default
#           it is 18 (image of size 400x702 & 780 sections).
#       sizeY: size (number of pixels) of the hyperpixels in Y axis. By default
#           it is 20 (image of size 400x702 & 780 sections).
#       num: Number of hyperpixels in Y axis (vertical). The number of 
#           hyperpixels in the horizontal axis will be num*2-1. By default it 
#           is 20 (image of size 400x702 & 780 sections). 
#   @Outputs:
#       tileNeigh: Pandas df with the next columns ordered:
#           - 0: ID of the tile.
#           - 1: ID of the hyperpixel A1
#           - 2: ID of the hyperpixel A2
#           - 3: ID of the hyperpixel A3
#           - 4: ID of the hyperpixel A4
#           - 5: ID of the hyperpixel A5
#           - 6: ID of the hyperpixel A9
#           - 7: ID of the hyperpixel A6
#           - 8: ID of the hyperpixel A8
#           - 9: ID of the hyperpixel B1
#           - 10: ID of the hyperpixel B2
#           - 11: ID of the hyperpixel B3
#           - 12: ID of the hyperpixel B4
#           - 13: ID of the hyperpixel B5
#           - 14: ID of the hyperpixel B9
#           - 15: ID of the previous tile.
#           - 16: ID of the next tile. 
def getNeighborsId(image, sizeX = 18, sizeY = 20, num = 20):
    hyperpixels = getHyperpixels(image, sizeX, sizeY, num)
    tileNeigh = np.zeros([len(hyperpixels), 17])
    for i in range(len(hyperpixels)):
        tileNeigh[i,0] = i
        if i == 0:
            tileNeigh[i,1] = float('nan') #A1
            tileNeigh[i,2] = float('nan') #A2
            tileNeigh[i,3] = int(5) #A3
            tileNeigh[i,4] = int(4) #A4
            tileNeigh[i,5] = float('nan') #A5
            tileNeigh[i,6] = int(14) #A9
            tileNeigh[i,7] = float('nan') #A6
            tileNeigh[i,8] = float('nan') #A8
            tileNeigh[i,9] = int(3) #B1
            tileNeigh[i,10] = int(2) #B2
            tileNeigh[i,11] = int(1) #B1
            tileNeigh[i,12] = float('nan') #B4
            tileNeigh[i,13] = int(10) #B5
            tileNeigh[i,14] = int(6) #B9
            tileNeigh[i,15] = float('nan') #ID of the previous tile
            tileNeigh[i,16] = int(3) #ID of the next tile
        elif hyperpixels[i,8] == 1: #ID1
            tileNeigh[i,1] = hyperpixels[i,3] #A1
            if not math.isnan(hyperpixels[i,3]): #Not in the lower part of the image
                tileNeigh[i,2] = hyperpixels[int(hyperpixels[i,3]),2] #A2
            else:
                tileNeigh[i,2] = float('nan')
            tileNeigh[i,3] = hyperpixels[i,2] #A3           
            tileNeigh[i,4] = hyperpixels[int(hyperpixels[i,2]),1] #A4
            if not math.isnan(hyperpixels[i,3]): #Not in the lower part of the image
                tileNeigh[i,5] = hyperpixels[int(hyperpixels[i,3]),3] #A5
            else:
                tileNeigh[i,5] = float('nan')
            tileNeigh[i,6] = hyperpixels[int(hyperpixels[i,2]),2]# A9
            if not math.isnan(tileNeigh[i,5]): #Not in the lower part of the image
                tileNeigh[i,7] = hyperpixels[int(tileNeigh[i,5]),2] #A6
            else:
                tileNeigh[i,7] = float('nan') #A6
            if not math.isnan(tileNeigh[i,2]): #Not in the lower part of the image
                tileNeigh[i,8] = hyperpixels[int(tileNeigh[i,2]),2] #A8
            else:
                tileNeigh[i,8] = float('nan') #A8
            tileNeigh[i,9] = hyperpixels[i,1] #B1
            if not math.isnan(hyperpixels[i,1]):
                tileNeigh[i,10] = hyperpixels[int(hyperpixels[i,1]), 4] #B2
            else:
                tileNeigh[i,10] = float('nan') #B2  
            tileNeigh[i,11] = hyperpixels[i,4] #B3
            if not math.isnan(hyperpixels[i,4]):
                tileNeigh[i,12] = hyperpixels[int(hyperpixels[i,4]),3]#B4
            else:
                tileNeigh[i,12] = float('nan')#B4
            if not math.isnan(hyperpixels[i,1]):
                tileNeigh[i,13] = hyperpixels[int(hyperpixels[i,1]),1] #B5
            else:
                tileNeigh[i,13] = float('nan') #B5
            if not math.isnan(hyperpixels[i,4]):
                tileNeigh[i,14] = hyperpixels[int(hyperpixels[i,4]),4] #B9
            else:
                tileNeigh[i,14] = float('nan') #B9
            tileNeigh[i,15] = hyperpixels[i,7] #ID of the previous tile
            tileNeigh[i,16] = hyperpixels[i,4] #ID of the next tile
        elif hyperpixels[i,8] == 2: #ID2
            tileNeigh[i,1] = hyperpixels[i,4] #A1
            tileNeigh[i,2] = hyperpixels[int(hyperpixels[i,4]),3] #A2
            tileNeigh[i,3] = hyperpixels[i,3] #A3
            tileNeigh[i,4] = hyperpixels[int(hyperpixels[i,3]),2] #A4
            tileNeigh[i,5] = hyperpixels[int(hyperpixels[i,4]),4] #A5
            tileNeigh[i,6] = hyperpixels[int(hyperpixels[i,3]),3] #A9
            tileNeigh[i,7] = hyperpixels[int(hyperpixels[int(hyperpixels[i,4]),3]),4] #A6
            tileNeigh[i,8] = hyperpixels[int(hyperpixels[int(hyperpixels[i,4]),3]),3] #A8     
            tileNeigh[i,9] = hyperpixels[i,2] #B1  
            tileNeigh[i,10] = hyperpixels[int(hyperpixels[i,2]),1] #B2
            tileNeigh[i,11] = hyperpixels[i,1] #B3
            tileNeigh[i,12] = hyperpixels[int(hyperpixels[i,4]),1] #B4           
            tileNeigh[i,13] = hyperpixels[int(hyperpixels[i,2]),2] #B5                
            if not math.isnan(hyperpixels[i,1]):
                tileNeigh[i,14] = hyperpixels[int(hyperpixels[i,1]),1] #B9
            else:
                tileNeigh[i,14] = float('nan') #B9                
            tileNeigh[i,15] = hyperpixels[i,7] #ID of the previous tile           
            tileNeigh[i,16] = hyperpixels[i,1] #ID of the next tile           
        elif hyperpixels[i,8] == 3: #ID3
            tileNeigh[i,1] = hyperpixels[i,3] #A1
            if not math.isnan(hyperpixels[i,3]): #Not in the lower part of the image
                tileNeigh[i,2] = hyperpixels[int(hyperpixels[i,3]),4] #A2
            else:
                tileNeigh[i,2] = float('nan')
            tileNeigh[i,3] = hyperpixels[i,4] #A3
            tileNeigh[i,4] = hyperpixels[int(hyperpixels[i,4]),1] #A4
            if not math.isnan(hyperpixels[i,3]): #Not in the lower part of the image
                tileNeigh[i,5] = hyperpixels[int(hyperpixels[i,3]),3] #A5
            else:
                tileNeigh[i,5] = float('nan')
            tileNeigh[i,6] = hyperpixels[int(hyperpixels[i,4]),4] #A9
            if not math.isnan(tileNeigh[i,5]): #Not in the lower part of the image
                tileNeigh[i,7] = hyperpixels[int(tileNeigh[i,5]),4] #A6
            else:
                tileNeigh[i,7] = float('nan') #A6
            if not math.isnan(tileNeigh[i,2]): #Not in the lower part of the image
                tileNeigh[i,8] = hyperpixels[int(tileNeigh[i,2]),4] #A8
            else:
                tileNeigh[i,8] = float('nan') #A8
            tileNeigh[i,9] = hyperpixels[i,1] #B1
            if not math.isnan(hyperpixels[i,1]):
                tileNeigh[i,10] = hyperpixels[int(hyperpixels[i,1]), 2] #B2
            else:
                tileNeigh[i,10] = float('nan') #B2 
            tileNeigh[i,11] = hyperpixels[i,2] #B3
            if not math.isnan(hyperpixels[i,2]):
                tileNeigh[i,12] = hyperpixels[int(hyperpixels[i,2]),3] #B4
            else:
                tileNeigh[i,12] = float('nan') #B4
            if not math.isnan(hyperpixels[i,1]):
                tileNeigh[i,13] = hyperpixels[int(hyperpixels[i,1]),1] #B5
            else:
                tileNeigh[i,13] = float('nan') #B5
            if not math.isnan(hyperpixels[i,2]):
                tileNeigh[i,14] = hyperpixels[int(hyperpixels[i,2]),2] #B9
            else:
                tileNeigh[i,14] = float('nan') #B9
            tileNeigh[i,15] = hyperpixels[i,7] #ID of the previous tile
            tileNeigh[i,16] = hyperpixels[i,2] #ID of the next tile
    tileNeigh = pd.DataFrame(data=tileNeigh[0:,0:], index=range(len(tileNeigh)), columns=['Tile_ID', 'A1', 'A2', 'A3', 'A4', 'A5', 'A9', 'A6', 'A8', 'B1', 'B2', 'B3', 'B4', 'B5', 'B9', 'Prev_tile_ID', 'Next_tile_ID']) #Create pandas df
    return tileNeigh



#INCREASECONTRAST
#Description: This function receives an image and increases the contrast of it
#   with the operation new_image = image*alpha + beta.     
#
#   @Inputs:
#       image: Image
#       alpha: alpha value of the previous equation. By default it is 2.
#       beta: beta value of the previous equation. By default it is 0.
#   @Outputs:
#       new_image: Image after transformation.
def increaseContrast(image, alpha = 2, beta = 0):
    new_image = np.zeros(image.shape, image.dtype)
    if len(image.shape) == 3:
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                for c in range(image.shape[2]):
                    new_image[y,x,c] = np.clip(alpha*image[y,x,c] + beta, 0, 255)
    elif len(image.shape) == 2:
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                    new_image[y,x] = np.clip(alpha*image[y,x] + beta, 0, 255)
    else:
        return image
    return new_image



#UNIQUELINES
#Description: This function receives an array of lines as the one given with 
#   the function cv2.HoughLines, and eliminates those lines that are 
#   repeated.
#
#   @Inputs:
#       lines: array of lines (cv2.HoughLines).
#       threshold_line: Max distance between the lines to be considered the 
#           same line.
#       threshold_angle: Max difference between the angles to be considered the
#           same line.
#   @Outputs:
#       different_lines: List with the different lines.  
def uniqueLines(lines, threshold_line = 50, threshold_angle = 0.15):
    different_lines = list()
    for i in reversed(range(len(lines))): #Loop backwards
        Flag_valid_line = 1
        for n in range(i):
            if (abs(lines[i,0,0] - lines[n,0,0]) < threshold_line) and (abs(lines[i,0,1] - lines[n,0,1]) < threshold_angle):
                Flag_valid_line = 0
        if Flag_valid_line:
            different_lines.append([lines[i,0,0], lines[i,0,1]])
    return different_lines



#UNIQUELINESP
#Description: This function receives an array of lines as the one given with 
#   the function cv2.HoughLinesP, and merges those that belong to the same line.
#
#   @Inputs:
#       lines: array of lines (cv2.HoughLinesP).
#       threshold_slope: Max distance between the slopes to be considered the 
#           same line.
#       threshold_distance: Max distance between the extreme points of the 
#           lines to be considered the same line.
#       threshold_perpendicular_distance: Perpendicular distance to the line.
#   @Outputs:
#       final_lines: List with the different lines. Each position of the list 
#           has the form [x1,y1,x2,y2]
def uniqueLinesP(lines, threshold_slope = 0.05, threshold_distance = 50, threshold_perpendicular_distance = 5):
    if not isinstance(lines,list): #If lines is not a list -> convert it to a list
        list_lines = list()
        for i in range(len(lines)):
            list_lines.append([lines[i,0,0], lines[i,0,1], lines[i,0,2], lines[i,0,3]]) 
        lines = list_lines
    final_lines = list()
    for i in range(len(lines)): 
        slope_line = (lines[i][1]-lines[i][3])/(lines[i][0]-lines[i][2]+0.01)
        Flag_merge_Lines = 0
        LinesToMerge = list()
        for n in range(i+1,len(lines), 1):
            slope_new_line = (lines[n][1]-lines[n][3])/(lines[n][0]-lines[n][2]+0.01)
            if (abs(slope_line - slope_new_line) < threshold_slope) and ((math.sqrt(pow((lines[i][1] - lines[n][3]),2) + pow((lines[i][0] - lines[n][2]),2)) < threshold_distance) or (math.sqrt(pow((lines[n][1] - lines[i][3]),2) + pow((lines[n][0] - lines[i][2]),2)) < threshold_distance) or (math.sqrt(pow((lines[i][1] - lines[n][1]),2) + pow((lines[i][0] - lines[n][0]),2)) < threshold_distance) or (math.sqrt(pow((lines[i][2] - lines[n][2]),2) + pow((lines[i][3] - lines[n][3]),2)) < threshold_distance) ):                         
                #distance_perpendicular line < threshold
                lineA = lines[i]
                p1 = np.array([lineA[0],lineA[1]])
                p2 = np.array([lineA[2],lineA[3]])
                p3 = np.array([lines[n][0],lines[n][1]])
                p4 = np.array([lines[n][2],lines[n][3]])
                d5 = np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1) #Perpendicular distance from extreme point 1 of the line that is being analysed point to lineA
                d6 = np.cross(p2-p1,p4-p1)/np.linalg.norm(p2-p1) #Perpendicular distance from extreme point 2 of the line that is being analysed point to lineA              
                if (d5 <= threshold_perpendicular_distance) and (d6 <= threshold_perpendicular_distance):
                    LinesToMerge.append([lines[n][0], lines[n][1], lines[n][2], lines[n][3], n])
                    Flag_merge_Lines = 1               
        if Flag_merge_Lines:
            x_min = lines[i][0]
            y_min = lines[i][1]
            x_max = lines[i][2]
            y_max = lines[i][3]
            listN = list()
            for line in LinesToMerge:               
                x_min_new = min(x_min, line[0])
                y_min = (y_min, line[1])[(x_min, line[0]).index(x_min_new)]
                x_min = x_min_new
                x_max_new = max(x_max, line[2])
                y_max = (y_max, line[3])[(x_max, line[2]).index(x_max_new)]
                x_max = x_max_new
                listN.append(line[4])  
            final_lines.append([x_min, y_min, x_max, y_max])
            for j in range(i+1,len(lines), 1):
                if j not in listN:
                    final_lines.append(lines[j])
            return uniqueLinesP(final_lines, threshold_slope, threshold_distance)
        final_lines.append([lines[i][0], lines[i][1], lines[i][2], lines[i][3]]) 
    return final_lines



#UNIQUELINESPIMPROVED
#Description: This function receives an array of lines as the one given with 
#   the function cv2.HoughLinesP, and merges those that belong to the same line.
#   This functionoptaines an improved result in comparison with uniqueLinesP as
#   there are no smaller lines left.
#
#   @Inputs:
#       listLines: array of lines (cv2.HoughLinesP).
#       threshold_slope: Max distance between the slopes to be considered the 
#           same line.
#       threshold_distance: Max distance between the extreme points of the 
#           lines to be considered the same line.
#       threshold_perpendicular_distance: Perpendicular distance to the line.
#   @Outputs:
#       final_lines: List with the different lines. Each position of the list 
#           has the form [x1,y1,x2,y2]
def uniqueLinesPImproved(listLines, threshold_slope = 0.1, threshold_distance = 30, threshold_perpendicular_distance = 10):
    if not isinstance(listLines,list): #If lines is not a list -> convert it to a list
        new_listLines = list()
        for i in range(len(listLines)):
            new_listLines.append([listLines[i,0,0], listLines[i,0,1], listLines[i,0,2], listLines[i,0,3]]) 
        listLines = new_listLines  
    merged_Lines = list()
    lineA = listLines[0]
    listLines.remove(lineA)
    mergeLines(lineA, listLines, merged_Lines, 0, threshold_slope, threshold_distance, threshold_perpendicular_distance)
    return merged_Lines
  
 

#INTERSECTION
#Description: This function returns the intersection of two lists
#
#   @Inputs:
#       lst1: list 1.
#       lst2: list 2.
#   @Outputs:
#       lst3: Intersected list.  
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3

 

#MERGELINES
#Description: This function complements uniqueLinesPImproved.
#
#   @Inputs:
#       lineA: line that is being analysed. 
#       listLines: array of lines (cv2.HoughLinesP).
#       mergedLines: Array with the lines already merged.
#       threshold_slope: Max distance between the slopes to be considered the 
#           same line.
#       threshold_distance: Max distance between the extreme points of the 
#           lines to be considered the same line.
#       threshold_perpendicular_distance: Perpendicular distance to the line.
#   @Outputs:
#       final_lines: List with the different lines. Each position of the list 
#           has the form [x1,y1,x2,y2].  
def mergeLines(lineA, listLines, merged_Lines = list(), checkingMatchingLine = 0, threshold_slope = 0.1, threshold_distance = 30, threshold_perpendicular_distance = 10):  
    slope_line = (lineA[1]-lineA[3])/(lineA[0]-lineA[2]+0.05)
    Flag_merge_Lines = 0
    x_min = lineA[0]
    y_min = lineA[1]
    x_max = lineA[2]
    y_max = lineA[3]
    num_lines_errased = 0
    verticalLines = 0
    n = 0
    while n < (len(listLines)): 
        n = n - num_lines_errased
        num_lines_errased = 0
        slope_new_line = (listLines[n][1]-listLines[n][3])/(listLines[n][0]-listLines[n][2]+0.05)
        if abs(lineA[0] - lineA[2]) <= 2 and abs(listLines[n][0] - listLines[n][2]) <= 2: #Vertical lines
            verticalLines = 1
            slope_new_line = slope_line        
        #First: dif(slopes) < threshold
        d0 = abs(slope_line - slope_new_line) #Difference between the slopes
        if (d0 < threshold_slope): 
            #Second: distance(extreme_points) < threshold
            d1 = math.sqrt(pow((lineA[1] - listLines[n][3]),2) + pow((lineA[0] - listLines[n][2]),2)) #Distance between extreme points of both lines
            d2 = math.sqrt(pow((listLines[n][1] - lineA[3]),2) + pow((listLines[n][0] - lineA[2]),2)) #Distance between extreme points of both lines
            d3 = math.sqrt(pow((lineA[1] - listLines[n][1]),2) + pow((lineA[0] - listLines[n][0]),2)) #Distance between extreme points of both lines
            d4 = math.sqrt(pow((lineA[2] - listLines[n][2]),2) + pow((lineA[3] - listLines[n][3]),2)) #Distance between extreme points of both lines
            if (d1 < threshold_distance) or (d2 < threshold_distance) or (d3 < threshold_distance) or (d4 < threshold_distance):                         
                #Third: distance_perpendicular line < threshold
                p1 = np.array([lineA[0],lineA[1]])
                p2 = np.array([lineA[2],lineA[3]])
                p3 = np.array([listLines[n][0],listLines[n][1]])
                p4 = np.array([listLines[n][2],listLines[n][3]])
                d5 = abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1)) #Perpendicular distance from extreme point 1 of the line that is being analysed point to lineA
                d6 = abs(np.cross(p2-p1,p4-p1)/np.linalg.norm(p2-p1)) #Perpendicular distance from extreme point 2 of the line that is being analysed point to lineA              
                if (d5 <= threshold_perpendicular_distance) and (d6 <= threshold_perpendicular_distance):
                    new_line = listLines[n]                    
                    prev_list = listLines[:n]   
                    listLines.remove(new_line)
                    finalNewLine, listLines = mergeLines(new_line, listLines, merged_Lines, 1, threshold_slope, threshold_distance, threshold_perpendicular_distance)
                    num_lines_errased = num_lines_errased + len(prev_list) - len(intersection(prev_list, listLines)) + 1
                    Flag_merge_Lines = 1  
        if Flag_merge_Lines:
            Flag_merge_Lines = 0
            if verticalLines:
                verticalLines = 0
                y_min_new = min(y_min, finalNewLine[1], y_max, finalNewLine[3])
                y_max_new = max(y_max, finalNewLine[3], y_min, finalNewLine[1])
                y_min = y_min_new
                y_max = y_max_new
                x_max = x_min
            else:            
                x_min_new = min(x_min, finalNewLine[0])
                y_min = (y_min, finalNewLine[1])[(x_min, finalNewLine[0]).index(x_min_new)]
                x_min = x_min_new
                x_max_new = max(x_max, finalNewLine[2])
                y_max = (y_max, finalNewLine[3])[(x_max, finalNewLine[2]).index(x_max_new)]
                x_max = x_max_new
        n = n + 1
    if not checkingMatchingLine: #If the list is not empty and we are not checking a matching line
        merged_Lines.append([x_min, y_min, x_max, y_max])
        if listLines:
            new_line = listLines[0]
            listLines.remove(new_line)
            finalNewLine, listLines = mergeLines(new_line, listLines, merged_Lines, 0, threshold_slope, threshold_distance, threshold_perpendicular_distance)
    return ([x_min, y_min, x_max, y_max]), listLines



#GETSLOPEPOLARCOORDINATES
#Description: This function receives a line in POLAR coordinates and returns 
#   the slope.
#
#   @Inputs:
#       line: Line in polar coordinates.
#   @Outputs:
#       slope: slope. 
def getSlopePolarCoordinates(line):
    2*np.pi*90/360
    a = np.cos(line[1])
    b = np.sin(line[1])
    x0 = a*line[0]
    y0 = b*line[0]
    slope = 1/((y0/x0))
    return slope



#GETSLOPEGIVENTWOPOINTS
#Description: This function receives a line in cartesian coordinates (it 
#   receives two points of this line) and returns it's slope.
#
#   @Inputs:
#       line: two points that belong to the line [x1,y1,x2,y2].
#   @Outputs:
#       slope: slope. 
def getSlopeGivenTwoPoints(line):
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    slope = -(y2-y1)/(x2-x1 + 0.1)
    return slope



#GETSLOPEGIVENTWOPOINTSDEGREES
#Description: This function receives a line in cartesian coordinates (it 
#   receives two points of this line) and returns it's slope.
#
#   @Inputs:
#       line: two points that belong to the line [x1,y1,x2,y2].
#   @Outputs:
#       degrees: Inclination of the slope. 
def getSlopeGivenTwoPointsDegrees(line):
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    slope = -(y2-y1)/(x2-x1 + 0.0001)
    degrees = math.degrees(math.atan(slope))
    return degrees



#GETLENGTH
#Description: This function receives a line in cartesian coordinates (it 
#   receives two points of this line) and returns it's length.
#
#   @Inputs:
#       line: two points that belong to the line [x1,y1,x2,y2].
#   @Outputs:
#       length: length. 
def getLength(line):
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    length = math.sqrt(math.pow(x2-x1,2)+math.pow(y2-y1,2))
    return length



#GETLINESOVER
#Description: This function receives a line in cartesian coordinates (it 
#   receives two points of this line) and a dataset with all the lines and
#   returns the number of non-vertical (degrees < 80) lines over it and the 
#   most similar slope of those lines.
#
#   @Inputs:
#       line: two points that belong to the line [x1,y1,x2,y2].
#       linesDB: Database with all the lines
#   @Outputs:
#       num_lines_over: Number of non-vertical lines (degrees < 80) over the 
#           line with the same slope (or when the line is horizontal < 5 degrees).
#       similar_slope: most similar slope of those lines that are over it.
def getLinesOver(line, linesDB):
    x1 = line[0]
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]
    slopesDB = np.zeros([len(linesDB),1]) #Will store all the slopes
    for i in range(len(linesDB)):
        slopesDB[i]=getSlopeGivenTwoPoints(linesDB[i])
    slope = getSlopeGivenTwoPoints(line)
    num_lines_over = 0
    similar_slope = -1000000 #Aux value
    for i in range(len(linesDB)):
        if line != linesDB[i]: #Different line
            x1lDB = linesDB[i][0]
            y1lDB = linesDB[i][1]
            x2lDB = linesDB[i][2]
            y2lDB = linesDB[i][3]
            #First check that the lineDB[i] is over the checked line:
            if (x1 <=x1lDB and x2 >= x1lDB and (y1lDB < y1 or y1lDB < y2)) or (x1 <=x2lDB and x2 >= x2lDB and (y2lDB < y2 or y2lDB < y1)) or (x1 <=x1lDB and x2 >= x2lDB and (y1lDB <= y1 or y2lDB <= y2 or y1lDB <= y2 or y2lDB <= y1)) or (x1 >=x1lDB and x2 <= x2lDB and (y1lDB <= y1 or y2lDB <= y2)): #Line over segment
                 #Second, check that the line is not vertical:
                 if abs(math.degrees(math.atan(slopesDB[i]))) <= 80: #Vertical if angle > 80 degrees
                     if (np.sign(slope) == np.sign(slopesDB[i])) or (abs(math.degrees(math.atan(slopesDB[i]))) <= 5) or (abs(math.degrees(math.atan(slope))) <= 5): #Same slope direction or the line is horizontal (<5 degrees)
                         num_lines_over = num_lines_over + 1
                         if abs(similar_slope - slope) >= abs(slopesDB[i] - slope):
                             similar_slope = slopesDB[i,0]
    return num_lines_over, similar_slope
    
    

#GETLINES
#Description: This function receives an image and returns a list with the 
#   detected lines and a db with the features of those lines.
#
#   @Inputs:
#       image: image (grayscale!!!).
#       sizeX: size (number of pixels) of the hyperpixels in X axis. By default
#           it is 18 (image of size 400x702 & 780 sections).
#       sizeY: size (number of pixels) of the hyperpixels in Y axis. By default
#           it is 20 (image of size 400x702 & 780 sections).
#       num: Number of hyperpixels in Y axis (vertical). The number of 
#           hyperpixels in the horizontal axis will be num*2-1. By default it 
#           is 20 (image of size 400x702 & 780 sections). 
#   @Outputs:
#       lines_db: DF with the line features. Per each line there is the next
#           information:
#           - 'id_line': ID of the line.
#           - 'x1': x1 coordinate.
#           - 'y1': y1 coordinate.
#           - 'x2': x2 coordinate.
#           - 'y2': y2 coordinate.
#           - 'slope': Slope of the line.
#           - 'length': length of line.
#           - 'number_lines_over': Number of lines over this line with the same 
#              slope (non-vertical lines).
#           - 'closest_slope_over': most similar slope of those lines that are 
#              over it.
#       mask_lines: Mask with the lines painted and with their ids as line's 
#           values.
def getLines(image, sizeX = 18, sizeY = 20, num = 20):
    #Normalize the number of pixels for doing the squared traverse
    numY = num
    numX = num*2-1
    dim = (sizeX*numX, sizeY*numY)
    mask_lines = np.zeros([image.shape[0], image.shape[1]])
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA) 
    size_upper_part_image = int(dim[1]*1/3)
    image = image[size_upper_part_image:,:] #Lower part of the image
    lines_db = pd.DataFrame(columns=['id_line', 'x1', 'y1', 'x2', 'y2', 'slope', 'degrees', 'length', 'number_lines_over', 'closest_slope_over'])
    edges = cv2.Canny(image,30,70,apertureSize = 3)
    minLineLength = 50
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,50,minLineLength,maxLineGap)
    lines = uniqueLinesPImproved(lines, threshold_slope = 0.1, threshold_distance = 30, threshold_perpendicular_distance = 10)
    if len(lines) != 0:
        for i in range(len(lines)):
            line = lines[i]
            lines_db.loc[i,'id_line'] = i
            lines_db.loc[i,'x1'] = line[0]
            lines_db.loc[i,'y1'] = line[1] + size_upper_part_image
            lines_db.loc[i,'x2'] = line[2]
            lines_db.loc[i,'y2'] = line[3] + size_upper_part_image    
            slope = getSlopeGivenTwoPoints(line)
            lines_db.loc[i,'slope'] = slope
            degrees = getSlopeGivenTwoPointsDegrees(line)
            lines_db.loc[i,'degrees'] = degrees
            length = getLength(line)
            lines_db.loc[i,'length'] = length     
            number_lines_over, closest_slope_over = getLinesOver(line, lines)
            lines_db.loc[i,'number_lines_over'] = number_lines_over
            lines_db.loc[i,'closest_slope_over'] = closest_slope_over
    lines_db = lines_db.sort_values(by=['length'], ascending = True)  
    lines_db.loc[:,'id_line'] = range(len(lines_db))
    lines_db.loc[:,'id_line'] = lines_db.loc[:,'id_line'] + 1
    lines_db.index = range(len(lines_db))
    for i in range(len(lines_db)):
        x1 = int(lines_db.loc[i,'x1'])
        y1 = int(lines_db.loc[i,'y1'])
        x2 = int(lines_db.loc[i,'x2'])
        y2 = int(lines_db.loc[i,'y2'])
        cv2.line(mask_lines, (x1, y1), (x2, y2), int(lines_db.loc[i,'id_line']), 1)
    return lines_db, mask_lines
    


#GETFELZENSZWALB
#Description: This function receives an image and returns a list with each of 
#   ids given by the FELZENSZWALB segmentation, and the image after the 
#   segmentation.
#
#   @Inputs:
#       image: image (grayscale!!!).
#       sizeX: size (number of pixels) of the hyperpixels in X axis. By default
#           it is 18 (image of size 400x702 & 780 sections).
#       sizeY: size (number of pixels) of the hyperpixels in Y axis. By default
#           it is 20 (image of size 400x702 & 780 sections).
#       num: Number of hyperpixels in Y axis (vertical). The number of 
#           hyperpixels in the horizontal axis will be num*2-1. By default it 
#           is 20 (image of size 400x702 & 780 sections). 
#   @Outputs:
#       f_db: Database with two columns. The first one is the id of each 
#           Felzenszwalb segmentation, and the second one is number of pixels 
#           that belong to that segmentation.
#       mask_f: Image with the segmented areas.
def getFelzenszwalb(image, sizeX = 18, sizeY = 20, num = 20):
    mask_f = felzenszwalb(image, scale=400, sigma=0.5, min_size=1000)
    f_db = np.zeros([np.max(mask_f)+1, 2])
    for i in range(len(f_db)):
        f_db[i,0] = i
        f_db[i,1] = np.count_nonzero(mask_f == i)
    return f_db, mask_f
    
   
    
#GETSURROUNDINGGRADIENT
#Description: This function receives an image and returns the biggest gradient
#    change in each of the four directions (up, down, right, left).
#
#   @Inputs:
#       hyperpixels_laplacian: Output of:
#           laplacian_image = abs(cv2.Laplacian(image,cv2.CV_64F))
#           hyperpixels_laplacian = kgf.getHyperpixels(laplacian_image, sizeX, sizeY, num)
#       ID: ID of the hyperpixel that is being analyzed.
#       sizeX: size (number of pixels) of the hyperpixels in X axis. By default
#           it is 18 (image of size 400x702 & 780 sections).
#       sizeY: size (number of pixels) of the hyperpixels in Y axis. By default
#           it is 20 (image of size 400x702 & 780 sections).
#       num: Number of hyperpixels in Y axis (vertical). The number of 
#           hyperpixels in the horizontal axis will be num*2-1. By default it 
#           is 20 (image of size 400x702 & 780 sections). 
#   @Outputs:
#       max_up: Maximum gradient value over the hyperpixel.
#       max_down: Maximum gradient value under the hyperpixel.
#       max_right: Maximum gradient value at right of the hyperpixel.
#       max_left: Maximum gradient value at left of the hyperpixel.
def getSurroundingGradient(hyperpixels_laplacian, ID, sizeX = 18, sizeY = 20, num = 20):
    pixels_up = np.array([0]) #It contains all the pixels that are over the hyperpixel ID
    pixels_down = np.array([0]) #It contains all the pixels that are under the hyperpixel ID
    pixels_right = np.array([0]) #It contains all the pixels that are at right of the hyperpixel ID
    pixels_left = np.array([0]) #It contains all the pixels that are at left of the hyperpixel ID
    ID_up = hyperpixels_laplacian[int(ID), 1] #ID of the superior hyperpixel
    ID_right = hyperpixels_laplacian[int(ID), 2] #ID of the hyperpixel at right
    ID_down = hyperpixels_laplacian[int(ID), 3] #ID of the lower hyperpixel
    ID_left = hyperpixels_laplacian[int(ID), 4] #ID of the left hyperpixel
    while not math.isnan(ID_up):
        pixels_up = np.append(pixels_up,hyperpixels_laplacian[int(ID_up), 9:]) 
        ID_up = hyperpixels_laplacian[int(ID_up), 1]
    while not math.isnan(ID_down):
        pixels_down = np.append(pixels_down,hyperpixels_laplacian[int(ID_down), 9:]) 
        ID_down = hyperpixels_laplacian[int(ID_down), 1]
    while not math.isnan(ID_right):
        pixels_right = np.append(pixels_right,hyperpixels_laplacian[int(ID_right), 9:]) 
        ID_right = hyperpixels_laplacian[int(ID_right), 1]
    while not math.isnan(ID_left):
        pixels_left = np.append(pixels_left,hyperpixels_laplacian[int(ID_left), 9:]) 
        ID_left = hyperpixels_laplacian[int(ID_left), 1]
    max_up = np.max(pixels_up)
    max_down = np.max(pixels_down)
    max_right = np.max(pixels_right)
    max_left = np.max(pixels_left)
    return max_up, max_down, max_right, max_left



#GETGRADIENTMAGNITUDE
#Description: This function receives an image and returns the mask of the 
#   gradient magnitude per each pixel
#
#   @Inputs:
#       image: image
#   @Outputs:
#      grad: Mask with the value of the gradient for each pixel
def getGradientMagnitude(image, ddepth = cv2.CV_64F):
    "Get magnitude of gradient for given image"
    dx = cv2.Sobel(image, ddepth, dx = 1, dy = 0, ksize = 3)
    dy = cv2.Sobel(image, ddepth, dx = 0, dy = 1, ksize = 3)
    grad = np.sqrt(np.power(dx,2)+ np.power(dy,2))
    grad = grad.astype(np.uint8)
    return grad



#PREDICTTREE
#Description: This function receives a collection of images and features and 
#   all the info of the decision trees and predicts the result in the correct 
#   order.
#
#   @Inputs:
#       X_pred: X features to predict the output.
#       decisionTree: Decision tree model.
#       image: example of an image for getting the neighbors.
#       num: Number of hyperpixels in Y axis (vertical). The number of 
#           hyperpixels in the horizontal axis will be num*2-1. By default it 
#           is 20 (image of size 400x702 & 780 sections). 
#       classes_names: Names of the classes that are being included.
#       columns_online: Names of tyhe coumns that are being actualized online.
#   @Outputs:
#      classes: Numpy array with two columns. The first one contains the ID of 
#          the tile. The second one contains the class of that tile.
#       X: df with the positions actualized
#   @Call:
#       classes = predictTree(X_pred, decisionTree = decisionTree, num = 20, image = df_merged_grayImages[0])
def predictTree(X_pred, decisionTree, image, num = 20, classes_names= ['class_previous_tile','A1_class','A2_class','A3_class','A4_class','A5_class','A9_class', 'A6_class', 'A8_class'], columns_online = ['class_previous_tile_back','class_previous_tile_edge', 'class_previous_tile_floor', 'A1_class_back', 'A1_class_edge', 'A1_class_floor', 'A2_class_back', 'A2_class_edge', 'A2_class_floor', 'A5_class_back', 'A5_class_edge', 'A5_class_floor', 'A6_class_back', 'A6_class_edge', 'A6_class_floor', 'A8_class_back', 'A8_class_edge', 'A8_class_floor']):
    X = X_pred
    num_hyperpixels = num*(2*num-1) #Set index (needed for painting the tiles)
    classes = np.zeros([num_hyperpixels,2]) #Output array
    classes[:,0] = range(num_hyperpixels) 
    order_of_analysis = getOrderOfAnalysis(num_hyperpixels) #Get the order of analysis
    classes[0:6,1] = 1
    neighbors_ID = getNeighborsId(image) #First get the ID of all the neighbors. As all the images are equaly segmented, it is only needed to get the neighbors of one image. Then it is applicable to the rest of the images.
    X.loc[:, columns_online] = 0 #Substitute by 0 (missing value -> They have to be inferred)
    order_of_analysis.loc[:,'ID'] = order_of_analysis.loc[:,'ID'].astype('int')
    already_analyzed_IDs = [0,1,2,3,4,5]
    for j in order_of_analysis.loc[6:,'ID']:
        already_analyzed_IDs.append(j)
        #Get the values of the previously analyzed tiles
        if not math.isnan(neighbors_ID.loc[j,'Prev_tile_ID']) and int(neighbors_ID.loc[j,'Prev_tile_ID']) in already_analyzed_IDs and 'class_previous_tile' in classes_names:
            if int(classes[int(neighbors_ID.loc[j,'Prev_tile_ID']),1]) == 0: #Background
                X.loc[j, 'class_previous_tile_back'] = 1 
            elif int(classes[int(neighbors_ID.loc[j,'Prev_tile_ID']),1]) == 1: #Floor
                X.loc[j, 'class_previous_tile_floor'] = 1 
            elif int(classes[int(neighbors_ID.loc[j,'Prev_tile_ID']),1]) == 2: #Edge
                X.loc[j, 'class_previous_tile_edge'] = 1 
        if not math.isnan(neighbors_ID.loc[j,'A1']) and int(neighbors_ID.loc[j,'A1']) in already_analyzed_IDs and 'A1_class' in classes_names:
            if classes[int(neighbors_ID.loc[j,'A1']),1] == 0: #Background
                X.loc[j, 'A1_class_back'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A1']),1] == 1: #Floor
                X.loc[j, 'A1_class_floor'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A1']),1] == 2: #EDge
                X.loc[j, 'A1_class_edge'] = 1 
        if not math.isnan(neighbors_ID.loc[j,'A2']) and int(neighbors_ID.loc[j,'A2']) in already_analyzed_IDs and 'A2_class' in classes_names:
            if classes[int(neighbors_ID.loc[j,'A2']),1] == 0: #Background
                X.loc[j, 'A2_class_back'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A2']),1] == 1: #Floor
                X.loc[j, 'A2_class_floor'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A2']),1] == 2: #Edge
                X.loc[j, 'A2_class_edge'] = 1 
        if not math.isnan(neighbors_ID.loc[j,'A3']) and int(neighbors_ID.loc[j,'A3']) in already_analyzed_IDs and 'A3_class' in classes_names:
            if classes[int(neighbors_ID.loc[j,'A3']),1] == 0: #Background
                X.loc[j, 'A3_class_back'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A3']),1] == 1: #Floor
                X.loc[j, 'A3_class_floor'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A3']),1] == 2: #Edge
                X.loc[j, 'A3_class_edge'] = 1 
        if not math.isnan(neighbors_ID.loc[j,'A4']) and int(neighbors_ID.loc[j,'A4']) in already_analyzed_IDs and 'A4_class' in classes_names:
            if classes[int(neighbors_ID.loc[j,'A4']),1] == 0: #Background
                X.loc[j, 'A4_class_back'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A4']),1] == 1: #Floor
                X.loc[j, 'A4_class_floor'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A4']),1] == 2: #Edge
                X.loc[j, 'A4_class_edge'] = 1 
        if not math.isnan(neighbors_ID.loc[j,'A5']) and int(neighbors_ID.loc[j,'A5']) in already_analyzed_IDs and 'A5_class' in classes_names:
            if classes[int(neighbors_ID.loc[j,'A5']),1] == 0: #Background
                X.loc[j, 'A5_class_back'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A5']),1] == 1: #Floor
                X.loc[j, 'A5_class_floor'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A5']),1] == 2: #Edge
                X.loc[j, 'A5_class_edge'] = 1 
        if not math.isnan(neighbors_ID.loc[j,'A9']) and int(neighbors_ID.loc[j,'A9']) in already_analyzed_IDs and 'A9_class' in classes_names:
            if classes[int(neighbors_ID.loc[j,'A9']),1] == 0: #Background
                X.loc[j, 'A9_class_back'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A9']),1] == 1: #Floor
                X.loc[j, 'A9_class_floor'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A9']),1] == 2: #Edge
                X.loc[j, 'A9_class_edge'] = 1 
        if not math.isnan(neighbors_ID.loc[j,'A6']) and int(neighbors_ID.loc[j,'A6']) in already_analyzed_IDs and 'A6_class' in classes_names:
            if classes[int(neighbors_ID.loc[j,'A6']),1] == 0: #Background
                X.loc[j, 'A6_class_back'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A6']),1] == 1: #Floor
                X.loc[j, 'A6_class_floor'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A6']),1] == 2: #Edge
                X.loc[j, 'A6_class_edge'] = 1 
        if not math.isnan(neighbors_ID.loc[j,'A8']) and int(neighbors_ID.loc[j,'A8']) in already_analyzed_IDs and 'A8_class' in classes_names:
            if classes[int(neighbors_ID.loc[j,'A8']),1] == 0: #Background
                X.loc[j, 'A8_class_back'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A8']),1] == 1: #Floor
                X.loc[j, 'A8_class_floor'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A8']),1] == 2: #Edge
                X.loc[j, 'A8_class_edge'] = 1 
        classes[j,1] = int(decisionTree.predict(X[X.index == j])[0])
    return classes, X



#PREDICTTREE4DT
#Description: This function receives a collection of images and features and 
#   all the info of the decision trees and predicts the result in the correct 
#   order.
#
#   @Inputs:
#       X_pred_main_ID1ID3: X features to predict the main hyperpixels at 
#           sections ID1 and ID3.
#       X_pred_main_ID2: X features to predict the main hyperpixels at section
#           ID2.
#       X_pred_low1: X features to predict the low1 hyperpixels.
#       X_pred_low2: X features to predict the low2 hyperpixels.
#       decisionTree_main_ID1ID3: Decision tree model of the main hyperpixels
#           at sections ID1 and ID3.
#       decisionTree_main_ID2: Decision tree model of the main hyperpixels
#           at section ID2.
#       decisionTree_low1: Decision tree model of the low1 hyperpixels.
#       decisionTree_low2: Decision tree model of the low2 hyperpixels.
#       image: example of an image for getting the neighbors.
#       num: Number of hyperpixels in Y axis (vertical). The number of 
#           hyperpixels in the horizontal axis will be num*2-1. By default it 
#           is 20 (image of size 400x702 & 780 sections). 
#       classes_names: Names of the classes that are being included.
#       columns_online: Names of tyhe coumns that are being actualized online.
#   @Outputs:
#      classes: Numpy array with two columns. The first one contains the ID of 
#          the tile. The second one contains the class of that tile.
#   @Call:
#       classes = predictTree3DT(X_pred_main, X_pred_low1, X_pred_low2, decisionTree_main, decisionTree_low1, decisionTree_low2, num = 20, image = df_merged_grayImages[0], classes_names= ['class_previous_tile','A1_class','A2_class','A3_class','A4_class','A5_class','A9_class', 'A6_class', 'A8_class'], columns_online = ['class_previous_tile_back','class_previous_tile_edge', 'class_previous_tile_floor', 'A1_class_back', 'A1_class_edge', 'A1_class_floor', 'A2_class_back', 'A2_class_edge', 'A2_class_floor', 'A5_class_back', 'A5_class_edge', 'A5_class_floor', 'A6_class_back', 'A6_class_edge', 'A6_class_floor', 'A8_class_back', 'A8_class_edge', 'A8_class_floor'])
def predictTree4DT(X_pred_main_ID1_ID3, X_pred_main_ID2, X_pred_low1, X_pred_low2, decisionTree_main_ID1ID3, decisionTree_main_ID2, decisionTree_low1, decisionTree_low2, image, num = 20, classes_names= ['class_previous_tile','A1_class','A2_class','A3_class','A4_class','A5_class','A9_class', 'A6_class', 'A8_class'], columns_online = ['class_previous_tile_back','class_previous_tile_edge', 'class_previous_tile_floor', 'A1_class_back', 'A1_class_edge', 'A1_class_floor', 'A2_class_back', 'A2_class_edge', 'A2_class_floor', 'A5_class_back', 'A5_class_edge', 'A5_class_floor', 'A6_class_back', 'A6_class_edge', 'A6_class_floor', 'A8_class_back', 'A8_class_edge', 'A8_class_floor']):
    X_main_ID1ID3 = X_pred_main_ID1_ID3
    X_main_ID2 = X_pred_main_ID2
    X_low1 = X_pred_low1
    X_low2 = X_pred_low2
    num_hyperpixels = num*(2*num-1) #Set index (needed for painting the tiles)
    classes = np.zeros([num_hyperpixels,2]) #Output array
    classes[:,0] = range(num_hyperpixels) 
    order_of_analysis = getOrderOfAnalysis(num_hyperpixels) #Get the order of analysis
    classes[0:6,1] = 1
    neighbors_ID = getNeighborsId(image) #First get the ID of all the neighbors. As all the images are equaly segmented, it is only needed to get the neighbors of one image. Then it is applicable to the rest of the images.
    order_of_analysis.loc[:,'ID'] = order_of_analysis.loc[:,'ID'].astype('int')
    for col in columns_online:
        if col in X_main_ID1ID3.columns:
            X_main_ID1ID3.loc[:, col] = 0 #Substitute by 0 (missing value -> They have to be inferred)
        if col in X_main_ID2.columns:
            X_main_ID2.loc[:, col] = 0 #Substitute by 0 (missing value -> They have to be inferred)
        if col in X_low1.columns:
            X_low1.loc[:, col] = 0 #Substitute by 0 (missing value -> They have to be inferred)
        if col in X_low2.columns:
            X_low2.loc[:, col] = 0 #Substitute by 0 (missing value -> They have to be inferred)
    already_analyzed_IDs = [0,1,2,3,4,5]
    left1, left2 = getLeftHyperpixels(num_hyperpixels)
    right1, right2 = getRightHyperpixels(num_hyperpixels)
    low1, low2 = getLowerHyperpixels(num_hyperpixels)
    top1, top2 = getTopHyperpixels(num_hyperpixels)
    id1id3, id2 = getID1ID2ID3Hyperpixels(num_hyperpixels)
    drop_tiles = left1 + left2 + right1+ right2 + top1 + top2
    order_of_analysis = order_of_analysis[~order_of_analysis['ID'].isin(drop_tiles)]  
    for j in order_of_analysis.loc[6:,'ID']:
        if j in low1:
            X = X_low1
        elif j in low2:
            X = X_low2
        else:
            if j in id1id3:
                X = X_main_ID1ID3
            elif j in id2:
                X = X_main_ID2
        already_analyzed_IDs.append(j)            
        #Get the values of the previously analyzed tiles
        if not math.isnan(neighbors_ID.loc[j,'Prev_tile_ID']) and int(neighbors_ID.loc[j,'Prev_tile_ID']) in already_analyzed_IDs and 'class_previous_tile' in classes_names and ('class_previous_tile_back' in X.columns and 'class_previous_tile_floor' in X.columns and 'class_previous_tile_edge' in X.columns):
            if int(classes[int(neighbors_ID.loc[j,'Prev_tile_ID']),1]) == 0: #Background
                X.loc[j,'class_previous_tile_back'] = 1 
            elif int(classes[int(neighbors_ID.loc[j,'Prev_tile_ID']),1]) == 1: #Floor
                X.loc[j,'class_previous_tile_floor'] = 1 
            elif int(classes[int(neighbors_ID.loc[j,'Prev_tile_ID']),1]) == 2: #Edge
                X.loc[j,'class_previous_tile_edge'] = 1             
        if not math.isnan(neighbors_ID.loc[j,'A1']) and int(neighbors_ID.loc[j,'A1']) in already_analyzed_IDs and 'A1_class' in classes_names and ('A1_class_back' in X.columns and 'A1_class_floor' in X.columns and 'A1_class_edge' in X.columns):
            if classes[int(neighbors_ID.loc[j,'A1']),1] == 0: #Background
                X.loc[j,'A1_class_back'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A1']),1] == 1: #Floor
                X.loc[j,'A1_class_floor'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A1']),1] == 2: #Edge
                X.loc[j,'A1_class_edge'] = 1             
        if not math.isnan(neighbors_ID.loc[j,'A2']) and int(neighbors_ID.loc[j,'A2']) in already_analyzed_IDs and 'A2_class' in classes_names and ('A2_class_back' in X.columns and 'A2_class_floor' in X.columns and 'A2_class_edge' in X.columns):
            if classes[int(neighbors_ID.loc[j,'A2']),1] == 0: #Background
                X.loc[j,'A2_class_back'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A2']),1] == 1: #Floor
                X.loc[j,'A2_class_floor'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A2']),1] == 2: #Edge
                X.loc[j,'A2_class_edge'] = 1 
        if not math.isnan(neighbors_ID.loc[j,'A3']) and int(neighbors_ID.loc[j,'A3']) in already_analyzed_IDs and 'A3_class' in classes_names and ('A3_class_back' in X.columns and 'A3_class_floor' in X.columns and 'A3_class_edge' in X.columns):
            if classes[int(neighbors_ID.loc[j,'A3']),1] == 0: #Background
                X.loc[j,'A3_class_back'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A3']),1] == 1: #Floor
                X.loc[j,'A3_class_floor'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A3']),1] == 2: #Edge
                X.loc[j,'A3_class_edge'] = 1 
        if not math.isnan(neighbors_ID.loc[j,'A4']) and int(neighbors_ID.loc[j,'A4']) in already_analyzed_IDs and 'A4_class' in classes_names and ('A4_class_back' in X.columns and 'A4_class_floor' in X.columns and 'A4_class_edge' in X.columns):
            if classes[int(neighbors_ID.loc[j,'A4']),1] == 0: #Background
                X.loc[j,'A4_class_back'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A4']),1] == 1: #Floor
                X.loc[j,'A4_class_floor'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A4']),1] == 2: #Edge
                X.loc[j,'A4_class_edge'] = 1 
        if not math.isnan(neighbors_ID.loc[j,'A5']) and int(neighbors_ID.loc[j,'A5']) in already_analyzed_IDs and 'A5_class' in classes_names and ('A5_class_back' in X.columns and 'A5_class_floor' in X.columns and 'A5_class_edge' in X.columns):
            if classes[int(neighbors_ID.loc[j,'A5']),1] == 0: #Background
                X.loc[j,'A5_class_back'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A5']),1] == 1: #Floor
                X.loc[j,'A5_class_floor'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A5']),1] == 2: #Edge
                X.loc[j,'A5_class_edge'] = 1 
        if not math.isnan(neighbors_ID.loc[j,'A9']) and int(neighbors_ID.loc[j,'A9']) in already_analyzed_IDs and 'A9_class' in classes_names and ('A9_class_back' in X.columns and 'A9_class_floor' in X.columns and 'A9_class_edge' in X.columns):
            if classes[int(neighbors_ID.loc[j,'A9']),1] == 0: #Background
                X.loc[j,'A9_class_back'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A9']),1] == 1: #Floor
                X.loc[j,'A9_class_floor'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A9']),1] == 2: #Edge
                X.loc[j,'A9_class_edge'] = 1 
        if not math.isnan(neighbors_ID.loc[j,'A6']) and int(neighbors_ID.loc[j,'A6']) in already_analyzed_IDs and 'A6_class' in classes_names and ('A6_class_back' in X.columns and 'A6_class_floor' in X.columns and 'A6_class_edge' in X.columns):
            if classes[int(neighbors_ID.loc[j,'A6']),1] == 0: #Background
                X.loc[j,'A6_class_back'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A6']),1] == 1: #Floor
                X.loc[j,'A6_class_floor'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A6']),1] == 2: #Edge
                X.loc[j,'A6_class_edge'] = 1 
        if not math.isnan(neighbors_ID.loc[j,'A8']) and int(neighbors_ID.loc[j,'A8']) in already_analyzed_IDs and 'A8_class' in classes_names and ('A8_class_back' in X.columns and 'A8_class_floor' in X.columns and 'A8_class_edge' in X.columns):
            if classes[int(neighbors_ID.loc[j,'A8']),1] == 0: #Background
                X.loc[j,'A8_class_back'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A8']),1] == 1: #Floor
                X.loc[j,'A8_class_floor'] = 1 
            elif classes[int(neighbors_ID.loc[j,'A8']),1] == 2: #Edge
                X.loc[j,'A8_class_edge'] = 1 
        if j in low1:
            classes[j,1] = int(decisionTree_low1.predict(X[X.index == j])[0])
        elif j in low2:
            classes[j,1] = int(decisionTree_low2.predict(X[X.index == j])[0])
        else:
            if j in id1id3:
                classes[j,1] = int(decisionTree_main_ID1ID3.predict(X[X.index == j])[0])  
            elif j in id2:
                classes[j,1] = int(decisionTree_main_ID2.predict(X[X.index == j])[0])         
    return classes


