# RobotVision_MasterThesis
This project consists in the development of a robot vision system. The algorithm performs a floor segmentation by dividing the image in superpixels and analyzing some features of these superpixels. 

The aim of this thesis is to infer the knowledge needed by an autonomous system for performing image segmentation of a video footage that is taken inside a building. Precisely, the current document is based on the use case of an autonomous vehicle that has to move in a public building transporting goods and without posing any kind of hazard for the people moving around it. Thus, one of the most important capabilities is performing a semantic segmentation of the image in order to distinguish foreground (movable area, ie., floor) from background (non-movable area).
The approach taken in this work to discern between foreground and background consists in the segmentation of images in a fixed number of squared superpixels, followed by a posterior analysis of the features collected for each of those superpixels. The features analysed for each of the superpixels are divided into features extracted directly from the superpixel, local knowledge in the sense of relationships of the superpixel with its close neighbourhood of superpixels and global knowledge in the sense of features taken from the image as a whole. In this way, the image is summarized in a database of the features, from which a data analysis can be made. Accordingly, in the present work a feature selection process has been carried out and several tests have been performed for these feature collections and for the correct combination between them.
In addition, one of the cornerstones of this thesis is to facilitate the explainability of the decision-making process for the subsequently image segmentation. For this reason, the posterior analysis of the collected features has been done with a decision tree model. Thus, this work is at an intermediate point between entirely black box segmentation models such as CNN (global information completely learned) and segmentation models based purely on knowledge (global information user defined).
