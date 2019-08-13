#!/usr/bin/env python
# coding: utf-8

# # PCA on Faces Dataset

# In this assignment, we will use PCA for dimensionality reduction on the MIT Face Recognition Dataset (https://courses.media.mit.edu/2004fall/mas622j/04.projects/faces/). The following code collects images from the folder rawdata and stores them in a 128x128x(number of images) tensor - image_collection. Running this might take about 20 minutes. If it takes more for you, you can break the loop after 1000 images.
# 

# In[1]:


import os
import skimage
import numpy as np
import matplotlib.pyplot as plt

directory = os.fsencode('rawdata/')
image_collection = np.zeros((128,128,1))
count=0
for file in os.listdir(directory):
    count= count+1
    filename = os.fsdecode(file)
    temp_image = np.fromfile('rawdata/'+filename, dtype='uint8', sep="")
    ratio = temp_image.shape[0]/16384
    if ratio>1:
        temp_image = np.reshape(temp_image, (128*int(np.sqrt(ratio)), 128*int(np.sqrt(ratio))))
        temp_image = skimage.measure.block_reduce(temp_image, (int(np.sqrt(ratio)),int(np.sqrt(ratio))), np.mean)
    image_collection = np.concatenate((image_collection,np.reshape(temp_image,(128,128,1))),axis=2)
    #if count==1000:
    #    break
image_collection = image_collection[:,:,1:]


# # Problem 1

# Recall that PCA requires centered data(zero mean). Center image_collection and display the mean image in gray scale using matplotlib

# In[13]:


#========Your code here ======
new_image = image_collection.reshape((128*128, 3993))
mean = []
for i in range(128*128):
    mean.append(np.mean(new_image[i]))
    
centered = []
for i in range(3993):
    centered.append(new_image[:, i] - mean)
    
mean = np.asarray(mean)
mean = mean.reshape((128, 128))
plt.imshow(mean, cmap='gray')

#==============================
plt.show()


# # Problem 2

# Perform PCA on the images using sklearn. Plot the total percentage of variance explained versus number of components. Also plot the minimum percentage of variance explained versus number of components. Finally plot the singular values when number of components is 500. Justify the plots.

# In[20]:


num_comp = [2,5,7,10,20,30,40,50,100,500]
#========Your code here ======
from sklearn.decomposition import PCA

min_variance = []
singular_values = []
total_variance_explained = []

for i in num_comp:
    pca = PCA(n_components = i)
    pca.fit(centered)
    total = 0
    mini = 1
    for i in range(len(pca.explained_variance_ratio_)):
        total = total + pca.explained_variance_ratio_[i]
        mini = min(mini, pca.explained_variance_ratio_[i])
        
    total_variance_explained.append(total)
    min_variance.append(mini)

singular_values = pca.singular_values_

#==============================
plt.figure(1)
plt.plot(num_comp,total_variance_explained,label='total variance')
plt.plot(num_comp, min_variance,label = 'min variance')
plt.legend()
plt.show()
plt.figure(2)
plt.plot(range(500),singular_values)
plt.show()


# # As we can see from the above graph, total variance increases as the number of components increase, and total variance reaches 1 when no component is abandoned. More components means more attributes will be used for prediction, so the sum of variance explained ratio will increase and reach to 1.0 because all components are stored and the sum of the ratios is equal to 1.0.The minimum variance decreases as the number of components increase and it reaches 0 when the number of components is large enough. This is because attributes that are less useful for the classification of the data are also kept. Since the bad attribute cannot seperate the data very well, data has low variance in terms of the bad attribute.

# # Problem 3

# Display the first 5 principal components as images in gray scale using matplotlib. Explain your observations.

# In[21]:


#========Your code here ======
pca = PCA().fit(centered)
eigen = []
for i in range(5):
    image = pca.components_[i].reshape(128, 128)
    eigen.append(image)

for i in range(5):
    plt.figure()
    plt.imshow(eigen[i], cmap='gray')
    plt.show()
#==============================


# # The first 5 principal components have the most information in the faces dataset. They are the most important 5 principal components, we can see from the above plots, each principal component can well describe the face image and the importance is descending.

# Project the data matrix on the first two principal components and plot a scatter plot of the dataset

# In[25]:


#========Your code here ======
projection = []
for i in range(2):
    projection.append(pca.components_[i])
projection = np.asarray(projection)
projection = np.transpose((projection.dot(np.transpose(centered))))
#==============================
plt.scatter(projection[:,0], projection[:,1])
plt.show()


# In[ ]:




