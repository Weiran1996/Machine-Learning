#!/usr/bin/env python
# coding: utf-8

# # ECE 4950 Assignment 5 
# 
# ## Coding (Support Vector Machines): Digit classification using SVM

# We consider hand written digit recognition, MNIST. Please visit http://yann.lecun.com/exdb/mnist/ for more information about the original MNIST dataset.
# 
# In this competition, you will be given images of hand written digits. Each image is grayscale, and 28 by 28 pixels. Your goal is to design a classifier for this problem (output digits from 0 to 9).
# 
# Make sure you have installed the package scikit-image:
# 
# ``
#     pip3 install scikit-image
# ``
# 
# or 
# 
# 
# ``
#     conda install scikit-image
# ``
# 
# You can use the following script to load data.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from scipy.io import loadmat
import numpy as np
data_path = "./mnist.mat"
data_raw = loadmat(data_path)
images = data_raw['data'].T
label = data_raw['label'][0]


# Data will be an array of 70000 784-length numpy arrays. 50000 of them will be the training data and 20000 of them will be the testing data. The corresponding labels of training data are also provided. 
# 
# Each vector represents an image of size $28 \times 28$. The original black and white images were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. The images were centered in a 28x28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field. 
# 
# You can reshape each vector to be an $28 \times 28$ matrix and plot the image using the following script. Typical images will look like the following:

# In[2]:


import matplotlib.pyplot as plt
import random
plt.figure(figsize=(20,10))
for i in range(10,20):
    plt.subplot(2, 5, i-9)
    t = random.randint(0,70000)
    plt.imshow(np.reshape(images[t,:], (28,28)), cmap = plt.cm.gray)
    plt.title('Digit %i\n' %label[t], fontsize = 20)


# ### Sampling, Normalization and Data splitting
# 
# To make training faster, we only take 10% of the data randomly. Then we further reduce the dimension of the data by taking the average of each  Then we split the data into training and testing set and normalize them by max norm.

# In[3]:


from sklearn.model_selection import train_test_split
X_new, X_unused, Y_new, Y_unused = train_test_split(images, label, test_size = 0.9, random_state = 1000)
X_trn, X_tst, Y_trn, Y_tst = train_test_split(X_new, Y_new, test_size = 0.3, random_state = 1000) # split the dataset into training and testing sets
X_trn = X_trn/256
X_tst = X_tst/256


# In[4]:


print(X_trn)


# ### Image Rescaling
# To make the training even faster. The next code block rescale all the images by reducing the height and width of the image both by half. We make each $2\time 2$ block in the orginal image into a single pixel in the new image. The resulting value of the pixels in the new image will be the average of the original four pixels.
# 
# Then we get new images with $14 \times 14$ in size and stores in the rows of X_trn_new and X_tst_new.

# In[5]:


from skimage.transform import rescale, resize, downscale_local_mean
m, n = X_trn.shape
n_new = n//4
X_trn_new = np.zeros((m,n_new))
for i in range(m):
    image = np.reshape(X_trn[i,:], (28,28))
    image_rescaled = rescale(image, 1.0 / 2.0, anti_aliasing=False)
    X_trn_new[i,:] = np.reshape(image_rescaled, n_new)

m2 = X_tst.shape[0]
X_tst_new = np.zeros((m2,n_new))
for i in range(m2):
    image = np.reshape(X_tst[i,:], (28,28))
    image_rescaled = rescale(image, 1.0 / 2.0, anti_aliasing=False)
    X_tst_new[i,:] = np.reshape(image_rescaled, n_new)
    
print('The new training set has size: '+ str(X_trn_new.shape))
print('The new testing set has size: '+ str(X_tst_new.shape))


# In[7]:


print(X_trn_new)
np.shape(X_trn_new)


# ## 1. Linear SVM
# Run a linear SVM for the penalty parameter $$C \in \{2^i: i = 0, 1, ..., 19\}$$, and plot the training and testing accuracy as a function of $log C$ (semi-log plot).
# 
# Explain how the accuracy changes with repect to penalty parameter $C$ (describe the underfitting and overfitting phenomenon). 
# 
# What is the maximum testing accruracy achieved among all penalty parameters for linear SVM?

# In[ ]:


from sklearn.svm import SVC
n = np.array(range(20))
C = 0.001*2**n
accuracy_tst = []
accuracy_trn = []
#========Your Code Here============
for i in (C):
    clf = SVC(gamma='auto', kernel='linear', C=i)
    clf.fit(X_trn_new, Y_trn)
    accuracy_trn.append(clf.score(X_trn_new, Y_trn))
    accuracy_tst.append(clf.score(X_tst_new, Y_tst))

max_acc = max(accuracy_tst)
#=============================
plt.semilogx(C, accuracy_tst)
plt.semilogx(C, accuracy_trn)
plt.title("Linear SVM")
plt.xlabel('C')
plt.ylabel('accuracy')
plt.show()
print('The maximum testing accuracy achieved with Linear SVM is: ' + str(max_acc))



# # As we can see above, when C is big (lamda is small), the pealty term only has a minor weight in the optimization problem, so the training accuracy is almost 100% and testing accuracy is not so good compared to the training accuracy because overfitting tends to occur in this situation. As C gets smaller(lamda gets bigger) the effect of the penalty starts to show, the traing accuracy starts to decrease and the testing accuracy becomes better, because the classifier that we are getting is not very overfitting to the training data. When C is extremely small(lamda is huge), the penalty term donimates the optimization problem, underfitting is more likely to occur, so we get bad training accuracy and testing accuracy.

# ## 2. Polynomial SVM
# Run an SVM with polynomial kernal of degree $2, 3, 4$ with the penalty parameter $$C \in \{2^i: i = 0, 1, ..., 19\}$$, and plot the training and testing accuracy as a function of $log C$ (semi-log plot).
# 
# Explain how the accuracy changes with repect to penalty parameter $C$ (describe the underfitting and overfitting phenomenon). 
# 
# What is the maximum testing accruracy achieved among all penalty parameters for SVM with polynomial kernal of each degree? Compare it with linear SVM and explain.

# In[10]:


from sklearn.svm import SVC
D = [2, 3, 4]
n = np.array(range(20))
C = 2**n
max_acc = np.zeros(3)
for i in range(3):
    accuracy_tst = []
    accuracy_trn = []
    d = D[i]
    #========Your Code Here============
    for j in (C):
        clf = SVC(gamma='auto', kernel='poly', degree=d, C=j)
        clf.fit(X_trn_new, Y_trn)
        accuracy_trn.append(clf.score(X_trn_new, Y_trn))
        accuracy_tst.append(clf.score(X_tst_new, Y_tst))
    max_acc[i] = max(accuracy_tst)
    #=============================
    plt.semilogx(C, accuracy_tst)
    plt.semilogx(C, accuracy_trn)
    plt.title("Polynomial Kernel SVM, degree %i" %d)
    plt.xlabel('C')
    plt.ylabel('accuracy')
    plt.show()
    print('The maximum testing accuracy achieved with Polynomial Kernel SVM of degree ' + str(d) + ' is: ' + str(max_acc[i]))


# # The explanation for the result of polinomial SVM is similar to the previous one, when C is big (lamda is small), the pealty term only has a minor weight in the optimization problem, so the training accuracy is almost 100% and overfitting tends to occur in this situation. As C gets smaller(lamda gets bigger) the effect of the penalty starts to show, the traing accuracy starts to decrease and the testing accuracy becomes better, because the classifier that we are getting is not very overfitting to the training data. When C is extremely small(lamda is huge), underfitting occurs and we get bad training accuracy and testing accuracy. Since polynomial kernels have more complicated features than linear SVM, so the training accuracy and testing accuracy are higher than linear SVM. But overfitting can be a problem for polynomial SVM. As we can see the higher degree the polynomial kernel is, the less testing accuracy it has.

# ## 3. SVM with Gaussian Kernal.
# Run an SVM with Gaussian kernal with the penalty parameter $$C \in \{2^i: i = 0, 1, ..., 19\}$$, and plot the training and testing accuracy as a function of $log C$ (semi-log plot).
# 
# Explain how the accuracy changes with repect to penalty parameter $C$ (describe the underfitting and overfitting phenomenon). 
# 
# What is the maximum testing accruracy achieved among all penalty parameters for SVM with Gaussian kernal of each degree? Compare it with linear SVM and polynomial SVM and explain.

# In[11]:


from sklearn.svm import SVC
accuracy_tst = []
accuracy_trn = []
n = np.array(range(20))
C = 2**n
#========Your Code Here============
for i in (C):
    clf = SVC(gamma='auto', kernel='rbf', C=i)
    clf.fit(X_trn_new, Y_trn)
    accuracy_trn.append(clf.score(X_trn_new, Y_trn))
    accuracy_tst.append(clf.score(X_tst_new, Y_tst))
max_acc = max(accuracy_tst)
#=============================
plt.semilogx(C, accuracy_tst)
plt.semilogx(C, accuracy_trn)
plt.title("SVM with Gaussian kernel")
plt.xlabel('C')
plt.ylabel('accuracy')
plt.show()
print('The maximum testing accuracy achieved with SVM with Gaussian kernel is: ' + str(max_acc))


# # The overfitting underfitting phenomenon for Gaussian Kernel is similar to previous ones, when C is big (lamda is small), the pealty term only has a minor weight in the optimization problem, so the training accuracy is almost 100% and overfitting tends to occur in this situation. As C gets smaller(lamda gets bigger) the effect of the penalty starts to show, the traing accuracy starts to decrease and the testing accuracy becomes better, because the classifier that we are getting is not very overfitting to the training data. When C is extremely small(lamda is huge), underfitting occurs and we get bad training accuracy and testing accuracy. 

# In[ ]:




