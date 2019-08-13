#!/usr/bin/env python
# coding: utf-8

# # Naive Bayes for Flower Recognition
# In this assignment, you will be asked to implement Gaussian Naive Bayes by yourself (sci-kit learn is not allowed) and use it classify what kind of iris flower the sample is given its features listed below:
# 1. sepal length in cm 
# 2. sepal width in cm 
# 3. petal length in cm 
# 4. petal width in cm 
# 
# There are three kinds of flowers: Iris Setosa, Iris Versicolour and Iris Virginica.
# The data we get comes from [Kaggle: Iris Dataset Visualization and Machine Learning](https://www.kaggle.com/xuhewen/iris-dataset-visualization-and-machine-learning). Make sure you have installed pandas, numpy and seaborn before running the script.
# ```bash
#     conda install pandas numpy seaborn
# ```
# or
# 
# ```bash
#     pip3 install pandas numpy seaborn
# ```
# The following code loads the data and the dataset looks like the following:

# In[285]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.utils import shuffle
iris = pd.read_csv('iris_data.txt', header=None) #read dataset
iris.columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species'] # rename each column
iris_visual = shuffle(iris, random_state = 0) # shuffle the dataset
iris_visual.head(10) #print the top ten entries


# In[286]:


iris.info()


# ## Visualization of the dataset.
# The following code visualize the distribution of each pair of the features within each class. (Diagnals are probability density function for each feature).

# In[287]:


import seaborn as sns
sns.set()
sns.pairplot(iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Species']],
             hue="Species", diag_kind="kde")


# ## Data splitting
# Next, we split the data into training and testing sets according to 60/40 rule. 

# In[288]:


from sklearn.model_selection import train_test_split
iris_data = np.array(iris)
X_trn, X_tst, y_trn, y_tst = train_test_split(iris_data[:,0:4], iris_data[:,4], test_size = 0.4, random_state = 0) # split the dataset into training and testing sets


# # Problem 1 Write your own Gaussian Naive Bayes code
# Using the skeleton provided below, write your own code for learning and inference using Gaussian Naive Bayes model. You can use the skeleton provided in the second problem to verify whether you are writing it correctly. Reading through chapter 1.9.1 in the following link and the slide from last year can be useful:
# https://scikit-learn.org/stable/modules/naive_bayes.html
# 
# https://www.dropbox.com/s/6d5h6fig1fj44e4/Naive_Bayes.pdf?dl=0

# In[289]:


output_label = list(set(y_trn))
conditional_data = []
temp = []
mean = np.zeros((len(output_label), X_trn.shape[1]))
std = np.zeros((len(output_label), X_trn.shape[1]))
#按每一个label来看
for i in range(len(output_label)):
    #every feature
    for m in range(X_trn.shape[1]):
        #每一组数据只看一个feature
        for j in range(X_trn.shape[0]):
            if ( y_trn[j] == output_label[i] ) :
                temp.append(X_trn[j][m])
                
        conditional_data.append(temp)
        temp = []


for i in range(len(output_label)):
    for j in range(X_trn.shape[1]):
        mean[i][j] = np.mean(conditional_data[(i)*X_trn.shape[1] + j])
        std[i][j] = np.std(conditional_data[(i)*X_trn.shape[1] + j])
#print(conditional_data)
#print(conditional_data[1])
print(mean)
print(std)


# In[290]:


def gnb_train(X, y, output_label):
    output_size = len(output_label)
    prior = np.zeros(output_size)
    mean = np.zeros((output_size, X.shape[1]))
    std = np.zeros((output_size, X.shape[1]))
# ======= Your Code Here =======
    for i in range(output_size):
        prior[i] = (np.count_nonzero(y_trn == list(set(y_trn))[i])/len(y_trn))
    
    
    conditional_data = []
    temp = []
    #按每一个label来看
    for i in range(output_size):
        #every feature
        for m in range(X.shape[1]):
            #每一组数据只看一个feature
            for j in range(X.shape[0]):
                if ( y[j] == output_label[i] ) :
                    temp.append(X[j][m])
            conditional_data.append(temp)
            temp = []

    for i in range(len(output_label)):
        for j in range(X.shape[1]):
            mean[i][j] = np.mean(conditional_data[(i)*X.shape[1] + j])
            std[i][j] = np.std(conditional_data[(i)*X.shape[1] + j])
                
    return prior, mean, std


# In[291]:


prior, mean, std = gnb_train(X_trn, y_trn, list(set(y_trn)))


# In[292]:


import math
def gaussian(x, mu, sig):
    variance = math.pow(sig, 2)
    return math.exp(-(math.pow(x-mu, 2)/(2*variance)))/math.sqrt(2*math.pi*variance)
    


def gnb_predict(X, prior, mean, std, output_label):
    predict = []
    # ======= Your Code Here =======
    featurenum = X.shape[1]
    testlen = X.shape[0]

    for i in range(testlen):
        py0 = py1 = py2 = 1
        
        for j in range(featurenum):
            prob_y0 = gaussian(X[i, j],  mean[0][j],  std[0][j])
            prob_y1 = gaussian(X[i, j],  mean[1][j],  std[1][j])
            prob_y2 = gaussian(X[i, j],  mean[2][j],  std[2][j])
        
            py0 *= prob_y0
            py1 *= prob_y1
            py2 *= prob_y2
    
        prob = [prior[0]*py0, prior[1]*py1, prior[2]*py2]
        idx = prob.index(max(prob))
        predict.append(output_label[idx])



    return predict


# # Problem 2: Inference on IRIS dataset
# Using your own GNB functions, implementing Gaussian Naive Bayes algorithm for the first feature, the first two features, the first three features and the first four features. Output the error for each experiment and explain using the visualization of the dataset.

# In[293]:


from sklearn.metrics import hamming_loss
output_label = list(set(y_trn))
for i in range(1, 5):
    prior, mean, std = gnb_train(X_trn[:,0:i], y_trn, output_label)
    y_pred = gnb_predict(X_tst[:,0:i], prior, mean, std, output_label)
    error = hamming_loss(y_tst, y_pred)
    print("Test error using first", i, "features:",  error)


# # As the output errors show, the more features we use to predict the label, the less error we are going to get.  From the visualization of the dataset, we can see that feature 1 does not seperate the data very well, the data of feature 1 are superimposed badly for two kinds of flowers, so the predicting error are pretty high using only feature 1. With the combination of feature 1 and feature 2, the error decreses a little bit. Feature 3 and 4 seperate the data set very well, so we get much better predictions when these two features are used. Generally, the more features we use to predict the label, the less error we are going to get.

# ### Problem 3: The Limitation of Naive Bayes
# From the last feature, we can see the more features we use, we will get better performance. In this question, we show sometimes it is not the case. Repeat the first feature for $i = 1, 2, 3, ...., 300$ times and plot the testing error. Justify the plot. If we repeat the first
# feature for infinitely many number of times, will the test accuracy become zero? If yes, explain why and if not, what would be your guess for the final error?

# In[294]:


err = np.zeros(300)
for i in range(1,300):
    X_trn_new = np.hstack((X_trn, np.tile(X_trn[:, [0]], i)))
    X_tst_new = np.hstack((X_tst, np.tile(X_tst[:, [0]], i)))
    prior, mean, std = gnb_train(X_trn_new, y_trn, output_label)
    y_pred = gnb_predict(X_tst_new, prior, mean, std, output_label)
    err[i] = hamming_loss(y_tst, y_pred)
plt.plot(err)


# # As the above plot shows, the error is increasing and approaching to the fixed number 0.35 as the repetition of feature 1 increases. So if we repeat the first feature for infinitely many number of times, the error will stop growing and be fixed 0.35 as the plot shows. So the test accuracy will not become zero and the guess for the final error is 0.35.

# In[ ]:




