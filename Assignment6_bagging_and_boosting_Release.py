#!/usr/bin/env python
# coding: utf-8

# # Assignment 6: Bagging and Boosting
# In this assignment, we are going to revisit the chess(King-Rook vs. King) Endgame Classification problem we saw in the first assignment. Recall that using decision trees, we couldn't get good testing accuracy (around 55%).  We will try to improve this using ensemble methods.

# ## Chess(King-Rook vs. King) Endgame Classification
# For introduction and rules of Chess, see [Wiki page](https://en.wikipedia.org/wiki/Chess). 
# 
# <img src="chess.png" width="400">
# 
# We will use Chess(King-Rook vs. King) Data Set from UCI machine learning repository. (See introduction [here](https://archive.ics.uci.edu/ml/datasets/Chess+(King-Rook+vs.+King)). This database has 28056 possible instances of chess endgame situations where the white has a king and a rook and the black has only a king. The goal is to determine what is the minimum depth for the white to win.
# 
# The dataset has 6 attributes. Each of them can take 8 values, listed as following:
# 
# 1. White King file (column a - h) 
# 2. White King rank (row 1 - 8) 
# 3. White Rook file 
# 4. White Rook rank 
# 5. Black King file 
# 6. Black King rank 
# 
# And the label/class is the least number of steps that white must use to win. (draw if more than 16). The following is how the data set looks like.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.utils import shuffle
chess = pd.read_csv('./krkopt_data.txt', header=None)
chess.columns = ['wkf', 'wkr', 'wrf', 'wrr', 'bkf', 'bkr', 'class']
chess = shuffle(chess, random_state = 0)
chess.head(10)


# Next we convert these values into boolean features using the same one-hot encoding trick we described for TIC-TAC-TOE game. Deleting symmetric features for the white king and drop the first for the others, we get a data set with $36$ boolean features. 
# 
# Next we randomly pick $70\%$ of the data to  be our training set and the remaining for testing. Training set looks like the following:

# In[2]:


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
d_wkf = pd.get_dummies(chess['wkf'], prefix='wkf')
d_wkr = pd.get_dummies(chess['wkr'], prefix='wkr')
d_wrf = pd.get_dummies(chess['wrf'], prefix='wrf', drop_first=True)
d_wrr = pd.get_dummies(chess['wrr'], prefix='wrr', drop_first=True)
d_bkf = pd.get_dummies(chess['bkf'], prefix='bkf', drop_first=True)
d_bkr = pd.get_dummies(chess['bkr'], prefix='bkr', drop_first=True)
chess_new = pd.concat([d_wkf, d_wkr, d_wrf, d_wrr, d_bkf, d_bkr, chess['class']], axis=1)
X = chess_new.iloc[:, :-1]
y = chess_new['class']
le = LabelEncoder()
y = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train.head(10)


# ## 1. Bagging and Random Forest.
# Recall that the classifier we get by bagging with decision trees as our base classifier is called Random Forest. Using the Bagging Meta Classifier implemented in 'sklearn.ensemble'. See [Scikit Learn Bagging](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier) for how to use it. (Don't use the random forest classifier in 'sklearn.ensemble' directly since it has other parameters and you may get weird results). Using information gain as your splitting criterion and train a random forest with number of classifiers in the following set:
# 
# $$n = \{ 2^i | i = 0, 1, ..., 11\}$$
# 
# Plot the training and testing accruracy and justify your plot.

# In[3]:


from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
n_max = 12
Err_Train = np.zeros(n_max)
Err_Test = np.zeros(n_max)
indices = 2**np.array(range(0,n_max))
#==========Write your code below=====
for i in range(12):
    clf = BaggingClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 0 ), n_estimators=indices[i])
    clf.fit(X_train, y_train)
    Err_Train[i] = clf.score(X_train, y_train)
    Err_Test[i] = clf.score(X_test, y_test)

#================================

plt.semilogx(indices,Err_Train, label = "training")
plt.semilogx(indices,Err_Test, label = "testing")
plt.legend()

# As the plot shows above, as the number of classifier increases, both training accuracy and testing accuracy get better, and the classifier are less overfitting to the training data
# ## 2. Adaboost.
# 
# Decision trees with small maximum depth won't give us good performance because of limited complexity. In this problem, we use adaboost algorithm to reduce the bias of the model and hopefully this will give us better performance. Using decision trees with maximum depth 10, 20, 50, 100 as you base classifiers, try [Adaboost algorithm](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier) with number of iterations in the following set:
# 
# $$T = \{ 2^i | i = 0, 1, ..., 11\}$$
# 
# Plot your training and testing error and justify your plots. When do you get zero training error? Explain.

# In[5]:


from sklearn.ensemble import AdaBoostClassifier
n_max = 12
Err_Train = np.zeros(n_max)
Err_Test = np.zeros(n_max)
indices = 2**np.array(range(0,n_max))
#==========Write your code below=====
#You can repeat this block multiple times for your experiments.

for i in range(12):
    clf = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', max_depth = 10, random_state = 0 ), n_estimators=indices[i])
    clf.fit(X_train, y_train)
    Err_Train[i] = 1-clf.score(X_train, y_train)
    Err_Test[i] = 1-clf.score(X_test, y_test)

#================================
plt.semilogx(indices,Err_Train, label = "training")
plt.semilogx(indices,Err_Test, label = "testing")
plt.legend()


# In[6]:


for i in range(12):
    clf = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', max_depth = 20, random_state = 0 ), n_estimators=indices[i])
    clf.fit(X_train, y_train)
    Err_Train[i] = 1-clf.score(X_train, y_train)
    Err_Test[i] = 1-clf.score(X_test, y_test)

#================================
plt.semilogx(indices,Err_Train, label = "training")
plt.semilogx(indices,Err_Test, label = "testing")
plt.legend()


# In[7]:


for i in range(12):
    clf = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', max_depth = 50, random_state = 0 ), n_estimators=indices[i])
    clf.fit(X_train, y_train)
    Err_Train[i] = 1-clf.score(X_train, y_train)
    Err_Test[i] = 1-clf.score(X_test, y_test)

#================================
plt.semilogx(indices,Err_Train, label = "training")
plt.semilogx(indices,Err_Test, label = "testing")
plt.legend()


# In[8]:


for i in range(12):
    clf = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy', max_depth = 100, random_state = 0 ), n_estimators=indices[i])
    clf.fit(X_train, y_train)
    Err_Train[i] = 1-clf.score(X_train, y_train)
    Err_Test[i] = 1-clf.score(X_test, y_test)

#================================
plt.semilogx(indices,Err_Train, label = "training")
plt.semilogx(indices,Err_Test, label = "testing")
plt.legend()

As we can see from the plots above, as the maximum depth increases, the training error gets better. When the maximum depth are 50 and 100, we will always have 0 training error
# ## 3. Boosting Complex Classifiers.
# 
# Using random forest with 10 trees and max_depth 50 as your base classifier, train an AdaBoost classifier with number 
# of iterations from 
# 
# $$T = \{ 2^i | i = 0, 1, ..., 6\}$$
# 
# Plot the training and testing error. Justify your plot.

# In[9]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
n_max = 6
Err_Train = np.zeros(n_max)
Err_Test = np.zeros(n_max)
indices = 2**np.array(range(0,n_max))
#==========Write your code below=====
for i in range(6):
    clf = AdaBoostClassifier(BaggingClassifier(DecisionTreeClassifier(criterion = 'entropy', random_state = 0 ), n_estimators=10), n_estimators=indices[i], random_state = 0)
    clf.fit(X_train, y_train)
    Err_Train[i] = 1-clf.score(X_train, y_train)
    Err_Test[i] = 1-clf.score(X_test, y_test)

#================================
plt.semilogx(indices,Err_Train, label = "training")
plt.semilogx(indices,Err_Test, label = "testing")
plt.legend()


# In[ ]:




