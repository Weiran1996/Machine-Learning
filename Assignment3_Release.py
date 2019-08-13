#!/usr/bin/env python
# coding: utf-8

# # Wine Quality Classification
# 
# In this assignment, we will use logistic regression to judge the quality of wines. The dataset is taken from UCI machine learning repository. For description of the dataset, see [here](https://archive.ics.uci.edu/ml/datasets/wine+quality).
# 
# Attributes of the dataset are listed as following:
# 1. fixed acidity 
# 2. volatile acidity 
# 3. citric acid 
# 4. residual sugar 
# 5. chlorides 
# 6. free sulfur dioxide 
# 7. total sulfur dioxide 
# 8. density 
# 9. pH 
# 10. sulphates 
# 11. alcohol 
# 
# Output variable (based on sensory data): 
# 12. quality (score between 0 and 10)
# 
# The following code loads the dataset, and the dataset looks like the following:

# In[17]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
#train = np.genfromtxt('wine_training1.txt', delimiter=',')
red = pd.read_csv('winequality-red.csv')
white = pd.read_csv('winequality-white.csv')
red = shuffle(red, random_state = 10)
white = shuffle(white, random_state = 10)
red.head(10)
white.head(10)


# ## Data Splitting
# To get this into a binary classification task. We split the quality into a binary feature *good* or *bad* depending on whether the quality is larger than 6 or not.
# 
# Next we randomly pick $70\%$ of the data to be our training set and the remaining for testing for both red and white wines.

# In[18]:


from sklearn.model_selection import train_test_split
X_red = red.iloc[:, :-1]
y_red = red.iloc[:, -1] >= 6

X_train_red, X_test_red, y_train_red, y_test_red = train_test_split(X_red, y_red, test_size=0.3, random_state = 0)

X_white = white.iloc[:, :-1]
y_white = white.iloc[:, -1] >= 6
X_train_white, X_test_white, y_train_white, y_test_white = train_test_split(X_white, y_white, test_size=0.3, random_state = 0)

#y_red.head(10)
y_white.head(10)


# ## Problem 1 Logistic Regression for Red Wine
# 
# Using scikit learn, train a Logistic Regression classifier using 'X_trn_red, y_trn_red'. Use the
# solver sag, which stands for Stochastic Average Gradient. Set max iteration to be 10000. Test the model on X_test_red. Output the testing error.

# In[19]:


#========Your code here ======
from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression(random_state=0, solver='sag',max_iter=10000).fit(X_train_red, y_train_red)
print(clf1)
error_red = 1 - clf1.score(X_test_red, y_test_red)
#========================
print('The testing error for red wine is: ' + str(error_red) + '.')


# ## Problem 2 Logistic Regression for White Wine
# 
# Using scikit learn, train a Logistic Regression classifier using 'X_trn_white, y_trn_white'. Use the
# solver sag, which stands for Stochastic Average Gradient. Set max iteration to be 10000. Test the model on X_test_white. Output the testing error.

# In[20]:


#========Your code here ======
clf2 = LogisticRegression(random_state=0, solver='sag',max_iter=10000).fit(X_train_white, y_train_white)
print(clf2)
error_white = 1 - clf2.score(X_test_white, y_test_white)

#========================
print('The testing error for white wine is: ' + str(error_white) + '.')


# ## Problem 3 
# Use the model you trained using 'X_trn_white, y_trn_white' to test on 'X_test_red' and use the model you trained on 'X_test_white'. Print out the errors and compare with previous results. Explain.

# In[21]:


#========Your code here ======
error_red = 1 - clf2.score(X_test_red, y_test_red)
error_white = 1 - clf1.score(X_test_white, y_test_white)

#========================
print('The testing error for red wine using white wine training data is: ' + str(error_red) + '.')
print('The testing error for white wine using red wine training data is: ' + str(error_white) + '.')


# # The testing error for red wine using white wine training data is: 0.356, it is higher than the testing error for red wine using red wine training data which is 0.275. That means, to predict the quality of red wine, it's better to train the model with red wine training data rather than using white wine training data, which is intuitive and makes perfect sense. However, when we predict the quality of red wine, white wine training data is not all useless. Since we We split the quality into a binary feature good or bad depending on whether the quality is larger than 6 or not, so the chance of a random guess of red wine quality to be correct is 0.5. As we can see, with white wine training data, testing error of predicting red wine quality is 0.356 which is better than a random guess. This means that the features of white wine and red wine have something in common and it's helpful for predicting the quality of red wine. The same logic can be applied to predicting white wine quality with red wine training data.

# # Problem 4 The effect of regularization
# Using red wine dataset. Implement logistic regression in sklearn, using $\ell_2$ regularization with regularizer value C in the set $\{0.00001 \times 4^i: i = 0,1,2,..., 15\}$. (The regularization parameter is 'C' in scikit-learn, which is the inverse of $\lambda$ we see in class). Plot the training error and test error with respect to the regularizer value. Explain what you get.

# In[22]:


N = np.array(range(0,15))
alpha = 0.00001*(4**N)
error_trn = np.zeros(15)
error_tst = np.zeros(15)
#========Your code here ======
for i in range(15):
    clf = LogisticRegression(random_state=0, solver='sag',max_iter=10000, C=alpha[i]).fit(X_train_red, y_train_red)
    error_trn[i] = 1 - clf.score(X_train_red, y_train_red)
    error_tst[i] = 1 - clf.score(X_test_red, y_test_red)
    
#========================
plt.figure(1)
plt.semilogx(alpha, error_tst,label = 'Test')
plt.semilogx(alpha, error_trn, label = 'Train')
plt.legend()


# # As we can see from the above figure, when C is small (lamda is big), both training error and testing error are high, this is because the penalty term doninates the optimization function, so the function is no longer optimizing the error but the penalty term which gives us high training error and testing error. As C increases(lamda decreases), both training error and testing error get lower because the loss function takes more weight in the optimization function. When C is really large (lamda is really small), overfitting is more likely to occur. According to the above figure, the best C (or lamda) lies between 0.1-10 where overfitting is not likely to happen and it gives good training error and testing error.

# In[ ]:




