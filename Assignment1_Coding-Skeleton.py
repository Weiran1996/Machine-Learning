#!/usr/bin/env python
# coding: utf-8

# # Machine Learning for Games
# 
# It has been widely publicized that machine learning has achieved great success in game playing during recent years, including ancient games like [GO](https://en.wikipedia.org/wiki/Go_(game) to modern computer games like [Starcraft](https://starcraft2.com/en-us/). For news, see:
# 
# [The awful frustration of a teenage Go champion playing Google’s AlphaGo](https://qz.com/993147/the-awful-frustration-of-a-teenage-go-champion-playing-googles-alphago/)
# 
# [AI defeated humans at StarCraft II. Here’s why it matters.](https://www.wired.com/story/deepmind-beats-pros-starcraft-another-triumph-bots/)
# 
# 
# We don't have enough background for understanding these complicated algorithms yet. In this assignment, we are going to see how decision trees can help understand some simple games including TIC-TAC-TOE and chess(King-Rook vs. King).
# 
# Make sure you have installed [Pandas](https://pandas.pydata.org/), [numpy](http://www.numpy.org/), [graphviz](https://www.graphviz.org/) and [Scikit Learn](https://scikit-learn.org/) before running the script.
# 
# ```bash
#     conda install pandas numpy graphviz scikit-learn
# ```
# or
# 
# ```bash
#     pip3 install pandas numpy graphviz scikit-learn
# ```
# 
# You may find the following links useful:
# 
# http://scikit-learn.org/stable/modules/tree.html and
# http://scikitlearn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier

# ## 1. Tic-Tac-Toe Endgame Classification
# For introduction and rules of Tic-Tac-Toe, see [Wiki page](https://en.wikipedia.org/wiki/Tic-tac-toe). 
# 
# <img src="tic_tac.jpg" width="400">
# 
# We will use Tic-Tac-Toe Endgame Data Set from UCI machine learning repository. (See introduction [here](https://archive.ics.uci.edu/ml/datasets/Tic-Tac-Toe+Endgame)). This database encodes the complete set of possible board configurations at the end of tic-tac-toe games, where "x" is assumed to have played first. The target concept is "win for x" (i.e., true when "x" has one of 8 possible ways to create a "three-in-a-row"). 
# 
# The dataset has 9 attributes, each indicating the status of each squre. ('x' if "x" is placed, 'o' if "o" is placed and 'b' if blank). Examples of the dataset can be seen here:

# In[17]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.utils import shuffle
tic_toc = pd.read_csv('./tic-tac-toe.data', header=None) #read dataset
tic_toc.columns = ['top_left_sqr', 'top_middle_sqr', 'top_right_sqr',
             'mid_left_sqr', 'mid_mid_sqr', 'mid_right_sqr', 
             'btm_left_sqr', 'btm_mid_sqr', 'btm_right_sqr',
             'class'] # rename each column
tic_toc = shuffle(tic_toc, random_state = 0) # shuffle the dataset
tic_toc.head(10) #print the top ten entries


# ### Post-processing
# To get these features and labels fit into learning models, we need to convert them into integer or boolean variables. For features with $k$ possible values, a common trick is one-hot encoding (dummy variables). Suppose feature $X$ can take $k$ possible values, we encode it into a $k$-dimensional boolean vector where there is a one at location $i$ if $X = 1$ and zeros elsewhere. Suppose $X$ can take $5$ possible values and $X = 3$. Then
# 
# $$ Enc(X) = (0, 0, 1, 0, 0) $$
# 
# We do this for each feature with $k = 3$ and convert labels to $\{0, 1\}$. Note that for one hot encoding, we can always drop the first dimension since if all others are zeros, we know it has to be one.
# 
# Next we randomly pick $70\%$ of the data to  be our training set and the remaining for testing.
# 
# After this, each training example has $18$ features. Training set looks like the following:

# In[18]:


from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

tic_toc.loc[tic_toc['class'] == 'positive', 'class'] = 1 #change labels from words to integers
tic_toc.loc[tic_toc['class'] == 'negative', 'class'] = 0
X = pd.get_dummies(tic_toc.iloc[:, :-1], drop_first=True) # one-hot encoding
y = tic_toc['class'].astype('int')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 0) # split the dataset into training and testing sets
X.head()


# ### Problem 1
# Using information gain as you splitting criterion, set the maximum depth from 1 to 12. Plot the training and testing error with respect to each maximum depth. Justify your plot.

# In[19]:


Err_Train = np.zeros(12)
Err_Test = np.zeros(12)
indices = range(1,13)

#==================Your code ===================
for i in range(1,13):
    clfi = DecisionTreeClassifier(criterion = 'entropy', max_depth = i, random_state = 0)
    clfi.fit(X_train, y_train)
    Err_Train[i-1] = 1-clfi.score(X_train, y_train)
    Err_Test[i-1] = 1-clfi.score(X_test, y_test)
#==============================================

plt.plot(indices,Err_Train, label = "training")
plt.plot(indices,Err_Test, label = "testing")
plt.legend()


# # As the plot shows above, as the decision tree gets more deep and complicated, both training error and testing error get lower. This is because the more complicated the decision tree is, the more likely we'll get the correct label given the input data

# ### Problem 2
# Using GINI impurity as you splitting criterion, set the maximum depth from 1 to 12. Plot the training and testing error with respect to each maximum depth. Is it the same with information gain?

# In[20]:


Err_Train = np.zeros(12)
Err_Test = np.zeros(12)
indices = range(1,13)
#==================Your code ===================
for i in range(1,13):
    clfi = DecisionTreeClassifier(criterion = 'gini', max_depth = i, random_state = 0)
    clfi.fit(X_train, y_train)
    Err_Train[i-1] = 1-clfi.score(X_train, y_train)
    Err_Test[i-1] = 1-clfi.score(X_test, y_test)
#==============================================
plt.plot(indices,Err_Train, label = "training")
plt.plot(indices,Err_Test, label = "testing")
plt.legend()


# # As we can see, the plots of errors with information gain and Gini impurity criterions are pretty similar. Although the acurracy are not necessarily the same, the general trend of training error and testing error are the same

# ### Problem 3
# Pick any tree you have learned above. Let "model_dtc" be the model you have created using scikit-learn. The following script will print the tree into file 'TIC-TOC-TOE.pdf'. What is the root node? Explain whether it coincides with your intuition.

# In[21]:


from sklearn import tree
import graphviz 
clf = DecisionTreeClassifier(criterion = 'gini', max_depth = 10, random_state = 0)
clf.fit(X_train, y_train)
dot_data = tree.export_graphviz(clf, out_file=None, 
                    feature_names = X.columns,
                    class_names= 'win',  
                    filled=True, rounded=True,  
                    special_characters=True)  
graph = graphviz.Source(dot_data) 
graph.render("TIC-TOC-TOE") 


# # The root node of the tree is mid_mid_sqr_o, it coincides with my tuition. Because for both O and X, there are 8 ways to win the game. Position mid_mid_sqr appears in 4 of the 8 winning solutions which has the largest frequency campared with other positions in the 8 winning solutions, so it has larger weight than other positions. In other words, the player who gets position mid_mid_sqr is more likely to win the game, hence position mid_mid_sqr seperates the data better than any other posiitons. Intuitively mid_mid_sqr should be placed at the root node. 

# ## 2. Chess(King-Rook vs. King) Endgame Classification
# For introduction and rules of Chess, see [Wiki page](https://en.wikipedia.org/wiki/Chess). 
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
# And the label is the least number of steps that the white must use to win. (draw if more than 16). The following is how the data set looks like.

# In[22]:


chess = pd.read_csv('./krkopt_data.txt', header=None) # read data 
chess.columns = ['wkf', 'wkr', 'wrf', 'wrr', 'bkf', 'bkr', 'class'] # rename columns
chess = shuffle(chess, random_state = 0) # shuffle the data 
chess.head(10) # print top 10 labels


# Next we convert these values into boolean features using the same one-hot encoding trick we described for TIC-TAC-TOE game. Deleting symmetric features (the dataset only has white kings in the bottom-left corner) for the white king and drop the first for the others, we get a data set with $36$ boolean features. 
# 
# Next we randomly pick $70\%$ of the data to  be our training set and the remaining for testing. Training set looks like the following:

# In[23]:


from sklearn.preprocessing import LabelEncoder

d_wkf = pd.get_dummies(chess['wkf'], prefix='wkf')   # one hot encoding
d_wkr = pd.get_dummies(chess['wkr'], prefix='wkr')
d_wrf = pd.get_dummies(chess['wrf'], prefix='wrf', drop_first=True)
d_wrr = pd.get_dummies(chess['wrr'], prefix='wrr', drop_first=True)
d_bkf = pd.get_dummies(chess['bkf'], prefix='bkf', drop_first=True)
d_bkr = pd.get_dummies(chess['bkr'], prefix='bkr', drop_first=True)
chess_new = pd.concat([d_wkf, d_wkr, d_wrf, d_wrr, d_bkf, d_bkr, chess['class']], axis=1) # get new dataset with new features
X = chess_new.iloc[:, :-1] 
y = chess_new['class']
le = LabelEncoder()  # change labels into integers 
y = le.fit_transform(y) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # split the dataset into training and testing sets
X_train.head(10) # print top 10 entries


# ### Problem 4 
# Using information gain as you splitting criterion, set the maximum depth from 20 to 35. Plot the training and testing error with respect to each maximum depth. When is the maximum training accuracy achieved? When is the maximum testing accuracy achieved? Explain this phenomenon.

# In[24]:


start = 20
end = 36
indices = range(start,end)
Err_Train = np.zeros(end - start)
Err_Test = np.zeros(end - start)
#==================Your code ===================
for i in range(20,36):
    clfi = DecisionTreeClassifier(criterion = 'entropy', max_depth = i,  random_state = 0 )
    clfi.fit(X_train, y_train)
    Err_Train[i-20] = 1-clfi.score(X_train, y_train)
    Err_Test[i-20] = 1-clfi.score(X_test, y_test)
    
#print Err_Train
#print Err_Test
print ("The maximum training accuracy is: ", 1-min(Err_Train) )
print ("The maximum training accuracy occurs when the tree depth is: ", 20+np.argmin(Err_Train))
print ("The maximum testing accuracy is: ", 1-min(Err_Test))
print ("The maximum testing accuracy occurs when the tree depth is: ", 20+np.argmin(Err_Test))

#==============================================


plt.plot(indices,Err_Train, label = "training")
plt.plot(indices,Err_Test, label = "testing")
plt.legend()


# # The maximum training accuracy is:  1.0                                                           The maximum training accuracy occurs when the tree depth is:  31                The maximum testing accuracy is:  0.5446120945705121                                  The maximum testing accuracy occurs when the tree depth is:  26                                                                                                                                                          As the output result shows, for the training data, the maximum accuracy 1.0 occurs when the tree depths are 31-35 since as decision tree gets deeper and more complicated, the prediction of training data always gets better. As for training data, when the tree gets too complicated and too tilted to training data, overfitting occurs. That explains why testing accuracy increases then decreses as tree depth gets larger.

# ### Problem 5
# Let's take a step further towards real AI applications. For the same game set-up, suppose you have a perfect decision tree which can tell you the minimum number of moves the white need to win. Given any instance, can you tell us which move is the optimal move for the white?

# # Yes, we can tell which move is the optimal move for the white because the positions of chess are attributes of the tree and minimum number of moves are labels. Given a perfect decision tree, we can move white chess to any direction and check the decision tree see if the minimum numbers of moves is lower than before, if it is, then we find the optimal move, if the minimum numbers of moves is larger or equal to before, then it is not the optimal move

# In[ ]:




