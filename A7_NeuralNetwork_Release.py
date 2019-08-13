#!/usr/bin/env python
# coding: utf-8

# # Assignment 7: Neural Networks and Backpropagation
# In this assignment, you will be asked to write your own code to implement the learning process of a simple neural network. We will use a simple version of [MNIST dataset](http://yann.lecun.com/exdb/mnist/), which we have introduced in Assignment 5. To make the problem simpler, we only take images with label '8' and '9', which gives us a binary classification problem. Then we subsample the dataset and reduce the dimension of each image using average pooling. The following code loads the dataset and prints its dimension.

# In[1]:


import numpy as np 
import matplotlib.pyplot as plt
#Load data
import scipy.io as sio
a = sio.loadmat('mnist_binary.mat')
X_trn = a['X_trn']
X_tst = a['X_tst']
Y_trn = a['Y_trn'][0]
Y_tst = a['Y_tst'][0]
print(X_trn.shape)
print(X_tst.shape)
print(Y_trn.shape)
print(Y_tst.shape)


# ## Requirements
# 1. You are not allowed to use any machine learning libraries which have neural networks implemented.
# 
# 2. Notice here most of the problems you have will be regarding the dimensions of variables. In each skeleton function we provide, we have one assert line to help you verify whether you write your code correctly. Passing the assert line doesn't mean your code is correct. But it is a necessary condition.
# 
# 3. You don't need to strictly follow the skeleton we provide. As long as you answer the problems correctly, you can write in any style you prefer.

# ## Parameters
# Let's first implement a simple neural network with one hidden layer and one output layer. The hidden layer only has $n_h$ neurons. We assume the output layer has two neurons. Hence you will have 4 parameters to describe the neural network: 
# 
# 1. $W_1$, a $n_h$ by 196 matrix, which is the weight matrix between features and the hidder layer.
# 2. $b_1$, a scalar, which is the offset for the first layer.
# 3. $W_2$, a 2 by $n_h$ matrix, which is the weight matrix between the hidder layer and the output layer.
# 4. $b_2$, a scalar, which is the offset for the second layer.
# 
# The following script initializes the above four parameters and returns them as a dictionary.

# In[2]:


#Initialize parameters 
num_hidden = 20 #number of neurons in the hidden layer
num_op = 2 #number of neurons in the output layer

def initialize_parameters(size_input, size_hidden, size_output):
    np.random.seed(2)
    W1 = np.random.randn(size_hidden, size_input) * 0.01
    b1 = np.zeros(shape=(size_hidden, 1))
    W2 = np.random.randn(size_output, size_hidden) * 0.01
    b2 = np.zeros(shape=(size_output, 1))
    parameters = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2}
    return parameters
parameters = initialize_parameters(X_trn.shape[0], num_hidden, num_op)
print('W1',parameters['W1'].shape)
print('b1',parameters['b1'].shape)
print('W2',parameters['W2'].shape)
print('b2',parameters['b2'].shape)


# ## Softmax function.
# Given the output layer $z_1, z_2$. The softmax outputs are probability estimates for outputing the corresponding label:
# 
# $$Pr(Y = 1 | z_1, z_2) = \frac{e^{z_1}}{e^{z_1} + e^{z_2}}$$
# 
# Write code in the cell below to do the softmax computation. Note here Z2 should be a matrix of shape $2 \times n$, where $n$ is the number of training samples. Your output softmax should be of the same size.

# In[3]:


import math
def softmax(Z2):
    # ip - (M,N) array where M is no. of neurons in output layer, N is number of samples.
    # You can modify the code if your output layer is of different dimension
   # =========Write your code below ==============
    columnlen = len(Z2[0])
    softmax = np.zeros(shape = (2,columnlen))
    for i in range(columnlen):
        softmax[0][i] = math.exp(Z2[0][i])/(math.exp(Z2[0][i]) + math.exp(Z2[1][i]))
        softmax[1][i] = math.exp(Z2[1][i])/(math.exp(Z2[0][i]) + math.exp(Z2[1][i]))
    # =============================================
    assert(softmax.shape == Z2.shape)
    return softmax


# ## Activation function.
# The following function should be able to implement activation function given the input.

# In[4]:


def activ(ip,act):
    # ip - array obtained after multiplying inputs with weights (between input layer and hidden layer)
    # act - ReLU or Sigmoid
    row = len(ip)
    column = len(ip[0])
    out = np.zeros(shape = (row,column))
    if act =="ReLU":
        
        # =========Write your code below ==============
        for i in range(row):
            for j in range(column):
                if ip[i][j] < 0:
                    out[i][j] = 0
                else:
                    out[i][j] = ip[i][j]



    # =============================================
    elif act == "Sigmoid":
        # =========Write your code below ==============
          out = 1/(1+np.exp(-ip))


    # =============================================
    assert(out.shape == ip.shape)
    return out


# ## Forward Propagation
# Given $X, W_1, b_1, W_2, b_2$, the following function will compute the neurons and activated values in the hidden layer, denoted by $Z_1, A_1$ respectively. It will also return the neurons in the last layer and the softmax function computed from it, denoted by $Z_2, A_2$ respectively.

# In[42]:


#Forward Propagation   
def forward_propagation(X, parameters, act):
# =========Write your code below ==============
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    
    Z1 = np.dot(W1, X) + b1
    A1 = activ(Z1, act)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)


    # =============================================
    
    assert(A2.shape == (2, X.shape[1]))
    
    neuron = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    return neuron


# # Backward propagation
# In this assignment, we will use the cross-entropy loss defined below as our loss function. Let $\hat{y}$ be the outputs after the softmax layer corresponding to label 8 and let $y$ be the true labels (assume 1 for '8', 0 for '9')
# $$L(y,\hat{y}) = \frac{1}{m}\sum\limits_{i=1}^{m} -y_i\log(\hat{y}_i) - (1-y_i)\log(1-\hat{y}_i).$$
# 
# Given the parameters and the neuron values, we can calculate the derivative of the loss function w.r.t all the parameters $W_1, b_1, W_2, b_2$ using backward propagation. Note here, all the gradients should be of the same dimension as the corresponding parameters. 

# In[43]:


def backprop(parameters, neuron, X, Y, act):

    
# =========Write your code below ==============

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = neuron["A1"]
    A2 = neuron["A2"]
    Z1 = neuron["Z1"]
    Z2 = neuron["Z2"]
    c = len(X[0])
    
    Y = Y.reshape((1,Y.shape[0]))
    
    dZ2 = A2 - np.concatenate((Y,1-Y), axis = 0)
    dW2 = np.matmul(dZ2, A1.transpose())/c
    db2 = np.sum(dZ2, axis=1, keepdims=True)/c
    dZ1 = np.dot(W2.transpose(), dZ2)
    
    if act == 'ReLU':
        dZ1[Z1<=0] = 0
    else:
        dZ1 *= A1*(1-A1)
        
    dW1 = np.matmul(dZ1, X.transpose())/c
    db1 = np.sum(dZ1, axis=1, keepdims=True)/c


    # =============================================
    
    assert(dW1.shape == W1.shape)
    assert(dW2.shape == W2.shape)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads
#print(backprop(parameters, neuron, X_trn, Y_trn, act='Sigmoid')['dW1'].shape)
#print(backprop(parameters, neuron, X_trn, Y_trn, act='Sigmoid')['dW2'].shape)
#print(backprop(parameters, neuron, X_trn, Y_trn, act='Sigmoid')['db1'].shape)
#print(backprop(parameters, neuron, X_trn, Y_trn, act='Sigmoid')['db2'].shape)

def cross_entropy_loss(softmax, Y):
# =========Write your code below ==============
    d = len(Y)
    loss = np.zeros(shape= (d,1))
    for i in range(d):
        loss[i] = -(Y[i] * np.log(softmax[0][i]) + (1-Y[i] )* np.log(softmax[1][i]))
# =============================================        
 #   assert(loss.shape = Y.shape)
    return loss


# ## Parameter updates
# Given the parameters and the gradients, we simply update the parameters by the following:
# 
# $$W = W - \eta dW$$
# 
# where $\eta$ is the learning rate.

# In[44]:


def update_parameters(parameters, grads, learning_rate):

# =========Write your code below ==============

    W1 = parameters["W1"]
    W2 = parameters["W2"]
    b1 = parameters["b1"]
    b2 = parameters["b2"]
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    W1 = W1- learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2- learning_rate * dW2
    b2 = b2 - learning_rate * db2

# =============================================

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


# ## Neural network models
# Combining the above mentioned parameters, implement the following function to learn a neural network and do inference on it. For prediction, you take the argument that gives the largest softmax output at the last layer.

# In[45]:


from sklearn.metrics import accuracy_score
def nn_model1(X_trn, X_tst, Y_trn, Y_tst, n_h, n_o, epochs, act, learning_rate):
    #X_trn: the training set
    #X_tst: the test set
    #Y_trn: training labels
    #Y_tst: test labels
    #n_h: number of neurons in the hidden layer
    #n_o: number of neurons in the output layer
    #epochs: number of epochs for the training
    #act: the activation function you choose
    #learning_rate: a list of length epochs, which consists of the learning rate in each step
    
    assert(len(learning_rate) == epochs)
    
   # =========Write your code below ==============
    err_tst = np.zeros(shape = (epochs, 1))
    err_trn = np.zeros(shape = (epochs, 1))
    loss_trn = np.zeros(shape = (epochs, 1))
    parameters = initialize_parameters(X_trn.shape[0], n_h, n_o)
    
    for i in range(epochs):
        neuron = forward_propagation(X_trn, parameters, act)
        grads = backprop(parameters, neuron, X_trn, Y_trn, act)
        parameters = update_parameters(parameters, grads, learning_rate[i])
        
        predict_tst = np.argmax(forward_propagation(X_tst, parameters, act)['A2'], axis = 0)
        predict_trn = np.argmax(forward_propagation(X_trn, parameters, act)['A2'], axis = 0)
        
        err_tst[i] = accuracy_score(predict_tst, Y_tst)
        err_trn[i] = accuracy_score(predict_trn, Y_trn)
        en = cross_entropy_loss(neuron["A2"], Y_trn)
        loss_trn[i] = (1/len(en)) * np.sum(en)











    # =============================================    
    #err_tst: testing error (classification error) in each epoch
    #err_trn: training error (classification error) in each epoch
    #loss_trn: training loss (cross entropy loss) in each epoch
    #parameters: the final learned parameters
    return err_tst, err_trn, loss_trn, parameters


# ## Problem 0: Verify that your code is working well.
# Using ReLU (Sigmoid) as your activation function, implement a learning algorithm with fixed learning rate $\eta = 0.01$ at each step. Set the number of epochs to be 20000. Plot the cross entropy loss at each epoch to convince yourself that you are training well. (Your cross entropy loss should be decreasing smoothly. This part won't be graded.)

# In[22]:


epochs = 20000
lr1 = 0.01*np.ones(epochs)
# =========Write your code below ==============


err_tst, err_trn, loss_trn, parameters = nn_model1(X_trn, X_tst, Y_trn, Y_tst, 20, 2, epochs, "ReLU", lr1)


# =============================================
plt.figure(1, figsize=(12, 8))
plt.plot(range(epochs), loss_trn, '-', color='orange',linewidth=2, label='training loss (lr = 0.01)')
plt.title('Training loss')
plt.xlabel('epoch')
plt.ylabel('Cross entropy error')
plt.legend(loc='best')
plt.grid()
plt.show()


# # Problem 1: Learning with fixed learning rate.
# Using ReLU as your activation function, implement a learning algorithm with fixed learning rate $\eta = 0.01$ at each step. Plot the training and testing error (classification error) you get at each epoch. Justify your plot. (Set the number of hidden neurons in the hidden layer to be 20 for problem 1-3, for all problems below, set epochs = 20000).

# In[23]:


epochs = 20000
lr1 = 0.01*np.ones(epochs)
# =========Write your code below ==============


#err_tst, err_trn, loss_trn, parameters = nn_model1(X_trn, X_tst, Y_trn, Y_tst, 20, 2, epochs, "ReLU", lr1)


# =============================================
plt.figure(1, figsize=(12, 8))
plt.plot(range(epochs), err_trn, '-', color='orange',linewidth=2, label='training error (lr = 0.01)')
plt.plot(range(epochs), err_tst, '-b', linewidth=2, label='test error (lr = 0.01)')
#plt.plot(range(epochs), trn_loss, '-r', linewidth=2, label='loss (lr = 0.01)')

plt.title('ReLU(Learning rate=0.01)')
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend(loc='best')
plt.grid()
plt.show()


# # Problem 2: 
# Using ReLU as your activation function, change the learning rate to $\eta = 0.1$. Plot the plots on the same figure as in problem 1. Compare the plots and justify.

# In[28]:


lr2 = 0.1*np.ones(epochs)
# =========Write your code below ==============


err_tst2, err_trn2, loss_trn2, parameters2 = nn_model1(X_trn, X_tst, Y_trn, Y_tst, 20, 2, epochs, "ReLU", lr2)


# =============================================
plt.figure(2, figsize=(12, 8))
# Classification errors for learning rate = 0.01, Relu Activation, n_hidden = 20
plt.plot(range(epochs), err_trn, '-', color='orange',linewidth=2, label='training error (lr = 0.01)')
plt.plot(range(epochs), err_tst, '-b', linewidth=2, label='test error (lr = 0.01)')

# Classification errors for learning rate = 0.1, Relu Activation, n_hidden = 20
plt.plot(range(epochs), err_trn2, '-', linewidth=2, label='training error (lr = 0.1)')
plt.plot(range(epochs), err_tst2, '-b', color='yellow', linewidth=2,  label='test error (lr = 0.1)')

plt.title('ReLU(Learning rate=0.1)')
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend(loc='best')
plt.grid()
plt.show()


# # Problem 3: Learning with variable learning rate.
# Using ReLU as your activation function, implement a learning algorithm with variable learning rate $\eta = \frac1{\sqrt{i+1}}$ at the $i$th step. Plot the training and testing error you get at each iteration and compare it with the plots you get previously. Justify your plot.

# In[46]:


indices = np.array(range(epochs))
lr3 = 1/np.sqrt(indices + 1)
# =========Write your code below ==============

err_tst3, err_trn3, loss_trn3, parameters3 = nn_model1(X_trn, X_tst, Y_trn, Y_tst, 20, 2, epochs, "ReLU", lr3)

# =============================================
plt.figure(3, figsize=(12, 8))
# Classification errors for learning rate = 0.01, Relu Activation, n_hidden = 20
plt.plot(range(epochs), err_trn, '-', color='orange',linewidth=2, label='training error (lr = 0.01)')
plt.plot(range(epochs), err_tst, '-b', linewidth=2, label='test error (lr = 0.01)')

# Classification errors for learning rate = 0.1, Relu Activation, n_hidden = 20
plt.plot(range(epochs), err_trn2, '-', color='red', linewidth=2, label='training error (lr = 0.1)')
plt.plot(range(epochs), err_tst2, '-b', color='yellow', linewidth=2, label='test error (lr = 0.1)')

# Classification errors for variable learning rate, Relu Activation, n_hidden = 20
plt.plot(range(epochs), err_trn3, '-', color='purple', linewidth=2, label='training error (unfixed lr)')
plt.plot(range(epochs), err_tst3, '-b', color='green', linewidth=2, label='test error (unfixed lr)')
plt.title('ReLU(Variable learning rate)')
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend(loc='best')
plt.grid()
plt.show()


# # Problem 4: Larger hidden layer.
# Change the number of neurons in the hidden layer to be $50$. Redo the experiment in problem 1. Plot all four plots in the same figure and justify your plot.

# In[47]:


num_hidden2 = 50
# =========Write your code below ==============

err_tst4, err_trn4, loss_trn4, parameters4 = nn_model1(X_trn, X_tst, Y_trn, Y_tst, num_hidden2, 2, epochs, "ReLU", lr1)

# =============================================
plt.figure(4, figsize=(12, 8))
# Classification errors for learning rate = 0.01, Relu Activation, n_hidden = 20
plt.plot(range(epochs), err_trn, '-', color='orange',linewidth=2, label='training error (#hidden = 20)')
plt.plot(range(epochs), err_tst, '-b', linewidth=2, label='test error (#hidden = 20)')

# Classification errors for learning rate = 0.01, Relu Activation, n_hidden = 50
plt.plot(range(epochs), err_trn4, '-', color='red', linewidth=2, label='training error (#hidden = 50)')
plt.plot(range(epochs), err_tst4, '-b', color='grey', linewidth=2, label='test error (#hidden = 50)')

plt.title('ReLU(Learning rate=0.01)')
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend(loc='best')
plt.grid()
plt.show()


# # Problem 5: Sigmoid Activation.
# Change the activation function to be Sigmoid function. Redo the experiment in problem 1. Plot all four plots in the same figure and justify your plot.

# In[48]:


# =========Write your code below ==============

err_tst5, err_trn5, loss_trn5, parameters5 = nn_model1(X_trn, X_tst, Y_trn, Y_tst, 20, 2, epochs, "Sigmoid", lr1)


# =============================================
# Classification errors for learning rate = 0.01, Relu Activation, n_hidden = 20
plt.figure(5, figsize=(12, 8))
plt.plot(range(epochs), err_trn, '-', color='orange',linewidth=2, label='training error (ReLU)')
plt.plot(range(epochs), err_tst, '-b', linewidth=2, label='test error (ReLU)')

# Classification errors for learning rate = 0.01, Sigmoid Activation, n_hidden = 20
plt.plot(range(epochs), err_trn5, '-', color='red',  linewidth=2, label='training error (Sigmoid)')
plt.plot(range(epochs), err_tst5, '-b', color='green', linewidth=2, label='test error (Sigmoid)')

plt.title('Learning rate=0.01')
plt.xlabel('epoch')
plt.ylabel('error')
plt.legend(loc='best')
plt.grid()
plt.show()


# In[ ]:




