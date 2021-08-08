#!/usr/bin/env python
# coding: utf-8

# # Deep Neural Network and Regularization
# 
# **Problem Statement**: You have just been hired as an AI expert by the French Football Corporation. They would like you to recommend positions where France's goal keeper should kick the ball so that the French team's players can then hit it with their head. 
# 
# <img src="images/field_kiank.png" style="width:600px;height:350px;">
# <caption><center> <u> **Figure 1** </u>: **Football field**<br> The goal keeper kicks the ball in the air, the players of each team are fighting to hit the ball with their head </center></caption>
# 
# 
# They give you the following 2D dataset from France's past 10 games.

# In[1]:


# import packages to use
import numpy as np
import scipy
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as f

# for reproduciblity
torch.manual_seed(0)
np.random.seed(0)
import random
random.seed(0)


# In[2]:


# Some helper functions
def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:,0].min() - .1, X[:,0].max() + .1
    y_min, y_max = X[:,1].min() - .1, X[:,1].max() + .1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    d = np.c_[xx.ravel(), yy.ravel()]
    # Predict the function value for the whole grid
    Z = model.predict(torch.from_numpy(d.astype('float32')))
    Z = torch.argmax(Z, dim=1)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z.detach().numpy(), cmap=plt.cm.Spectral, alpha=.5)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.title('Decision Boundary')
    plt.show()


# In[3]:


# Loading the training datd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn import model_selection

X, Y = make_classification(n_samples = 300, n_features=2, 
                           n_redundant=0, n_informative=2,
                           random_state = np.random.seed(8),
                           flip_y = .001)
Y = 1-Y
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.50, random_state=2)

model_selection.train_test_split(X)

train_X =  torch.from_numpy(train_X.astype('float32'))
train_Y =  torch.from_numpy(train_Y.astype('int64'))
test_X =  torch.from_numpy(test_X.astype('float32'))
test_Y =  torch.from_numpy(test_Y.astype('int64'))


plt.scatter(train_X[:,0], train_X[:,1], c=train_Y, s=40, cmap=plt.cm.Spectral);
plt.show()


# Each dot corresponds to a position on the football field where a football player has hit the ball with his/her head after the French goal keeper has shot the ball from the left side of the football field.
# - If the dot is blue, it means the French player managed to hit the ball with his/her head
# - If the dot is red, it means the other team's player hit the ball with their head
# 
# **Your goal**: Use a classification algorithm to find the positions on the field where the goalkeeper should kick the ball.

# ## Logistic Regression Model
# You will first try a logistic regression model, which has been implemented below. Go through the model, which is similar to the last lab logistic regression model. However, here Object oriented programming (using classes) principles have been used. From now onwards this is how you will implement a model/hypothesis. Secondly, we have used nn.CrossEntropyLoss(). This criterion combines LogSoftmax and NLLLoss in one single class. https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html

# In[5]:


# simple logistic regression model
class model01(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        self.linear = nn.Linear(in_dim, out_dim)
        
    def predict(self, x):
        return f.softmax(self.linear(x), dim=1)
        
    def forward(self, x):
        return self.linear(x)


# In[6]:


# Training the model
model = model01(2,2)
cost = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = .1, weight_decay=0)

J_history = []
for iterations in range(10000):
    optimizer.zero_grad()
    # forward pass
    out = model(train_X)
    
    loss = cost(out, train_Y)
    
    # backward pass
    loss.backward()
    
    # update paramters
    optimizer.step()
    
    J_history += [loss.item()]


# In[7]:


# plot the results
from matplotlib import pyplot as plt
plt.plot(J_history)
plt.title('Convergence plot of gradient descent')
plt.xlabel('No of iterations')
plt.ylabel('J')
plt.show()

plot_decision_boundary(model, train_X, train_Y)

# Print train and test error and accuracy
out = model(train_X)
loss_train = cost(out, train_Y)
out = model(test_X)
loss_test = cost(out, test_Y)
print(f'Train Error: {loss_train.item()}, Test Error: {loss_test.item()}')

out = torch.argmax(model.predict(train_X), dim=1)
loss_train = 100.*torch.sum(out == train_Y)/out.shape[0]
out = torch.argmax(model.predict(test_X), dim=1)
loss_test = 100.*torch.sum(out == test_Y)/out.shape[0]
print(f'Train Accuracy: {loss_train}, Test Accuracy: {loss_test}')


# The model you implement above (also called Perceptron) can be represented pictorially as:
# 
# <img src="images/perceptron.png" style="width:600px;height:350px;">
# <caption><center> <u> **Figure 2** </u>: **Perceptron**<br> </center></caption>
# 
# This is a linear model and therefore it has a linear decision boundary. This is the best you can get from a linear decision boundary. In order to have a non-linear decision boundary, we need to have Deep Learning model. 
# 
# Deep learning models have so much flexibility and capacity that overfitting can be a serious problem, if the training dataset is not big enough. Sure it does well on the training set, but the learned network doesn't generalize to new examples that it has never seen!
# 
# <img src="images/dl.png" style="width:600px;height:350px;">
# <caption><center> <u> **Figure 3** </u>: **Deep Neural Network**<br> </center></caption>
# 

# ## Lab Task 1 - Implement Deep Learning Model
# Implement a neural network with two hidden layers. In the first hidden layer, you will have 20 neuron and in the second 3 neurons. Use ReLU activation unit.
# 
# Use following to implement the deep network
#  - nn.Sequential
#  - nn.Linear
#  - nn.ReLU

# In[53]:


class model02(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        
        # todo. Chamnge the model to include more layers 
        self.feat = nn.Sequential(nn.Linear(in_dim, 2), # Input layer
                                  nn.ReLU(),
                                  nn.Linear(2,20),    # First hidden layer
                                  nn.ReLU(),
                                  nn.Linear(20, 3),   # Second Hidden Layer
                                  nn.ReLU(),
                                  nn.Linear(3,2)      # Output Layer
                                 )
        
        
    def predict(self, x):
        return f.softmax(self.feat(x), dim=1)
        
    def forward(self, x):
        return self.feat(x)


# In[57]:


model = model02(2,2)
cost = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = .1, weight_decay=0)

J_history = []
for iterations in range(10000):
    optimizer.zero_grad()
    # forward pass
    out = model(train_X)
    
    loss = cost(out, train_Y)
    
    # backward pass
    loss.backward()
    
    # update paramters
    optimizer.step()
    
    J_history += [loss.item()]


# In[58]:


# plot the results
from matplotlib import pyplot as plt
plt.plot(J_history)
plt.title('Convergence plot of gradient descent')
plt.xlabel('No of iterations')
plt.ylabel('J')
plt.show()

plot_decision_boundary(model, train_X, train_Y)

#print accuracy
out = torch.argmax(model.predict(train_X), dim=1)
loss_train = 100*torch.sum(out == train_Y)/out.shape[0]
out = torch.argmax(model.predict(test_X), dim=1)
loss_test = 100*torch.sum(out == test_Y)/out.shape[0]
print(f'Train Accuracy: {loss_train}, Test Accuracy: {loss_test}')


# The model training accuracy has improved, however the test accuracy has not improved relatively. This is obviously overfitting the training set. It is fitting the noisy points! Lets now look at two techniques to reduce overfitting. 

# ## Lab Task 2 - L2 Regularization
# 
# The standard way to avoid overfitting is called **L2 regularization**. L2 regularization makes your decision boundary smoother. To use L2 Regularization, use **weight_decay** parameter of the optimizer above (optimizer = optim.SGD(model.parameters(), lr = .1, weight_decay=0)). <p style="color:red;">Report train and test accuracy by using different values of weight_decay and find the optimal value.</p> If weight_decay is too large, it is also possible to "oversmooth", resulting in a model with high bias.
# 
# L2-regularization relies on the assumption that a model with small weights is simpler than a model with large weights. Thus, by penalizing the square values of the weights in the cost function you drive all the weights to smaller values. It becomes too costly for the cost to have large weights! This leads to a smoother model in which the output changes more slowly as the input changes. [Please revisit the lab after we have covered Regularization in the class]
# 

# ### Doing some modifications for ease of use

# In[60]:


model = model02(2,2)
cost = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = .1, weight_decay=0)


# In[62]:


# Converting the training to a function for ease of use
def train_model(num_iter, model, optimizer, cost):
    J_history = []
    for iterations in range(num_iter):
        optimizer.zero_grad()
        # forward pass
        out = model(train_X)

        loss = cost(out, train_Y)

        # backward pass
        loss.backward()

        # update paramters
        optimizer.step()

        J_history += [loss.item()]
    return model, J_history


# ### Testing the function with `weight_decay = 0`

# In[64]:


model1, J_history1 = train_model(10000, model, optimizer, cost)


# In[65]:


out = torch.argmax(model1.predict(train_X), dim=1)
loss_train = 100*torch.sum(out == train_Y)/out.shape[0]
out = torch.argmax(model1.predict(test_X), dim=1)
loss_test = 100*torch.sum(out == test_Y)/out.shape[0]
print(f'Train Accuracy: {loss_train}, Test Accuracy: {loss_test}')


# ### Testing on varying values of `weight_decay` to find optimal value

# In[66]:


weight_decay_values=[0.1,0.01,0.001,0.0001,0.00001]
model = model02(2,2)
cost = nn.CrossEntropyLoss()
for weight_decay in weight_decay_values:
    optimizer = optim.SGD(model.parameters(), lr = .1, weight_decay=weight_decay)
    temp_model, temp_J_history = train_model(10000, model, optimizer, cost)
    out = torch.argmax(temp_model.predict(train_X), dim=1)
    loss_train = 100*torch.sum(out == train_Y)/out.shape[0]
    out = torch.argmax(temp_model.predict(test_X), dim=1)
    loss_test = 100*torch.sum(out == test_Y)/out.shape[0]
    print(f'Weight Decay Value: {weight_decay} Train Accuracy: {loss_train}, Test Accuracy: {loss_test}')


# From the output it is clear that `weight_decay = 0.01` performed the best in minimizing overfitting.
# 
# Using this to train the model and then plotting the results.

# In[77]:


# Would have to redefine Optimizer as the last value was not optimal
model = model02(2,2)
cost = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = .1, weight_decay=0.01)


# In[78]:


model, J_history = train_model(10000, model, optimizer, cost)


# In[79]:


# plot the results
from matplotlib import pyplot as plt
plt.plot(J_history)
plt.title('Convergence plot of gradient descent')
plt.xlabel('No of iterations')
plt.ylabel('J')
plt.show()

plot_decision_boundary(model, train_X, train_Y)

#print accuracy
out = torch.argmax(model.predict(train_X), dim=1)
loss_train = 100*torch.sum(out == train_Y)/out.shape[0]
out = torch.argmax(model.predict(test_X), dim=1)
loss_test = 100*torch.sum(out == test_Y)/out.shape[0]
print(f'Train Accuracy: {loss_train}, Test Accuracy: {loss_test}')


# ## Lab Task 3 - Dropout
# 
# Finally, **dropout** is a widely used regularization technique that is specific to deep learning. 
# **It randomly shuts down some neurons in each iteration.** Watch these two videos to see what this means!
# 
# <center>
# <video width="620" height="440" src="images/dropout1_kiank.mp4" type="video/mp4" controls>
# </video>
# </center>
# <br>
# <caption><center> <u> Figure 4 </u>: Drop-out on the second hidden layer. <br> At each iteration, you shut down (= set to zero) each neuron of a layer with probability $1 - keep\_prob$ or keep it with probability $keep\_prob$ (50% here). The dropped neurons don't contribute to the training in both the forward and backward propagations of the iteration. </center></caption>
# 
# 
# When you shut some neurons down, you actually modify your model. The idea behind drop-out is that at each iteration, you train a different model that uses only a subset of your neurons. With dropout, your neurons thus become less sensitive to the activation of one other specific neuron, because that other neuron might be shut down at any time. 
# 
# <p style="color:red;"> Use nn.DropOut in your model to use the dropout regularization and report train and test errors. What is the best Dropout rate for the problem above? <p>

# ### Defining the new model with `dropout`

# In[94]:


class model03(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_val):
        super().__init__()
        
        # todo. Chamnge the model to include more layers 
        self.feat = nn.Sequential(nn.Linear(in_dim, 2), # Input layer
                                  nn.ReLU(),
                                  nn.Linear(2,20),    # First hidden layer
                                  nn.ReLU(),
                                  nn.Linear(20, 3),   # Second Hidden Layer
                                  nn.ReLU(),
                                  nn.Linear(3,2),      # Output Layer
                                  nn.Dropout(dropout_val)
                                 )
        
        
    def predict(self, x):
        return f.softmax(self.feat(x), dim=1)
        
    def forward(self, x):
        return self.feat(x)


# ### Trying for `Dropout` value of `0.25`

# In[114]:


model = model03(2,2,dropout_val=0.25)
cost = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = .1, weight_decay=0.01)


# Using the `train_model()` function defined earlier

# In[96]:


model, J_history = train_model(10000, model, optimizer, cost)


# In[97]:


# plot the results
from matplotlib import pyplot as plt
plt.plot(J_history)
plt.title('Convergence plot of gradient descent')
plt.xlabel('No of iterations')
plt.ylabel('J')
plt.show()

plot_decision_boundary(model, train_X, train_Y)

#print accuracy
out = torch.argmax(model.predict(train_X), dim=1)
loss_train = 100*torch.sum(out == train_Y)/out.shape[0]
out = torch.argmax(model.predict(test_X), dim=1)
loss_test = 100*torch.sum(out == test_Y)/out.shape[0]
print(f'Train Accuracy: {loss_train}, Test Accuracy: {loss_test}')


# ### Trying for varying `Dropout` values to find the optimal value

# In[118]:


dropout_values=[0.1,0.2,0.3,0.4,0.5]
for dropout_val in dropout_values:
    temp_model = model03(2,2,dropout_val)
    cost = nn.CrossEntropyLoss()
    optimizer = optim.SGD(temp_model.parameters(), lr = .1, weight_decay=0.01)
    temp_model, temp_J_history = train_model(10000, temp_model, optimizer, cost)
    out = torch.argmax(temp_model.predict(train_X), dim=1)
    loss_train = 100*torch.sum(out == train_Y)/out.shape[0]
    out = torch.argmax(temp_model.predict(test_X), dim=1)
    loss_test = 100*torch.sum(out == test_Y)/out.shape[0]
    print(f'Weight Decay Value: {weight_decay} Train Accuracy: {loss_train}, Test Accuracy: {loss_test}')


# The optimal value is obtained with `Dropout` value of 0.2

# ### Doing a second run to further optimize the value
# 
# - Changing test dropout values so that they are close to the previously obtained dropout value

# In[119]:


dropout_values=[0.20,0.21, 0.22,0.23,0.24,0.25]
for dropout_val in dropout_values:
    temp_model = model03(2,2,dropout_val)
    cost = nn.CrossEntropyLoss()
    optimizer = optim.SGD(temp_model.parameters(), lr = .1, weight_decay=0.01)
    temp_model, temp_J_history = train_model(10000, temp_model, optimizer, cost)
    out = torch.argmax(temp_model.predict(train_X), dim=1)
    loss_train = 100*torch.sum(out == train_Y)/out.shape[0]
    out = torch.argmax(temp_model.predict(test_X), dim=1)
    loss_test = 100*torch.sum(out == test_Y)/out.shape[0]
    print(f'Weight Decay Value: {weight_decay} Train Accuracy: {loss_train}, Test Accuracy: {loss_test}')


# It can be concluded from the results of the above output that `0.21` gives the optimal value.
