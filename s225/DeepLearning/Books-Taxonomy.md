# Required Stack

## Packages
***The d2l package is lightweight and only requires the following***
***dependencies:***

```python
#@save
import collections
import hashlib
import inspect
import math
import os
import random
import re
import shutil
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
import pandas as pd
import requests
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline

d2l = sys.modules[__name__]
```

## PyTorch

# Math Notation

### Num Objs
• 𝑥: a scalar
• x: a vector
• X: a matrix
• X: a general tensor
• I: the identity matrix (of some given dimension), i.e., a square matrix with 1 on all
diagonal entries and 0 on all off-diagonals
• 𝑥𝑖 , [x] 𝑖 : the 𝑖 th element of vector x
• 𝑥𝑖 𝑗 , 𝑥𝑖, 𝑗 ,[X] 𝑖 𝑗 , [X] 𝑖, 𝑗 : the element of matrix X at row 𝑖 and column 𝑗.

### Sets
• X: a set
• Z: the set of integers
• Z+ : the set of positive integers
• R: the set of real numbers
• R𝑛 : the set of 𝑛-dimensional vectors of real numbers

• R𝑎×𝑏 : The set of matrices of real numbers with 𝑎 rows and 𝑏 columns
• |X|: cardinality (number of elements) of set X
• A ∪ B: union of sets A and B
• A ∩ B: intersection of sets A and B
*• A \ B: set subtraction of B from A (contains only those elements of A that do not*
*belong to B)*

### Functions
• 𝑓 (·): a function
• log(·): the natural logarithm (base 𝑒)
• log2 (·): logarithm to base 2
• exp(·): the exponential function
• 1(·): the indicator function; evaluates to 1 if the boolean argument is true, and 0 other-
wise
• 1 X (𝑧): the set-membership indicator function; evaluates to 1 if the element 𝑧 belongs to
the set X and 0 otherwise
• (·) > : transpose of a vector or a matrix
• X−1 : inverse of matrix X
•
: Hadamard (elementwise) product
• [·, ·]: concatenation
• k · k 𝑝 : ℓ 𝑝 norm
• k · k: ℓ2 norm
• hx, yi: inner (dot) product of vectors x and y
•
•
Í
: summation over a collection of elements
Î
def
: product over a collection of elements
• = : an equality asserted as a definition of the symbol on the left-hand side

### Calculus
*• ∇x 𝑦: gradient of 𝑦 with respect to x*

##  Wk6
#### Adversarial Attacks

A critical vulnerability of deep learning models which,$$x_{adv}=argmax_{x'\in B_c}L(f(x;\theta),y)$$
L - CE loss function
f(x; theta) - model to predict the prediction probabilities
y - truth labels

input that causes a model to make confident errors. 
Do this by *increasing* chance of predicting $y'\not =y$, where the Pr of x is towards $x \not = y$ , we *maximize L*, and *untargeted attack* because we are **moving away** from y. 

#### CNN Code Block
```python
import torch
import torch.nn as nn

cnn = nn.Sequential(
    nn.Conv2d(in_channels=3,  out_channels=16, kernel_size=3, stride=1, padding=1),
"""purpose:""" 
primes teh *filter* to go across images and turns input channels into feature maps
"""params""":
#kernel_size - small matrix of weights across input (like a NxN window)
#strides - how many pixels the filter jumps when moving across image (p=1 means move 1 pixel at a time) 
#padding - tells the filter what to do when reaches end of image    
    
    nn.ReLU(),
    # activation function for nonlinearity
    
    nn.MaxPool2d(kernel_size=2, stride=2),
    # halve the spatial resolution and changes the shape
    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.Flatten()
    # flattens all excpt batch into single feture vector
)

# Example batch: 64 images, 3 channels, 224x224
x = torch.randn(64, 3, 224, 224)
y = cnn(x)  # forward pass
print(y.shape)  # -> torch.Size([64, 401408])

```



#### Feed Forward NN

***HELP***
Given the following feed-forward neural network. Assume that we input to the network a mini-batch X with the batch size 32. What is the shape of the input X if we use activation ReLU?

![[Pasted image 20250926181659.png]]

***HELP***

Feed a mini batch $x\in R^{32\times3}$ to network. what is the shape of $h^1$?
![[Pasted image 20250926183037.png]]


#### Input Tensor Shape to Output Shape

**Over a Conv2D** of [3,32,32]
32 filters [3,7,7]
strides [2,2]
passing 1

**The output Shape**: 
$$\begin{matrix}Tshape=\frac{width + 2\times padding-filter_{width}}{strides}+1= \\
[filters, Tshape, Tshape]\end{matrix}$$


#### Calculating Batch Gradient Descent

input: obj function $J(\theta)$
*cost function to minimize*
Output: optimal $\theta^*$
init params randomly by $\tilde {}N(0, \sigma^2)$
*the guassian distro with mean 0 and variance sigma^2*
for t=1 to T
compute $\nabla_{\theta}J(\theta)=\frac{\partial J}{\partial \theta}(\theta)$
*compute gradient of ost function at each iteration*
update $\theta_{t+1}=\theta_t - \eta\nabla_{\theta}J(\theta)$
*update params in opposite direction of gradient*
($\eta$ is the learning rate)
return $\theta^*$
$\nabla_{\theta}J(\theta)$ - the direction with steepest increase in J
$\theta^*$ - updated in opposite dir


# Ch 1

Deep learning is differentiated from classical approaches
principally by the set of powerful models that it focuses on. These models consist of many
successive transformations of the data that are chained together top to bottom, thus the
name deep learning. On our way to discussing deep models, we will also discuss some
more traditional methods.


## Learning
In order to develop a formal mathematical system of learning machines, we need to have
formal measures of how good (or bad) our models are. In machine learning, and optimiza-
tion more generally, we call these objective functions.

## if we want to create learning machines, we need a means of evaluating them
common loss function is squared error,
i.e., the square of the difference between the prediction and the ground truth target.

## What about training?
We learn the best values of our model’s parameters by
minimizing the loss incurred on a set consisting of some number of examples collected for
training. However, doing well on the training data does not guarantee that we will do well
on unseen data. So we will typically want to split the available data into two partitions:
the training dataset (or training set), for learning model parameters; and the test dataset
(or test set), which is held out for evaluation

training is like test score of a student

## What about scoring too well?
Over time, the model might begin to memorize the
practice questions, appearing to master the topic but faltering when faced with previously
unseen questions on the actual final exam. When a model performs well on the training set
but fails to generalize to unseen data, we say that it is overfitting to the training data.


## How do we optimize?
Once we have got some data source and representation, a model, and a well-defined objec-
tive function, we need an algorithm capable of searching for the best possible parameters
for minimizing the loss function. Example is **gradient descent**

## Types of ML Problems that 

### Supervised
given a dataset containing both features
and labels and asked to produce a model that predicts the labels when given input features.
Sometimes, when the context is clear, we
may use the term examples to refer to a collection of inputs
In probabilistic terms, we typically are interested in estimating the conditional probability
of a label given input features

#### process of learning
the learning process looks something like the following. First, grab a big col-
lection of examples for which the features are known and select from them a random subset,
acquiring the ground truth labels for each. Together, these inputs and corresponding labels comprise the train-
ing set. We feed the training dataset into a supervised learning algorithm, a function that
takes as input a dataset and outputs another function: the learned model. Finally, we can
feed previously unseen inputs to the learned model, using its outputs as predictions of the
corresponding label

![[Pasted image 20250802162342.png]]

#### regression
Answers to the learning problems of *how many* .
The goal is to produce a
model whose predictions closely approximate the actual label values.

#### classifying
Answers to the learning problems of *how much*. 
In classification, we want our model to look at features, 
and then predict to which category (sometimes called a class) among some discrete set
of options, an example belongs

##### Types of classifications
1. **Binary** 
2. **Multiclass**
3. **Hierarchical** (like Linnean hierarchies)
4. **Multi-label/Tagging** - predict classes that are not mutually exclusive

#### Searching (ranking)
The goal is less to determine whether a particular page is relevant for a
query, but rather which, among a set of relevant results, should be shown most prominently
to a particular user.

#### Recommender Systems
Same as search, but emphasis on personalization to specific users. Given such a model, for any given user, we could retrieve the set of objects with the largest
scores, which could then be recommended to the user

### Sequence Learning
Specifically, sequence-to-sequence
learning considers problems where both inputs and outputs consist of variable-length se-
quences. Examples include machine translation and speech-to-text transcription.

### Unsupervised (self supervised) Learning
can train models to “fill in the blanks” by predicting some aspect of the unlabeled data to provide supervision.

### Offline Learning
the learning takes place
after the algorithm is disconnected from the environment, 
The upside is that we can worry about
pattern recognition in isolation, with no concern about complications arising from interac-
tions with a dynamic environment
BUT!
Limitations with remembering, leveraging enviro, competing with enviro, etc.?
These questions raise the problem of distribution shift, where training and test data are
different.

### Reinforcement Learning
machine learning to develop an agent that interacts with an
environment and takes actions,

## Perceptrons and core properties
The alternation of linear and nonlinear processing units, often referred to as layers.
• The use of the chain rule (also known as backpropagation) for adjusting parameters in
the entire network at once.

Deep learning is the subset
of machine learning concerned with models based on many-layered neural networks. It is
deep in precisely the sense that its models learn many layers of transformations.

