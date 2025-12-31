# Part 1:  Document Clustering

This assignment assesses your understanding of
1. Document Clustering,
2. Perceptrons,
3. Neural Networks,
4. Unsupervised Learning
covered in Modules 4 and 5 in written (Jupyter Notebook) form.
In your Jupyter notebook you can use any import from the libraries numpy, scipy, matplotlib, pandas, and scikit-learn to solve the tasks except those imports that would render a task trivial (exception is Question 3 where importing pytorch is permitted to base answer on lab code). The maximum number of marks of this assignment is 100 (sum of grades from each section). This as-
signment constitutes 16% of your final mark for this unit. Make sure you read and understand not only all the tasks but also the notes at the end of this document about submission, assessment criteria, and penalties.
## Background
In this part, you solve a document clustering problem using unsupervised learning algorithms (Mixture Models with Expectation Maximization) for document clustering. We follow the notations in Module 4. Let $D = {d_1 , d_2 , ..., d_N }$ be a set of $N$ documents and partition them into $K$ clusters. Please use the parameters $φ_k$ and $µ_{wk}$ to represent the prior of the k-th cluster and the occurence probability of the w-th word in the k-th cluster, respectively. All documents share the same vocabulary A.

### Section 1
1. Please briefly answer the following high-level questions:
– Write mathematical formulations of the optimization functions of maximum likelihood estimation (MLE) for the document clustering model with complete data and incomplete data, respectively. Then briefly describe why MLE with incomplete data is hard to optimize.
– Briefly explain the high-level idea of the EM algorithm to find MLE parameter estimates.
2. Derive Expectation and Maximization steps of the (soft)-EM algorithm for Document Clustering, in a markdown cell (ideally using Latex for clean typesetting and show your work in your submitted PDF report.) In particular, include all model parameters that should be learnt and the exact expression (using the same math convention that we saw in the Module 3) that should be used to update these parameters during the learning process (ie., E-step, M-step and assignments).

3. Load Task2A.txt file (if needed, perform text preprocessing similar to what we did in Activity 4.2).

4. Implement the EM algorithm (derived in Chapter 5 of Module 4). Please provide enough comments in your submitted code.

**Hint**: 
a) If it helps, feel free to base your code on the provided code for EM algorithm for GMM in Activity 4.1. However, please remember that the EM algorithm is not limited to solving GMMs; based on the lecture, make sure to assess and choose the appropriate model that fits your specific needs. 
b) When implementing the M-step, you will need to normalize the posterior probability matrix in a row-wise manner. Formally, if we need to normalize an N dimensional vector of log probabilities $x_i = log p_i$ using Softmax, we might naively compute the following equation,
$$p_i=\frac{exp(x_i)}{\sum_{n=1}^{N}exp(x_n)}, \sum_{n=1}^{N}p_n=1$$
Since each $x_i$ is a log probability which may be very large, and either negative or positive, then exponentiating might result in under- or overflow respectively. For example, $exp(10000) = inf$. We can simply use the log-sum-exp trick to avoid this numerical unstability. We consider the log-sum-exp operation:
$$\begin{matrix} LSE(x_1,...,x_N)=log(\sum_{n=1}^{N}exp(x_n)) \\
=c+log(\sum_{n=1}^{N}exp(x_n-c))\end{matrix}$$
where we usually set $c = max{x_1 , ..., x_N }$ to ensure that the largest positive exponentiated term is $exp(0) = 1,$ so you will definitely not overflow, and even if you underflow, the answer will be sensible. Then the final equation will be
$$p_i=exp(x_i-c-log(\sum_{n=1}^{N}exp(x_n-c)))$$
**You should use this log-sum-exp trick during implementation.**

5. Set the number of clusters K=4, and run the hard clustering (using hard-EM) and soft clustering (using soft-EM) on the provided data.
6. Perform a PCA on the clusterings that you get based on the Mixture Models in the same way we did in Activity 4.2. Then, plot the obtained clusters with different colors where x and y axes are the first two principal components (similar to Activity 4.2). Based on your plots, discuss how and why the hard and soft clustering are different in a markdown cell.

# Part 2: Perceptron vs Neural Networks
## Background
In this part, you will be working on a binary classification task on a given synthetic dataset. You will use machine learning tools including a Perceptron and a 3-layer Neural Network to solve the task. Here, we are looking for your meaningful observations and discussions towards the differences between Perceptron and Neural Networks

### Section 1
1. Load Task2B train.csv and Task2B test.csv datasets, plot the training and testing data separately in two plots. Mark the data with different labels in different colors.
2. Train two Perceptron models on the provided training data: one with early stopping andone without. For the model with early stopping, use a validation set comprising 20% of the training samples and you can set the threshold to 0.001. For both models, evaluate combinations of learning rates $η ∈ {0.001, 0.01, 0.1}$ and L2 regularisation strengths $λ ∈ {0.0001, 0.001, 0.01, 0.1, 1.0}.$ Calculate the test errors for both models across all configurations, and identify the best-performing combination of η and λ for each model. Finally, plot the decision boundaries of both models with best η and λ along with the test data in a single figure.
**Hint:**  We expect the decision boundary of your perceptron to be a linear function that
separates the testing data into two parts. You may also choose to change the labels from [0,1] to [-1, +1] for your convenience.

4. Train two 3-layer neural network models with different L2 regularisation strengths: one with $λ = 0.001$ and the other with $λ = 1.0$. For both models, evaluate combinations of hidden layer sizes $K ∈ {5, 10, 15, ..., 40},$ (i.e. from 5 to 40 with a step size of 5) and learning rates η ∈ {0.001, 0.01, 0.1}. Calculate and record testing error for each of them. Find the best combination of K and η for both models. Then, plot the decision boundaries of each model alongside the test data in two separate plots. Based on the test errors and the decision boundary plots, explain which model performs better
5. Explain the reason(s) responsible for such difference between Perceptron and a 3-layer Neural Network by comparing the plots you generated in Steps II and III.
**Hint:**  Look at the plotsand think about the model assumptions.

# Part 3: Unsupervised Learning
## Background
In this part, you will implement self-taught learning using an Autoencoder and a
3-layer Neural Network to solve a multi-class classification task on real-world data.

### Section 1
1. I Load Task2C labeled.csv, Task2C unlabeled.csv, and Task2C test.csv datasets, along with the required libraries. Note that we will use both Task2C labeled.csv and Task2C unlabeled.csv to train the autoencoder, and only Task2C labeled.csv to train the classifiers. Finally, we will evaluate the trained classifier on the test dataset Task2C test.csv.
2. Train an autoencoder with only one hidden layer and change the number of its neurons to 20, 60, 100, ..., 220 (i.e. from 20 to 220 with a step size of 40).
3. For each model in Step II, calculate and record the reconstruction error for the autoencoder, which is simply the average of Euclidean distances between the input and output of the autoencoder. Plot these values where the x-axis is the number of units in the middle layer and the y-axis is the reconstruction error. Then, explain your findings based on the plot.
4. Build the 3-layer NN to build a classification model using all the original attributes from the training set and change the number of its neurons to 20, 60, 100, ..., 220 (i.e. from 20 to 220 with a step size of 40). For each model, calculate and record the test error.
5. Build augmented self-taught networks using the models learnt in Step II. For each model: 
	1. Add the output of the middle layer of an autoencoder as extra features to the original feature set; 
	2. Train a new 3-layer Neural Network using all features (original + extra) and varying the number of hidden neurons (like Step IV) as well. 
	3. Then calculate and record the test error.
For example, each model should be developed as follows: Model 1: 20 hidden neurons + extra 20 features (from an autoencoder), Model 2: 60 hidden neurons + extra 60 features (from an autoencoder), ..., Model 5: 220 hidden neurons + extra 220 features (from an autoencoder).

6. Plot the error rates for the 3-layer neural networks from Step IV and the augmented selftaught networks from Step V, while the x-axis is the number of hidden neurons and y-axis is the classification error. Explain how the performance of the 3-layer neural networks and the augmented self-taught networks is different and why they are different or why they are not different, based on the plot.