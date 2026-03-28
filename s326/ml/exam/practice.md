## stion 2: Goal of Supervised Learning [3 marks]

### **What You Need to Know**

**Key Concepts**:

1. **Ultimate Goal**: Minimize **generalization error** (test error on unseen data)
2. **Training Error**: $$E_{train} = \frac{1}{N} \sum_{i=1}^{N} L(y_i, f(x_i)) $$Error on the training dataset
3. **Test Error** (Generalization Error): $$E_{test} = \mathbb{E}_{(x,y) \sim P_{data}}[L(y, f(x))] $$Expected error on new, unseen data from the true distribution
4. **The Tension**:
    - Easy to minimize training error (just memorize!)
    - Hard to minimize test error (need to generalize)
    - **Overfitting**: Low training error, high test error
    - **Underfitting**: High training error, high test error

$$\frac{\partial \sigma(w^T x)}{\partial w_i} = \frac{\partial \sigma(a)}{\partial a} \cdot \frac{\partial a}{\partial w_i}$$ where $a = w^T x$

$$\log p(X, Z | \theta) = \sum_{n=1}^{N} \sum_{k=1}^{K} z_{nk} \left[\log \pi_k + \log p(x_
n | \mu_k)\right]$$

$$\frac{\partial E}{\partial w_1} = 37.4 \times 6 \times 0.5 = 112.2$$

---


How to use this collection to practice: 
It is fundamentally important that you attempt every question alone without help before you refer to the teaching material, discuss with your peers, or even look at the separately published reference solutions. If you are like most students (and teachers) this approach will be scary because it requires you to confront your gaps in knowledge. However, this is exactly why it will lead you to a good exam performance (very much in contrast to just scanning the solutions). It gives yourself the opportunity to really understand what you do not understand before looking for help. This way, once you receive the information to fill t he gap, you will connect it better with the other things you know, causing it to be available to you under exam conditions. 
Elements of Statistical Learning 
# 1 Classification versus Regression [ 3 marks] 
## Question 
*What is the statistical supervised learning problem on a high level and what is difference between a classification and a regression problem? Answer in two to three sentences.* 

## Answer:
In statistical supervised learning we are predicting the value y(x) of a target (random) variable T
given the value of an input variable X = x such that y(X) is on average a good approximation to
T where this average is taken with respect to the joint distribution of X and T that governs both,
the process of sampling training data as well as the occurrence of data later when the prediction
model is deployed. In regression the response variable T is assumed to be continuous, e.g., the predicted future price of a financial a sset o r t he p redicted t emperature i n a s pecific re gion. In co ntrast, in classification the response variable has a discrete domain of a finite number of categories, e.g., ‘spam’/‘no-spam’ for e-mail classification o r ‘ tuna’, ‘ salmon’, ‘ bass’ for the fish classification example.
# 2 Goal of Supervised Learning [3 marks] 

## Question 1
What is the ultimate goal of statistical supervised learning? How are the concepts of training error and test error related to this goal? Answer in three to four sentences. 


## Answer
The ultimate goal of predictive (supervised) learning is to find a prediction function y(x) with small
generalization error, i.e., expected loss between target value T and prediction y(X) based on input
value X:
$$E(l(T,y(X)))$$
One aims to find such a prediction function by minimizing the training error,
$$\frac{1}{N}\sum^N_{n=1}l(t_n,y(x_n))$$
on a set of training data $(x_1,t_1)...(x_n,t_n)$ that has been sampled independently and identically distributed according to the joint distribution of X and T. The test error,
$$\frac{1}{M}\sum^M_{=1}l(t_{N+m},y(x_{N+m}))$$
on further i.i.d. test data $(x_{N+1},t_{N+1})...(x_{N+M},t_{N+M})$ is then used as an unbiased estimate on the predicted function that has been found

# 3 Model Selection [6 marks] 
## Question
Consider the following scenario: • Alice wants to model the species (‘setosa’, ‘versicolor’, or ‘virginica’) of iris flowers as a function of four variables (their sepal and petal length and width) • She has collected a dataset of 150 examples • She wants to use the kNN classifier but she does not know what is a suitable value for k, hence shes want to choose k from the set of candidates {1, 2, …, 10} based on the collected data • Finally, she wants a reliable estimate of the performance of the model that has been learned. 
Currently Alice plans to proceed with the following machine learning workflow: 1. Split the data into 10 folds of roughly equal size. 2. Pick the value of k with the second best average test error across all folds (for each using the remaining folds as training data). In particular, she plans to use the second best “to avoid overfitting”. 3. Use this test error as the final performance estimate. 
Answer the following two questions: (a) Point out in up to two sentences, what is the most substantial problem with Alice’s proposed workflow and why. (b) Describe an improved machine learning process that adequately addresses this problem. 

## Answer

a) The average test error of the chosen model (corresponding to the second best k) obtained by
Alice’s cross validation procedure is optimistically biased, because the test data is used as part of the overall training data to learn the value for k.

b) The procedure can be corrected by first performing a training/test split, carrying out the cross validation procedure to choose k only based on the training data, and obtaining a final estimate of the generalization error on the test data.

# 4 Normal Distribution and Maximum Likelihood Estimation [10 marks] 

## Question
The (uni-variate) normal distribution is an extremely important distribution describing the behaviour of continuous random variables. It is parameterised by a mean μ and a standard deviation parameters σ (or more typically by the corresponding variance parameter σ2 ). Given a dataset {x1 , … , xN } of independent realizations of a normal random variable X, we can use the principal of maximum likelihood to find guesses for the unknown parameters. In particular, these guesses have simple closed form solutions. 

Answer each of the following questions with one to two sentences and give mathematical derivations as appropriate. 
(a) What is the definition of the normal density function p(x|μ, σ)? What is the key component of the definition that gives rise to the characteristic bell shape? 
(b) What is the key idea of the maximum likelihood estimation of the parameters μ and σ, i.e., what is the defining property of the maximum likelihood estimates μ{ML} and σ{ML} . 
(c) How can we derive the closed form solution of the maximum likelihood estimation for the mean μ? Apply this approach to derive it. 
(d) How can we derive the closed form solution of the maximum likelihood estimation for the standard deviation σ? Apply this approach to derive it. 

## Answer

(a) The density function of N (µ, σ 2 ), the normal distribution with mean µ and variance σ 2 is,
$$p(x|\mu,\sigma)=-\frac{1}{\sqrt{2\pi\sigma}}exp(-\frac{(x-\mu)^2}{2\sigma^2})$$

The central component is the exponential reduction of the density in the normalized square of the difference of x from the mean µ. This creates the bell shaped curve, because of a slow reduction close to the mean (where the normalized difference is less than 1) and a rapid reduction further away from the mean

(b) Given a dataset D = (x1 , . . . , xN ) of observations drawn independently and identically distributed according to N (µ, σ 2 ), the idea of maximum likelihood estimation is to find estimates
µML and σML that maximize the probability of the data, i.e.,
$$p(D|\mu_{ML},\sigma_{ML})=max\{p(D|\mu,\sigma):\mu \in \mathbb{R}, \sigma \in \mathbb{R_+}\}$$

(c) All partial derivatives of the likelihood function have to be 0 at the maximum likelihood estimates. Additionally, we can optimize the log likelihood instead of the likelihood (because this is a monotone transformation). With this and the assumption that the sample elements are drawn independently, a method for identifying the maximum likelihood parameters is to check where the partial derivatives of
$$lnp(D|\mu,\sigma)=ln\prod^N_{n=1}p(x_n|\mu,\sigma)$$
$$=\sum^N_{n=1}ln(p(x_n|\mu,\sigma))$$
is zero. Applying this idea to $\mu$, we find the partial derivatives $\mu$ as,
$$\frac{\partial}{\partial \mu}\sum^N_{n=1}ln(p(x_n|\mu,\sigma))=\sum^N_{n=1}\frac{\partial}{\partial \mu}ln(p(x_n|\mu,\sigma))$$
$$=\sum^N_{n=1}\frac{\partial}{\partial \mu}(-ln\sqrt{2\pi\sigma}-\frac{(x_n-\mu)^2}{2\sigma^2})$$
$$=-\sum^N_{n=1}\frac{x_n-\mu}{\sigma^2}$$
Setting the last expression to 0 and solving for µ, we find that,
$$\mu_{ML}=\frac{1}{N}\sum^N_{n=1}x_n$$

# Linear Regression 
5 Derivation of Squared Error [6 marks] 
For fitting the model parameters w of a linear regression model, we used the approach to minimise the squared error: E(w) = \frac{1}{2}\sum^{N}_{n=1}(t_n-y_n)2 
where {(x1, t1 ), … , (xN , tn)} is the given training data and yn = \sum^{p}_{i=1}wiφi (xn) are the model predictions. To justify this error function, we showed that it can be derived as a maximum likelihood parameter estimation for a probabilistic model p(t|x, w). Answer each of the following questions with one to two sentences (including mathematical equations as appropriate). (a) What is the form of the probabilistic model that we assumed for the regression problem, i.e., how are the target values generated given the input vectors? (b) What is the likelihood function corresponding to this model? © Why is maximising this likelihood function equivalent to minimising the squared error function? 
Linear Classification 
6 Logistic Regression [8 marks] 
When using the logistic regression model for binary classification, we model the probability of the positive class (t = 1) given input x via the sigmoid transformation σ of a linear function w · x of model parameters w. (a) Give the log likelihood function log p(t|x, w) of the logistic regression model for a single data point (x, t). Hint: We used the fact that we encode the positive class with t = 1 and the negative class with t = 0 to give a compact formula. (b) As a step towards the gradient descent algorithm for logistic regression, derive the partial derivative of the negative log likelihood (error function) − log p(t|x, w) with respect to parameter w_i . Derive the result in individual steps, noting what results you are using (all correct steps give partial marks). © Extend your result from part (b) to the full gradient of the negative log likelihood when observing a set of n training data points {(x1, t1 ), … , (xN , tn)}. 
Latent Variable Models 
7 Document Clustering Model [9 marks] 
Suppose we are given a collection of documents D. The data set D is represented as {x1 , x2, x3 , …, xN } where x_i is a d-dimensional “count vector” representing the i-th document, based on bag-of-words and with respect to a word vocabulary of size d. We are interested in fitting a Mixture multinomial model onto this dataset. (a) An individual cluster is described by a vector of word occurrence probabilities μ where μ_j describes the probability of a word in a document to be the j-th word in the vocabulary. Give a formula of the probability p(x|μ) of a count vector x given word occurrence probabilities μ and give a brief explanation of the formula (one to two sentences). Hint: remember that, for simplicity, we assumed the individual counts to be independent. (b) Write down the “Q-function”, which is the basis of the Expectation-Maximization (EM) algorithm for maximizing the log-likelihood. Notice that you do not need to write the EM algorithm in this part. © Write down the “hard” as well as the ”soft” Expectation-Maximization (EM) algorithm for estimating the parameters of the model. If necessary, provide enough explanation to under- stand the algorithm that you have written. Also briefly explain what is the main difference between hard and soft EM. 
Neural Networks 
8 Forward and Backward Propagation (9 marks) 
Given a neural network f(·) and a dataset D = {(x1, y1), (x2 , y2), …, (xn , y n)} where xi is a 2-dimensional vector and yi is a scalar value which represents the target. {w1, w2 , …, wn } are learnable parameters. h represents a linear unit. For example ti = h_1w_7 +h_2w_8. The error function for training this neural network is the sum of squared error: E(w)=\frac{1}{2}\sum^{N}_{n=1}(yi-ti)2, 
  x₁        x₂        x₃
  │         │         │
 
w₁─┼─w₂ w₃─┼─w₄ w₅─┼─w₆ │ │ │ │ │ ▼ ▼ ▼ ▼ ▼ h₁ h₂ │ │ w₇─┼─────w₈─┘ │ │ │ w₉ (Bias) ▼ t 
(a) Suppose we have a sample x, where x1 =0.5, x2 =0.6, x3 =0.7. The network parameters are w1 =2, w2 =3 w3 =2, w4 =1.5 w5 =3, w6 =4 w7 =6, w8 =3 Next, let’s suppose the target value y for this example is 4. Write down the forward steps and the prediction error for this given sample. Hint: you need to write down the detailed computational steps. (b) Given the prediction error in the previous question, calculate the gradient of w1, namely \frac{∂E}{∂w1}. Please also write down all involved derivatives.