## Question 2: Goal of Supervised Learning [3 marks]

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

---

# FIT5021 — Solutions for Exam Practice Questions

---

## 1 Classification versus Regression

In statistical supervised learning we are predicting the value $y(x)$ of a target (random) variable $T$ given the value of an input variable $X = x$ such that $y(X)$ is on average a good approximation to $T$ where this average is taken with respect to the joint distribution of $X$ and $T$ that governs both, the process of sampling training data as well as the occurrence of data later when the prediction model is deployed.

In regression the response variable $T$ is assumed to be continuous, e.g., the predicted future price of a financial asset or the predicted temperature in a specific region. In contrast, in classification the response variable has a discrete domain of a finite number of categories, e.g., 'spam'/'no-spam' for e-mail classification or 'tuna', 'salmon', 'bass' for the fish classification example.

---

## 2 Goal of Supervised Learning

The ultimate goal of predictive (supervised) learning is to find a prediction function $y(x)$ with small generalisation error, i.e., expected loss between target value $T$ and prediction $y(X)$ based on input value $X$:

$$\mathbb{E}(\ell(T,, y(X)))$$

One aims to find such a prediction function by minimising the training error

$$\frac{1}{N} \sum_{n=1}^{N} \ell(t_n,, y(x_n))$$

on a set of training data $(x_1, t_1), \ldots, (x_N, t_N)$ that has been sampled independently and identically distributed according to the joint distribution of $X$ and $T$. The test error

$$\frac{1}{M} \sum_{m=1}^{M} \ell(t_{N+m},, y(x_{N+m}))$$

on further iid test data $(x_{N+1}, t_{N+1}), \ldots, (x_{N+M}, t_{N+M})$ is then used as an unbiased estimate of the prediction function that has been found.

---

## 3 Model Selection

### (a)

The average test error of the chosen model (corresponding to the second best $k$) obtained by Alice's cross validation procedure is optimistically biased, because the test data is used as part of the overall training data to learn the value for $k$.

### (b)

The procedure can be corrected by first performing a training/test split, carrying out the cross validation procedure to choose $k$ only based on the training data, and obtaining a final estimate of the generalisation error on the test data.

---

## 4 Normal Distribution and Maximum Likelihood Estimation

### (a)

The density function of $\mathcal{N}(\mu, \sigma^2)$, the normal distribution with mean $\mu$ and variance $\sigma^2$ is

$$p(x \mid \mu, \sigma) = \frac{1}{\sqrt{2\pi},\sigma} \exp!\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$

The central component is the exponential reduction of the density in the normalised square of the difference of $x$ from the mean $\mu$. This creates the bell shaped curve, because of a slow reduction close to the mean (where the normalised difference is less than 1) and a rapid reduction further away from the mean.

### (b)

Given a dataset $\mathcal{D} = (x_1, \ldots, x_N)$ of observations drawn independently and identically distributed according to $\mathcal{N}(\mu, \sigma^2)$, the idea of maximum likelihood estimation is to find estimates $\mu_\text{ML}$ and $\sigma_\text{ML}$ that maximise the probability of the data, i.e.,

$$p(\mathcal{D} \mid \mu_\text{ML}, \sigma_\text{ML}) = \max{p(\mathcal{D} \mid \mu, \sigma) : \mu \in \mathbb{R},, \sigma \in \mathbb{R}^+}$$

### (c)

All partial derivatives of the likelihood function have to be 0 at the maximum likelihood estimates. Additionally, we can optimise the log likelihood instead of the likelihood (because this is a monotone transformation). With this and the assumption that the sample elements are drawn independently, a method for identifying the maximum likelihood parameters is to check where the partial derivatives of

$$\ln p(\mathcal{D} \mid \mu, \sigma) = \ln \prod_{n=1}^{N} p(x_n \mid \mu, \sigma) = \sum_{n=1}^{N} \ln p(x_n \mid \mu, \sigma)$$

is zero. Applying this idea to $\mu$, we find the partial derivatives with respect to $\mu$ as

$$\frac{\partial}{\partial \mu} \sum_{n=1}^{N} \ln p(x_n \mid \mu, \sigma) = \sum_{n=1}^{N} \frac{\partial}{\partial \mu} \ln p(x_n \mid \mu, \sigma)$$

$$= \sum_{n=1}^{N} \frac{\partial}{\partial \mu} \left(- \ln \sqrt{2\pi},\sigma - \frac{(x_n - \mu)^2}{2\sigma^2}\right)$$

$$= -\sum_{n=1}^{N} \frac{x_n - \mu}{\sigma^2}$$

Setting the last expression to 0 and solving for $\mu$, we find that

$$\mu_\text{ML} = \frac{1}{N} \sum_{n=1}^{N} x_n$$

### (d)

Applying the same principles as in (c), we find the partial derivative of the log likelihood with respect to $\sigma$ as

$$\frac{\partial}{\partial \sigma} \sum_{n=1}^{N} \ln p(x_n \mid \mu, \sigma) = \sum_{n=1}^{N} \frac{\partial}{\partial \sigma} \left(- \ln\sqrt{2\pi} - \ln \sigma - \frac{(x_n - \mu)^2}{2\sigma^2}\right)$$

$$= \sum_{n=1}^{N} \left(-\frac{1}{\sigma} + \frac{(x_n - \mu)^2}{\sigma^3}\right)$$

Setting this expression to 0, plugging in our ML estimate for $\mu$, and solving for $\sigma$ gives

$$\sigma_\text{ML} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_n - \mu_\text{ML})^2}$$

---

## 5 Derivation of Squared Error

### (a)

We assumed that the conditional distribution of the target variable $T$ given the input variable $X$ follows a deterministic function $y(X)$ plus normally distributed noise (with mean 0), i.e.,

$$T = y(X) + \epsilon$$

$$\epsilon \sim \mathcal{N}(0, \sigma^2)$$

### (b)

With the above model, the conditional distribution of $T$ given $X$ is a normal distribution with mean $y(X)$. For a prediction function $y(x, \mathbf{w})$ with parameter vector $\mathbf{w}$, the likelihood function for an individual data point $(x, t)$ is

$$p(t \mid x, \mathbf{w}) = \frac{1}{\sqrt{2\pi},\sigma} \exp!\left(-\frac{(t - y(x, \mathbf{w}))^2}{2\sigma^2}\right)$$

### (c)

Maximising the likelihood function is equivalent to minimising the negative log likelihood. The negative log likelihood for a single data point is

$$-\ln p(t \mid x, \mathbf{w}) = \ln\sqrt{2\pi},\sigma + \frac{1}{2\sigma^2}(t - y(x, \mathbf{w}))^2$$

which is the same as the squared error up to a (positive) constant factor and a constant additional term. Hence, minimising the sum of squared errors is equivalent to minimising the sum of negative log likelihood terms, which is the negative log likelihood given an i.i.d. dataset.

---

## 6 Logistic Regression

### (a)

Given that the sigmoid transform of $\mathbf{w} \cdot \mathbf{x}$ is the modelled probability of $y = 1$, we can write the likelihood as

$$p(t \mid x, \mathbf{w}) = \begin{cases} \sigma(\mathbf{w} \cdot \mathbf{x}) & \text{if } t = 1 \ 1 - \sigma(\mathbf{w} \cdot \mathbf{x}) & \text{if } t = 0 \end{cases}$$

For the following steps, however, it is useful to write this more compactly without case distinction as

$$p(t \mid x, \mathbf{w}) = \sigma(\mathbf{w} \cdot \mathbf{x})^t,(1 - \sigma(\mathbf{w} \cdot \mathbf{x}))^{(1-t)}$$

from which we obtain the log likelihood as

$$\ln p(t \mid x, \mathbf{w}) = t\ln\sigma(\mathbf{w} \cdot \mathbf{x}) + (1 - t)\ln(1 - \sigma(\mathbf{w} \cdot \mathbf{x}))$$

### (b)

Using basic properties of derivatives and the property of the sigmoid function that $1 - \sigma(a) = \sigma(-a)$, we can start deriving the partial derivative with respect to $w_i$ of the log likelihood as

$$\frac{\partial}{\partial w_i}\bigl[-\ln p(t \mid x, \mathbf{w})\bigr] = -t,\frac{\partial}{\partial w_i}\ln\sigma(\mathbf{w} \cdot \mathbf{x}) - (1-t),\frac{\partial}{\partial w_i}\ln\sigma(-\mathbf{w} \cdot \mathbf{x})$$

then applying the chain rule and the derivative of the natural logarithm we can continue with

$$= -t,\frac{1}{\sigma(\mathbf{w} \cdot \mathbf{x})},\frac{\partial}{\partial w_i}\sigma(\mathbf{w} \cdot \mathbf{x}) - (1-t),\frac{1}{\sigma(-\mathbf{w} \cdot \mathbf{x})},\frac{\partial}{\partial w_i}\sigma(-\mathbf{w} \cdot \mathbf{x})$$

followed by another application of the chain rule and the fact that the derivative of the sigmoid function is given as $\sigma'(a) = \sigma(a)\sigma(-a)$, we reach the form

$$= -t,\sigma(-\mathbf{w} \cdot \mathbf{x}),\frac{\partial}{\partial w_i}(\mathbf{w} \cdot \mathbf{x}) - (1-t),\sigma(\mathbf{w} \cdot \mathbf{x}),\frac{\partial}{\partial w_i}(-\mathbf{w} \cdot \mathbf{x})$$

$$= -t(1 - \sigma(\mathbf{w} \cdot \mathbf{x})),x_i + (1-t),\sigma(\mathbf{w} \cdot \mathbf{x}),x_i$$

which we can further simplify to

$$= -(t - \sigma(\mathbf{w} \cdot \mathbf{x})),x_i$$

### (c)

When we look at a whole training dataset $\mathcal{D} = {(x_1, t_1), \ldots, (x_N, t_N)}$ drawn independently and identically distributed, the corresponding negative log likelihood function becomes

$$L(\mathbf{w}) = -\ln \prod_{n=1}^{N} p(t_n \mid x_n, \mathbf{w}) = -\sum_{n=1}^{N} \ln p(t_n \mid x_n, \mathbf{w})$$

Using the result from (b) and the sum rule of derivatives, the partial derivative can then be computed as

$$\frac{\partial}{\partial w_i} L(\mathbf{w}) = -\sum_{n=1}^{N} (t_n - \sigma(\mathbf{w} \cdot \mathbf{x}_n)),x_{n,i}$$

The gradient $\nabla L(\mathbf{w})$ can therefore be compactly written as

$$\nabla L(\mathbf{w}) = -\sum_{n=1}^{N} (t_n - \sigma(\mathbf{w} \cdot \mathbf{x}_n)),\mathbf{x}_n$$

---

## 7 Document Clustering Model

### (a)

Following the notation used in Module 4, we reiterate the probabilistic model. We would like to partition $N$ documents into $K$ clusters. We represent each document $x_n$ under a bag of words representation (BoW) coming from a dictionary denoted by $\mathcal{A}$. We use mixture multinomial models to model our latent variable model. Each document $x_n$ is first generated by being allocated to a cluster $k$ under a multinomial distribution (parameter $\phi$). Given the cluster $k$ for $x_n$, each word is generated from a multinomial model (parameters $\mu_k$). As such, we define the following constraints:

$$\ln p(\mathbf{x}) = \ln\left(\prod_{n=1}^{N} p(x_n)\right)$$

$$= \ln\left(\prod_{n=1}^{N} \sum_{k=1}^{K} p(x_n \mid z_{n,k}=1),p(z_{n,k}=1)\right)$$

$$= \sum_{n=1}^{N} \ln \sum_{k=1}^{K} \left(\phi_k \prod_{w \in \mathcal{A}} \mu_{k,w}^{c(w,, x_n)}\right)$$

### (b)

We use expectation maximisation to find the parameters $\theta$ which are $\phi_k$ and $\mu_{k,w}$. As such, we need to find $Q$ — a tractable lower bound function of the likelihood function. For the typical 'soft' expectation maximisation function, this $Q$ function is derived by using Jensen's inequality and the Kullback-Leibler (KL) divergence:

$$Q(\theta, \theta^\text{old}) = \sum_{n=1}^{N} \sum_{k=1}^{K} p(z_{n,k}=1 \mid x_n, \theta^\text{old}),\ln p(x_n, z_{n,k} \mid \theta)$$

$$\phi_k = \frac{1}{N} \sum_{n=1}^{N} z^*_{n,k}$$

$$\mu_{k,w} = \frac{\sum_{n=1}^{N} z^__{n,k},c(w,, x_n)}{\sum_{w' \in \mathcal{A}} \sum_{n=1}^{N} z^__{n,k},c(w',, x_n)}$$

### (c)

For the hard EM, we do not need the expectation over all $K$ clusters, rather, we just need the 'most likely' $k$ for the calculation of the lower bound $Q$ function, and thus, the updated parameters $\theta$. Therefore, for each individual document $x_n$, we define:

$$z^*_{n,k} = \begin{cases} 1 & k = \arg\max_k, p(z_{n,k}=1 \mid x_n, \theta^\text{old}) \ 0 & \text{else} \end{cases}$$

Our lower bound $Q$ function is now:

$$Q(\theta, \theta^\text{old}) = \sum_{n=1}^{N} \sum_{k=1}^{K} z^__{n,k},\ln p(x_n, z^__{n,k} \mid \theta)$$

---

## 8 Forward and Backward Propagation

### (a)

$$h_1 = X_1 \cdot W_1 + X_2 \cdot W_3 + X_3 \cdot W_5 = 4.3$$

$$h_2 = X_1 \cdot W_2 + X_2 \cdot W_4 + X_3 \cdot W_6 = 5.2$$

$$t = h_1 \cdot W_7 + h_2 \cdot W_8 = 41.4$$

$$E = \frac{1}{2}(y - t)^2 = 699.38$$

### (b)

According to the chain rule, we have

$$\frac{\partial E}{\partial W_1} = \frac{\partial E}{\partial t} \cdot \frac{\partial t}{\partial h_1} \cdot \frac{\partial h_1}{\partial W_1}$$

where $\dfrac{\partial E}{\partial t} = t - y = 37.4$, $\dfrac{\partial t}{\partial h_1} = W_7 = 6$, $\dfrac{\partial h_1}{\partial W_1} = X_1 = 0.5$. Hence the solution is

$$\frac{\partial E}{\partial W_1} = 37.4 \times 6 \times 0.5 = 112.2$$