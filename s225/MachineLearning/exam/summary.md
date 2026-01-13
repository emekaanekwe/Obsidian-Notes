 **Step A — finish linear regression “properly”**

1. Design matrix view:
    

y=Φw\mathbf{y}=\Phi\mathbf{w}y=Φw

2. Normal equations / closed-form (when applicable)
    
3. Gradient of SSE and one GD update
    
4. Regularization:
    

- Ridge (L2) and why it’s MAP under a Gaussian prior
    
- LASSO (L1) conceptually (subgradient / sparsity)
    

**Step B — transition to classification**

1. What changes: target type + likelihood
    
2. Logistic regression:
    

- sigmoid, decision boundary, NLL / cross-entropy
    
- gradient intuition: “error times features” still shows up, just with probabilities
    

**Step C — probabilistic discriminative models (broader view)**
1. Softmax regression (multiclass)
2. link to neural nets: logits -> softmax/sigmoid -> NLL
3. Stohastic gradient descent, minibatches
# 1. Classification versus Regression 

### What is the statistical supervised learning problem on a high level, and what is difference between a classification and a regression problem? Answer in two to three sentences.


**The statistical supervised learning problem (high level)**  
We are given input–output pairs  
$${(x_1, t_1), \dots, (x_N, t_N)}$$  
drawn from some unknown joint distribution over $(X,T)$, and we want to learn a function $f$ such that $f(x)$ accurately predicts $t$ for **new**, unseen inputs.

- In a **regression** problem, the target $t$ is **continuous** (e.g. house price).
    
- In a **classification** problem, the target $t$ is **categorical** (e.g. class labels like “bird”, “cat”, “dog”).
    
# 2. Goal of Supervised Learning [3 marks]

### What is the ultimate goal of statistical supervised learning? How are the concepts of training error and test error related to this goal? Answer in three to four sentences.

The ultimate goal of supervised learning is to **minimize the expected prediction error on new, unseen data**, i.e. the **generalization error** (also called test error), not just error on the training set.

- The **training error** is the average loss on the training data; it measures how well the model fits the **seen** data.
    
- The **test error** is the average loss on a separate test set drawn from the same distribution; it estimates generalization performance.
    
- A good model balances low training error **and** low test error (avoids overfitting/underfitting).
    

---

# 3. Model Selection (kNN on Iris) [6 marks]

## Consider the following scenario:

**a) Alice wants to model the species (‘setosa’, ‘versicolor’, or ‘virginica’) of iris flowers as a function of four variables (their sepal and petal length and width)**

**b) She has collected a dataset of 150 examples**

**c) She wants to use the kNN classifier but she does not know what is a suitable value for k, hence shes want to choose k from the set of candidates {1, 2, ..., 10} based on the collected data**

**d) Finally, she wants a reliable estimate of the performance of the model that has been learned.**

#### Currently Alice plans to proceed with the following machine learning workflow:

1. **Split the data into 10 folds of roughly equal size.**
2. **Pick the value of k with the second best average test error across all folds (for each using the remaining folds as training data). In particular, she plans to use the second best “to avoid overfitting”.**
3. **Use this test error as the final performance estimate.**

### (a) Point out in up to two sentences, what is the most substantial problem with Alice’s proposed workflow and why.

Alice uses the **same 10-fold cross-validation both to choose $k$ and to report the final test error**. This “double use” of the data for both **model selection** and **performance estimation** leads to an optimistically biased estimate of the performance (she is overfitting to the folds).

### (b) Describe an improved machine learning process that adequately addresses this problem.

A correct process:

1. **Hold out a test set** once at the start (e.g. 20–30% of data), never touch it during model selection.
    
2. Use the **remaining training data** to select $k$:
    
    - Either do **inner cross-validation** (nested CV) on the training set to pick the best $k \in {1,\dots,10}$.
        
    - Or split the training set into **train/validation**: fit models for each $k$ on the inner train set, choose $k$ with the smallest validation error.
        
3. **Retrain** the final model on the full training set using the chosen $k$.
    
4. Finally, evaluate this chosen model **once on the held-out test set** to get an unbiased performance estimate.
    

Alternative: full **nested cross-validation**, where outer folds are used for performance estimation and inner folds for model selection.

---

# 4. Normal Distribution & Maximum Likelihood

	The (uni-variate) normal distribution is an extremely important distribution describing the behaviour of continuous random variables. It is parameterised by a mean μ and a standard deviation parameters σ (or more typically by the corresponding variance parameter σ2 ). Given a dataset {x1 , . . . , xN } of independent realisations of a normal random variable X, we can use the principal
of maximum likelihood to find guesses for the unknown parameters. In particular, these guesses have simple closed form solutions.

***Answer each of the following questions with one to two sentences and give mathematical derivations as appropriate.***


We assume $x_1,\dots,x_N$ are i.i.d. from a normal distribution $X \sim \mathcal{N}(\mu,\sigma^2)$.

### a) What is the definition of the normal density function p(x|μ, σ)? What is the key component of the definition that gives rise to the characteristic bell shape?

**Definition of the (univariate) normal density**:

$$  
p(x \mid \mu,\sigma)  
= \frac{1}{\sqrt{2\pi}\sigma} \exp!\left( -\frac{(x - \mu)^2}{2\sigma^2} \right).  
$$

**Bell shape source:**  
The **quadratic term** $(x-\mu)^2$ inside the exponential is the key: as $x$ moves away from $\mu$, $(x-\mu)^2$ grows, the exponent becomes more negative, and $p(x)$ decays smoothly on both sides, giving the characteristic symmetric bell shape.

---

### (b) ### b) What is the key idea of the maximum likelihood estimation of the parameters μ and σ, i.e., what is the defining property of the maximum likelihood estimates $μ_{ML}$ and $σ_{ML}$

Given data $x_1,\dots,x_N$, the **likelihood** of parameters $(\mu,\sigma)$ is

$$  
L(\mu,\sigma) = \prod_{n=1}^N p(x_n \mid \mu,\sigma).  
$$

The **maximum likelihood estimates** $(\mu_{\text{ML}}, \sigma_{\text{ML}})$ are the parameter values that **maximize** this likelihood:

$$  
(\mu_{\text{ML}}, \sigma_{\text{ML}}) = \arg\max_{\mu,\sigma} L(\mu,\sigma),  
$$

equivalently, they **maximize the log-likelihood** $\log L(\mu,\sigma)$.

---

### (c) How can we derive the closed form solution of the maximum likelihood estimation for the mean μ? Apply this approach to derive it.

**Prerequisite refresher**

- Logs turn products into sums:  
    $$\log \prod_n a_n = \sum_n \log a_n.$$
    
- Derivative of a quadratic:  
    $$\frac{\partial}{\partial \mu}(x_n - \mu)^2 = -2(x_n - \mu).$$
    

**Step 1: Write the log-likelihood**

[  
\begin{aligned}  
\log L(\mu,\sigma)  
&= \sum_{n=1}^N \log p(x_n \mid \mu,\sigma) \  
&= \sum_{n=1}^N \left( -\frac{1}{2}\log(2\pi) - \log\sigma

- \frac{(x_n - \mu)^2}{2\sigma^2} \right).  
    \end{aligned}  
    ]
    

We treat $\sigma$ as fixed for this part and maximize w.r.t. $\mu$.

**Step 2: Differentiate w.r.t. $\mu$**

Only the last term depends on $\mu$:

$$  
\frac{\partial}{\partial \mu} \log L(\mu,\sigma)  
= \sum_{n=1}^N \frac{\partial}{\partial \mu} \left(- \frac{(x_n-\mu)^2}{2\sigma^2}\right)  
= \sum_{n=1}^N \frac{(x_n - \mu)}{\sigma^2}.  
$$

**Step 3: Set derivative = 0 and solve**

$$  
\sum_{n=1}^N (x_n - \mu) = 0  
;;\Rightarrow;;  
\sum_{n=1}^N x_n - N\mu = 0  
;;\Rightarrow;;  
\mu_{\text{ML}} = \frac{1}{N} \sum_{n=1}^N x_n.  
$$

So the MLE of the mean is the **sample mean**.

---

### (d) How can we derive the closed form solution of the maximum likelihood estimation for the standard deviation σ? Apply this approach to derive it.

It’s easier to derive the MLE for $\sigma^2$ and then convert to $\sigma$.

**Step 1: Using log-likelihood**

Using the same log-likelihood:

[  
\log L(\mu,\sigma)  
= -\frac{N}{2}\log(2\pi) - N\log \sigma

- \frac{1}{2\sigma^2} \sum_{n=1}^N (x_n - \mu)^2.  
    ]
    

Now treat $\mu=\mu_{\text{ML}}$ as fixed and maximize w.r.t. $\sigma$ (or $\sigma^2$).

**Step 2: Differentiate w.r.t. $\sigma$**

Refresher:

- $\frac{\partial}{\partial \sigma} \log\sigma = \frac{1}{\sigma}$
    
- $\frac{\partial}{\partial \sigma} \frac{1}{\sigma^2} = -\frac{2}{\sigma^3}$
    

Derivative:

[  
\begin{aligned}  
\frac{\partial}{\partial \sigma} \log L  
&= - N \frac{1}{\sigma}

- \frac{\partial}{\partial \sigma} \left[  
    \frac{1}{2\sigma^2} \sum_{n=1}^N (x_n - \mu)^2  
    \right] \  
    &= - \frac{N}{\sigma}
    

- \frac{1}{\sigma^3} \sum_{n=1}^N (x_n - \mu)^2.  
    \end{aligned}  
    ]
    

**Step 3: Set derivative = 0 and solve**

[

- \frac{N}{\sigma} + \frac{1}{\sigma^3} \sum_{n=1}^N (x_n - \mu)^2 = 0  
    ]
    

Multiply both sides by $\sigma^3$:

[

- N \sigma^2 + \sum_{n=1}^N (x_n - \mu)^2 = 0  
    ;;\Rightarrow;;  
    \sigma^2_{\text{ML}} = \frac{1}{N} \sum_{n=1}^N (x_n - \mu_{\text{ML}})^2.  
    ]
    

Thus the MLE for the **standard deviation** is:

$$  
\sigma_{\text{ML}} = \sqrt{ \frac{1}{N} \sum_{n=1}^N (x_n - \mu_{\text{ML}})^2 }.  
$$

Small example: if $x = [1,3]$, $\mu_{\text{ML}}=2$, so  
$\sigma^2_{\text{ML}} = \frac{1}{2}[(1-2)^2 + (3-2)^2] = 1$, so $\sigma_{\text{ML}} = 1$.

---

# 5. Derivation of Squared Error in Linear Regression 

For fitting the model parameters w of a linear regression model, we used the approach to minimise the squared error:
$E(w) = \frac{1}{2}\sum^{N}_{n=1}(t_n-y_n)^2$
where ${(x1, t1 ), . . . , (xN , tn)}$ is the given training data and $y_n = \sum^{p}_{i=1}w_iφ_i (x_n)$ are the model predictions. To justify this error function, we showed that it can be derived as a maximum likelihood parameter estimation for a probabilistic model p(t|x, w). Answer each of the following questions with one to two sentences (including mathematical equations as appropriate).

We model targets with a linear model in basis functions:

$$  
y_n = y(x_n, \mathbf{w}) = \sum_{i=1}^p w_i ,\phi_i(x_n),  
$$

and we assume noisy observations $t_n$.

### (a) What is the form of the probabilistic model that we assumed for the regression problem, i.e., how are the target values generated given the input vectors?

We assume **Gaussian noise**:

$$  
t_n = y(x_n, \mathbf{w}) + \varepsilon_n,  
\qquad  
\varepsilon_n \sim \mathcal{N}(0, \sigma^2), \text{ independent}.  
$$

Equivalently,

$$  
p(t_n \mid x_n, \mathbf{w})  
= \mathcal{N}\big(t_n \mid y(x_n,\mathbf{w}), \sigma^2\big).  
$$

---

### (b) What is the likelihood function corresponding to this model?

Because observations are i.i.d.,

$$  
p(\mathbf{t} \mid X, \mathbf{w})  
= \prod_{n=1}^N \mathcal{N}\big(t_n \mid y(x_n,\mathbf{w}), \sigma^2\big).  
$$

Log-likelihood (dropping constants) is:

$$  
\log p(\mathbf{t} \mid X, \mathbf{w})  
= -\frac{N}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{n=1}^N \big(t_n - y_n\big)^2.  
$$

---

### (c) Why is maximising this likelihood function equivalent to minimising the squared error function?

The **negative log-likelihood** (up to constants) is:

$$  
-\log p(\mathbf{t} \mid X, \mathbf{w})  
= \frac{1}{2\sigma^2}\sum_{n=1}^N (t_n - y_n)^2 + \text{const}.  
$$

Since $\sigma^2$ and the constant don’t depend on $\mathbf{w}$, maximizing likelihood is equivalent to **minimizing**:

$$  
E(\mathbf{w}) = \frac{1}{2} \sum_{n=1}^N (t_n - y_n)^2.  
$$

We can ignore the $\frac{1}{2\sigma^2}$ scaling factor for argmin; that’s exactly the usual squared-error objective.

---

## 6. Logistic Regression

When using the logistic regression model for binary classification, we model the probability of the positive class (t = 1) given input x via the sigmoid transformation σ of a linear function $w · x$ of model parameters w.

We model the probability of the positive class as:

$$  
p(t=1 \mid \mathbf{x},\mathbf{w}) = \sigma(z),  
\quad  
z = \mathbf{w}^\top\mathbf{x},  
\quad  
\sigma(z) = \frac{1}{1 + e^{-z}}.  
$$

We encode $t \in {0,1}$.

---

### (a) Give the log likelihood function log p(t|x, w) of the logistic regression model for a single data point (x, t). Hint: We used the fact that we encode the positive class with t = 1 and the negative class with t = 0 to give a compact formula.

For one example $(\mathbf{x}, t)$:

$$  
p(t \mid \mathbf{x},\mathbf{w})  
= \sigma(z)^t \big(1 - \sigma(z)\big)^{1-t},  
\quad z = \mathbf{w}^\top \mathbf{x}.  
$$

So the **log likelihood** is:

$$  
\log p(t \mid \mathbf{x},\mathbf{w})  
= t ,\log \sigma(z)

- (1 - t), \log\big(1 - \sigma(z)\big).  
    $$
    

Small check: if $t=1$, we get $\log \sigma(z)$; if $t=0$, we get $\log(1-\sigma(z))$.

---

### (b) As a step towards the gradient descent algorithm for logistic regression, derive the partial derivative of the negative log likelihood (error function) − log p(t|x, w) with respect to parameter w_i . Derive the result in individual steps, noting what results you are using (all correct steps give partial marks).

Define the **negative log-likelihood** (error for one sample):

$$  
E(\mathbf{w}) = -\log p(t\mid\mathbf{x},\mathbf{w})  
= -\left[  
t \log \sigma(z) + (1-t)\log(1-\sigma(z))  
\right],  
\quad z = \mathbf{w}^\top\mathbf{x}.  
$$

We want $\dfrac{\partial E}{\partial w_i}$.

**Prerequisite: derivatives**

- $\dfrac{d}{dz} \sigma(z) = \sigma(z)(1 - \sigma(z))$
    
- $\dfrac{d}{dz} \log \sigma(z) = \dfrac{1}{\sigma(z)} \sigma'(z) = 1 - \sigma(z)$
    
- $\dfrac{d}{dz} \log (1-\sigma(z)) = \dfrac{-\sigma'(z)}{1-\sigma(z)} = -\sigma(z)$
    
- $\dfrac{\partial z}{\partial w_i} = x_i$ since $z = \sum_j w_j x_j$.
    

**Step-by-step**

1. Differentiate w.r.t. $z$ first:
    

[  
\begin{aligned}  
\frac{\partial E}{\partial z}  
&= -\left[  
t \cdot \frac{d}{dz}\log \sigma(z)

- (1-t)\cdot \frac{d}{dz}\log(1-\sigma(z))  
    \right] \  
    &= -\left[  
    t \cdot (1 - \sigma(z))
    
- (1-t)\cdot (-\sigma(z))  
    \right] \  
    &= -\left[  
    t - t\sigma(z) -\sigma(z) + t\sigma(z)  
    \right] \  
    &= -\left[  
    t - \sigma(z)  
    \right]  
    = \sigma(z) - t.  
    \end{aligned}  
    ]
    

2. Now apply chain rule to $w_i$:
    

$$  
\frac{\partial E}{\partial w_i}  
= \frac{\partial E}{\partial z}\cdot \frac{\partial z}{\partial w_i}  
= (\sigma(z) - t) , x_i.  
$$

So the derivative is:

$$  
\boxed{  
\frac{\partial}{\partial w_i}\big(-\log p(t\mid\mathbf{x},\mathbf{w})\big)  
= (\sigma(\mathbf{w}^\top\mathbf{x}) - t) , x_i.  
}  
$$

---

### (c) Extend your result from part (b) to the full gradient of the negative log likelihood when observing a set of n training data points {(x1, t1 ), . . . , (xN , tn)}.

For data ${(\mathbf{x}_n,t_n)}_{n=1}^N$, the total error is:

$$  
E(\mathbf{w}) = - \sum_{n=1}^N \log p(t_n \mid \mathbf{x}_n,\mathbf{w}).  
$$

Using the one-example result:

$$  
\frac{\partial E}{\partial w_i}  
= \sum_{n=1}^N \big(\sigma(\mathbf{w}^\top \mathbf{x}_n) - t_n\big) x_{n,i}.  
$$

Vector form:

$$  
\nabla_{\mathbf{w}} E(\mathbf{w})  
= \sum_{n=1}^N \big(\sigma(\mathbf{w}^\top \mathbf{x}_n) - t_n\big)\mathbf{x}_n.  
$$

This is precisely the gradient used in gradient descent for logistic regression.

---

# 7. Document Clustering with Mixture of Multinomials 

Suppose we are given a collection of documents D. The data set D is represented as {x1 , x2, x3 , ..., xN } where x_i is a d-dimensional “count vector” representing the i-th document, based on bag-of-words and with respect to a word vocabulary of size d. We are interested in fitting a Mixture multinomial model onto this dataset.

We have $N$ documents, each represented as a **word count vector**:

$$  
\mathbf{x}_i = (x_{i1}, \dots, x_{id})^\top \in \mathbb{N}^d,  
$$

where $d$ is the vocabulary size.

### (a) An individual cluster is described by a vector of word occurrence probabilities μ where $μ_j$ describes the probability of a word in a document to be the j-th word in the vocabulary. Give a formula of the probability p(x|μ) of a count vector x given word occurrence probabilities μ and give a brief explanation of the formula (one to two sentences). Hint: remember that, for simplicity, we assumed the individual counts to be independent.

Each cluster has word probabilities:

$$  
\boldsymbol{\mu} = (\mu_1,\dots,\mu_d),\quad  
\mu_j \ge 0,\quad \sum_{j=1}^d \mu_j = 1.  
$$

**Under the simplified “independent counts” assumption** (as in the hint), we write:

$$  
p(\mathbf{x} \mid \boldsymbol{\mu}) = \prod_{j=1}^d \mu_j^{x_j}.  
$$

Short explanation:

- $x_j$ is the count of the $j$-th word in the document.
    
- $\mu_j$ is the probability of that word.
    
- Each occurrence contributes a factor of $\mu_j$, so $x_j$ occurrences give $\mu_j^{x_j}$.
    
- Multiplying over all words yields the probability of the whole document.
    

(We are ignoring the multinomial coefficient, which does not depend on $\mu$ and thus drops out in maximum likelihood / EM.)

---

### (b) Write down the “Q-function”, which is the basis of the Expectation-Maximization (EM) algorithm for maximizing the log-likelihood. Notice that you do not need to write the EM algorithm in this part.

We have a **mixture of $K$ multinomials** with mixing weights $\pi_k$ and cluster-specific word distributions $\boldsymbol{\mu}_k$.

- Latent variable for document $i$: $z_i \in {1,\dots,K}$.
    
- Responsibility (posterior cluster probability):
    
    $$  
    \gamma_{ik} = p(z_i = k \mid \mathbf{x}_i, \theta^{\text{old}}),  
    $$
    
    where $\theta = {\pi_k, \boldsymbol{\mu}_k}_{k=1}^K$.
    

The **complete-data log-likelihood** (if $z_i$ were known) is:

$$  
\log p(\mathbf{X},\mathbf{Z} \mid \theta) =  
\sum_{i=1}^N \sum_{k=1}^K \mathbb{1}[z_i=k]  
\left(\log \pi_k + \log p(\mathbf{x}_i \mid \boldsymbol{\mu}_k)\right).  
$$

For EM, the **$Q$-function** is the expected complete-data log-likelihood under the posterior of $Z$ using old parameters:

$$  
\boxed{  
Q(\theta,\theta^{\text{old}})  
= \sum_{i=1}^N \sum_{k=1}^K  
\gamma_{ik}\Big( \log \pi_k + \log p(\mathbf{x}_i \mid \boldsymbol{\mu}_k)\Big).  
}  
$$

With $p(\mathbf{x}_i \mid \boldsymbol{\mu}_k) = \prod_{j=1}^d \mu_{jk}^{x_{ij}}$, we can write:

$$  
\log p(\mathbf{x}_i \mid \boldsymbol{\mu}_k)  
= \sum_{j=1}^d x_{ij} \log \mu_{jk}.  
$$

---

### (c) Write down the “hard” as well as the ”soft” Expectation-Maximization (EM) algorithm for estimating the parameters of the model. If necessary, provide enough explanation to understand the algorithm that you have written. Also briefly explain what is the main difference between hard and soft EM.

We want to estimate ${\pi_k,\boldsymbol{\mu}_k}$.

#### Soft EM

**E-step (Soft responsibilities):**

For each document $i$ and cluster $k$:

$$  
\gamma_{ik} = p(z_i = k \mid \mathbf{x}_i,\theta^{\text{old}})  
= \frac{\pi_k^{\text{old}} , p(\mathbf{x}_i \mid \boldsymbol{\mu}_k^{\text{old}})}  
{\sum_{j=1}^K \pi_j^{\text{old}} p(\mathbf{x}_i \mid \boldsymbol{\mu}_j^{\text{old}})}.  
$$

**M-step (Update parameters):**

Effective cluster sizes:

$$  
N_k = \sum_{i=1}^N \gamma_{ik}.  
$$

Update mixing weights:

$$  
\pi_k^{\text{new}} = \frac{N_k}{N}.  
$$

Update word probabilities for each cluster:

$$  
\mu_{jk}^{\text{new}}  
= \frac{\sum_{i=1}^N \gamma_{ik} x_{ij}}  
{\sum_{j'=1}^d \sum_{i=1}^N \gamma_{ik} x_{ij'}}.  
$$

Repeat E/M steps until convergence.

---

#### Hard EM

**E-step (Hard assignments):**

Instead of fractional responsibilities, assign each document to its **most likely** cluster:

$$  
z_i^{*} = \arg\max_{k} p(z_i = k \mid \mathbf{x}_i,\theta^{\text{old}}),  
$$

then define:

$$  
\gamma_{ik} =  
\begin{cases}  
1, & k = z_i^* \  
0, & \text{otherwise}.  
\end{cases}  
$$

**M-step:**

Same update formulas as in soft EM, but now $\gamma_{ik}$ are **0/1**:

- $N_k = \sum_i \gamma_{ik}$ = number of documents assigned to cluster $k$.
    
- Update $\pi_k^{\text{new}}$ and $\mu_{jk}^{\text{new}}$ with these hard counts.
    

---

#### Main difference (Hard vs Soft EM)

- **Soft EM:** Documents have **fractional membership** in all clusters; assignments are probabilities $\gamma_{ik} \in (0,1)$.
    
- **Hard EM:** Documents are assigned to **exactly one cluster**; $\gamma_{ik} \in {0,1}$.
    

Soft EM is smoother and generally yields better log-likelihood; Hard EM is more like k-means-style hard clustering and can converge faster but is greedier.

---

# 8. Neural Networks – Forward & Backward Propagation

Given a neural network f(·) and a dataset D = {(x1, y1), (x2 , y2), ..., (xn , y n)} where xi is a 2-dimensional vector and yi is a scalar value which represents the target. {w1, w2 , ..., wn } are learnable parameters. h represents a linear unit. For example t^i = h_1w_7 +h_2w_8. The error function for training
this neural network is the sum of squared error:

$E(w)=\frac{1}{2}\sum^{N}_{n=1}(y^i-t^i)^2$

	---Neural Network Structure---
      x₁        x₂        x₃
      │         │         │
   w₁─┼─w₂  w₃─┼─w₄  w₅─┼─w₆
      │    │    │    │    │
      ▼    ▼    ▼    ▼    ▼
      h₁        h₂
      │         │
   w₇─┼─────w₈─┘
      │    │
      │  w₉ (Bias)
      ▼
      t


We have:

- Inputs: $x_1, x_2, x_3$
    
- Hidden linear units: $h_1, h_2$
    
- Output:  
    $$  
    t = w_7 h_1 + w_8 h_2 + w_9  
    $$  
    (using $w_9$ as bias; they give example $t^i = h_1 w_7 + h_2 w_8$ plus bias).
    

Hidden units are **linear**:

$$  
h_1 = w_1 x_1 + w_3 x_2 + w_5 x_3,  
\qquad  
h_2 = w_2 x_1 + w_4 x_2 + w_6 x_3.  
$$

Error for one sample:

$$  
E = \frac{1}{2}(y - t)^2.  
$$

---

### (a) Suppose we have a sample x, where x1 =0.5, x2 =0.6, x3 =0.7. The network parameters are
### w1 =2, w2 =3
### w3 =2, w4 =1.5
### w5 =3, w6 =4
### w7 =6, w8 =3
### Next, let’s suppose the target value y for this example is 4. Write down the forward steps and the prediction error for this given sample. Hint: you need to write down the detailed computational steps.

Given:

- $x_1 = 0.5,; x_2 = 0.6,; x_3 = 0.7$
    
- $w_1 = 2,; w_2 = 3$  
    $w_3 = 2,; w_4 = 1.5$  
    $w_5 = 3,; w_6 = 4$  
    $w_7 = 6,; w_8 = 3$  
    (no value given for $w_9$, so we treat $w_9 = 0$ for the numeric example)
    

Target: $y = 4$.

**Step 1: compute $h_1$**

$$  
\begin{aligned}  
h_1  
&= w_1 x_1 + w_3 x_2 + w_5 x_3 \  
&= 2 \cdot 0.5 + 2 \cdot 0.6 + 3 \cdot 0.7 \  
&= 1.0 + 1.2 + 2.1 = 4.3.  
\end{aligned}  
$$

**Step 2: compute $h_2$**

$$  
\begin{aligned}  
h_2  
&= w_2 x_1 + w_4 x_2 + w_6 x_3 \  
&= 3 \cdot 0.5 + 1.5 \cdot 0.6 + 4 \cdot 0.7 \  
&= 1.5 + 0.9 + 2.8 = 5.2.  
\end{aligned}  
$$

**Step 3: compute output $t$**

(assuming $w_9 = 0$ for this numeric example)

$$  
\begin{aligned}  
t  
&= w_7 h_1 + w_8 h_2 + w_9 \  
&= 6 \cdot 4.3 + 3 \cdot 5.2 + 0 \  
&= 25.8 + 15.6 = 41.4.  
\end{aligned}  
$$

**Step 4: compute prediction error**

Error function:

$$  
E = \frac{1}{2}(y - t)^2.  
$$

Plug in $y=4,\ t=41.4$:

$$  
\begin{aligned}  
E  
&= \frac{1}{2}(4 - 41.4)^2  
= \frac{1}{2}(-37.4)^2  
= \frac{1}{2}\cdot 37.4^2.  
\end{aligned}  
$$

Compute $37.4^2$:

- $37.4^2 = (37 + 0.4)^2 = 37^2 + 2\cdot 37\cdot 0.4 + 0.4^2 = 1369 + 29.6 + 0.16 = 1398.76$.
    

So

$$  
E = \frac{1}{2} \cdot 1398.76 = 699.38.  
$$

So for this sample, the prediction error is $E \approx 699.38$.

---

### (b) Given the prediction error in the previous question, calculate the gradient of $w_1$, namely $\frac{∂E}{∂w1}$. Please also write down all involved derivatives.

We have:

- $E = \dfrac{1}{2}(y - t)^2$
    
- $t = w_7 h_1 + w_8 h_2 + w_9$
    
- $h_1 = w_1 x_1 + w_3 x_2 + w_5 x_3$
    

We want $\dfrac{\partial E}{\partial w_1}$, so we use **chain rule**:

$$  
\frac{\partial E}{\partial w_1}  
= \frac{\partial E}{\partial t}  
\cdot \frac{\partial t}{\partial h_1}  
\cdot \frac{\partial h_1}{\partial w_1}.  
$$

**Step 1: $\partial E / \partial t$**

Let $e = y - t$. Then $E = \tfrac{1}{2} e^2$.

- $\dfrac{dE}{de} = e$
    
- $\dfrac{de}{dt} = -1$
    

So

$$  
\frac{\partial E}{\partial t}  
= \frac{dE}{de}\cdot \frac{de}{dt}  
= e \cdot (-1)  
= -(y - t) = t - y.  
$$

**Step 2: $\partial t / \partial h_1$**

From $t = w_7 h_1 + w_8 h_2 + w_9$, treating $h_2$ and bias as independent of $h_1$:

$$  
\frac{\partial t}{\partial h_1} = w_7.  
$$

**Step 3: $\partial h_1 / \partial w_1$**

From $h_1 = w_1 x_1 + w_3 x_2 + w_5 x_3$:

$$  
\frac{\partial h_1}{\partial w_1} = x_1.  
$$

**Combine via chain rule**

$$  
\frac{\partial E}{\partial w_1}  
= (t - y) \cdot w_7 \cdot x_1.  
$$

**Numeric value** (using our numbers: $t=41.4$, $y=4$, $w_7=6$, $x_1=0.5$):

- $t - y = 41.4 - 4 = 37.4$
    
- So
    
    $$  
    \frac{\partial E}{\partial w_1}  
    = 37.4 \cdot 6 \cdot 0.5  
    = 37.4 \cdot 3  
    = 112.2.  
    $$
    

So:

$$  
\boxed{  
\frac{\partial E}{\partial w_1} = (t - y),w_7,x_1 \approx 112.2.  
}  
$$

---

If you’d like next, we can turn each of these exam questions into **flashcard-style prompts** or condensed “exam answers” you can recall under time pressure (e.g. 2–3 lines per part), while this document can stay as your full reference.

---

# General ML Cheat Sheet

    **(1) Explanation**  
    **(2) Required math prerequisites**  
    **(3) A tiny numerical example** _(when useful)_  
    *_(4) Intuition that is interview-ready_
    

_(Built from your handwritten keyword list)_

---

# **1. Data, Weights, Biases, Parameters, Model Type**

### **Model Parameters**

Parameters are tunable values (learned from data) that define the model.

- In linear regression:  
    $\mathbf{w} = (w_1, \dots, w_d)$`, bias `$b$
    
- In neural networks:  
    Weights = connections, Biases = shifts before activation.
    

### **Mathematical Form**

$$  
y = \mathbf{w}^\top \mathbf{x} + b  
$$

### **Refresher: Vector Dot Product**

$$  
\mathbf{w}^\top \mathbf{x} = \sum_{i=1}^d w_i x_i  
$$

**Example:**  
If $\mathbf{w} = (2,3)$ and $\mathbf{x}=(1,4)$,  
$\mathbf{w}^\top\mathbf{x} = 2\cdot1 + 3\cdot4 = 14$.

---

# **2. Error Function (Loss) & Model Complexity**

### **Squared Error Loss**

$$  
E = \frac12 \sum_n (t_n - y_n)^2  
$$

**Why:** Comes from assuming Gaussian noise.

### **Model Complexity**

- Too simple → **underfit**
    
- Too flexible → **overfit**
    

---

# **3. Generalization, Overfitting, Underfitting**

### **Training vs Test Error**

- Training error ↓ as model becomes more complex
    
- Test error forms a **U-shape**
    

### **Bias–Variance Tradeoff**

$$  
\text{Test Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}  
$$

---

# **4. Regularization & Validation**

### **L2 Regularization (Ridge)**

$$  
E(\mathbf{w}) = \frac12 \sum_n (t_n - y_n)^2 + \frac{\lambda}{2}|\mathbf{w}|^2  
$$

**Prerequisite:**  
Norm squared:  
$|\mathbf{w}|^2 = \sum_i w_i^2$

### **Why:** Reduces overfitting by shrinking weights.

---

# **5. Cross-Validation, k-fold, LOOCV**

### **k-Fold CV**

Split data into `$k$` folds, train `$k$` times, average error.

### **Leave-One-Out CV (LOOCV)**

$N$ folds with exactly 1 test point each.

**Why:** Nearly unbiased but computationally expensive.

---

# **6. Bootstrap**

Sampling _with_ replacement to estimate variability (variance of estimator).

---

# **7. Bayesian Concepts**

- Prior: `$p(\theta)$`
    
- Likelihood: `$p(D \mid \theta)$`
    
- Posterior:  
    $$  
    p(\theta \mid D) = \frac{p(D\mid\theta)p(\theta)}{p(D)}  
    $$
    

**Tiny Example:**  
If prior = 0.6, likelihood = 0.5, evidence = 0.5 → posterior = 0.6.

---

# **8. Variance & Covariance**

### **Variance**

$$  
\mathrm{Var}(X) = \mathbb{E}[(X - \mu)^2]  
$$

### **Covariance**

$$  
\mathrm{Cov}(X,Y) = \mathbb{E}[(X-\mu_X)(Y-\mu_Y)]  
$$

---

# **9. Basis Functions (Feature Engineering)**

Transform data:  
$$  
\phi(\mathbf{x}) = [1, x, x^2, \sin x, \dots]  
$$

Model becomes:  
$$  
y = \mathbf{w}^\top \phi(\mathbf{x})  
$$

Basis → increases model flexibility.

---

# **10. Linear Regression**

### **Normal Equation**

$$  
\mathbf{w} = (X^\top X)^{-1} X^\top \mathbf{t}  
$$

**Prerequisite:**  
Matrix inverse & multiplication.

---

# **11. Logistic Regression**

### **Sigmoid Function**

$$  
\sigma(z) = \frac{1}{1+e^{-z}}  
$$

### **Model**

$$  
p(t=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x})  
$$

### **Binary Cross-Entropy Loss**

$$  
E = -\sum_{n=1}^N \big[ t_n \log y_n + (1-t_n)\log(1-y_n) \big]  
$$

---

# **12. Gradient & Gradient Descent**

### **Gradient Descent Update**

$$  
\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}}E  
$$

Where η = learning rate.

**Example:**  
If `$w=3$`, gradient `$=2$`, η `$=0.1$`  
→ new `$w = 3 - 0.1\cdot2 = 2.8$`.

---

# **13. Perceptron**

### **Decision Function**

$$  
t = \text{sign}(\mathbf{w}^\top\mathbf{x})  
$$

### **Update Rule**

$$  
\mathbf{w} \leftarrow \mathbf{w} + \eta (t_n - y_n)\mathbf{x}_n  
$$

---

# **14. k-Means Clustering**

### **Objective Function**

$$  
J = \sum_{n=1}^N | x_n - \mu_{z_n} |^2  
$$

Alternates between:

- E-step: assign to nearest centroid
    
- M-step: recompute centroids
    

---

# **15. Mixture Models & Gaussian Mixture Models (GMM)**

### **Mixture Distribution**

$$  
p(x) = \sum_{k=1}^K \pi_k , \mathcal{N}(x \mid \mu_k, \Sigma_k)  
$$

---

# **16. Expectation–Maximization (EM) Algorithm**

### **E-step (Soft Assignments)**

$$  
\gamma_{nk} =  
\frac{  
\pi_k, p(x_n \mid \theta_k)  
}{  
\sum_{j=1}^K \pi_j, p(x_n \mid \theta_j)  
}  
$$

### **M-step**

Update parameters using responsibilities.

### **Hard EM**

Replace $\gamma_{nk}$ with 1 for cluster with max probability.

---

# **17. Complete & Incomplete Data Log-Likelihood**

### **Complete Data Log-Likelihood**

For mixture models:  
$$  
\ln p(X,Z\mid\theta)  
= \sum_{n=1}^N \sum_{k=1}^K  
z_{nk}\big( \ln\pi_k + \ln p(x_n\mid\theta_k) \big)  
$$

### **Q-Function**

Expectation under old parameters:  
$$  
Q(\theta,\theta^{old}) =  
\mathbb{E}_{Z \mid X,\theta^{old}}[\ln p(X,Z\mid\theta)]  
$$

---

# **18. Neural Networks**

## **Forward Propagation**

For hidden unit:  
$$  
h = \sigma(\mathbf{w}_h^\top \mathbf{x} + b_h)  
$$

Output:  
$$  
y = \mathbf{w}_o^\top \mathbf{h} + b_o  
$$

## **Backpropagation (Core Derivative)**

Example of chain rule for $w_1$:  
$$  
\frac{\partial E}{\partial w_1}  
= (y - t), w_{ho}, x_1  
$$

---

# **19. Hidden Units & Generalization**

- Too few → underfit
    
- Too many → overfit
    
- Regularization, early stopping, dropout prevent this
    

---

# **20. Autoencoders**

### **Structure**

$$  
x \to h = \sigma(W_e x + b_e)  
$$  
$$  
\hat{x} = \sigma(W_d h + b_d)  
$$

### **Loss**

$$  
E = \frac12 |x - \hat{x}|^2  
$$

Autoencoders learn **compressed hidden features**.

---

# **21. PCA (Principal Component Analysis)**

### **Goal:**

Find directions of maximum variance.

### **Covariance Matrix**

$$  
C = \frac{1}{N}X^\top X  
$$

### **Eigenvalue Problem**

$$  
C v = \lambda v  
$$

Top eigenvectors = principal components.

---



# **📌 PART B — FLASHCARDS FOR THE ENTIRE CHEAT SHEET**

# -----------------------------------------------------

Flashcards use:

- **Q:** question
    
- **A:** answer
    
- Math in `$...$` or `$$...$$`
    
- Short, exam-ready explanations
    
- No redundant filler
    

---

# **1. Data, Weights, Biases, Parameters**

### **Card 1.1**

**Q:** What are model parameters in machine learning?  
**A:** Parameters are learned values such as **weights** and **biases** that define the behavior of a model.  
Example (linear regression):  
$$ y = \mathbf{w}^\top \mathbf{x} + b. $$

---

### **Card 1.2**

**Q:** What is the difference between weights and biases?  
**A:**

- **Weights:** scale each input feature
    
- **Bias:** shifts the activation so the model is not forced to pass through the origin
    

---

# **2. Error Functions & Model Complexity**

### **Card 2.1**

**Q:** What is the squared error loss?  
**A:**  
$$ E = \frac12 \sum_n (t_n - y_n)^2. $$

---

### **Card 2.2**

**Q:** What is model complexity?  
**A:** The expressive power of the model.  
More complexity → potential overfitting.  
Less complexity → potential underfitting.

---

# **3. Generalization, Overfitting, Underfitting**

### **Card 3.1**

**Q:** How do overfitting and underfitting differ?  
**A:**

- **Underfit:** model too simple → high bias
    
- **Overfit:** model too flexible → high variance  
    Goal: minimize test/generalization error.
    

---

### **Card 3.2**

**Q:** What is the bias–variance decomposition?  
**A:**  
$$ \text{Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}. $$

---

# **4. Regularization**

### **Card 4.1**

**Q:** Write the L2-regularized objective (ridge regression).  
**A:**  
$$  
E(\mathbf{w}) = \frac12\sum_n (t_n - y_n)^2 + \frac{\lambda}{2}|\mathbf{w}|^2.  
$$

---

# **5. Cross-Validation**

### **Card 5.1**

**Q:** Why use k-fold cross-validation?  
**A:** To estimate generalization error and select hyperparameters.

---

### **Card 5.2**

**Q:** Why is nested CV needed for model selection?  
**A:** To avoid optimistic bias when hyperparameters are tuned on the same folds used for evaluation.

---

# **6. Bootstrap**

### **Card 6.1**

**Q:** What is bootstrap sampling used for?  
**A:** Estimating variance/uncertainty by resampling _with_ replacement.

---

# **7. Bayesian Concepts**

### **Card 7.1**

**Q:** What is Bayes’ rule?  
**A:**  
$$  
p(\theta\mid D)=\frac{p(D\mid\theta)p(\theta)}{p(D)}.  
$$

---

# **8. Variance & Covariance**

### **Card 8.1**

**Q:** What is covariance?  
**A:**  
$$  
\mathrm{Cov}(X,Y) = \mathbb{E}[(X-\mu_X)(Y-\mu_Y)].  
$$

---

# **9. Basis Functions**

### **Card 9.1**

**Q:** Why use basis functions?  
**A:** They transform data to allow linear models to fit nonlinear relationships.

---

# **10. Linear Regression**

### **Card 10.1**

**Q:** What is the normal equation for linear regression?  
**A:**  
$$  
\mathbf{w} = (X^\top X)^{-1}X^\top\mathbf{t}.  
$$

---

# **11. Logistic Regression**

### **Card 11.1**

**Q:** What is the logistic function?  
**A:**  
$$  
\sigma(z)=\frac{1}{1+e^{-z}}.  
$$

---

### **Card 11.2**

**Q:** What is the cross-entropy loss for logistic regression?  
**A:**  
$$  
E=-\sum_n \big[t_n\log y_n + (1-t_n)\log(1-y_n)\big].  
$$

---

# **12. Gradient Descent**

### **Card 12.1**

**Q:** What is the GD update rule?  
**A:**  
$$  
\mathbf{w}\leftarrow\mathbf{w}-\eta\nabla_{\mathbf{w}}E.  
$$

---

# **13. Perceptron**

### **Card 13.1**

**Q:** What is the perceptron decision rule?  
**A:**  
$$  
y = \text{sign}(\mathbf{w}^\top\mathbf{x}).  
$$

---

# **14. Clustering: k-Means**

### **Card 14.1**

**Q:** What does k-Means minimize?  
**A:**  
$$  
\sum_n |x_n - \mu_{z_n}|^2.  
$$

---

# **15. Gaussian Mixture Models (GMM)**

### **Card 15.1**

**Q:** What is a mixture distribution?  
**A:**  
$$  
p(x)=\sum_k \pi_k , \mathcal{N}(x\mid\mu_k,\Sigma_k).  
$$

---

# **16. EM Algorithm**

### **Card 16.1**

**Q:** What is the E-step?  
**A:**  
$$  
\gamma_{nk} =  
\frac{\pi_k p(x_n\mid\theta_k)}  
{\sum_j \pi_j p(x_n\mid\theta_j)}.  
$$

### **Card 16.2**

**Q:** What is the M-step for mixture weights?  
**A:**  
$$  
\pi_k = \frac{1}{N}\sum_n \gamma_{nk}.  
$$

---

# **17. Log-Likelihood (Complete vs Incomplete)**

### **Card 17.1**

**Q:** What is the complete-data log-likelihood for mixtures?  
**A:**  
$$  
\ln p(X,Z\mid\theta)=\sum_{n,k} z_{nk}(\ln\pi_k + \ln p(x_n\mid\theta_k)).  
$$

---

# **18. Neural Networks (Forward + Backward)**

### **Card 18.1**

**Q:** What is the forward propagation rule for a hidden unit?  
**A:**  
$$  
h = \sigma(Wx + b).  
$$

---

### **Card 18.2**

# **Q:** Give the chain rule expansion for $\frac{\partial E}{\partial w_1}$.  
**A:**  
$$  
\frac{\partial E}{\partial w_1}

\frac{\partial E}{\partial y}  
\cdot  
\frac{\partial y}{\partial h}  
\cdot  
\frac{\partial h}{\partial w_1}.  
$$

---

# **19. Hidden Units & Early Stopping**

### **Card 19.1**

**Q:** Why can too many hidden units cause overfitting?  
**A:** Model capacity increases → memorizes training data.

---

# **20. Autoencoders**

### **Card 20.1**

**Q:** What is the objective of an autoencoder?  
**A:**  
$$  
E = \frac12 |x - \hat{x}|^2.  
$$

---

# **21. PCA**

### **Card 21.1**

**Q:** How is PCA computed?  
**A:**

1. Compute covariance matrix
    
2. Take its eigenvectors
    
3. Use top eigenvectors as principal components
    

---

# -----------------------------------------------------

# **📌 PART C — WORKED NUMERICAL EXAMPLES (WITH MATH REQUIREMENTS)**

# -----------------------------------------------------

# **Example 1 — Logistic Function**

### **Formula**

$$  
\sigma(z)=\frac{1}{1+e^{-z}}  
$$

### **Requires knowledge of:**

- Exponentials
    
- Fractions
    
- Basic nonlinear functions
    

### **Example**

Compute $\sigma(2)$:

$$  
\sigma(2)=\frac{1}{1+e^{-2}} \approx \frac{1}{1+0.1353}  
=0.881  
$$

### **Interpretation**

Input $z=2$ corresponds to an 88% probability of class 1.

---

# **Example 2 — Gradient Descent Step**

### **Formula**

$$  
w_{\text{new}} = w_{\text{old}} - \eta \frac{dE}{dw}  
$$

### **Requires knowledge of:**

- Derivatives
    
- Multiplying scalars
    
- Gradient descent meaning
    

### **Example**

Let:

- $w=3$
    
- Gradient = $2$
    
- Learning rate $\eta=0.1$
    

Then:  
$$  
w_{\text{new}} = 3 - 0.1 \cdot 2 = 2.8  
$$

---

# **Example 3 — k-Means Assignment Step**

### **Objective**

Assign each point to the nearest centroid.

### **Requires knowledge of:**

- Euclidean distance
    
- Squared difference
    

### **Example**

Point: $x = 5$  
Centroids: $\mu_1=2$, $\mu_2=8$

Distances:

- To $\mu_1$: $(5-2)^2 = 9$
    
- To $\mu_2$: $(5-8)^2 = 9$
    

Tie → either cluster is acceptable.


​

---

## 🔍 **Requires knowledge of:**

- **Euclidean distance**
    
    $∥x−μ∥=(x−μ)2\|x - \mu\| = \sqrt{(x - \mu)^2}∥x−μ∥=(x−μ)2​$
- **Squared distance (used in k-Means because it removes the square root)**
    
    $(x−μ)2(x - \mu)^2(x−μ)2$
- **Minimization**  
    Select the smallest value.
---

# **Example 4 — PCA Variance Explained**

### **Requires knowledge of:**

- Eigenvalues
    
- Variance
    

If eigenvalues are: `[5, 2, 1]`  
Total variance = 8

First two PCs explain:  
$$  
\frac{5+2}{8} = 0.875 = 87.5%  
$$

---

# **Example 5 — EM E-Step for GMM**

### **Requires knowledge of:**

- Bayes rule
    
- Gaussian PDF
    
- Fractions and normalization
    

### **Scenario**

Two clusters, equal weights $\pi_1=\pi_2=0.5$.

Likelihoods:

- $p(x|1)=0.2$
    
- $p(x|2)=0.8$
    

### **Responsibility**

$$  
\gamma_1 = \frac{0.5\cdot0.2}{0.5\cdot0.2 + 0.5\cdot0.8}  
= \frac{0.1}{0.5} = 0.2  
$$  
$$  
\gamma_2 = 0.8  
$$

Interpretation: 20% cluster 1, 80% cluster 2.

---

# **Example 6 — Autoencoder Reconstruction Error**

### **Requires knowledge of:**

- Vector subtraction
    
- Norms
    

### **Example**

Input $x=(1,2)$, reconstruction $\hat{x}=(0.9, 2.1)$  
Error:  
$$  
|x-\hat{x}|  
= \sqrt{(1-0.9)^2 + (2-2.1)^2}  
= \sqrt{0.01 + 0.01}  
= \sqrt{0.02}  
= 0.141  
$$

---

# **Example 7 — Linear Regression Prediction**

### **Requires knowledge of:**

- Dot product
    

Let:  
$\mathbf{x}=(2,3)$,  
$\mathbf{w}=(1,4)$,  
$b=2$

$$  
y = 1\cdot2 + 4\cdot3 + 2 = 16  
$$

---

# Extra Terms

### Linear Discriminant


To optimize the training objective: $\frac{\partial}{\partial w_i}L(\mathbf{w})=\frac{1}{\sigma^2}\sum^N_{n=1}(t_n-\mathbf{w}\cdot\mathbf{\phi}(\mathbf{x_n}))\phi_i(x_n)=0$
The error function to minimize:
$E(w):=\frac{1}{2}\sum^N_{n=1}(t_n-w\cdot\phi(x_n))^2$

The gradient of the training objective:
$\nabla E(w):=-\sum^N_{n=1}(t_n-w\cdot\phi(x_n))^2$
---

Data: $(x_1,y_1)=(1,2)$, $(x_2,y_2)=(2,2)$, so $n=2$.  
Start $m=0$, $b=0$.



---
From https://www.geeksforgeeks.org/machine-learning/gradient-descent-in-linear-regression/
1. Calculate the cost function using MSE: $J(m,b)=n1​∑i=1n​(yi​−(mxi​+b))^2$
2. compute the gradient for slope m: $∂m∂J​=−n2​∑i=1n​xi​(yi​−(mxi​+b))$
3. compute the gradient for intercept b: $∂J​=−n2​∑i=1n​(yi​−(mxi​+b))$

---

Examples of basis functions:

- Polynomial: $\phi(x)=[1,x,x^2]^\top$
    
- Gaussian/RBF: $\phi_k(x)=\exp!\left(-\frac{(x-\mu_k)^2}{2\sigma^2}\right)$
    
- “Intercept” is just $\phi_0(x)=1$


# Foundations of a Linear Regression Model

$$y(w,x)=wx+b$$
	taken as the core structure of model

$$\begin{align}1.&& y(w,x)=\begin{bmatrix}w &b\end{bmatrix}\begin{bmatrix}x&1\end{bmatrix}\end{align}$$
	with w and x as row vectors

**But this not a usable model due to the laws of linear algebra**

$$\begin{align}2.&& \mathbf{w}\in\mathbb{R}^{M\times1}, \phi(x)\in\mathbb{R}^{M\times1}, \mathbf{w}^T\in \mathbb{R}^{1\times M} \\ &&\text{where}\ \phi(x)=\begin{bmatrix}x_1\\.\\.\\x_d\\1\end{bmatrix}\mathbf{w}=\begin{bmatrix}w_1\\.\\.\\w_d\\b\end{bmatrix}\end{align}$$
	using Bishop's convention for structuring the model parameters, set both as row vectors

$$\begin{align}3. && y=(x,w)=\begin{bmatrix}x_1\\.\\.\\x_d\\1\end{bmatrix}\begin{bmatrix}w_1\\.\\.\\w_d\\b\end{bmatrix} \end{align} 1,2$$
**We can perform a transposition of one of the vectors to get a clean output**

$$\begin{align}4. \begin{bmatrix}x_1\\.\\.\\x_d\\1\end{bmatrix}\begin{bmatrix}w_1\\.\\.\\w_d\\b\end{bmatrix}^T=\phi(x)w^T \\\\ y(w,x)=\mathbf{w}^T\phi(x) &&3\end{align}$$

	where phi is a basis function (sigmoid/tanh/etc.), and we get a clean map to feature vectors

$$\begin{align}5. \Phi=\begin{bmatrix}\phi_1(x_1) & \phi_2(x_1) ...&\phi_{M-1}(x_1)\\\phi_1(x_2) & .&.\\\phi_1(x_N) & ...&\phi_{M-1}(x_N) \end{bmatrix}&& 4\end{align}$$
	based on the rule of transposition

$$\begin{align}6. && y=\mathbf{w}\Phi &&4,5\end{align}$$
	is the core model

$$\begin{align}7. && y(w,x)=\sum^N\mathbf{w}^T\Phi && 6\end{align}$$
	making the model linear in its parameters


**Assume we have some data**
$$\begin{align*}D=\{d_1,d_2,...,d_n\}\end{align*}$$
	where d are the data values

$$8. \ \ \hat y=y(w,x_n)=w^T\phi(x_n)$$
	and we make a prediction

$$\begin{align}9.&& r_n=\hat y_n -y(x_n,w)= \hat y - w^T\phi(x_n)&& 8\end{align}$$
	calculate the residual error on the sample

**At this point, we need to justify how the loss is calculated. We do so with Gaussian Noise**

$$\begin{align}10.&& \ E(w)=\frac{1}{2}\sum^N_{n=1}(t_n-w^T\phi(x_n))^2&& \end{align}$$
	start with the squared error function

$$\begin{align}11.&& \underset{w}{argmin}E(w)=\frac{1}{2}\sum^N_{n=1}(t_n-w^T\phi(x_n))^2&& 10\end{align}$$

$$\begin{align} 12. \ \ t_n=w^T\phi(x_n)+\epsilon_n, \\ \epsilon_n \sim \mathcal{N}(0,\sigma^2) \ i.i.d.\end{align}$$
	make the assumption that there are noisy versions of the prediction

$$\begin{align}13.&&p(t_n|x_n, w)=\mathcal{N}(t_n|w^T\phi(x_n),\sigma^2&&12\end{align}$$
	we can rewrite as the conditional density

$$\begin{align}14.&& p(t_n|w)=\prod^N_{n=1}p(t_n|w^T\phi(x_n), \sigma^2)&& 13\end{align}$$
	which can be re-written as a product of probabilities, given i.i.d

$$\begin{align}15. && \ln p(t_n|w)=\sum^N_{n=1}\ln p(t_n|x_n,w)&&14\end{align}$$
	borrowing from the rule of logarithms, we can change the products into sums

$$\begin{align}16.&& \ln p(t|w)=c-\frac{1}{2\sigma^2}\sum^N_{n=1}(t_n-w^T\phi(x_n))^2&& 14 \end{align}$$

$$\begin{align}17. \mathbf{W}_{ML}=\underset{w}{argmax} \ln p(t|w)=\underset{w}{argmin}\sum^N_{n=1}(t_n-w^T\phi(x_n))^2&& 15\end{align}$$

	demonstrating that maximizing the log-likelihood is equivalent to minimizing the sum of squred errors.

**Our next step is to optimize, i.e., how to readjust the parameters that minimizes loss.**
***note:*** **derive the gradient and set it to zero** → _closed-form solution (normal equations)_
***note*** **derive the gradient and use it in an iterative update** → _gradient descent / SGD_
    


$$\begin{align}18. && \mathcal{L}(\mathbf{w})  
\propto  
\sum_{n=1}^N \left(t_n-\mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x}_n)\right)^2  
&& 16\end{align}$$
	SSE re-written with assumed Gaussian noise

$$  
\Phi_{n j}=\phi_j(\mathbf{x}_n),  
\quad  
\Phi\in\mathbb{R}^{N\times M},  
\quad  
\mathbf{t}\in\mathbb{R}^{N\times 1},  
\quad  
\mathbf{w}\in\mathbb{R}^{M\times 1}.  
$$


$$\begin{align}19.&&
\mathbf{y}=\Phi\mathbf{w}.  
&& 6\end{align}$$
	with predictions  

$$  \begin{align}20. &&
E(\mathbf{w})=\frac{1}{2}\lVert \mathbf{t}-\Phi\mathbf{w}\rVert^2&& 9  \end{align}
$$
	and with squared error


$$  \begin{align}21. &&
\nabla E(\mathbf{w})=\Phi^\top(\Phi\mathbf{w}-\mathbf{t}).  \end{align}
$$
	KEY: compute the gradient 

Equivalently:  
$$  
\nabla E(\mathbf{w})=-\Phi^\top(\mathbf{t}-\Phi\mathbf{w}).  
$$

**The issue with optimizing is that it is prone to OVERFITTING. 
(overfitting understood as  resultant complex model with High Variance and poor performance on test data) overfitting**

**So, we add a Regularization term to control it**

*L2 Ridge*
$$  
E_{\text{ridge}}(\mathbf{w})

\frac{1}{2}\lVert \mathbf{t}-\Phi\mathbf{w}\rVert^2  
+\frac{\lambda}{2}\lVert \mathbf{w}\rVert^2  
$$
	where lambda is the regularization parameter

$$  
\nabla E_{\text{ridge}}(\mathbf{w})=\Phi^\top(\Phi\mathbf{w}-\mathbf{t})+\lambda \mathbf{w}. 
$$
	then calculate the gradient of E

$$  
(\Phi^\top\Phi+\lambda I)\mathbf{w}=\Phi^\top\mathbf{t}  
\quad\Rightarrow\quad  
\mathbf{w}=(\Phi^\top\Phi+\lambda I)^{-1}\Phi^\top\mathbf{t}.  
$$
	with closed-form (regularized normal equations)

$$  
\Phi^\top(\Phi\mathbf{w}-\mathbf{t})=\mathbf{0}  
$$
	set gradient to zero


$$  
\Phi^\top\Phi,\mathbf{w}=\Phi^\top\mathbf{t}.  
$$
	rearrange to get the least squares / MLE solution for linear-in-parameters regression.

**We could add an L1 penalty**  
$$  
E_{\text{lasso}}(\mathbf{w})

\frac{1}{2}\lVert \mathbf{t}-\Phi\mathbf{w}\rVert^2  
+\lambda\lVert \mathbf{w}\rVert_1,  
\qquad  
\lVert \mathbf{w}\rVert_1=\sum_i |w_i|  
$$
	with many weights set to 0, which does not provide a simple closed-form solution, because $|w_i|$ is not differentiable at $0$

# Worked Example

### Data (tiny)

Two training points:  
$$  
(x_1,t_1)=(1,2),\qquad (x_2,t_2)=(2,3).  
$$
### Step 1: Model

$$  
y(x;m,b)=mx+b  
$$

(Here parameters are $\theta=(m,b)$.)

### Step 2: Loss (sum-of-squares with Bishop’s $\tfrac12$)

#### Residuals:  
$$\begin{align}  
r_1= t_1-y(x_1)=2-(m\cdot 1+b)=2-m-b  
\end{align}$$  
$$  
r_2=t_2-y(x_2)=3-(m\cdot 2+b)=3-2m-b  
$$

#### Error function:  
$$  
E(m,b)=\frac12\left[r_1^2+r_2^2\right]  
=\frac12\left( (2-m-b)^2+(3-2m-b)^2\right)  
$$

---

### Step 3: Compute the gradient (partials)

#### Partial $\frac dd$ (m)

- $\frac{\partial r_1}{\partial m}=-1$
    
- $\frac{\partial r_2}{\partial m}=-2$
    

So:  
$$  
\frac{\partial E}{\partial m} r_1(-1)+r_2(-2)

-(2-m-b)-2(3-2m-b)  
$$

Simplify:  
$$  
\frac{\partial E}{\partial m} -(2-m-b)-6+4m+2b
(-8)+5m+3b  
$$

So:  
$$  
\boxed{\frac{\partial E}{\partial m}=5m+3b-8}  
$$

#### Partial $\frac dd$ (b)

- $\frac{\partial r_1}{\partial b}=-1$
    
- $\frac{\partial r_2}{\partial b}=-1$
    

So:  
$$  
\frac{\partial E}{\partial b} r_1(-1)+r_2(-1)
-(2-m-b)-(3-2m-b)  
$$

Simplify:  
$$  
\frac{\partial E}{\partial b}

-5+3m+2b  
$$

So:  
$$  
\boxed{\frac{\partial E}{\partial b}=3m+2b-5}  
$$

if $\frac12r^2$, then $\frac{\partial E}{\partial \theta}=r\frac{\partial r}{\partial \theta}$

**Partial derivative with respect to m**

Treat b as a constant when differentiating w.r.t. m:

- $\frac{\partial}{\partial m}(3m)=3$ (because $\frac{\partial}{\partial m}(m)=1$ and constant multiples pull out)
    
- $\frac{\partial}{\partial m}(2b)=0$ (no m in that term)
    
- $\frac{\partial}{\partial m}(-5)=0$ (constant) 

Thus, $\frac{\partial}{\partial m}(3m+2b-5)=3$

 **Partial derivative with respect to b**

Now treat m as a constant when differentiating w.r.t. b:

- $\frac{\partial}{\partial b}(3m)=0$ (no b in that term)
    
- $\frac{\partial}{\partial b}(2b)=2$ (because $\frac{\partial}{\partial b}(b)=1$)
    
- $\frac{\partial}{\partial b}(-5)=0$

Thus, $\frac{\partial}{\partial b}(3m+2b-5)=2$

---

### Step 4A: Set gradient to zero (closed-form solution)

#### Solve:  
$$  
5m+3b-8=0  
$$  
$$  
3m+2b-5=0  
$$

Rewrite:  
$$  
5m+3b=8 \quad (1)  
$$  
$$  
3m+2b=5 \quad (2)  
$$

Eliminate $b$.

Multiply (2) by 3:  
$$  
9m+6b=15  
$$

Multiply (1) by 2:  
$$  
10m+6b=16  
$$

Subtract:  
$$  
(10m+6b)-(9m+6b)=16-15  
\Rightarrow m=1  
$$

Plug into (2):  
$$  
3(1)+2b=5 \Rightarrow 2b=2 \Rightarrow b=1  
$$

✅ Closed-form optimum:  
$$  \boxed{m^{*}=1,\quad b^{*}=1}  $$

**Check**: $y(1)=2$, $y(2)=3$ fits perfectly.

---

### Step 4B: Gradient descent (do it by hand)

Update rule:  
$$  
m_{k+1}=m_k-\eta\frac{\partial E}{\partial m}(m_k,b_k)  
$$  
$$  
b_{k+1}=b_k-\eta\frac{\partial E}{\partial b}(m_k,b_k)  
$$

#### Choose a simple start and learning rate:  
$$  
(m_0,b_0)=(0,0),\qquad \eta=0.1  
$$

#### Compute gradient at ((0,0))

$$  
\frac{\partial E}{\partial m}(0,0)=5(0)+3(0)-8=-8  
$$  
$$  
\frac{\partial E}{\partial b}(0,0)=3(0)+2(0)-5=-5  
$$

Update:  
$$  
m_1=0-0.1(-8)=0.8  
$$  
$$  
b_1=0-0.1(-5)=0.5  
$$

So after 1 step:  
$$  
(m_1,b_1)=(0.8,0.5)  
$$

#### Do one more step (optional but clarifying)

Gradient at $(0.8,0.5)$:  
$$  
\frac{\partial E}{\partial m}=5(0.8)+3(0.5)-8=4+1.5-8=-2.5  
$$  
$$  
\frac{\partial E}{\partial b}=3(0.8)+2(0.5)-5=2.4+1-5=-1.6  
$$

Update:  
$$  
m_2=0.8-0.1(-2.5)=1.05  
$$  
$$  
b_2=0.5-0.1(-1.6)=0.66  
$$

Now:  
$$  
(m_2,b_2)=(1.05,0.66)  
$$

You can see it moving toward $(1,1)$.

**Common confusion:** If $\eta$ is too big, you overshoot and bounce; too small, you crawl.

---

### Summary (what you should remember)

- Loss:  
    $$  
    E(m,b)=\frac12\sum_{n=1}^N (t_n-(mx_n+b))^2  
    $$
    
- Gradients:  
    $$  
    \frac{\partial E}{\partial m}=\sum_{n=1}^N (mx_n+b-t_n)x_n  
    $$  
    $$  
    \frac{\partial E}{\partial b}=\sum_{n=1}^N (mx_n+b-t_n)  
    $$  
    (These are the same as what we computed above, just in general form.)
    
- Closed-form: solve $\nabla E=0$.
    
- GD: update $(m,b)$ by subtracting $\eta$ times the gradient.
# Bias-Variance Decomposition
![[Pasted image 20251229192535.png|500]]
![[Pasted image 20251229192545.png|500]]
![[Pasted image 20251229192617.png]]
![[Pasted image 20251229192626.png]]
![[Pasted image 20251229192632.png]]

# Linear Models for Classification

Awesome — Part B is basically: **same training pipeline as regression**, but with (i) **discrete targets**, and (ii) a **Bernoulli likelihood** instead of Gaussian. That one change forces a different loss (cross-entropy) and a different link function (sigmoid).

---

# Part B1: What changes when you move from regression → classification?

## Target type changes

- Regression: $t\in\mathbb{R}$ (continuous)
    
- Binary classification: $t\in{0,1}$
    

## Likelihood changes (this is the real pivot)

- Regression (Gaussian noise):  
    $$p(t\mid \mathbf{x},\mathbf{w})=\mathcal{N}(t\mid y(\mathbf{x},\mathbf{w}),\sigma^2)$$
    
- Classification (Bernoulli):  
    $$p(t\mid \mathbf{x},\mathbf{w})=\text{Bernoulli}(t\mid \pi(\mathbf{x}))$$
    

So now the model must output a **probability** $\pi(\mathbf{x})\in(0,1)$.

---

# Part B2: Logistic regression model

## Step 1: Linear “score” (same as regression)

Define features (basis functions) $\boldsymbol{\phi}(\mathbf{x})$ and a linear score:  
$$  
a(\mathbf{x})=\mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x})  
$$

## Step 2: Sigmoid turns score into a probability

$$  
\pi(\mathbf{x}) = p(t=1\mid \mathbf{x},\mathbf{w})=\sigma(a)=\frac{1}{1+e^{-a}}  
$$

Then:  
$$  
p(t=0\mid \mathbf{x},\mathbf{w})=1-\pi(\mathbf{x})  
$$

**Common confusion:** logistic regression is “linear” in the parameters $\mathbf{w}$ at the score level $a=\mathbf{w}^\top\phi(\mathbf{x})$, but the final mapping to probability is nonlinear via $\sigma(\cdot)$.

---

# Part B3: Decision boundary

Predicted class is usually:  
$$  
\hat t =  
\begin{cases}  
1 & \text{if } \pi(\mathbf{x})\ge 0.5\  
0 & \text{otherwise}  
\end{cases}  
$$

Since $\sigma(a)\ge 0.5 \iff a\ge 0$, the boundary is:  
$$  
\mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x})=0  
$$

So the **decision boundary is linear in feature space**.

---

# Part B4: Likelihood, log-likelihood, NLL (cross-entropy)

For one data point $(\mathbf{x}_n,t_n)$ with $t_n\in{0,1}$, define:  
$$  
a_n=\mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x}_n),\qquad \pi_n=\sigma(a_n)  
$$

## Bernoulli likelihood for one point

$$  
p(t_n\mid \mathbf{x}_n,\mathbf{w})=\pi_n^{t_n}(1-\pi_n)^{1-t_n}  
$$

## Likelihood for i.i.d. dataset

$$  
p(\mathbf{t}\mid \mathbf{w})=\prod_{n=1}^N \pi_n^{t_n}(1-\pi_n)^{1-t_n}  
$$

## Log-likelihood (product → sum)

$$  
\ln p(\mathbf{t}\mid \mathbf{w})  
=\sum_{n=1}^N\left[t_n\ln \pi_n+(1-t_n)\ln(1-\pi_n)\right]  
$$

## Negative log-likelihood (this is the loss)

$$  
\mathcal{L}(\mathbf{w})=-\ln p(\mathbf{t}\mid \mathbf{w})  
=-\sum_{n=1}^N\left[t_n\ln \pi_n+(1-t_n)\ln(1-\pi_n)\right]  
$$

That is exactly **cross-entropy loss** for binary classification.

---

# Part B5: Gradient intuition (“error times features” still shows up)

You need one derivative fact:

$$  
\sigma'(a)=\sigma(a)(1-\sigma(a))  
$$

The magic result (PRML-standard) is:

# $$  
\boxed{  
\nabla \mathcal{L}(\mathbf{w})

\sum_{n=1}^N(\pi_n-t_n),\boldsymbol{\phi}(\mathbf{x}_n)  
}  
$$

# Compare with least squares gradient:  
$$  
\nabla E(\mathbf{w})=  
-\sum_{n=1}^N(t_n-\mathbf{w}^\top\phi(\mathbf{x}_n)),\phi(\mathbf{x}_n)

\sum_{n=1}^N(\hat y_n-t_n),\phi(\mathbf{x}_n)  
$$

So the pattern is the same:

- regression “error”: $(\hat y_n-t_n)$
    
- logistic “error”: $(\pi_n-t_n)$ (probability error)
    

Then gradient descent is:  
$$  
\mathbf{w}\leftarrow \mathbf{w}-\eta \nabla\mathcal{L}(\mathbf{w})  
$$

**Common confusion:** logistic regression does _not_ use squared error as the “natural” loss; NLL/cross-entropy comes directly from the Bernoulli likelihood and gives nicer optimization behavior.

---

## A “handwritten” example you can compute (one GD step)

Use one feature + bias:  
$$  
\boldsymbol{\phi}(x)=\begin{bmatrix}x\1\end{bmatrix},  
\quad  
\mathbf{w}=\begin{bmatrix}w_1\w_0\end{bmatrix},  
\quad  
a=w_1x+w_0  
$$

Data:  
$$  
(x_1,t_1)=(0,0),\qquad (x_2,t_2)=(1,1)  
$$

Start at $\mathbf{w}=\begin{bmatrix}0\0\end{bmatrix}$.

### Step 1: compute probabilities

For $x_1=0$:  
$$  
a_1=0,\quad \pi_1=\sigma(0)=0.5  
$$  
For $x_2=1$:  
$$  
a_2=0,\quad \pi_2=\sigma(0)=0.5  
$$

### Step 2: compute gradient

$$  
\nabla\mathcal{L}(\mathbf{w})=\sum_{n=1}^2(\pi_n-t_n)\phi(x_n)  
$$

Compute each term:

- For $n=1$:  
    $$  
    (\pi_1-t_1)=0.5-0=0.5,\quad \phi(x_1)=\begin{bmatrix}0\1\end{bmatrix}  
    \Rightarrow 0.5\phi(x_1)=\begin{bmatrix}0\0.5\end{bmatrix}  
    $$
    
- For $n=2$:  
    $$  
    (\pi_2-t_2)=0.5-1=-0.5,\quad \phi(x_2)=\begin{bmatrix}1\1\end{bmatrix}  
    \Rightarrow -0.5\phi(x_2)=\begin{bmatrix}-0.5\-0.5\end{bmatrix}  
    $$
    

#### Sum:  
$$  
\nabla\mathcal{L}(\mathbf{w})

\begin{bmatrix}0\\0.5\end{bmatrix}  
+  
\begin{bmatrix}-0.5\\-0.5\end{bmatrix}

\begin{bmatrix}-0.5\\0\end{bmatrix}  
$$

### Step 3: gradient descent update

 **Pick $\eta=1$ (just to see movement clearly):**  
$$  
\mathbf{w}_{new}=\mathbf{w}-\eta\nabla\mathcal{L}

 \begin{bmatrix}0&0\end{bmatrix}

 \begin{bmatrix}-0.5&0\end{bmatrix}

\begin{bmatrix}0.5&0\end{bmatrix}  
$$

Now the model has $w_1>0$, so it pushes $x=1$ toward class 1 more strongly than $x=0$.

---

##### What to do next (your “Part B” roadmap)

1. Get comfortable with:
    
    - $a=\mathbf{w}^\top\phi(\mathbf{x})$
        
    - $\pi=\sigma(a)$
        
    - $\mathcal{L}(\mathbf{w})$ as Bernoulli NLL
        
2. Memorize/derive the key gradient:  
    $$  
    \nabla \mathcal{L}(\mathbf{w})=\sum(\pi_n-t_n)\phi(\mathbf{x}_n)  
    $$        




















---
***COMPARE TOPIC PROGRESS TO BELOW ONLY***


Here’s an order that “clicks together” conceptually and mathematically, so each topic sets up the next.

---

## A. The supervised-learning spine (models + objectives + training)

1. **Supervised learning**  
    (Defines the setting: labeled data, prediction, evaluation.)
    
2. **Optimization**  
    (Training is posed as minimizing an objective.)
    
3. **Gradient descent**  
    (The default tool to optimize most objectives.)
    
4. **Maximum Likelihood Principle**  
    (Connects probability to training: “fit parameters that make the observed data most likely.”)
    
5. **Linear regression**  
    (First full worked example of MLE: Gaussian noise ⇒ squared error.)
    
6. **Classification**  
    (Changes the target type: discrete labels instead of real values.)
    
7. **Non-Probabilistic Discriminative Models**  
    (Classify via a score + threshold; no probabilities yet.)
    
8. **The Perceptron**  
    (The canonical non-probabilistic discriminative algorithm; introduces mistake-driven updates.)
    
9. **Probabilistic Discriminative Models**  
    (Model (p(t\mid x)) directly; training via log-likelihood / cross-entropy.)
    
10. **Logistic regression**  
    (The canonical probabilistic discriminative model; sigmoid, decision boundary, NLL, gradients.)
    
11. **Probabilistic Generative Models**  
    (Model $(p(x,t)=p(x\mid t)p(t))$; classify via Bayes rule (bridges to latent-variable thinking later).)
    

---

## B. The unsupervised / latent-variable spine (structure discovery + EM)

12. **Clustering and K-Means Algorithm**  
    (First unsupervised method; introduces “cluster assignments” and alternating optimization intuition.)


13. **Latent Variables**  
    (Formalizes hidden/unknown assignments (e.g., which cluster generated each point).)
    
14. **Gaussian Mixture Models (GMMs)**  
    (K-means becomes “soft” clustering with probabilities; introduces mixture weights, means, covariances.)
    
15. **Complete Data Likelihood**  
    (What the likelihood would look like if the latent assignments were known.)
    
16. **Incomplete Data Likelihood**  
    (What you actually have: latent variables summed out; becomes harder to optimize directly.)
    
17. **Expectation–Maximization (EM)**  
    (The general method for MLE with latent variables; E-step computes responsibilities, M-step updates parameters.)
    

---

## C. Neural networks + representation learning (connects back to both spines)

18. **Feed-Forward Networks**  
    (Generalizes “linear score → nonlinearity” and trains with gradient descent on NLL/cross-entropy.)
    
19. **Unsupervised and Self-Taught Learning**  
    (Uses unlabeled data to learn representations; connects unsupervised objectives to better supervised performance.)
    
20. **Autoencoding**  
    (A core unsupervised/self-taught method: learn an encoding that reconstructs inputs; conceptually related to latent-variable models and to feature learning for discriminative tasks.)
    

---

### What you’ll notice as the “thread”

- **Optimization + GD** keeps showing up everywhere (regression, logistic regression, neural nets).
    
- **MLE + (log-)likelihood** explains why the “right” loss changes (MSE vs cross-entropy) and sets up EM.
    
- **Latent variables** are the bridge from k-means → GMM → EM.
    
- **Autoencoders** are another way to learn latent representations, but trained via backprop rather than EM.
