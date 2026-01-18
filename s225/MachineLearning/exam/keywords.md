

1. explain process of k-fold cross validation
2. ho do you derive a closed form solution for MLE and EM algorithm?
3. 








---
## 1) Weights

### Informal definition

Numbers that tell the model **how strongly** each input feature influences the prediction.

### Formal definition

In linear regression, $\mathbf{w}\in\mathbb{R}^d$ and  
$$\hat y=\mathbf{w}^\top\mathbf{x}+b.$$

### Example

If “bedrooms” has a larger weight than “age”, the model changes more when bedrooms changes.

### Mathematics behind it

Requires: **dot product**  
$$\mathbf{w}^\top\mathbf{x}=\sum_{i=1}^d w_ix_i.$$

### Hand-worked example

Let $\mathbf{w}=\begin{bmatrix}1&2\end{bmatrix}$, $b=0$.  
For $\mathbf{x}^{(1)}$:  
$$\hat y^{(1)}=1\cdot 1+2\cdot 2=5.$$

---

## 2) Bias

### Informal definition

A constant offset: what the model predicts when inputs are “zero”.

### Formal definition

$b\in\mathbb{R}$ in  
$$\hat y=\mathbf{w}^\top\mathbf{x}+b.$$

### Example

House prices might have a baseline even if all “features” were zero.

### Mathematics behind it

Bias shifts predictions without changing slopes.

### Hand-worked example

With $\mathbf{w}=\begin{bmatrix}1&2\end{bmatrix}$ and $b=3$:  
$$\hat y^{(1)}=5+3=8.$$

---

## 3) Parameters

### Informal definition

All the knobs the model can tune during learning.

### Formal definition

Parameters are typically denoted $\theta$. For linear regression:  
$$\theta=(\mathbf{w},b).$$

### Example

Neural nets: parameters are all weights and biases across layers.

### Mathematics behind it

Training means choosing $\theta$ to minimize a loss:  
$$\theta^*=\arg\min_\theta \mathcal{L}(\theta).$$

### Hand-worked example

If $\theta=(\mathbf{w},b)$ and you change $b$ from $0$ to $1$, $\theta$ changed.

---

## 4) Model / Type

### Informal definition

The rule that maps inputs to outputs (e.g., linear, tree, neural net).

### Formal definition

A model is a function class: $$f_\theta:\mathcal{X}\to\mathcal{Y}.$$

### Example

Linear regression: $f_\theta(\mathbf{x})=\mathbf{w}^\top\mathbf{x}+b$.

### Mathematics behind it

Choosing the model type chooses the _functional form_ you can represent.

### Hand-worked example

If you choose linear: outputs are planes/hyperplanes in feature space.

---

## 5) Train

### Informal definition

Adjust parameters using data so predictions improve.

### Formal definition

Given data ${(\mathbf{x}^{(n)},y^{(n)})}_{n=1}^N$, training solves:  
$$\theta^*=\arg\min_\theta \frac{1}{N}\sum_{n=1}^N \ell\big(f_\theta(\mathbf{x}^{(n)}),y^{(n)}\big).$$

### Example

Gradient descent updates weights to reduce mean squared error.

### Mathematics behind it

Requires: sums, derivatives/gradients.

### Hand-worked example

Compute loss on sample (see “Error function”), then update $\theta$ to reduce it.

---

## 6) Error function (Loss function)

### Informal definition

A number that measures “how wrong” the model is on data.

### Formal definition

Per-sample loss $\ell(\hat y,y)$, e.g. squared error:  
$$\ell(\hat y,y)=(\hat y-y)^2.$$

### Example

Classification often uses cross-entropy; regression often uses MSE.

### Mathematics behind it

Requires: basic algebra; for optimization, derivatives.

### Hand-worked example

For sample 2 with $\mathbf{w}=\begin{bmatrix}1&2\end{bmatrix}$, $b=0$:  
$$\hat y^{(2)}=1\cdot2+2\cdot0=2,$$  
$$\ell^{(2)}=(2-4)^2=4.$$

---

## 7) Training objective

### Informal definition

The exact thing you’re trying to minimize (or maximize) during training.

### Formal definition

Empirical risk (average loss), optionally plus regularization:  
$$\mathcal{L}(\theta)=\frac{1}{N}\sum_{n=1}^N \ell\big(f_\theta(\mathbf{x}^{(n)}),y^{(n)}\big)+\lambda \Omega(\theta).$$

### Example

Ridge regression uses $\Omega(\theta)=\lVert\mathbf{w}\rVert_2^2$.

### Mathematics behind it

Requires: sums, norms, and “argmin”.

### Hand-worked example

If $N=2$ and $\Omega=0$:  
$$\mathcal{L}=\frac{1}{2}\big(\ell^{(1)}+\ell^{(2)}\big).$$

---

## 8) Model classes

### Informal definition

A _set_ of models you’re allowed to choose from.

### Formal definition

A hypothesis class $\mathcal{H}={f_\theta:\theta\in\Theta}$.

### Example

“All linear functions in $\mathbb{R}^d$” is one model class; “all depth-3 decision trees” is another.

### Mathematics behind it

Generalization depends on the capacity/complexity of $\mathcal{H}$.

### Hand-worked example

Linear class in 1D: $\mathcal{H}={x\mapsto wx+b\mid w,b\in\mathbb{R}}$.

---

## 9) Optimization

### Informal definition

The procedure for finding parameters that minimize the objective.

### Formal definition

Solve:  
$$\min_\theta \mathcal{L}(\theta).$$

### Example

Gradient descent, SGD, Adam.

### Mathematics behind it

Requires: derivative/gradient.  
Gradient descent update:  
$$\theta_{t+1}=\theta_t-\eta\nabla_\theta \mathcal{L}(\theta_t).$$

### Hand-worked example

If $\theta=w$ (1D) and $\mathcal{L}(w)=w^2$, then $\nabla \mathcal{L}=2w$.  
With $w_0=3$, $\eta=0.1$:  
$$w_1=3-0.1(2\cdot 3)=3-0.6=2.4.$$

---

## 10) Model complexity

### Informal definition

How “flexible” the model is (how many patterns it can fit).

### Formal definition

Often tied to degrees of freedom, parameter count, tree depth, VC dimension, norm size, etc.

### Example

A 10th-degree polynomial is more complex than a line.

### Mathematics behind it

More complexity can reduce training error but increase overfitting risk.

### Hand-worked example

Line: $y=wx+b$ (2 parameters).  
Quadratic: $y=ax^2+bx+c$ (3 parameters) → more flexible.

---

## 11) Generalization

### Informal definition

How well the model performs on **new unseen data**.

### Formal definition

Difference between training performance and test/true performance. Ideal goal: low expected risk:  
$$\mathbb{E}_{(\mathbf{x},y)\sim \mathcal{D}}\big[\ell(f_\theta(\mathbf{x}),y)\big].$$

### Example

Great training accuracy but poor validation accuracy = weak generalization.

### Mathematics behind it

Requires: expectation concept.

### Hand-worked example

Training loss small, validation loss large → generalization gap.

---

## 12) Overfit

### Informal definition

Model learns noise/idiosyncrasies of training data; fails on new data.

### Formal definition

Training loss decreases while validation/test loss increases (or stays high).

### Example

Very deep tree perfectly classifies training data but misclassifies new points.

### Mathematics behind it

Often happens when complexity is high relative to data.

### Hand-worked example

If train MSE $=0.1$ but validation MSE $=10$, that’s classic overfit.

---

## 13) Underfit

### Informal definition

Model is too simple to capture the real pattern.

### Formal definition

Both training and validation errors are high.

### Example

Trying to fit a line to a strongly curved relationship.

### Mathematics behind it

Complexity too low or training not converged.

### Hand-worked example

If train MSE $=8$ and validation MSE $=9$, likely underfit.

---

## 14) Regularization

### Informal definition

A penalty that discourages overly complex solutions.

### Formal definition

Add a penalty term:  
$$\mathcal{L}(\theta)=\frac{1}{N}\sum_{n=1}^N \ell(\cdot)+\lambda\Omega(\theta).$$  
Common:

- L2: $\Omega(\theta)=\lVert\mathbf{w}\rVert_2^2=\sum_i w_i^2$
    
- L1: $\Omega(\theta)=\lVert\mathbf{w}\rVert_1=\sum_i |w_i|$
    

### Example

Ridge (L2) shrinks weights; Lasso (L1) can drive some weights to zero.

### Mathematics behind it

Requires: norms, derivatives/subgradients.

### Hand-worked example

If $\mathbf{w}=\begin{bmatrix}1&2\end{bmatrix}$:  
$$\lVert\mathbf{w}\rVert_2^2=1^2+2^2=5.$$

---

## 15) Validation set

### Informal definition

Data held out from training, used to tune choices (hyperparameters, model selection).

### Formal definition

Split dataset into training and validation; train on training, evaluate on validation.

### Example

Pick $\lambda$ for ridge regression by best validation error.

### Mathematics behind it

Not a new formula—just correct experimental design.

### Hand-worked example

If you have 100 samples: 80 train, 20 validation.

---

## 16) Training set

### Informal definition

The data you actually use to fit parameters.

### Formal definition

Training minimizes empirical risk on this set:  
$$\min_\theta \frac{1}{N_{\text{train}}}\sum_{n\in \text{train}}\ell(\cdot).$$

### Example

SGD uses mini-batches drawn from training set.

### Hand-worked example

If $N=100$ and you use 80 for training, $N_{\text{train}}=80$.

---

## 17) k-fold CV (cross-validation)

### Informal definition

Estimate performance by training/evaluating $k$ times on different splits.

### Formal definition

Partition data into $k$ folds. For fold $j$:

- train on $k-1$ folds
    
- validate on fold $j$  
    Average metric over folds.
    

### Example

$k=5$ is common.

### Mathematics behind it

CV estimate:  
$$\text{CV}=\frac{1}{k}\sum_{j=1}^k \text{Metric}_j.$$

### Hand-worked example

If accuracies across 5 folds are $[0.80,0.78,0.81,0.79,0.82]$:  
$$\text{CV}=\frac{0.80+0.78+0.81+0.79+0.82}{5}=0.80.$$

---

## 18) Samples

### Informal definition

Individual data points (observations).

### Formal definition

A dataset is ${(\mathbf{x}^{(n)},y^{(n)})}_{n=1}^N$, each pair is one sample.

### Example

One image + its label is one sample.

### Mathematics behind it

Often assume i.i.d. samples:  
$$ (\mathbf{x}^{(n)},y^{(n)})\overset{i.i.d.}{\sim}\mathcal{D}. $$

### Hand-worked example

Our toy dataset has $N=2$ samples.

---

## 19) Leave-one-out CV (LOOCV)

### Informal definition

Extreme CV where each validation set is exactly 1 sample.

### Formal definition

$k=N$. For each $n$, train on $N-1$ points, validate on the left-out point.

### Example

Useful when data is tiny; computationally expensive for big models.

### Mathematics behind it

LOOCV estimate:  
$$\text{LOOCV}=\frac{1}{N}\sum_{n=1}^N \text{Metric}_{(n)}.$$

### Hand-worked example

If $N=4$, you train 4 times, each time leaving out one different point.

---

## 20) k-NN (k-nearest neighbors)

### Informal definition

Predict based on the labels/values of the $k$ closest training points.

### Formal definition

Given a distance $d(\mathbf{x},\mathbf{x}')$, find neighbor set $\mathcal{N}_k(\mathbf{x})$.

- Classification: majority vote
    
- Regression: average  
    $$\hat y=\frac{1}{k}\sum_{\mathbf{x}^{(i)}\in \mathcal{N}_k(\mathbf{x})} y^{(i)}.$$
    

### Example

$k=3$ classifier for spam based on nearest emails.

### Mathematics behind it

Requires: distance metrics (e.g., Euclidean)  
$$\lVert \mathbf{x}-\mathbf{x}'\rVert_2=\sqrt{\sum_{i=1}^d (x_i-x_i')^2}.$$

### Hand-worked example (1D)

Training: $(1\to A),(3\to A),(10\to B)$. Query $x=4$, $k=2$.  
Distances: $|4-3|=1$, $|4-1|=3$, $|4-10|=6$.  
2-NN labels: $A,A$ → predict $A$.

---

## 21) Maximum Likelihood

### Informal definition

Choose parameters that make the observed data most probable under the model.

### Formal definition

Given likelihood $p(\mathcal{D}\mid\theta)$:  
$$\theta_{\text{ML}}=\arg\max_\theta p(\mathcal{D}\mid\theta).$$  
Often maximize log-likelihood:  
$$\theta_{\text{ML}}=\arg\max_\theta \sum_{n=1}^N \ln p(x^{(n)}\mid\theta).$$

### Example

Fit Gaussian mean/variance to data.

### Mathematics behind it

Requires: logs, derivatives.

### Hand-worked example (Gaussian mean, known variance)

Assume $x^{(n)}\sim \mathcal{N}(\mu,\sigma^2)$ with known $\sigma^2$.  
MLE for $\mu$ is the sample average:  
$$\mu_{\text{ML}}=\frac{1}{N}\sum_{n=1}^N x^{(n)}.$$  
If data $[2,4,6]$:  
$$\mu_{\text{ML}}=\frac{2+4+6}{3}=4.$$

---

## 22) Bootstrap

### Informal definition

Estimate uncertainty/performance by resampling the dataset **with replacement**.

### Formal definition

Create bootstrap datasets $\mathcal{D}^*_b$ by sampling $N$ points from $\mathcal{D}$ with replacement; compute statistic $T(\mathcal{D}^*_b)$; look at its distribution over $b=1,\dots,B$.

### Example

Bootstrap confidence interval for model accuracy.

### Mathematics behind it

Requires: sampling concept; empirical distribution.

### Hand-worked example

If original indices are $[1,2,3,4]$, one bootstrap sample might be:  
$$[2,2,4,1]$$  
(note duplicates allowed, some points missing).

---

## 23) Assumption priors

### Informal definition

The _assumptions you bake in before seeing data_ (often informal, sometimes formal priors).

### Formal definition

Anything that constrains the hypothesis space before data:

- “weights should be small” (often corresponds to L2 prior)
    
- “sparse weights” (often corresponds to L1 prior)
    
- “smooth function” assumptions, etc.
    

### Example

Assuming noise is Gaussian → squared error becomes natural.

### Mathematics behind it

Often corresponds to placing a prior distribution on parameters, e.g.  
$$\mathbf{w}\sim \mathcal{N}(\mathbf{0},\sigma^2 I).$$

### Hand-worked example

“Small weights preferred” means $p(\mathbf{w})$ is larger near $\mathbf{0}$ than far away.

---

## 24) Prior distribution

### Informal definition

A probability distribution over parameters/hypotheses **before** seeing data.

### Formal definition

$$p(\theta).$$

### Example

Coin bias $\theta$ might have prior $\theta\sim \text{Beta}(\alpha,\beta)$.

### Mathematics behind it

Combines with likelihood via Bayes.

### Hand-worked example

If $\theta\in{F,B}$:  
$$p(B)=0.2,\quad p(F)=0.8.$$

---

## 25) Posterior

### Informal definition

Updated belief about parameters/hypotheses **after** seeing data.

### Formal definition

$$p(\theta\mid x).$$

### Example

After observing many heads, posterior shifts toward “coin is biased”.

### Mathematics behind it

Computed by Bayes’ rule.

### Hand-worked example (discrete)

If $p(B\mid x)=0.6$, you now believe 60% chance the coin is biased, given data $x$.

---

## 26) Bayes’ rule (Bayes’ theorem)

### Informal definition

The rule for updating beliefs: **posterior ∝ likelihood × prior**.

### Formal definition

$$p(\theta\mid x)=\frac{p(x\mid\theta)p(\theta)}{p(x)}$$  
with evidence (via sum/integral):  
$$p(x)=\sum_\theta p(x\mid\theta)p(\theta)\quad \text{or}\quad p(x)=\int p(x\mid\theta)p(\theta),d\theta.$$

### Example

MAP estimation:  
$$\theta_{\text{MAP}}=\arg\max_\theta p(\theta\mid x)=\arg\max_\theta \big(\ln p(x\mid\theta)+\ln p(\theta)\big).$$

### Hand-worked example (tiny discrete)

Let $p(B)=0.2$, $p(F)=0.8$, and data $x=$ “HHH”.  
Likelihoods: $p(x\mid B)=0.9^3=0.729$, $p(x\mid F)=0.5^3=0.125$.  
Evidence:  
$$p(x)=0.729\cdot 0.2+0.125\cdot 0.8=0.2458.$$  
Posterior:  
$$p(B\mid x)=\frac{0.729\cdot 0.2}{0.2458}\approx 0.593.$$

Got it — here’s **Part 2**, with the same structure **plus a one-line “common confusion”** under each keyword.

I’ll reuse small hand-worked numbers and keep math in MathJAX.

---

## 1) Variance

### Informal definition

How spread out a variable is around its mean.

### Formal definition

For a random variable $X$:  
$$\mathrm{Var}(X)=\mathbb{E}\big[(X-\mathbb{E}[X])^2\big].$$  
For samples $x_1,\dots,x_N$ (population form):  
$$s^2=\frac{1}{N}\sum_{n=1}^N (x_n-\bar x)^2,\quad \bar x=\frac{1}{N}\sum_{n=1}^N x_n.$$

### Example

Two datasets can have the same mean but different spread.

### Mathematics behind it

Requires: mean, deviations, squaring, expectation/sum.

### Hand-worked example

Data $[2,4,6]$, mean $\bar x=4$:  
$$s^2=\frac{(2-4)^2+(4-4)^2+(6-4)^2}{3}=\frac{4+0+4}{3}=\frac{8}{3}.$$

**Common confusion:** Variance is in **squared units**; standard deviation is $\sqrt{\mathrm{Var}}$.

---

## 2) Covariance

### Informal definition

How two variables vary _together_ (positive: move together; negative: move opposite).

### Formal definition

$$\mathrm{Cov}(X,Y)=\mathbb{E}\big[(X-\mathbb{E}[X])(Y-\mathbb{E}[Y])\big].$$  
Sample (population form):  
$$\mathrm{cov}(x,y)=\frac{1}{N}\sum_{n=1}^N (x_n-\bar x)(y_n-\bar y).$$

### Example

Height and weight typically have positive covariance.

### Mathematics behind it

Requires: means, products of deviations.

### Hand-worked example

$x=[1,2,3]$, $y=[2,4,6]$  
$\bar x=2$, $\bar y=4$:  
$$\mathrm{cov}(x,y)=\frac{(1-2)(2-4)+(2-2)(4-4)+(3-2)(6-4)}{3}$$  
$$=\frac{(-1)(-2)+0\cdot 0+(1)(2)}{3}=\frac{2+0+2}{3}=\frac{4}{3}.$$

**Common confusion:** Covariance depends on **scale**; correlation is the normalized version in $[-1,1]$.

---

## 3) Basis functions

### Informal definition

A way to transform inputs so a “linear” model can fit nonlinear patterns.

### Formal definition

Choose functions $\phi_1,\dots,\phi_M$ and define:  
$$\boldsymbol{\phi}(\mathbf{x})=  
\begin{bmatrix}  
\phi_1(\mathbf{x})\  
\vdots\  
\phi_M(\mathbf{x})  
\end{bmatrix},\quad  
\hat y=\mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x})+b.$$

### Example

Polynomial basis: $\phi(,x,)=[x,;x^2,;x^3]$.

### Mathematics behind it

Requires: function composition + dot product.

### Hand-worked example

Let $\phi_1(x)=x$, $\phi_2(x)=x^2$, $\mathbf{w}=[1,1]^\top$, $b=0$, $x=2$:  
$$\boldsymbol{\phi}(2)=\begin{bmatrix}2\4\end{bmatrix},\quad \hat y=1\cdot2+1\cdot4=6.$$

**Common confusion:** “Linear model” can still be **nonlinear in $x$** if it’s linear in parameters $\mathbf{w}$.

---

## 4) Linear Regression

### Informal definition

Predict a real number using a weighted sum of features.

### Formal definition

$$\hat y=\mathbf{w}^\top\mathbf{x}+b.$$  
Typical objective (MSE):  
$$\min_{\mathbf{w},b};\frac{1}{N}\sum_{n=1}^N (\mathbf{w}^\top\mathbf{x}^{(n)}+b-y^{(n)})^2.$$

### Example

Predict house price from size and location features.

### Mathematics behind it

Requires: dot product, squared error, minimization.

### Hand-worked example

$\mathbf{x}=\begin{bmatrix}1\2\end{bmatrix}$, $\mathbf{w}=\begin{bmatrix}1\2\end{bmatrix}$, $b=0$:  
$$\hat y=1\cdot1+2\cdot2=5.$$

**Common confusion:** Regression outputs **continuous values**, not class labels.

---

## 5) Logistic Regression (general)

### Informal definition

A linear model that outputs a probability between 0 and 1.

### Formal definition

Define score $z$:  
$$z=\mathbf{w}^\top\mathbf{x}+b,$$  
probability:  
$$p(y=1\mid\mathbf{x})=\sigma(z)=\frac{1}{1+e^{-z}}.$$

### Example

Predict probability an email is spam.

### Mathematics behind it

Requires: sigmoid, log-likelihood / cross-entropy.

### Hand-worked example

If $z=0$, then:  
$$\sigma(0)=\frac{1}{1+1}=0.5.$$

**Common confusion:** Logistic regression is a **linear classifier in feature space** even though it uses a nonlinear sigmoid.

---

## 6) Sigmoid

### Informal definition

Squashes any real number into $(0,1)$, often interpreted as a probability.

### Formal definition

$$\sigma(z)=\frac{1}{1+e^{-z}}.$$

### Example

Turn a linear score $z$ into $p(y=1\mid x)$.

### Mathematics behind it

Key derivative (used in gradients):  
$$\sigma'(z)=\sigma(z)\big(1-\sigma(z)\big).$$

### Hand-worked example

If $z=2$:  
$$\sigma(2)=\frac{1}{1+e^{-2}}\approx \frac{1}{1+0.135}=0.881.$$

**Common confusion:** $\sigma(z)$ is not “the class”; you still need a threshold (e.g., $\ge 0.5$).

---

## 7) ReLU activation

### Informal definition

Keeps positives, zeros out negatives; helps deep nets learn efficiently.

### Formal definition

$$\mathrm{ReLU}(z)=\max(0,z).$$

### Example

Used as default activation in many neural nets.

### Mathematics behind it

Derivative (piecewise):  
$$\frac{d}{dz}\mathrm{ReLU}(z)=  
\begin{cases}  
0,& z<0\  
1,& z>0  
\end{cases}  
$$  
(At $z=0$ it’s not differentiable; implementations pick 0 or 1.)

### Hand-worked example

If $z=-3$, ReLU$(z)=0$. If $z=2$, ReLU$(z)=2$.

**Common confusion:** ReLU can cause “dead neurons” when $z$ stays negative so gradient becomes 0.

---

## 8) Gradient

### Informal definition

The direction of steepest increase of a function.

### Formal definition

For scalar loss $\mathcal{L}(\theta)$ with vector parameters $\theta\in\mathbb{R}^d$:  
$$\nabla_\theta \mathcal{L}=  
\begin{bmatrix}  
\frac{\partial \mathcal{L}}{\partial \theta_1}\  
\vdots\  
\frac{\partial \mathcal{L}}{\partial \theta_d}  
\end{bmatrix}.$$

### Example

Used to update weights in gradient descent.

### Mathematics behind it

Requires: partial derivatives.

### Hand-worked example

If $\mathcal{L}(w_1,w_2)=w_1^2+3w_2^2$:  
$$\nabla \mathcal{L}=\begin{bmatrix}2w_1&6w_2\end{bmatrix}.$$

**Common confusion:** A gradient is a **vector**, not a single slope number (unless 1D).

---

## 9) Negative gradient

### Informal definition

The direction of steepest **decrease** of the loss.

### Formal definition

If $\nabla \mathcal{L}$ points uphill, then $-\nabla \mathcal{L}$ points downhill.

### Example

Gradient descent moves in the negative gradient direction.

### Mathematics behind it

Update step:  
$$\theta_{t+1}=\theta_t-\eta\nabla \mathcal{L}(\theta_t).$$

### Hand-worked example

If $\nabla \mathcal{L}=\begin{bmatrix}4&-2\end{bmatrix}$ then $-\nabla \mathcal{L}=\begin{bmatrix}-4\2\end{bmatrix}$.

**Common confusion:** “Negative gradient” is not “negative derivative”; it’s the **opposite direction** vector.

---

## 10) Iterative / sequential algorithms

### Informal definition

Algorithms that improve the solution step-by-step, rather than solving in one shot.

### Formal definition

Generate a sequence ${\theta_t}$ via an update rule:  
$$\theta_{t+1}=g(\theta_t).$$

### Example

Gradient descent, EM, k-means.

### Mathematics behind it

Requires: sequences, convergence idea.

### Hand-worked example

Start $w_0=10$, update $w_{t+1}=0.9w_t$ → $10,9,8.1,\dots$

**Common confusion:** “Iterative” doesn’t guarantee convergence; it depends on the function and step sizes.

---

## 11) Gradient descent

### Informal definition

A downhill walk on the loss surface to find low loss.

### Formal definition

$$\theta_{t+1}=\theta_t-\eta\nabla_\theta \mathcal{L}(\theta_t).$$

### Example

Train linear regression by minimizing MSE.

### Mathematics behind it

Requires: gradients and learning rate $\eta$.

### Hand-worked example

$\mathcal{L}(w)=w^2$, $\nabla \mathcal{L}=2w$.  
$w_0=3$, $\eta=0.1$:  
$$w_1=3-0.1(6)=2.4.$$

**Common confusion:** Gradient descent is not guaranteed to find the global minimum for non-convex losses.

---

## 12) Stochastic / sequential GD (SGD)

### Informal definition

Gradient descent using **one sample (or mini-batch)** at a time.

### Formal definition

Full gradient uses all data:  
$$\nabla \mathcal{L}=\frac{1}{N}\sum_{n=1}^N \nabla \ell_n.$$  
SGD uses a random sample $i$:  
$$\theta_{t+1}=\theta_t-\eta\nabla \ell_i(\theta_t).$$

### Example

Standard for deep learning due to huge datasets.

### Mathematics behind it

Requires: expectation idea (SGD gradient is a noisy estimate).

### Hand-worked example

If true gradient is 10, SGD might use 8 one step, 12 next step, but average toward 10.

**Common confusion:** SGD’s “noise” is a feature (helps escape shallow minima) but also adds variance.

---

## 13) Learning rate

### Informal definition

Step size: how far you move each update.

### Formal definition

$\eta>0$ in:  
$$\theta_{t+1}=\theta_t-\eta\nabla \mathcal{L}(\theta_t).$$

### Example

Too large → diverge; too small → painfully slow.

### Mathematics behind it

Controls stability vs speed.

### Hand-worked example

If $\nabla \mathcal{L}=6$ and $\eta=0.1$, step is $0.6$; if $\eta=1$, step is $6$.

**Common confusion:** Learning rate is **not** the same as “number of epochs” or “how long you train”.

---

## 14) Least mean squares algorithm (LMS)

### Informal definition

An online/stochastic method for linear regression using instantaneous squared error.

### Formal definition

For sample $(\mathbf{x},y)$, prediction error:  
$$e=y-(\mathbf{w}^\top\mathbf{x}+b).$$  
Update (common form, bias omitted for simplicity):  
$$\mathbf{w}_{t+1}=\mathbf{w}_t+\eta, e,\mathbf{x}.$$

### Example

Adaptive filtering / signal processing classic.

### Mathematics behind it

Requires: gradient of squared error; note sign becomes $+\eta e\mathbf{x}$ because $e$ includes a minus.

### Hand-worked example

$\mathbf{w}=\begin{bmatrix}0&0\end{bmatrix}$, $\mathbf{x}=\begin{bmatrix}1&2\end{bmatrix}$, $y=5$, $b=0$  
$\hat y=0$, $e=5$. With $\eta=0.1$:  
$$\mathbf{w}_{new}=\begin{bmatrix}0&0\end{bmatrix}+0.1\cdot5\cdot\begin{bmatrix}1&2\end{bmatrix}=\begin{bmatrix}0.5&1.0\end{bmatrix}.$$

**Common confusion:** LMS is essentially **SGD on MSE** for linear models (not a different objective).

---

## 15) Ridge Regression

### Informal definition

Linear regression with L2 penalty to shrink weights and reduce overfitting.

### Formal definition

$$\min_{\mathbf{w},b};\frac{1}{N}\sum_{n=1}^N (\mathbf{w}^\top\mathbf{x}^{(n)}+b-y^{(n)})^2+\lambda\lVert\mathbf{w}\rVert_2^2.$$

### Example

Many correlated features → ridge stabilizes coefficients.

### Mathematics behind it

Requires: L2 norm:  
$$\lVert\mathbf{w}\rVert_2^2=\sum_i w_i^2.$$

### Hand-worked example

If $\mathbf{w}=[3,4]^\top$, then:  
$$\lVert\mathbf{w}\rVert_2^2=3^2+4^2=25.$$

**Common confusion:** Ridge usually **does not** produce exact zeros (that’s LASSO).

---

## 16) LASSO Regression

### Informal definition

Linear regression with L1 penalty that can produce sparse (zero) weights.

### Formal definition

$$\min_{\mathbf{w},b};\frac{1}{N}\sum_{n=1}^N (\mathbf{w}^\top\mathbf{x}^{(n)}+b-y^{(n)})^2+\lambda\lVert\mathbf{w}\rVert_1,$$  
where  
$$\lVert\mathbf{w}\rVert_1=\sum_i |w_i|.$$

### Example

Feature selection when you have many irrelevant features.

### Mathematics behind it

Requires: absolute value; subgradient at 0.

### Hand-worked example

If $\mathbf{w}=[3,-4]^\top$:  
$$\lVert\mathbf{w}\rVert_1=|3|+|-4|=7.$$

**Common confusion:** LASSO sparsity depends on $\lambda$ and feature scaling; it’s not “always sparse”.

---

## 17) Weight decay

### Informal definition

A training rule that gradually shrinks weights; in many contexts equivalent to L2 regularization.

### Formal definition

Often corresponds to adding $\lambda\lVert\mathbf{w}\rVert_2^2$ to the loss; under GD this yields an update like:  
$$\mathbf{w}_{t+1}=(1-\eta\lambda)\mathbf{w}_t-\eta\nabla_{\mathbf{w}}\mathcal{L}_{data}.$$

### Example

Common in deep learning optimizers.

### Mathematics behind it

Requires: gradient of $\lVert\mathbf{w}\rVert_2^2$ is $2\mathbf{w}$ (up to convention constants).

### Hand-worked example

If $(1-\eta\lambda)=0.99$ and $w_t=10$, decay alone gives $w_{t+1}=9.9$.

**Common confusion:** “Weight decay” equals “L2 regularization” for plain SGD, but can differ subtly with adaptive optimizers unless implemented as decoupled weight decay.

---

## 18) Sparsity

### Informal definition

Most components are zero (or near zero).

### Formal definition

A vector is sparse if many $w_i=0$. One measure:  
$$\lVert\mathbf{w}\rVert_0={i:w_i\neq 0}.$$  
($\lVert\cdot\rVert_0$ isn’t a true norm, but is common notation.)

### Example

Only a few features matter for prediction.

### Mathematics behind it

L1 regularization encourages sparsity.

### Hand-worked example

$\mathbf{w}=[0,0,5,0]^\top$ has $\lVert\mathbf{w}\rVert_0=1$.

**Common confusion:** Sparsity is about **many zeros**, not “small variance” or “small weights”.

---

## 19) Linear Classification

### Informal definition

Classify using a linear score; decision boundary is a hyperplane.

### Formal definition

Score:  
$$z=\mathbf{w}^\top\mathbf{x}+b.$$  
Predict (binary):  
$$\hat y=\begin{cases}  
1,& z\ge 0\  
0,& z<0  
\end{cases}$$

### Example

Perceptron, linear SVM, logistic regression (after thresholding).

### Mathematics behind it

Requires: dot product; hyperplane equation.

### Hand-worked example

$\mathbf{w}=[1,-1]^\top$, $b=0$, $\mathbf{x}=[2,1]^\top$:  
$$z=1\cdot2-1\cdot1=1\Rightarrow \hat y=1.$$

**Common confusion:** “Linear classifier” means linear in **features**, not linear in probability.

---

## 20) Logistic Regression (classification context)

### Informal definition

A linear classifier that models class probability with a sigmoid.

### Formal definition

$$p(y=1\mid \mathbf{x})=\sigma(\mathbf{w}^\top\mathbf{x}+b).$$  
Training via negative log-likelihood (cross-entropy):  
$$\mathcal{L}(\mathbf{w},b)=-\sum_{n=1}^N \Big(y^{(n)}\ln p^{(n)}+(1-y^{(n)})\ln(1-p^{(n)})\Big),$$  
where $p^{(n)}=\sigma(\mathbf{w}^\top\mathbf{x}^{(n)}+b)$.

### Example

Spam vs not-spam.

### Mathematics behind it

Requires: sigmoid, log rules, gradients.

### Hand-worked example

If $y=1$ and model predicts $p=0.8$, contribution to loss:  
$$-\ln(0.8)\approx 0.223.$$

**Common confusion:** Don’t use MSE as the default for logistic regression; cross-entropy matches the Bernoulli likelihood.

---

## 21) Gaussian Basis Function

### Informal definition

A “bump” feature that activates strongly near a center and fades with distance.

### Formal definition

For center $\boldsymbol{\mu}$ and width $\sigma$:  
$$\phi(\mathbf{x})=\exp\left(-\frac{\lVert \mathbf{x}-\boldsymbol{\mu}\rVert_2^2}{2\sigma^2}\right).$$

### Example

Turn distances to landmarks into features for nonlinear regression/classification.

### Mathematics behind it

Requires: Euclidean norm squared.

### Hand-worked example (1D)

Let $\mu=0$, $\sigma=1$, $x=2$:  
$$\phi(2)=\exp\left(-\frac{(2-0)^2}{2}\right)=e^{-2}\approx 0.135.$$

**Common confusion:** Gaussian basis functions are features; they are not the same thing as assuming Gaussian noise.

---

## 22) Decision boundary

### Informal definition

The set of points where the classifier is exactly “undecided” between classes.

### Formal definition

For linear classification:  
$$\mathbf{w}^\top\mathbf{x}+b=0.$$  
For logistic regression at threshold $0.5$:  
$$\sigma(z)=0.5 \iff z=0 \iff \mathbf{w}^\top\mathbf{x}+b=0.$$

### Example

A line in 2D, a plane in 3D, a hyperplane in higher dimensions.

### Mathematics behind it

Requires: solving equations.

### Hand-worked example

If $w_1=1,w_2=1,b=-3$, boundary:  
$$x_1+x_2-3=0 \Rightarrow x_2=3-x_1.$$

**Common confusion:** Decision boundary is about **classification**, not the regression “best-fit line”.

---

## 23) Features

### Informal definition

The input variables you feed into the model.

### Formal definition

A feature vector $\mathbf{x}\in\mathbb{R}^d$ with components $x_i$.

### Example

For images: pixel intensities; for text: TF-IDF, embeddings.

### Mathematics behind it

Features define the space where dot products/distances happen.

### Hand-worked example

If $\mathbf{x}=[\text{size},\text{bedrooms}]^\top=[120,3]^\top$, then $d=2$.

**Common confusion:** Features are not parameters; features are **inputs**, parameters are what you learn.

---

## 24) Linear separability

### Informal definition

You can draw a single straight line (hyperplane) that perfectly separates classes.

### Formal definition

Binary labels $y\in{-1,+1}$. Data are linearly separable if $\exists(\mathbf{w},b)$ such that for all $n$:  
$$y^{(n)}(\mathbf{w}^\top\mathbf{x}^{(n)}+b)>0.$$

### Example

AND and OR in 2D are separable; XOR is not.

### Mathematics behind it

Requires: hyperplanes and sign.

### Hand-worked example (XOR non-separable)

Points: $(0,0)\to 0$, $(1,1)\to 0$, $(0,1)\to 1$, $(1,0)\to 1$.  
No single line can separate the 1’s from the 0’s.

**Common confusion:** If data aren’t linearly separable, logistic regression can still be trained—it just won’t reach zero training error.


## 1) Generative Model

### Informal definition

Models how the data is generated (often models $p(\mathbf{x},y)$ or $p(\mathbf{x}\mid y)$ plus $p(y)$).

### Formal definition

A common generative classifier specifies:  
$$p(y),\quad p(\mathbf{x}\mid y) \quad\Rightarrow\quad p(y\mid \mathbf{x})=\frac{p(\mathbf{x}\mid y)p(y)}{p(\mathbf{x})}.$$

### Example

Naive Bayes, Gaussian Discriminant Analysis (GDA), GMMs.

### Mathematics behind it

Bayes’ rule + probability factorization.

### Hand-worked example

Binary classes with priors $p(y=1)=0.25$, $p(y=0)=0.75$.  
If $p(x\mid y=1)=0.20$ and $p(x\mid y=0)=0.05$:  
$$p(y=1\mid x)\propto 0.20\cdot 0.25=0.05,$$  
$$p(y=0\mid x)\propto 0.05\cdot 0.75=0.0375,$$  
Normalize:  
$$p(y=1\mid x)=\frac{0.05}{0.05+0.0375}=\frac{0.05}{0.0875}\approx 0.571.$$

**Common confusion:** Generative models don’t “generate labels”; they model the **joint** generation of $(\mathbf{x},y)$ (or $\mathbf{x}$ given $y$).

---

## 2) Discriminative Model

### Informal definition

Directly models the decision rule or $p(y\mid \mathbf{x})$ without modeling how $\mathbf{x}$ was generated.

### Formal definition

Typical forms:  
$$p(y\mid \mathbf{x};\theta) \quad \text{or} \quad \hat y=f_\theta(\mathbf{x}).$$

### Example

Logistic regression, neural nets, SVMs.

### Mathematics behind it

Conditional probability models; optimization of conditional likelihood.

### Hand-worked example

Logistic regression:  
$$p(y=1\mid \mathbf{x})=\sigma(\mathbf{w}^\top\mathbf{x}+b).$$

**Common confusion:** Discriminative $\neq$ “better”; it’s just a different modeling target ($p(y\mid x)$ vs $p(x,y)$).

---

## 3) Pros vs. Cons of generative models

### Informal definition

Tradeoffs of modeling $p(\mathbf{x},y)$ vs only $p(y\mid \mathbf{x})$.

### Formal definition

Generative often uses:  
$$p(y\mid \mathbf{x})=\frac{p(\mathbf{x}\mid y)p(y)}{\sum_{k=1}^K p(\mathbf{x}\mid y=k)p(y=k)}.$$

### Example

With little data, a correct generative assumption can generalize well.

### Mathematics behind it

Bias–variance + model assumptions.

### Hand-worked “pro/con” snapshot

- Pro: if $p(\mathbf{x}\mid y)$ is correct, you can get good performance with fewer samples.
    
- Con: if $p(\mathbf{x}\mid y)$ is wrong, posterior can be badly biased.
    

**Common confusion:** “Generative has more information” is true only if its assumptions are not badly wrong.

---

## 4) Logistic Function

### Informal definition

Maps any real number to $(0,1)$.

### Formal definition

$$\sigma(z)=\frac{1}{1+e^{-z}}.$$

### Example

Convert a linear score into a probability.

### Mathematics behind it

Exponentials + monotone squashing.

### Hand-worked example

$$\sigma(0)=0.5,\quad \sigma(\ln 3)=\frac{1}{1+e^{-\ln 3}}=\frac{1}{1+1/3}=0.75.$$

**Common confusion:** Logistic function is the **same thing** as the sigmoid.

---

## 5) K-Class Discriminant

### Informal definition

For $K$ classes, compute $K$ scores and choose the biggest.

### Formal definition

Discriminant functions $g_k(\mathbf{x})$:  
$$\hat y=\arg\max_{k\in{1,\dots,K}} g_k(\mathbf{x}).$$  
For probabilistic models, often $g_k(\mathbf{x})=\ln p(y=k\mid \mathbf{x})$ or proportional scores.

### Example

Softmax classifier uses linear scores.

### Mathematics behind it

Argmax decision rule.

### Hand-worked example

If $g_1(\mathbf{x})=2.0$, $g_2(\mathbf{x})=1.2$, $g_3(\mathbf{x})=2.5$, then $\hat y=3$.

**Common confusion:** Discriminants don’t need to be probabilities; only **relative ordering** matters for argmax.

---

## 6) Perceptron

### Informal definition

A simple linear classifier trained by correcting mistakes.

### Formal definition

Binary labels $y\in{-1,+1}$, prediction:  
$$\hat y=\mathrm{sign}(\mathbf{w}^\top\mathbf{x}+b).$$  
Update on misclassified sample ($y(\mathbf{w}^\top\mathbf{x}+b)\le 0$):  
$$\mathbf{w}\leftarrow \mathbf{w}+\eta y\mathbf{x},\quad b\leftarrow b+\eta y.$$

### Example

Works when classes are linearly separable.

### Mathematics behind it

Dot products + sign + iterative updates.

### Hand-worked example

Let $\mathbf{w}=\begin{bmatrix}0&0\end{bmatrix}$, $b=0$, $\eta=1$, sample $\mathbf{x}=\begin{bmatrix}1&2\end{bmatrix}$, $y=+1$.  
Score $=0$ → treat as mistake:  
$$\mathbf{w}\leftarrow \begin{bmatrix}0&0\end{bmatrix}+1\cdot(+1)\begin{bmatrix}1&2\end{bmatrix}=\begin{bmatrix}1&2\end{bmatrix}.$$

**Common confusion:** Perceptron is **not** logistic regression; it doesn’t output probabilities and uses a different loss idea.

---

## 7) Class-conditional densities

### Informal definition

The distribution of features given the class: “what does class $k$ look like in feature space?”

### Formal definition

$$p(\mathbf{x}\mid y=k).$$

### Example

Gaussian class-conditional:  
$$p(\mathbf{x}\mid y=k)=\mathcal{N}(\mathbf{x}\mid \boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k).$$

### Mathematics behind it

Density functions; Bayes classifier.

### Hand-worked example

If $p(\mathbf{x}\mid y=1)$ is larger than $p(\mathbf{x}\mid y=0)$ at a point, that point “looks like” class 1.

**Common confusion:** $p(\mathbf{x}\mid y)$ is not the same as $p(y\mid \mathbf{x})$.

---

## 8) Maximum likelihood (MLE)

### Informal definition

Pick parameters that make observed data most probable.

### Formal definition

$$\theta_{\text{ML}}=\arg\max_\theta p(\mathbf{X}\mid \theta).$$

### Example

Fit Gaussian mean/variance.

### Mathematics behind it

Products over i.i.d. samples + optimization.

### Hand-worked example

For Gaussian with known $\sigma^2$, MLE mean:  
$$\mu_{\text{ML}}=\frac{1}{N}\sum_{n=1}^N x_n.$$

**Common confusion:** MLE is not “Bayesian”—it does **not** include a prior.

---

## 9) Maximum Log-Likelihood

### Informal definition

Same as MLE, but optimize the log (easier math, stable numerics).

### Formal definition

Since $\ln$ is monotone:  
$$\arg\max_\theta p(\mathbf{X}\mid\theta)=\arg\max_\theta \ln p(\mathbf{X}\mid\theta).$$  
For i.i.d.:  
$$\ln p(\mathbf{X}\mid\theta)=\sum_{n=1}^N \ln p(\mathbf{x}_n\mid\theta).$$

### Example

Logistic regression training = maximize conditional log-likelihood (equivalently minimize cross-entropy).

### Mathematics behind it

Log rules: $\ln\prod = \sum \ln$.

### Hand-worked example

If $p(\mathbf{X}\mid\theta)=0.001$, then $\ln p(\mathbf{X}\mid\theta)\approx -6.907$.

**Common confusion:** “Maximum log-likelihood” isn’t a different estimator—just a different objective form.

---

## 10) Bayesian classifier

### Informal definition

Classify by computing posterior class probabilities and picking the largest.

### Formal definition

$$p(y=k\mid \mathbf{x})=\frac{p(\mathbf{x}\mid y=k)p(y=k)}{\sum_{j=1}^K p(\mathbf{x}\mid y=j)p(y=j)},\quad \hat y=\arg\max_k p(y=k\mid\mathbf{x}).$$

### Example

Naive Bayes, GDA.

### Mathematics behind it

Bayes’ rule + normalization.

### Hand-worked example

Two classes:  
$$\text{score}_1=p(\mathbf{x}\mid 1)p(1),\quad \text{score}_0=p(\mathbf{x}\mid 0)p(0),$$  
choose larger score.

**Common confusion:** Bayesian classifier refers to using **Bayes rule**; it doesn’t necessarily mean “Bayesian parameter inference”.

---

## 11) Latent variables

### Informal definition

Hidden variables not observed in data but explain structure (clusters, components, topics).

### Formal definition

Observed $\mathbf{X}$, latent $\mathbf{Z}$. Model uses joint:  
$$p(\mathbf{X},\mathbf{Z}\mid \theta).$$  
Marginal (what you observe):  
$$p(\mathbf{X}\mid\theta)=\sum_{\mathbf{Z}} p(\mathbf{X},\mathbf{Z}\mid\theta) \quad \text{or}\quad \int p(\mathbf{X},\mathbf{Z}\mid\theta),d\mathbf{Z}.$$

### Example

GMM: latent variable = which Gaussian generated each point.

### Mathematics behind it

Marginalization (sum/integral).

### Hand-worked example

If $z\in{1,2}$:  
$$p(x)=p(x,z=1)+p(x,z=2).$$

**Common confusion:** Latent variables are not “errors”; they are **structured hidden causes**.

---

## 12) Expectation–Maximization (EM)

### Informal definition

An iterative method to fit models with latent variables by alternating “guess hidden stuff” and “update parameters.”

### Formal definition

Maximize incomplete-data log-likelihood:  
$$\max_\theta \ln p(\mathbf{X}\mid\theta).$$  
Define:  
$$Q(\theta,\theta^{old})=\mathbb{E}_{\mathbf{Z}\mid \mathbf{X},\theta^{old}}\big[\ln p(\mathbf{X},\mathbf{Z}\mid \theta)\big].$$  
E-step: compute posterior over $\mathbf{Z}$.  
M-step: $\theta^{new}=\arg\max_\theta Q(\theta,\theta^{old})$.

### Example

GMM parameter fitting.

### Mathematics behind it

Expectation + logs + optimization.

### Hand-worked micro-idea

E-step gives “soft memberships” $\gamma_{nk}$; M-step recomputes parameters from weighted averages.

**Common confusion:** EM does not guarantee global optimum; it increases (or doesn’t decrease) $\ln p(\mathbf{X}\mid\theta)$ each iteration.

---

## 13) Log-likelihood

### Informal definition

Log of the probability (or density) the model assigns to observed data.

### Formal definition

$$\ell(\theta)=\ln p(\mathbf{X}\mid\theta).$$  
For i.i.d.:  
$$\ell(\theta)=\sum_{n=1}^N \ln p(\mathbf{x}_n\mid\theta).$$

### Example

Used for MLE, EM, logistic regression.

### Mathematics behind it

Log transforms products into sums.

### Hand-worked example

If $p(x_1\mid\theta)=0.2$, $p(x_2\mid\theta)=0.5$ (independent):  
$$\ln p(x_1,x_2\mid\theta)=\ln(0.2\cdot 0.5)=\ln 0.1.$$

**Common confusion:** For continuous data, “likelihood” uses **densities**; values can exceed 1, but log-likelihood is still valid.

---

## 14) Document clustering

### Informal definition

Group documents into topics without labels.

### Formal definition

Represent each document as a vector $\mathbf{x}_n$ (e.g., TF-IDF). Cluster by minimizing within-cluster distortion (k-means) or maximizing likelihood (GMM).

### Example

Cluster news articles into sports/politics/tech.

### Mathematics behind it

Vector spaces, distances, or mixture likelihoods.

### Hand-worked example

Two documents: $\mathbf{x}_1=[3,0]$, $\mathbf{x}_2=[0,3]$ are far apart → likely different clusters.

**Common confusion:** Clustering is unsupervised—there is no “accuracy” unless you have ground truth labels.

---

## 15) k-Means algorithm

### Informal definition

Find $K$ centers so points are close to their assigned center.

### Formal definition

Minimize distortion:  
$$J=\sum_{n=1}^N \lVert \mathbf{x}_n-\boldsymbol{\mu}_{c_n}\rVert_2^2.$$  
Iterate:

1. Assign: $c_n\leftarrow \arg\min_k \lVert \mathbf{x}_n-\boldsymbol{\mu}_k\rVert^2$
    
2. Update: $\boldsymbol{\mu}_k\leftarrow \frac{1}{N_k}\sum_{n:c_n=k}\mathbf{x}_n$
    

### Example

Customer segmentation.

### Mathematics behind it

Euclidean distance + averaging.

### Hand-worked example (1D)

Points $[1,2,10]$, $K=2$, initial centers $\mu_1=1$, $\mu_2=10$.  
Assignments: ${1,2}\to 1$, ${10}\to 2$.  
Update:  
$$\mu_1=\frac{1+2}{2}=1.5,\quad \mu_2=10.$$

**Common confusion:** k-means minimizes squared distances, not general “cluster goodness”; it assumes roughly spherical clusters.

---

## 16) Gaussian Mixture Models (GMMs)

### Informal definition

A probabilistic clustering model: data comes from a mixture of Gaussians.

### Formal definition

$$p(\mathbf{x})=\sum_{k=1}^K \pi_k ,\mathcal{N}(\mathbf{x}\mid \boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k),\quad \sum_{k=1}^K \pi_k=1,;\pi_k\ge 0.$$

### Example

Clusters with different shapes (covariances), unlike k-means.

### Mathematics behind it

Mixtures + Gaussian density + latent component.

### Hand-worked example (mixture weight intuition)

If $\pi_1=0.7$ and $\pi_2=0.3$, then before seeing $\mathbf{x}$:  
$$p(z=1)=0.7,\quad p(z=2)=0.3.$$

**Common confusion:** A GMM is not “one Gaussian”; it’s a weighted **sum** of Gaussians.

---

## 17) EM algorithm (duplicate of EM)

Same as “Expectation–Maximization” above.

**Common confusion:** People say “EM” for GMMs, but EM is a general method for many latent-variable models.

---

## 18) E-step

### Informal definition

Compute the posterior over latent variables given current parameters.

### Formal definition (GMM responsibilities)

$$\gamma_{nk}=p(z_n=k\mid \mathbf{x}_n,\theta^{old})  
=\frac{\pi_k \mathcal{N}(\mathbf{x}_n\mid \boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)}  
{\sum_{j=1}^K \pi_j \mathcal{N}(\mathbf{x}_n\mid \boldsymbol{\mu}_j,\boldsymbol{\Sigma}_j)}.$$

### Example

“Soft cluster membership” of each point.

### Hand-worked example (toy numbers)

Suppose numerator scores for a point are $s_1=0.2$, $s_2=0.8$. Then:  
$$\gamma_{n1}=\frac{0.2}{1.0}=0.2,\quad \gamma_{n2}=\frac{0.8}{1.0}=0.8.$$

**Common confusion:** E-step does **not** change parameters; it changes latent posterior estimates.

---

## 19) M-step

### Informal definition

Update parameters to best fit data given the current soft assignments.

### Formal definition (GMM core updates)

Define $N_k=\sum_{n=1}^N \gamma_{nk}$. Then:  
$$\pi_k^{new}=\frac{N_k}{N},$$  
$$\boldsymbol{\mu}_k^{new}=\frac{1}{N_k}\sum_{n=1}^N \gamma_{nk}\mathbf{x}_n,$$  
$$\boldsymbol{\Sigma}_k^{new}=\frac{1}{N_k}\sum_{n=1}^N \gamma_{nk}(\mathbf{x}_n-\boldsymbol{\mu}_k)(\mathbf{x}_n-\boldsymbol{\mu}_k)^\top.$$

### Example

Recompute means as weighted averages.

### Hand-worked example (mean update)

Two points $x_1=0$, $x_2=10$ for one component, responsibilities $\gamma_{11}=0.9$, $\gamma_{21}=0.1$:  
$$N_1=1.0,\quad \mu_1=\frac{0.9\cdot 0+0.1\cdot 10}{1.0}=1.$$

**Common confusion:** M-step is not “maximize the original log-likelihood directly”; it maximizes the **Q-function**.

---

## 20) Mean ($\mu$) parameter estimation

### Informal definition

Compute the component mean that best matches assigned (soft) data.

### Formal definition (GMM)

$$\boldsymbol{\mu}_k=\frac{\sum_{n=1}^N \gamma_{nk}\mathbf{x}_n}{\sum_{n=1}^N \gamma_{nk}}.$$

### Example

Cluster center for GMM component.

### Hand-worked example

Same as above: responsibilities weight the average.

**Common confusion:** Don’t average raw points unless responsibilities are all 0/1 (hard assignments).

---

## 21) Soft EM

### Informal definition

EM where assignments are probabilistic (responsibilities in $(0,1)$).

### Formal definition

E-step computes $\gamma_{nk}\in[0,1]$ with $\sum_k \gamma_{nk}=1$.

### Example

GMM EM is soft EM.

### Hand-worked example

A point can be 60% in cluster 1 and 40% in cluster 2:  
$$\gamma_{n1}=0.6,;\gamma_{n2}=0.4.$$

**Common confusion:** “Soft” doesn’t mean “approximate”; it means **probabilistic membership**.

---

## 22) Hard EM

### Informal definition

Assignments are hard: each point belongs to exactly one cluster.

### Formal definition

Replace responsibilities with:  
$$\gamma_{nk}\in{0,1},\quad \sum_k \gamma_{nk}=1.$$

### Example

k-means is essentially hard EM for a special GMM case.

### Hand-worked example

If point assigned to cluster 2:  
$$\gamma_{n2}=1,;\gamma_{n1}=0.$$

**Common confusion:** Hard EM is not always the same as k-means unless you assume equal spherical covariances and equal priors (or related constraints).

---

## 23) Complete data log-likelihood

### Informal definition

Log-likelihood if you could see the hidden variables too.

### Formal definition

$$\ln p(\mathbf{X},\mathbf{Z}\mid\theta).$$  
For GMM with one-hot $z_{nk}$:  
$$\ln p(\mathbf{X},\mathbf{Z}\mid\theta)=\sum_{n=1}^N\sum_{k=1}^K z_{nk}\Big(\ln \pi_k + \ln \mathcal{N}(\mathbf{x}_n\mid \boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)\Big).$$

### Example

Key object EM “wants” to maximize indirectly.

### Hand-worked example

If for a point $n$, $z_{n2}=1$ then only $k=2$ term contributes.

**Common confusion:** Complete-data log-likelihood uses **latent indicators** $z_{nk}$; incomplete-data does not.

---

## 24) Q-function

### Informal definition

The expected complete-data log-likelihood under the current posterior over latent variables.

### Formal definition

$$Q(\theta,\theta^{old})=\mathbb{E}_{\mathbf{Z}\mid\mathbf{X},\theta^{old}}\big[\ln p(\mathbf{X},\mathbf{Z}\mid\theta)\big].$$  
In GMM, expectation replaces $z_{nk}$ with $\gamma_{nk}$:  
$$Q(\theta,\theta^{old})=\sum_{n=1}^N\sum_{k=1}^K \gamma_{nk}\Big(\ln \pi_k + \ln \mathcal{N}(\mathbf{x}_n\mid \boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)\Big).$$

### Example

M-step maximizes $Q$.

### Hand-worked example

A hard assignment uses $\gamma_{nk}\in{0,1}$; soft uses fractional weights.

**Common confusion:** $Q$ is not the same as the actual log-likelihood $\ln p(\mathbf{X}\mid\theta)$, but EM uses it to improve that.

---

## 25) Incomplete-data log-likelihood

### Informal definition

Log-likelihood of observed data when latent variables are hidden.

### Formal definition

$$\ln p(\mathbf{X}\mid\theta)=\sum_{n=1}^N \ln \left(\sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x}_n\mid\boldsymbol{\mu}_k,\boldsymbol{\Sigma}_k)\right).$$

### Example

This is the objective EM is trying to increase.

### Hand-worked example

For one point:  
$$\ln p(\mathbf{x})=\ln(\pi_1 \mathcal{N}_1(\mathbf{x})+\pi_2\mathcal{N}_2(\mathbf{x})).$$

**Common confusion:** People try to “take the log inside the sum” incorrectly: $\ln(a+b)\ne \ln a+\ln b$.

---

## 26) Constrained optimization

### Informal definition

Optimize an objective while respecting constraints.

### Formal definition

$$\min_\theta f(\theta)\quad \text{s.t.}\quad g_i(\theta)=0,;h_j(\theta)\le 0.$$

### Example

GMM mixing weights must satisfy $\sum_k \pi_k=1$, $\pi_k\ge 0$.

### Mathematics behind it

Lagrange multipliers / KKT conditions.

### Hand-worked example

Constraint: $\pi_1+\pi_2=1$.

**Common confusion:** Ignoring constraints can produce “optimal” parameters that are invalid probabilities.

---

## 27) Lagrange multipliers

### Informal definition

A technique to handle equality constraints by adding them to the objective.

### Formal definition

For maximize $f(\pi)$ s.t. $c(\pi)=0$:  
$$\mathcal{L}(\pi,\lambda)=f(\pi)+\lambda,c(\pi).$$  
Solve $\nabla_\pi \mathcal{L}=0$ plus constraint.

### Example (derive GMM $\pi_k$ update idea)

Maximize $\sum_k N_k \ln \pi_k$ s.t. $\sum_k \pi_k=1$.

### Hand-worked derivation sketch

Lagrangian:  
$$\mathcal{L}=\sum_{k=1}^K N_k \ln \pi_k+\lambda\left(\sum_{k=1}^K \pi_k-1\right).$$  
Derivative:  
$$\frac{\partial \mathcal{L}}{\partial \pi_k}=\frac{N_k}{\pi_k}+\lambda=0\Rightarrow \pi_k=-\frac{N_k}{\lambda}.$$  
Sum constraint gives:  
$$\sum_k \pi_k=1\Rightarrow -\frac{1}{\lambda}\sum_k N_k=1\Rightarrow \lambda=-N,$$  
so  
$$\pi_k=\frac{N_k}{N}.$$

**Common confusion:** The “$\lambda$” here is not the same as regularization strength $\lambda$ used in ridge/LASSO—unfortunate notation clash.

---

## 28) Neural networks

### Informal definition

A flexible function approximator built from layers of linear maps + nonlinear activations.

### Formal definition

Layer $l$:  
$$\mathbf{z}^{(l)}=W^{(l)}\mathbf{a}^{(l-1)}+\mathbf{b}^{(l)},\quad \mathbf{a}^{(l)}=\phi(\mathbf{z}^{(l)}).$$  
Output $\hat y$ depends on task (regression/classification).

### Example

Image classification, language modeling, regression.

### Mathematics behind it

Matrix multiplication + nonlinear functions + gradients.

### Hand-worked example (one neuron)

If $z=2x+1$ and ReLU, at $x=-1$:  
$$z=-1,;\mathrm{ReLU}(z)=0.$$

**Common confusion:** A network is not “just linear”—nonlinear activations are what make it powerful.

---

## 29) Feed-forward

### Informal definition

Information flows from input to output with no cycles.

### Formal definition

A DAG of computations; no recurrent connections.

### Example

MLP (multi-layer perceptron).

### Math behind it

Composition of functions:  
$$f(\mathbf{x})=f^{(L)}(\cdots f^{(2)}(f^{(1)}(\mathbf{x}))\cdots).$$

### Hand-worked example

$$f(x)=\sigma(2\cdot \mathrm{ReLU}(3x)).$$

**Common confusion:** “Feed-forward” describes the architecture (no loops), not the training method.

---

## 30) Forward propagation

### Informal definition

Compute the output by pushing inputs through layers.

### Formal definition

Repeatedly apply:  
$$\mathbf{z}^{(l)}=W^{(l)}\mathbf{a}^{(l-1)}+\mathbf{b}^{(l)},\quad \mathbf{a}^{(l)}=\phi(\mathbf{z}^{(l)}).$$

### Example

Compute logits then softmax probabilities.

### Hand-worked example (2-layer scalar)

Let $a^{(0)}=x=2$, first layer $z^{(1)}=3x=6$, $a^{(1)}=\mathrm{ReLU}(6)=6$, second layer $z^{(2)}=a^{(1)}-1=5$.

**Common confusion:** Forward propagation computes predictions; it does **not** compute gradients.

---

## 31) Backpropagation

### Informal definition

Efficiently compute gradients of loss w.r.t. all parameters using the chain rule.

### Formal definition (core recursion)

Let $\delta^{(l)}=\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(l)}}$. Then:  
$$\delta^{(L)}=\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{(L)}}\odot \phi'(\mathbf{z}^{(L)}),$$  
$$\delta^{(l)}=\left(W^{(l+1)}\right)^\top \delta^{(l+1)} \odot \phi'(\mathbf{z}^{(l)}).$$  
Parameter gradients:  
$$\frac{\partial \mathcal{L}}{\partial W^{(l)}}=\delta^{(l)}\left(\mathbf{a}^{(l-1)}\right)^\top,\quad \frac{\partial \mathcal{L}}{\partial \mathbf{b}^{(l)}}=\delta^{(l)}.$$

### Example

Training deep nets with SGD/Adam.

### Hand-worked example (tiny chain rule)

If $\mathcal{L}=(\hat y-y)^2$ and $\hat y=2w$, then:  
$$\frac{d\mathcal{L}}{dw}=2(\hat y-y)\cdot \frac{d\hat y}{dw}=2(2w-y)\cdot 2=4(2w-y).$$

**Common confusion:** Backprop is not the optimizer; it’s how you compute gradients for _any_ gradient-based optimizer.

---

## 32) Error surface

### Informal definition

The loss value as a function of parameters—think “landscape” you optimize over.

### Formal definition

$$\mathcal{L}(\theta) \quad \text{viewed as a surface over } \theta.$$

### Example

Non-convex for neural nets.

### Hand-worked example

For $\mathcal{L}(w)=w^2$, the surface is a parabola with minimum at $w=0$.

**Common confusion:** Local minima/saddles are properties of the **loss landscape**, not of the data alone.

---

## 33) Hidden units

### Informal definition

Neurons in hidden layers (not input or output).

### Formal definition

Components of $\mathbf{a}^{(l)}$ for hidden layers $l=1,\dots,L-1$.

### Example

A network with 10 hidden units has 10 neurons in its hidden layer.

### Hand-worked example

If hidden layer activation is $\mathbf{a}^{(1)}\in\mathbb{R}^{10}$, that’s 10 hidden units.

**Common confusion:** Hidden units are not “latent variables” in the probabilistic EM sense (though both are “unobserved”).

---

## 34) Early stopping

### Informal definition

Stop training when validation performance stops improving to reduce overfitting.

### Formal definition

Choose stopping time $t^*$:  
$$t^*=\arg\min_t \mathcal{L}_{val}(\theta_t).$$

### Example

Train until validation loss rises for several epochs (“patience”).

### Hand-worked example

If validation loss over epochs is $[0.9,0.7,0.6,0.62,0.65]$, stop around epoch 3.

**Common confusion:** Early stopping uses the **validation** set, not the training set.

---

## 35) 3-layer neural networks

### Informal definition

Usually means input layer + one hidden layer + output layer.

### Formal definition

One hidden layer:  
$$\mathbf{a}^{(1)}=\phi(W^{(1)}\mathbf{x}+\mathbf{b}^{(1)}),\quad \hat y=g(W^{(2)}\mathbf{a}^{(1)}+\mathbf{b}^{(2)}).$$

### Example

Classic MLP for classification.

### Hand-worked example

Dimensions: $\mathbf{x}\in\mathbb{R}^d$, hidden $h$ units, output $K$ classes:  
$$W^{(1)}\in\mathbb{R}^{h\times d},\quad W^{(2)}\in\mathbb{R}^{K\times h}.$$

**Common confusion:** Some texts count layers differently (sometimes they don’t count the input layer).

---

## 36) Supervised

### Informal definition

Learn from labeled examples $(\mathbf{x},y)$.

### Formal definition

Given ${(\mathbf{x}^{(n)},y^{(n)})}$, minimize expected/empirical loss:  
$$\min_\theta \frac{1}{N}\sum_{n=1}^N \ell\big(f_\theta(\mathbf{x}^{(n)}),y^{(n)}\big).$$

### Example

Classification, regression.

### Hand-worked example

Predict $y$ from $\mathbf{x}$ using MSE or cross-entropy.

**Common confusion:** “Supervised” doesn’t mean “human in the loop during training”; it means labels are provided.

---

## 37) Unsupervised

### Informal definition

Learn structure from unlabeled data $\mathbf{X}$ only.

### Formal definition

Optimize objectives without labels, e.g. clustering distortion or log-likelihood:  
$$\min_{{\mu_k}} \sum_n \min_k \lVert \mathbf{x}_n-\mu_k\rVert^2\quad \text{or}\quad \max_\theta \ln p(\mathbf{X}\mid\theta).$$

### Example

k-means, PCA, autoencoders, GMMs.

### Hand-worked example

Cluster points into $K$ groups without labels.

**Common confusion:** Unsupervised learning can still have an objective; it’s not “learning with no goal”.

---

## 38) Autoencoders

### Informal definition

Neural networks trained to reconstruct inputs; learn compressed representations.

### Formal definition

Encoder $f_\theta$, decoder $g_\phi$:  
$$\mathbf{z}=f_\theta(\mathbf{x}),\quad \hat{\mathbf{x}}=g_\phi(\mathbf{z}).$$  
Minimize reconstruction loss:  
$$\min_{\theta,\phi}\frac{1}{N}\sum_{n=1}^N \lVert \hat{\mathbf{x}}^{(n)}-\mathbf{x}^{(n)}\rVert^2.$$

### Example

Dimensionality reduction, denoising.

### Hand-worked example

If $\mathbf{x}=[1,0]$ and model outputs $\hat{\mathbf{x}}=[0.8,0.1]$:  
$$\lVert \hat{\mathbf{x}}-\mathbf{x}\rVert^2=(0.8-1)^2+(0.1-0)^2=0.04+0.01=0.05.$$

**Common confusion:** Autoencoders are not automatically “clustering”; reconstruction alone doesn’t guarantee clusters.

---

## 39) Principal Component Analysis (PCA)

### Informal definition

Find directions of maximum variance; project data onto those directions.

### Formal definition

Center data: $\tilde{\mathbf{x}}_n=\mathbf{x}_n-\bar{\mathbf{x}}$. Covariance:  
$$S=\frac{1}{N}\sum_{n=1}^N \tilde{\mathbf{x}}_n\tilde{\mathbf{x}}_n^\top.$$  
First principal component $\mathbf{u}_1$ solves:  
$$\mathbf{u}_1=\arg\max_{\lVert \mathbf{u}\rVert_2=1}\mathbf{u}^\top S\mathbf{u},$$  
and $\mathbf{u}_1$ is the top eigenvector of $S$.

### Example

Compress 100D data to 2D for visualization.

### Hand-worked example (simple 2D where it’s obvious)

# Points: $(1,0)$ and $(3,0)$. Mean is $(2,0)$. Centered points: $(-1,0)$ and $(1,0)$.  
Covariance (population form):  
$$S=\frac{1}{2}\left(  
\begin{bmatrix}-1&0\end{bmatrix}\begin{bmatrix}-1&0\end{bmatrix}+  
\begin{bmatrix}1&0\end{bmatrix}\begin{bmatrix}1&0\end{bmatrix}  
\right)  
=\frac{1}{2}\left(  
\begin{bmatrix}1&070&0\end{bmatrix}+  
\begin{bmatrix}1&070&0\end{bmatrix}  
\right)

\begin{bmatrix}1&0&0&0\end{bmatrix}.$$  
Largest-variance direction is the $x$-axis: $\mathbf{u}_1=[1,0]^\top$.

**Common confusion:** PCA is unsupervised and variance-based; it does not “use labels” or directly optimize classification accuracy.

---

### Quick note on duplicates in your list

- “Maximum likelihood” and “Log-likelihood” appeared earlier too; the key difference is **probability vs log-probability** and **joint vs conditional**.
    

If you want, I can turn this into a **one-page “EM/GMM map”** (complete-data LL → Q-function → E-step responsibilities → M-step updates) because that cluster of terms is where exams often try to trick you.