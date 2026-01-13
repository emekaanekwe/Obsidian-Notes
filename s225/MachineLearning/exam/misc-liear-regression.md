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
$$\nabla \mathcal{L}=\begin{bmatrix}2w_1\6w_2\end{bmatrix}.$$

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

If $\nabla \mathcal{L}=\begin{bmatrix}4\-2\end{bmatrix}$ then $-\nabla \mathcal{L}=\begin{bmatrix}-4\2\end{bmatrix}$.

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

$\mathbf{w}=\begin{bmatrix}0\0\end{bmatrix}$, $\mathbf{x}=\begin{bmatrix}1\2\end{bmatrix}$, $y=5$, $b=0$  
$\hat y=0$, $e=5$. With $\eta=0.1$:  
$$\mathbf{w}_{new}=\begin{bmatrix}0\0\end{bmatrix}+0.1\cdot5\cdot\begin{bmatrix}1\2\end{bmatrix}=\begin{bmatrix}0.5\1.0\end{bmatrix}.$$

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
$$\lVert\mathbf{w}\rVert_0=#{i:w_i\neq 0}.$$  
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

---

If you paste **Part 3**, I’ll keep the same format. Also: if your exam expects specific conventions (e.g., $y\in{0,1}$ vs $y\in{-1,1}$; MSE with $\frac{1}{2}$ factors), tell me and I’ll match them.

---

- **posterior** $p(t\mid \mathbf{x})$
    
- **likelihood** $p(\mathbf{x}\mid t)$
    
- **prior** $p(t)$
    
- **evidence** $p(\mathbf{x})$
    

Not “$(\text{likelihood})p(x)/p(x)$”; it’s **likelihood × prior / evidence**.

