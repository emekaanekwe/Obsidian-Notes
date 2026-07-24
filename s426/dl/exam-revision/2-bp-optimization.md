# FIT5215 — Deep Learning Revision Notes (Part 2: Backprop Mechanics & Optimization)

_Source: Final Exam Revision slides, pp. 21–40_

## 1. Jacobian Worked Example

**Function**: $f:\mathbb{R}^3\to\mathbb{R}^2$, $y=f(x)=f(x_1,x_2,x_3)=(x_1^2+x_2^2,\ x_2^2+x_3^2 x_2)$

- $f_1(x) = x_1^2+x_2^2$
- $f_2(x) = x_2^2+x_3^2 x_2$

**Jacobian**:
$$\frac{\partial y}{\partial x} = \nabla_x f = \begin{bmatrix}\dfrac{\partial f_1}{\partial x_1} & \dfrac{\partial f_1}{\partial x_2} & \dfrac{\partial f_1}{\partial x_3}\\[6pt] \dfrac{\partial f_2}{\partial x_1} & \dfrac{\partial f_2}{\partial x_2} & \dfrac{\partial f_2}{\partial x_3}\end{bmatrix} = \begin{bmatrix} 2x_1 & 2x_2 & 0 \\ 0 & 2x_2+x_3^2 & 2x_2 x_3\end{bmatrix} \in \mathbb{R}^{2\times3}$$

---

## 2. Gradient of Cross-Entropy w.r.t. Logits (Key Backprop Result)

**Setup**: 4-class output, logits $h=[h_1,h_2,h_3,h_4]$, softmax probabilities $p=[p_1,p_2,p_3,p_4]$, true label $y=2$.

**Loss** (cross-entropy with true class 2): $$l = -\log\frac{e^{h_2}}{e^{h_1}+e^{h_2}+e^{h_3}+e^{h_4}} = \log(e^{h_1}+e^{h_2}+e^{h_3}+e^{h_4}) - h_2$$

Let $u = e^{h_1}+e^{h_2}+e^{h_3}+e^{h_4}$ (so $l = \log u - h_2$).

**Partial derivatives**: $$\frac{\partial l}{\partial h_1} = \frac{\nabla_{h_1} u}{u} = \frac{e^{h_1}}{u} = p_1$$ $$\frac{\partial l}{\partial h_2} = \frac{\nabla_{h_2} u}{u} - 1 = \frac{e^{h_2}}{u} - 1 = p_2 - 1$$ $$\frac{\partial l}{\partial h_3} = \frac{\nabla_{h_3} u}{u} = p_3$$ $$\frac{\partial l}{\partial h_4} = \frac{\nabla_{h_4} u}{u} = p_4$$

**Combined (vector form)** — this is the single most important backprop identity for classification networks: $$\boxed{\frac{\partial l}{\partial \mathbf{h}} = [p_1, p_2-1, p_3, p_4] = p - \mathbf{1}_y}$$

Where $\mathbf{1}_y$ is the one-hot vector for the true class. **This generalizes to any number of classes**: the gradient of CE loss w.r.t. the logits is simply "predicted probabilities minus one-hot true label."

---

## 3. Jacobian of an Activation Layer

**Setup**: $\bar h = xW+b$, $h=\sigma(\bar h)$, so $h=\sigma(xW+b)$.

**Chain rule**: $$\frac{\partial h}{\partial x} = \frac{\partial h}{\partial \bar h}\times\frac{\partial \bar h}{\partial x} = \text{diag}(\sigma'(\bar h)), W^T$$

**Why diagonal?** Because $\sigma$ is applied **element-wise**: $h_i$ depends only on $\bar h_i$, not on $\bar h_j$ for $j\ne i$. So: 
$$\frac{\partial h}{\partial \bar h} = \begin{bmatrix}\sigma'(\bar h_1) & 0 & 0 & 0 \\ 0 & \sigma'(\bar h_2) & 0 & 0 \\ 0&0&\sigma'(\bar h_3)&0 \\ 0&0&0&\sigma'(\bar h_4)\end{bmatrix} = \text{diag}(\sigma'(\bar h))$$

And since $\bar h = xW+b$ is linear in $x$: $$\frac{\partial \bar h}{\partial x} = W^T$$

**For sigmoid specifically** ($\sigma'(\bar h)=\sigma(\bar h)(1-\sigma(\bar h))=h\otimes(1-h)$): $$\frac{\partial h}{\partial x} = \text{diag}(h\otimes(1-h)), W^T$$ where $\otimes$ is the element-wise product.

### PyTorch verification

```python
x = torch.tensor([1,-1,1], dtype=torch.float32)
W = torch.rand(3,4); b = torch.rand(1,4)

hbar = x @ W + b
h = torch.nn.Sigmoid()(hbar)

v = (h*(1-h)).squeeze()          # σ'(h̄) as a vector
A = torch.diag(v)                # diag(σ'(h̄))
derivative = A @ W.T              # ∂h/∂x
```

This matches the formula exactly — `A` is the diagonal matrix of $\sigma'(\bar h)$ values, and `derivative = A @ W.T` recovers $\frac{\partial h}{\partial x}$.

---

## 4. The Deep Learning Optimization Problem (Formal Statement)

**Parameters**: $\theta := {(W^l,b^l)}_{l=1}^L$

**Objective**: $$\min_\theta L(D;\theta) := -\frac1N\sum_{i=1}^N \log p_{y_i}(x_i) = -\frac1N\sum_{i=1}^N \log\frac{\exp{h^L_{y_i}(x_i)}}{\sum_{m=1}^M \exp{h^L_m(x_i)}}$$

**Generalized form** (any loss, any model): $$\min_\theta J(\theta) := \frac1N\sum_{i=1}^N l(f(x_i;\theta), y_i)$$

### Full ML/DL optimization form (with regularization)

$$\min_\theta J(\theta) = \Omega(\theta) + \frac1N\sum_{i=1}^N l(y_i, f(x_i;\theta))$$

**Regularization term** — encourages simpler models, helps avoid overfitting: $$\Omega(\theta) = \lambda\sum_k\sum_{i,j}(W^k_{i,j})^2 = \lambda\sum_k |W^k|_F^2$$

This reflects **Occam's Razor**: prefer the simplest model that still predicts the data well.

**Empirical loss term** — measures how well the model fits the training set.

### Two families of solvers

|Method|Uses|Update rule basis|
|---|---|---|
|**First-order** (GD, steepest descent)|Gradient $g=\nabla_\theta J(\theta)$|Linear approximation|
|**Second-order** (Newton, quasi-Newton)|Hessian $H=\nabla_\theta^2 J(\theta)$|Quadratic approximation|

**Practical constraint**: $N$ (training size) can be enormous ($N\approx10^6$), and $P$ (number of trainable parameters) can also be huge ($P\approx20\times10^6$) — this drives the need for efficient methods (SGD) over exact full-batch or second-order methods.

---

## 5. Gradient and Hessian — Formal Definitions

Given objective $J(\theta)$ with $\theta=[\theta_1,\dots,\theta_P]$ (for DL: weights, filters, biases; $P$ = number of trainable parameters).

**Gradient** (first-order derivative): $$\nabla J(\theta) = g = \begin{bmatrix}\dfrac{\partial J}{\partial \theta_1}(\theta)\ \vdots \ \dfrac{\partial J}{\partial \theta_P}(\theta)\end{bmatrix} \in \mathbb{R}^{P}$$

**Hessian matrix** (second-order derivative): $$\nabla^2 J(\theta) = H(\theta) = \begin{bmatrix} \dfrac{\partial^2 J}{\partial\theta_1\partial\theta_1}(\theta) & \cdots & \dfrac{\partial^2 J}{\partial\theta_1\partial\theta_P}(\theta)\ \vdots & \ddots & \vdots\ \dfrac{\partial^2 J}{\partial\theta_P\partial\theta_1}(\theta) & \cdots & \dfrac{\partial^2 J}{\partial\theta_P\partial\theta_P}(\theta) \end{bmatrix} \in \mathbb{R}^{P\times P}$$

---

## 6. Critical Points: Local Minima, Local Maxima, Saddle Points

**Critical point**: $\theta$ such that $\nabla J(\theta) = \mathbf{0}$.

Let $\lambda_1\le\lambda_2\le\dots\le\lambda_P$ be the eigenvalues of $H(\theta)=\nabla^2 J(\theta)$ at a critical point.

|Type|Condition on gradient|Condition on Hessian|Eigenvalue pattern|
|---|---|---|---|
|**Local minimum**|$\nabla J(\theta)=\mathbf{0}$|positive semi-definite|$0\le\lambda_1\le\dots\le\lambda_P$|
|**Local maximum**|$\nabla J(\theta)=\mathbf{0}$|negative semi-definite|$\lambda_1\le\dots\le\lambda_P\le0$|
|**Saddle point**|$\nabla J(\theta)=\mathbf{0}$|indefinite|$\lambda_1\le\dots<0<\dots\le\lambda_P$|


### Worked example: $f(\theta) = \theta_1^2 - \theta_2^2$

**Gradient**: 
$$g = \begin{bmatrix}\dfrac{\partial f}{\partial \theta_1} \\[4pt] \dfrac{\partial f}{\partial \theta_2}\end{bmatrix} = \begin{bmatrix}2\theta_1 \\ -2\theta_2\end{bmatrix} = \begin{bmatrix}0 \\ 0\end{bmatrix} \Rightarrow \theta=(0,0) \text{ is a critical point}$$

**Hessian**: $$H = \begin{bmatrix}\dfrac{\partial^2 f}{\partial\theta_1^2} & \dfrac{\partial^2 f}{\partial\theta_1\partial\theta_2} \\[4pt] \dfrac{\partial^2 f}{\partial\theta_2\partial\theta_1} & \dfrac{\partial^2 f}{\partial\theta_2^2}\end{bmatrix} = \begin{bmatrix}2 & 0 \\ 0 & -2\end{bmatrix}$$

**Eigenvalues**: $\lambda_1=-2 < 0 < 2=\lambda_2$ → $(0,0)$ is a **saddle point**.

---

## 7. Why Saddle Points Dominate in High Dimensions

**Assumption**: Hessian $H(\theta)$ is a random matrix with i.i.d. random eigenvalues, and $\mathbb{P}(\lambda_i\ge0)=0.5$ for each $i$ independently.

**Probabilities**: $$\mathbb{P}(\text{minima}) = \mathbb{P}(\lambda_1\ge0)\mathbb{P}(\lambda_2\ge0)\cdots\mathbb{P}(\lambda_P\ge0) = 0.5^P$$ $$\mathbb{P}(\text{maxima}) = \mathbb{P}(\lambda_1\le0)\cdots\mathbb{P}(\lambda_P\le0) = 0.5^P$$ $$\mathbb{P}(\text{saddle point}) = 1 - \mathbb{P}(\text{minima}) - \mathbb{P}(\text{maxima}) = 1 - 0.5^{P-1}$$

**Ratio**: $${localMinima} :{localMaxima} : {saddlePoint} = 1 : 1 : (2^P - 2)$$

**Key takeaway**: since $P$ (parameter count) can be in the millions, saddle points **exponentially outnumber** local minima/maxima. This is why optimization in deep learning is dominated by concerns about escaping saddle points rather than avoiding poor local minima — empirically, most local minima found by SGD tend to be "good enough," but saddle points can stall training.

The DL loss surface itself (e.g., a ResNet without skip connections) is visually extremely rugged — highly non-linear, non-convex, with many local minima but even more saddle points.

---

## 8. Gradient Descent

**Update rule**: $$\theta_{t+1} = \theta_t - \eta\nabla_\theta J(\theta_t)$$

where $\eta>0$ is the **learning rate**.

**Convergence guarantee**:

- If $J(\cdot)$ is **convex** → guaranteed to converge to the **global minimum**.
- If $J(\cdot)$ is **non-convex** (the DL case) → can get stuck in a local minimum or, more likely given Section 7, a **saddle point**.

### Algorithm

```
Input: objective function J(θ)
Output: optimal solution θ*

1. Initialize θ₀ ~ N(0, σ²) randomly
2. for t = 1 to T:
3.     Compute gradient ∇_θ J(θ_t) = ∂J/∂θ (θ_t)
4.     Update θ_{t+1} = θ_t − η_t ∇_θ J(θ_t)
5. Return θ* = θ_{T+1}
```

### Worked numeric example

**Problem**: $\min_w f(w) = (w-1)^2 + (w-3)^2$

Currently $w_t=1$, so $f(w_t)=f(1)=0^2+2^2=4$. Learning rate $\eta=0.1$.

**Derivative**: $$f'(w) = 2(w-1) + 2(w-3) = 4w-8$$ $$f'(w_t) = f'(1) = -4$$

**Update**: $$w_{t+1} = w_t - \eta f'(w_t) = 1 - 0.1(-4) = 1.4$$

**Verification**: $f(w_{t+1}) = f(1.4) = 2.72 < f(w_t)=4$ ✓ (loss decreased, as expected)

---

## 9. Gradient Descent for Deep Learning — The Cost Problem

**Full training objective**: $$\min_\theta L(D;\theta) := \frac1N\sum_{i=1}^N l(x_i,y_i;\theta) = \frac1N\sum_{i=1}^N l(y_i, f(x_i;\theta))$$

where: $$l(x_i,y_i;\theta) = -\log p(y=y_i|x_i) = -\log\frac{\exp{h^L_{y_i}(x_i)}}{\sum_{m=1}^M \exp{h^L_m(x_i)}}$$

**Gradient descent update**: $$\theta_{t+1} = \theta_t - \eta\nabla_\theta L(D;\theta_t) = \theta_t - \frac{\eta}{N}\sum_{i=1}^N \nabla_\theta l(x_i,y_i;\theta_t)$$

**The problem**: computing $\nabla_\theta L(D;\theta_t)$ requires summing over **all $N$ data points** → computational cost $O(N)$. For big datasets ($N\approx10^6$), this is prohibitively expensive **per single parameter update**.

**Question**: how to estimate $\nabla_\theta L(D;\theta_t)$ more efficiently?

---

## 10. Stochastic Gradient Descent (SGD)

**Idea**: instead of the full gradient, sample a **mini-batch** and use its gradient as an unbiased estimate.

**Sampling**: draw indices $i_1,\dots,i_b \sim \text{Uni}({1,\dots,N})$, where $b$ is the batch size (commonly 32, 64, 128, 256, ...).

**Mini-batch loss estimator**: $$\tilde L(\theta) := \frac1b\sum_{k=1}^b l(x_{i_k}, y_{i_k}; \theta)$$

**Unbiasedness**: $$\mathbb{E}_{i_1,\dots,i_b}\left[\nabla_\theta \tilde L(\theta_t)\right] = \nabla_\theta L(D;\theta_t)$$

Where $\nabla_\theta \tilde L(\theta_t) = \frac1b\sum_{k=1}^b \nabla_\theta l(x_{i_k},y_{i_k};\theta_t)$ is an **unbiased estimator** of the true full-dataset gradient, computed at cost $O(b)$ instead of $O(N)$.

**Update rule**: $$\theta_{t+1} = \theta_t - \eta_t \nabla_\theta \tilde L(\theta_t), \quad \text{with } \eta_t \propto O\left(\frac1t\right)$$

### Worked numeric example

**Function**: $f(w) = \frac{1}{1000}\sum_{i=1}^{1000}(w-i)^2$, solve $\min_w f(w)$ with $\eta=0.1$.

Sample batch indices $i_1=1, i_2=2, i_3=3, i_4=4$. At iteration $t$, $w_t=10$.

**Mini-batch approximation**: $$\tilde f(w) = \frac14\left[(w-1)^2+(w-2)^2+(w-3)^2+(w-4)^2\right]$$

**Derivative**: $$\tilde f'(w) = 2w-5$$ $$\tilde f'(w_t) = \tilde f'(10) = 2(10)-5 = 15$$

**Update**: $$w_{t+1} = w_t - \eta \tilde f'(w_t) = 10 - 0.1(15) = 8.5$$

> Note: the slide's derivation shows an intermediate typo (writing $(w-3)^4$ in one place but arriving at the correct linear $\tilde f'(w)=2w-5$), so the final numeric answer $w_{t+1}=8.5$ is the authoritative result to remember.

---

## 11. SGD Training Loop for Deep Learning

```python
b = 32                          # batch size
iter_per_epoch = N / b          # one epoch = one full pass through all data
n_epoch = 50                    # number of epochs

for epoch in range(1, n_epoch+1):
    for i in range(1, iter_per_epoch+1):
        # Sample a minibatch B = {(x_ij, y_ij)} for j=1..b
        # Forward propagation for B
        # Backward propagation to compute ∂l/∂W^k, ∂l/∂b^k for k=1..L
        for k in range(1, L+1):
            W_k = W_k - eta * dl_dW_k
            b_k = b_k - eta * dl_db_k
```

This is the general training loop pattern used across virtually all deep learning models.

---

## 12. Full Forward–Backward Propagation Derivation (3-Layer Network)

This is the **master template** — derive it once and the pattern applies to networks of any depth.

**Architecture**: $h^0=x \to \bar h^1,h^1=\sigma(\bar h^1) \to \bar h^2,h^2=\sigma(\bar h^2) \to h^3 \to \text{softmax} \to p \to \text{CE loss } l$

### Forward pass

$$\bar h^1 = h^0 W^1+b^1,\quad h^1=\sigma(\bar h^1)$$ $$\bar h^2 = h^1 W^2+b^2,\quad h^2=\sigma(\bar h^2)$$ $$h^3 = h^2 W^3+b^3,\quad p=\text{softmax}(h^3)$$ $$l = CE(1_y, p)$$

### Backward pass (derived layer by layer, from output to input)

**Output layer gradient** (using the "$p-\mathbf1_y$" result from Section 2): $$g^3 = \frac{\partial l}{\partial h^3} = p - \mathbf1_y \in \mathbb{R}^{1\times n_3}$$

**Gradients for $W^3, b^3$**: $$\frac{\partial l}{\partial W^3} = \frac{\partial l}{\partial h^3}\cdot\frac{\partial h^3}{\partial W^3} = (h^2)^T g^3 \in \mathbb{R}^{n_2\times n_3}$$ $$\frac{\partial l}{\partial b^3} = \frac{\partial l}{\partial h^3}\cdot\frac{\partial h^3}{\partial b^3} = g^3 \in \mathbb{R}^{1\times n_3}$$

**Updates**: $$W^3 = W^3 - \eta\frac{\partial l}{\partial W^3}, \qquad b^3 = b^3 - \eta\frac{\partial l}{\partial b^3}$$

**Propagate gradient back to $h^2$**: $$g^2 = \frac{\partial l}{\partial h^2} = \frac{\partial l}{\partial h^3}\cdot\frac{\partial h^3}{\partial h^2} = g^3 W^3 \in \mathbb{R}^{1\times n_2}$$

**Push through the activation** (using the diagonal Jacobian from Section 3): $$\bar g^2 = \frac{\partial l}{\partial \bar h^2} = \frac{\partial l}{\partial h^2}\cdot\frac{\partial h^2}{\partial \bar h^2} = g^2, \text{diag}(\sigma'(\bar h^2)) \in \mathbb{R}^{1\times n_2}$$

**Gradients for $W^2, b^2$**: $$\frac{\partial l}{\partial W^2} = (h^1)^T \bar g^2 \in \mathbb{R}^{n_1\times n_2}, \qquad \frac{\partial l}{\partial b^2} = \bar g^2 \in \mathbb{R}^{1\times n_2}$$

**Updates**: $$W^2 = W^2-\eta\frac{\partial l}{\partial W^2}, \qquad b^2 = b^2-\eta\frac{\partial l}{\partial b^2}$$

**Propagate to $h^1$**: $$g^1 = \frac{\partial l}{\partial h^1} = \bar g^2 W^2 \in \mathbb{R}^{1\times n_1}$$

**Push through activation**: $$\bar g^1 = \frac{\partial l}{\partial \bar h^1} = g^1,\text{diag}(\sigma'(\bar h^1)) \in \mathbb{R}^{1\times n_1}$$

**Gradients for $W^1, b^1$**: $$\frac{\partial l}{\partial W^1} = (h^0)^T \bar g^1 \in \mathbb{R}^{n_0\times n_1}, \qquad \frac{\partial l}{\partial b^1} = \bar g^1 \in \mathbb{R}^{1\times n_1}$$

**Updates**: $$W^1 = W^1-\eta\frac{\partial l}{\partial W^1}, \qquad b^1 = b^1-\eta\frac{\partial l}{\partial b^1}$$

### The repeating pattern (memorize this!)

For each layer $k$ going backward:

1. **Gradient at $\bar h^k$**: $\bar g^k = g^k ,\text{diag}(\sigma'(\bar h^k))$ — pushes gradient through the non-linearity
2. **Weight gradient**: $\dfrac{\partial l}{\partial W^k} = (h^{k-1})^T \bar g^k$ — outer product of previous-layer activation and current gradient
3. **Bias gradient**: $\dfrac{\partial l}{\partial b^k} = \bar g^k$ — same as the pushed-through gradient
4. **Propagate further back**: $g^{k-1} = \bar g^k W^k{}^T$ (or equivalently $\bar g^k$ times $W^k$ depending on row/column convention) — passes gradient to the previous layer

---

## 13. Mini-Batch Feed-Forward with Forward/Backward Arrows

Extends the single-example version to full batches. With batch size $b$, $d=n_0=5$, hidden sizes $n_1=7, n_2=5$, output $n_3=M=4$:

**Forward**: $$h^1 = \sigma(XW^1+b^1),\quad \text{tensor } [b,n_1=7]$$ $$h^2 = \sigma(h^1 W^2+b^2),\quad \text{tensor } [b,n_2=5]$$ $$h^3 = h^2 W^3+b^3,\quad P=\text{softmax}(h^3,\text{dim}=1),\quad \text{tensor } [b,n_3=M=4]$$

**Batch loss**: $$\text{batch_loss} = \frac1b\sum_{i=1}^b CE(1_{y_i},p_i) = -\frac1b\sum_{i=1}^b \log p^i_{y_i}$$

**Backward**: gradients flow in the reverse direction through the same graph, following exactly the layer-by-layer pattern derived in Section 12 — but now every quantity ($h^1, h^2, h^3, g^1, g^2, g^3$, etc.) is a **batch of vectors** (shape $[b, n_k]$) rather than a single row vector, and weight/bias gradients are **summed (or averaged) across the batch dimension**.

---

## Summary Table: Key Formulas (Parts 1+2 Combined)

|Concept|Formula|
|---|---|
|CE gradient w.r.t. logits|$\dfrac{\partial l}{\partial h} = p - \mathbf1_y$|
|Activation layer Jacobian|$\dfrac{\partial h}{\partial x} = \text{diag}(\sigma'(\bar h)),W^T$|
|Sigmoid Jacobian|$\text{diag}(h\otimes(1-h)),W^T$|
|Regularized DL objective|$J(\theta)=\Omega(\theta)+\frac1N\sum_i l(y_i,f(x_i;\theta))$|
|L2 regularization|$\Omega(\theta)=\lambda\sum_k\|W^k\|_F^2$|
|Gradient (1st order)|$g=\nabla J(\theta)\in\mathbb{R}^P$|
|Hessian (2nd order)|$H(\theta)=\nabla^2 J(\theta)\in\mathbb{R}^{P\times P}$|
|Local min/max/saddle|via sign pattern of Hessian eigenvalues $\lambda_1\le\dots\le\lambda_P$|
|Saddle:minima ratio|$1:1:(2^P-2)$|
|GD update|$\theta_{t+1}=\theta_t-\eta\nabla_\theta J(\theta_t)$|
|Full-batch GD cost|$O(N)$ per update|
|SGD mini-batch loss|$\tilde L(\theta)=\frac1b\sum_{k=1}^b l(x_{i_k},y_{i_k};\theta)$|
|SGD cost|$O(b)$ per update, unbiased estimator of full gradient|
|Backprop weight gradient (layer $k$)|$\dfrac{\partial l}{\partial W^k}=(h^{k-1})^T \bar g^k$|
|Backprop bias gradient (layer $k$)|$\dfrac{\partial l}{\partial b^k}=\bar g^k$|
|Backprop gradient through activation|$\bar g^k = g^k,\text{diag}(\sigma'(\bar h^k))$|