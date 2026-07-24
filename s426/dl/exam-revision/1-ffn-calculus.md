| Week | Topic                                                         |
| ---- | ------------------------------------------------------------- |
| 1    | ML & math background revisit                                  |
| 2    | Prelude to DL, Feed-forward NNs                               |
| 3    | DL for Vision I: CNNs                                         |
| 4    | Backpropagation & Optimization                                |
| 5    | Practical skills in DL                                        |
| 6    | DL for Vision II: architectures, visualization, robustness    |
| 7    | DL for time-series: RNN, LSTM                                 |
| 8    | Representation learning: Word2Vec                             |
| 9    | Advanced sequential models (Seq2Seq, attention, Transformers) |
| 10   | ViT & model fine-tuning                                       |
| 11   | Deep generative models (GAN, Diffusion)                       |
# FIT5215 — Deep Learning Revision Notes (Part 1: Feed-Forward Networks & Calculus Foundations)

_Source: Final Exam Revision slides, pp. 1–20_

## 1. Unit Overview

**Model families covered in this unit:**

- **Supervised models**: Feed-forward NN, CNN, RNN, Advanced sequential models
- **Self-supervised models**: Word2Vec, Seq2Seq, Transformer
- **Generative models**: GAN, Diffusion

**General supervised training loop** (applies across all model types): $$\min_\theta J(\theta) = \Omega(\theta) + \frac{1}{N}\sum_{i=1}^{N} l(y_i, f(x_i;\theta))$$

where $\Omega(\theta)$ is an (optional) regularization term.

Training proceeds in two passes:

- **Forward propagation**: Input → Layers → Prediction probabilities → Loss
- **Backward propagation**: Loss → Layers → Input (computes gradients)

**Week-to-topic map:**



---

## 2. Feed-Forward Neural Networks: Parameterisation

### Layer notation

- Input layer: $h^0(x) = x$, a $d$-dimensional vector, $x = [x^1,\dots,x^d] \in \mathbb{R}^{1\times d}$
- Hidden layers: $h^1(x), \dots, h^{L-1}(x)$
- Output layer: $h^L(x)$, has $M$ nodes providing discriminative values, followed by a **softmax layer** for classification, giving prediction $\hat y$

### Layer sizes

- Layer $k$ has $n_k$ neurons
- $n_0 = d$ (input dimension), $n_L = M$ (output dimension / number of classes)

### Weight matrices and biases

$$W^k \in \mathbb{R}^{n_{k-1}\times n_k}, \quad b^k \in \mathbb{R}^{1\times n_k} \quad \text{for } k = 1,2,\dots,L$$

Each layer transition ($h^0\to h^1$, $h^1 \to h^2$, ..., $h^{L-1}\to h^L$) has its own weight matrix / bias pair.

---

## 3. Forward Propagation — Classification

**Notation**: $h^0(x) = x$ with ground-truth label $y$.

**Algorithm:** $$\text{for } k = 1 \text{ to } L-1: \quad \bar h^k(x) = h^{k-1}(x)W^k + b^k \quad \text{}$$ $$h^k(x) = \sigma(\bar h^k(x)) \quad \text{(activation — reduces non-linearity to linear function)}$$ $$h^L(x) = h^{L-1}(x)W^L + b^L$$ $$p(x) = \text{softmax}(h^L(x)) \quad \text{(prediction probabilities)}$$

**Parameters**: $\theta = {(W^k,b^k)}_{k=1}^{L}$ (given/learned)

**Activation function** $\sigma:\mathbb{R}\to\mathbb{R}$ — e.g. sigmoid, tanh, ReLU

**Output**: $$\Pr(y=k\mid x) = p_k(x) = \frac{\exp{h_k^L(x)}}{\sum_{j=1}^{M}\exp{h_j^L(x)}}$$ $$\hat y = \arg\max_{1\le k \le M} p_k(x)$$

**Cross-entropy loss for one sample:** $$CE(x,y) = -\log p_y(x)$$

---

## 4. Modern Activation Functions

All three activations below are logistic S-shaped (sigmoid, tanh) or piecewise-linear (ReLU), continuous, and (mostly) differentiable.

### Sigmoid

$$s(z) = \sigma(z) = \frac{1}{1+e^{-z}}$$ $$\sigma'(z) = \sigma(z)(1-\sigma(z))$$

- Output range: $(0,1)$

### Hyperbolic tangent (tanh)

$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$ $$\sigma'(z) = 1-\sigma^2(z)$$

- Output range: $(-1,1)$ — zero-centred output helps speed up convergence relative to sigmoid.

### ReLU (rectified linear unit)

$$\text{ReLU}(z) = \max{0,z}$$
$$\sigma'(z) = \begin{cases}1 &  z \ge \\ 0 & otherwise \end{cases}$$
- Continuous, but **not differentiable at 0** — can make gradient descent "bounce around" near 0.
- Very fast to compute; works extremely well in practice.
- No upper cap on output — helps counter the vanishing-gradient problem (sigmoid/tanh saturate for large $|z|$, driving gradients toward zero; ReLU's gradient is exactly 1 for all $z>0$, so it doesn't shrink).

### Why activations matter

Without any activation function, stacking multiple linear layers is mathematically equivalent to a single linear layer — the whole network can only ever draw a **straight-line (or hyperplane) decision boundary**, regardless of depth.

With non-linear activations, the same network can carve out an **arbitrarily complex, curved decision boundary** that fits non-linearly-separable data.

---

## 5. Softmax Layer

**Purpose**: transforms real-valued discriminative scores (logits) $h$, ranged in $(-\infty,+\infty)$, into a valid probability distribution $p$ ranged in $[0,1]$ — **while preserving the relative order** of the scores.

**Input**: vector of dimension $M$ (e.g. $h=[h_1,h_2,h_3]=[1,2,3]$, $M=3$)

**Output**: discrete distribution of dimension $M$ (e.g. $p=[0.09, 0.24, 0.67]$)

**Formula:** $$p = \text{softmax}(h) := \left[\frac{\exp{h_m}}{\sum_{i=1}^{M}\exp{h_i}}\right]_{m=1}^{M}$$

**Properties** — $p$ is a proper distribution over classes ${1,\dots,M}$: $$p_m \ge 0 \ (1\le m \le M), \qquad \sum_{m=1}^{M} p_m = 1$$

**Prediction rule**: return the class with highest probability $$\hat y = \arg\max_{1\le m\le M} p_m$$

For the example above ($h=[1,2,3]$), the prediction is class 3, since it has the largest logit and therefore the largest softmax probability (softmax is monotonic, so order is preserved from logits to probabilities).

---

## 6. Worked Example: Spam Email Detection (Forward Prop)

**Setup**: 3 extracted email features, $x=[x_1,x_2,x_3]\in\mathbb{R}^3$. Two classes: spam ($y=1$), non-spam ($y=2$).

**Architecture**: $3 \to 4\ (\text{sigmoid}) \to 3\ (\text{sigmoid}) \to 2\ (\text{softmax})$

$$h^0(x) = x = [1.2,\ -1,\ 2.2] \in \mathbb{R}^{1\times 3}$$

**Hidden layer 1:** $$\bar h^1(x) = h^0(x)W^1 + b^1 \in \mathbb{R}^{1\times 4}, \qquad W^1\in\mathbb{R}^{3\times4},\ b^1\in\mathbb{R}^{1\times4}$$ $$h^1(x) = \text{sigmoid}(\bar h^1(x)) \in \mathbb{R}^{1\times4}$$

**Hidden layer 2:** $$\bar h^2(x) = h^1(x)W^2 + b^2 \in \mathbb{R}^{1\times 3}, \qquad W^2\in\mathbb{R}^{4\times3},\ b^2\in\mathbb{R}^{1\times3}$$ $$h^2(x) = \text{sigmoid}(\bar h^2(x))$$

**Output layer:** $$h^3(x) = h^2(x)W^3 + b^3 \in \mathbb{R}^{1\times 2}, \qquad W^3\in\mathbb{R}^{3\times2},\ b^3\in\mathbb{R}^{1\times2}$$ $$p(x) = \text{softmax}(h^3(x)) \in \mathbb{R}^{1\times2}$$

**Result**: assume $p(x) = [0.3, 0.7]$. Prediction $\hat y = 2$ (non-spam), but true label $y=1$ (spam) → **incorrect prediction**.

> Exercise from slides: check all dimensions for matrix-multiplication consistency, and compute the cross-entropy loss for this case ($CE = -\log p_1(x) = -\log(0.3)$).

---

## 7. Forward Propagation — Regression

Same structure as classification, but **no softmax** at the end — the output is the raw prediction directly.

$$h^0(x) = x$$ $$\text{for } k=1 \text{ to } L-1: \quad \bar h^k(x) = h^{k-1}(x)W^k + b^k \quad \text{(linear)}$$ $$h^k(x) = \sigma(\bar h^k(x)) \quad \text{(activation)}$$ $$h^L(x) = h^{L-1}(x)W^L + b^L$$ $$\hat y = h^L(x)$$

---

## 8. Training Deep Networks — Loss Function

**Parameters**: $\theta := {(W^l,b^l)}_{l=1}^{L}$

**Training set**: $D = {(x_1,y_1),\dots,(x_N,y_N)}$

**Goal**: find $\theta$ so the model's predictions fit the training set as well as possible: $$\min_\theta L(D;\theta)$$

**Loss function** (average cross-entropy over the dataset, i.e. negative log-likelihood): $$L(D;\theta) := \frac{1}{N}\sum_{i=1}^{N} CE(1_{y_i}, p(x_i)) = -\frac{1}{N}\sum_{i=1}^{N}\log p_{y_i}(x_i)$$

**Per-sample loss:** $$l(y,\hat y) = CE(1_y, p(x)) = -\log p_y(x)$$

**Optimizers** used to minimize this loss: SGD, Adagrad, Adam, RMSProp (covered in Lecture 4). Frameworks like `TorchOpt` map string names directly to optimizer classes, e.g. `"Adam": torch.optim.Adam`.

---

## 9. Mini-Batch Training

Rather than computing the loss over one example at a time, training operates on **batches** of examples simultaneously (tensor operations), which is far more computationally efficient.

### Shapes through the network

Given a batch of size $b$ and input dimension $d=n_0$:

- **Input**: Batch $X$: $[b, n_0=d]$
- **Hidden layer 1**: $h^1 = \sigma(XW^1+b^1)$, shape $[b, n_1]$
- **Hidden layer 2**: $h^2 = \sigma(h^1 W^2+b^2)$, shape $[b, n_2]$
- **Output layer**: $h^3 = h^2 W^3+b^3$; $P=\text{softmax}(h^3,\ \text{dim}=1)$, shape $[b, n_L=M]$

### Batch loss

$$BatchLoss = \frac{1}{b}\sum_{i=1}^{b} CE(1_{y_i}, p_i) = -\frac{1}{b}\sum_{i=1}^{b}\log p^i_{y_i}$$

Weight matrices and biases are updated to **minimize the batch loss**, not just a single example's loss.

### One epoch

One epoch = passing through **every** mini-batch that partitions the full training set exactly once: $$(x_1,y_1),\dots,(x_b,y_b), (x_{b+1},y_{b+1}),\dots (x_{2b},y_{2b}),\ \dots \ (\phi,\psi),\dots (x_N,y_N)$$

After training, the model is evaluated on a held-out **testing set** to obtain test accuracy.

### PyTorch worked example (batch of 32, $d=5$, hidden sizes 7 and 5, $M=4$ output classes)

```python
# Declare FFN
torch.manual_seed(1234)
W1 = torch.randn(5, 7);  b1 = torch.randn(1, 7)
W2 = torch.randn(7, 5);  b2 = torch.randn(1, 5)
W3 = torch.randn(5, 4);  b3 = torch.randn(1, 4)

# Declare a batch
x = torch.rand(size=(4, 5))

# Forward propagation
hbar_1 = torch.matmul(x, W1) + b1
h1 = torch.relu(hbar_1)                 # shape: [4, 7]

hbar_2 = torch.matmul(h1, W2) + b2
h2 = torch.relu(hbar_2)                 # shape: [4, 5]

h3 = torch.matmul(h2, W3) + b3          # logits, shape: [4, 4]
p = torch.softmax(h3, dim=1)            # prediction probabilities

# Making prediction
yhat = torch.argmax(p, dim=1)

# Loss (assume ground-truth labels y = [0,1,0,1])
y = torch.tensor([0, 1, 0, 1])
one_hot_y = torch.nn.functional.one_hot(y, num_classes=4)
loss = -torch.mean(one_hot_y * torch.log(p))
```

**Sample output** (with `torch.manual_seed(1234)` and this batch of $x$):

```
p: tensor([[0.7601, 0.0210, 0.0257, 0.1932],
           [0.5877, 0.0427, 0.0372, 0.3323],
           [0.7814, 0.0410, 0.0526, 0.1251],
           [0.7531, 0.0460, 0.0509, 0.1499]])
yhat: tensor([0, 0, 0, 0])
loss: 0.4221
```

> Note: with this random seed, the model predicts class 0 for every sample, which mismatches 2 of the 4 true labels — illustrating an under-trained (randomly initialized) network.

### Single-example worked version (for comparison)

```python
x = torch.tensor([[1.0, 2.0, -2.0, 0.2, 5.7]])   # shape [1, 5]
# ... same W1,b1,W2,b2,W3,b3 as above ...
hbar_1 = torch.matmul(x, W1) + b1        # tensor([[8.9391, -4.6345, 7.4453, -0.9053, -3.9509, 1.2647, -8.7399]])
h1 = torch.relu(hbar_1)                  # negative entries zeroed out

hbar_2 = torch.matmul(h1, W2) + b2
h2 = torch.relu(hbar_2)

h3 = torch.matmul(h2, W3) + b3           # logit h3: [[3.5699, -10.3066, -5.5052, -2.0499]]
p = torch.softmax(h3, dim=1)             # p: [[9.9627e-01, 9.37e-07, 1.14e-04, 3.61e-03]]
yhat = torch.argmax(p, dim=1)            # yhat: 0

# Assume ground truth y = 1 (0-indexed)
loss = -torch.log(p[0, 1])               # loss: 3.8616
```

This shows a case where the model confidently (99.6%) predicts class 0, but the true label is class 1 — producing a **large** loss (3.86), consistent with the low probability assigned to the correct class.

---

## 10. Calculus Foundations (for Backpropagation)

> "Calculus = mathematics of change" — essential for understanding backprop.

### Single-variable derivative rules

$$f'(x) = \nabla f(x) = \lim_{h\to0}\frac{f(x+h)-f(x)}{h}$$

$$(uv)' = u'v + uv'$$ $$\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$$ $$(e^u)' = u'e^u$$ $$(\log u)' = \frac{u'}{u}$$

### Gradient of a multivariate scalar function

For $f:\mathbb{R}^n\to\mathbb{R}$ with $y=f(x)=f(x_1,\dots,x_n)$: $$\frac{\partial f}{\partial x}(a) = \nabla_x f(a) = [\nabla_{x_1}f(a), \nabla_{x_2}f(a),\dots,\nabla_{x_n}f(a)]$$

### Basic chain rule

$$\frac{\partial u}{\partial x} = \frac{\partial u}{\partial v}\times\frac{\partial v}{\partial x}$$

### Derivative for vector-valued multivariate functions (Jacobian)

Given $f:\mathbb{R}^m\to\mathbb{R}^n$, $f(x)=(f_1(x),\dots,f_n(x))$ where each $f_i:\mathbb{R}^m\to\mathbb{R}$ and $x=(x_1,\dots,x_m)$. Let $y=f(x)$.

The derivative of $f$ at point $a\in\mathbb{R}^m$ — written $\nabla f(a)$ (function notation) or $\frac{\partial y}{\partial x}(a)$ (variable notation) — is an $n\times m$ matrix, the **Jacobian matrix**:

$$\frac{\partial y}{\partial x}(a) = \nabla f(a) = \begin{bmatrix} \dfrac{\partial f_1}{\partial x_1}(a) & \cdots & \dfrac{\partial f_1}{\partial x_m}(a) \\[6pt] \vdots & \ddots & \vdots \\[6pt] \dfrac{\partial f_n}{\partial x_1}(a) & \cdots & \dfrac{\partial f_n}{\partial x_m}(a) \end{bmatrix} \quad \in \mathbb{R}^{n \times m}$$
### Chain rule (multivariate / matrix form)

Given $f:\mathbb{R}^m\to\mathbb{R}^n$ and $g:\mathbb{R}^n\to\mathbb{R}^p$, define $h = g\circ f : \mathbb{R}^m \to \mathbb{R}^p$, i.e. $h(x) = g(f(x))$.

Let $y=f(x)$ and $z=g(y)=g(f(x))=h(x)$.

For $x\in\mathbb{R}^m$: $$\nabla h(x) = \nabla g(f(x)) \times \nabla f(x) \quad \text{or equivalently} \quad \frac{\partial z}{\partial x} = \frac{\partial z}{\partial y}\cdot\frac{\partial y}{\partial x}$$

**Dimension check** (this is the key practical takeaway for backprop bookkeeping): $$\underbrace{\frac{\partial z}{\partial x}}_{p\times m} = \underbrace{\frac{\partial z}{\partial y}}_{p\times n} \cdot \underbrace{\frac{\partial y}{\partial x}}_{n\times m}$$

This is exactly the mechanism that lets backpropagation chain gradients backward through arbitrarily many layers: each layer contributes one Jacobian factor, and the matrix dimensions must line up for the multiplication to be valid — a useful sanity check when deriving backprop by hand.

---

## Summary Table: Key Formulas

| Concept              | Formula                                                                                         |
| -------------------- | ----------------------------------------------------------------------------------------------- |
| Layer weights/biases | $W^k\in\mathbb{R}^{n_{k-1}\times n_k},\ b^k\in\mathbb{R}^{1\times n_k}$                         |
| Linear step          | $\bar h^k(x) = h^{k-1}(x)W^k+b^k$                                                               |
| Activation step      | $h^k(x)=\sigma(\bar h^k(x))$                                                                    |
| Softmax              | $p_m = \dfrac{\exp{h_m}}{\sum_i \exp{h_i}}$                                                     |
| Prediction           | $\hat y=\arg\max_m p_m(x)$                                                                      |
| Per-sample CE loss   | $CE(x,y)=-\log p_y(x)$                                                                          |
| Dataset loss         | $L(D;\theta)=-\frac1N\sum_i \log p_{y_i}(x_i)$                                                  |
| Batch loss           | $\frac1b\sum_{i=1}^b CE(1_{y_i},p_i)$                                                           |
| Sigmoid              | $\sigma(z)=\frac{1}{1+e^{-z}},\ \sigma'=\sigma(1-\sigma)$                                       |
| Tanh                 | $\tanh(z)=\frac{e^z-e^{-z}}{e^z+e^{-z}},\ \sigma'=1-\sigma^2$                                   |
| ReLU                 | $\max(0,z),\ \sigma'=\mathbb{1}[z\ge0]$                                                         |
| Jacobian             | $\nabla f(a) \in \mathbb{R}^{n\times m}$ for $f:\mathbb{R}^m\to\mathbb{R}^n$                    |
| Chain rule (matrix)  | $\frac{\partial z}{\partial x}=\frac{\partial z}{\partial y}\cdot\frac{\partial y}{\partial x}$ |
chunk 9
