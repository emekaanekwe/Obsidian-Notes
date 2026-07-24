

## tags: [FIT5215, deep-learning, data-augmentation, adversarial-robustness, RNN] pages: 101-120

# FIT5215 — Deep Learning: Pages 101–120

## 1. Advanced Data Augmentation: Mixup & CutMix

### 1.1 Mixup (p. 101)

**Idea:** create synthetic training examples by linearly blending pairs of images **and** their labels.

**Algorithm** — for $(x_1,y_1), (x_2,y_2)$ sampled from batch 1 and batch 2:

1. Sample mixing coefficient: $\lambda \sim \text{Beta}(\alpha,\alpha)$
2. Blend the images: $\tilde{x} = \lambda \times x_1 + (1-\lambda)\times x_2$
3. Blend the (one-hot) labels: $\tilde{y} = \lambda \times \mathbf{1}_{y_1} + (1-\lambda)\times \mathbf{1}_{y_2}$
4. Update the optimizer to minimize: $CE(\tilde{y}, p(\tilde{x}))$

> **Intuition:** the model is trained to output _proportionally blended_ predictions for _proportionally blended_ inputs — encourages linear behaviour between classes and improves generalization/calibration.

**Paper:** [Zhang et al., ICLR 2018 — mixup: Beyond Empirical Risk Minimization](https://openreview.net/pdf?id=r1Ddp1-Rb)

```python
import numpy as np
import torch

def mixup_batch(x1, y1_onehot, x2, y2_onehot, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    x_tilde = lam * x1 + (1 - lam) * x2
    y_tilde = lam * y1_onehot + (1 - lam) * y2_onehot
    return x_tilde, y_tilde

# Training loop usage:
# x_tilde, y_tilde = mixup_batch(x1, y1_onehot, x2, y2_onehot)
# loss = cross_entropy(p(x_tilde), y_tilde)   # soft-label CE
```

---

### 1.2 CutMix (p. 102)

**Idea:** instead of blending pixel intensities, **cut a patch** from one image and **paste it** into another, and mix the labels in proportion to the patch area.

For images $x_1, x_2 \in \mathbb{R}^{C\times W \times H}$ with labels $\mathbf{1}_{y_1}, \mathbf{1}_{y_2}$:

$$\tilde{x} = M\odot x_1 + (1-M)\odot x_2, \qquad M \in {0,1}^{H\times W}$$

$$\tilde{y} = \lambda \times \mathbf{1}_{y_1} + (1-\lambda) \times \mathbf{1}_{y_2}$$

**Algorithm** — for $(x_1,y_1), (x_2,y_2)$:

1. $\lambda \sim \text{Beta}(\alpha,\alpha)$
2. Sample a bounding box $B = [r_x, r_y, r_w, r_h]$:
    - $r_x \sim \text{Uni}[0,W]$, $r_w \sim W\sqrt{1-\lambda}$
    - $r_y \sim \text{Uni}[0,H]$, $r_h \sim H\sqrt{1-\lambda}$
3. Construct binary mask $M \in {0,1}^{W\times H}$: fill **1** outside $B$, **0** inside $B$
4. $\tilde{x} = M\odot x_1 + (1-M)\odot x_2$
5. $\tilde{y} = \lambda\times \mathbf{1}_{y_1} + (1-\lambda)\times \mathbf{1}_{y_2}$
6. Minimize $CE(\tilde{y}, p(\tilde{x}))$

**Key relationship** (area of cut patch ↔ $\lambda$):

$$\frac{\text{area}(B)}{\text{area}(\text{image})} = \frac{WH(1-\lambda)}{WH} = 1-\lambda$$

> The proportion of the image replaced by the patch is exactly $1-\lambda$ — so the label mix ratio matches the visible pixel ratio from each source image.

**Paper:** [Yun et al. — CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/pdf/1905.04899)

```python
import numpy as np
import torch

def rand_bbox(W, H, lam):
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def cutmix_batch(x1, y1_onehot, x2, y2_onehot, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    _, _, H, W = x1.shape
    bx1, by1, bx2, by2 = rand_bbox(W, H, lam)
    x_tilde = x1.clone()
    x_tilde[:, :, by1:by2, bx1:bx2] = x2[:, :, by1:by2, bx1:bx2]
    lam_adj = 1 - ((bx2 - bx1) * (by2 - by1) / (W * H))  # adjust lambda to actual area
    y_tilde = lam_adj * y1_onehot + (1 - lam_adj) * y2_onehot
    return x_tilde, y_tilde
```

---

## 2. Adversarial Attack and Defense

### 2.1 Core Notation (p. 104)

$$x,\ y \in {1,2,\dots,M}$$

$$h(x;\theta) \quad \text{(logits)}$$

$$f(x;\theta) = \text{softmax}(h(x;\theta)) \quad \text{(prediction probabilities)}$$

$$f(x;\theta) = \big[f_m(x;\theta)\big]_{m=1}^{M}, \qquad p(y=m\mid x) = f_m(x;\theta) \ \ \text{for } 1\le m \le M$$

**Cross-entropy loss for a single instance:**

$$l\big(f(x;\theta), y\big) = CE\big(\mathbf{1}_y, f(x;\theta)\big) = -\log f_y(x;\theta)$$

---

### 2.2 Distance Between Two Images (p. 105)

For pixel $x_{ijk}$ ($1\le i \le C$ channels, $1\le j \le H$, $1 \le k \le W$), with $0\le x_{ijk}\le 255$ (colour) or $0\le x_{ijk}\le 1$ (greyscale):

**L2 distance:**

$$|x-x'|_2 = \sqrt{\sum_{i=1}^{C}\sum_{j=1}^{H}\sum_{k=1}^{W}\big(x_{ijk}-x'_{ijk}\big)^2}$$

**L1 distance:**

$$|x-x'|_1 = \sum_{i=1}^{C}\sum_{j=1}^{H}\sum_{k=1}^{W}\left|x_{ijk}-x'_{ijk}\right|$$

**L∞ distance:**

$$|x-x'|_\infty = \max_{1\le i\le C,, 1\le j\le H,, 1\le k\le W} \left|x_{ijk}-x'_{ijk}\right|$$

---

### 2.3 Adversarial Examples: Definition (p. 106–107)

**The true sense of a robust model:** if $x'$ and $x$ are close, their predictions must be the same:

$$\text{If } x' \in B_\epsilon(x) \ \big(\text{i.e., } |x'-x|\le\epsilon\big), \quad \arg\max_{1\le i \le M} f_i(x;\theta) = \hat{y} = \hat{y}' = \arg\max_{1\le i\le M} f_i(x';\theta)$$

where the **$\epsilon$-ball** around $x$ is:

$$B_\epsilon(x) = {x' : |x'-x| \le \epsilon}, \quad \text{small } \epsilon > 0$$

**Unfortunately**, adversarial examples $x_{adv}\in B_\epsilon(x)$ can easily be found that fool the classifier:

$$\arg\max_{1\le i\le M} f_i(x;\theta) = \hat{y} \ne \hat{y}_{adv} = \arg\max_{1\le i\le M} f_i(x_{adv};\theta)$$

**Formal definition of an adversarial example** $x_{adv}$ of clean example $x$ w.r.t. model $f(\cdot;\theta)$:

- $d(x_{adv}, x) = |x_{adv}-x| \le \epsilon$ for some norm $|\cdot|$ (e.g. $|\cdot|_1, |\cdot|_2, |\cdot|_\infty$)
- $f$ predicts $x$ and $x_{adv}$ with **different** labels: $\arg\max_i f_i(x;\theta) \ne \arg\max_i f_i(x_{adv};\theta)$

> **Real-world relevance:** small, human-imperceptible perturbations ("crafted noise") added to an image can flip a classifier's prediction — a key vulnerability e.g. for misleading autonomous vehicle vision systems.

---

### 2.4 Decision Regions & Loss Surface Geometry (p. 108–110)

**Setup:** define two orthogonal directions in input space:

- $g = \nabla_x l\big(f(x;\theta), y\big)$ — the **gradient direction**
- $g^{\perp}$ — a direction orthogonal to $g$, i.e., $g^T g^{\perp} = 0$

Points near $x$ are parameterized as:

$$x' = x + \alpha g + \beta g^{\perp}$$

Plotting the model's predicted class (colour) as a function of $(\alpha,\beta)$ reveals **decision regions**: starting at $x = x + 0\cdot g + 0\cdot g^{\perp}$ (coordinate $(0,0)$), moving along the gradient direction $g$ quickly exits the current decision region — showing that:

- $x_{adv} = x + \alpha g + 0\cdot g^{\perp}$ is **visually indistinguishable** from $x$ (same class to a human)
- $x_{adv}$ is classified into a **different class** by the model
- This demonstrates that **deep learning models are fragile** and easy to attack — observed consistently across VGG-16, ResNet-50/101/152, and GoogLeNet architectures

**Paper:** [Liu et al., ICLR 2016 — Delving into Transferable Adversarial Examples and Black-box Attacks](https://arxiv.org/) _(decision region visualizations)_

**Loss surface interpretation (p. 110):**

Using a first-order Taylor expansion around $x$:

$$l\big(f(x+h;\theta), y\big) \approx l\big(f(x;\theta), y\big) + g^{T}h, \qquad g = \nabla_x l\big(f(x;\theta), y\big)$$

- The gradient direction $g$ is the **steepest local direction to increase the loss**
- Plotting the loss surface on $x' = x + \epsilon_1 g + \epsilon_2 g^{\perp}$ shows a **sharp ridge along $g$** but flatness along $g^{\perp}$ — confirming that moving along the gradient is the most efficient way to increase loss (i.e., to craft an adversarial example)

**Paper:** [Tramèr et al., ICLR 2017 — Ensemble Adversarial Training: Attacks and Defences]

---

### 2.5 Adversarial Attacks: FGSM and PGD

#### Untargeted Attacks (p. 111)

**Goal:** push $x$ across the decision boundary to _any_ incorrect class — maximize the loss w.r.t. the _true_ label $y$:

$$x_{adv} = \arg\max_{x' \in B_\epsilon(x)} l\big(f(x';\theta), y\big)$$

**Fast Gradient Sign Method (FGSM)** — one-step update:

$$x_{adv} = x + \epsilon \cdot \text{sign}\big(\nabla_x l(f(x;\theta), y)\big)$$

$$ \text{sign}(t) = \begin{cases} 1 & \text{if } t > 0 \ -1 & \text{if } t < 0 \ 0 & \text{otherwise} \end{cases} $$

**Paper:** [Goodfellow et al., ICLR 2015 — Explaining and Harnessing Adversarial Examples]

**Projected Gradient Descent / Ascent (PGD)** — iterative, multi-step:

$$x_0 = x + \text{Uniform}([-\epsilon,\epsilon])$$

$$\tilde{x}_{t+1} = x_t + \eta\cdot\text{sign}\big(\nabla_x l(f(x_t;\theta),y)\big)$$

$$x_{t+1} = \text{Proj}_{B_\epsilon(x)}(\tilde{x}_{t+1})$$

- Run for $k$ steps (commonly $k=20$), $\eta > 0$ is the step size (learning rate)
- Final adversarial example: $x_{adv} = x_k$

**Paper:** [Madry et al., ICLR 2017 — Towards Deep Learning Models Resistant to Adversarial Attacks]

```python
import torch

def fgsm_untargeted(x, y, model, loss_fn, epsilon):
    x = x.clone().detach().requires_grad_(True)
    loss = loss_fn(model(x), y)
    loss.backward()
    x_adv = x + epsilon * x.grad.sign()
    return x_adv.detach()

def pgd_untargeted(x, y, model, loss_fn, epsilon, eta, k=20):
    x_adv = x + torch.empty_like(x).uniform_(-epsilon, epsilon)
    for _ in range(k):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        loss = loss_fn(model(x_adv), y)
        loss.backward()
        x_adv = x_adv + eta * x_adv.grad.sign()
        # Project back into the epsilon-ball around x
        x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
        x_adv = x_adv.detach()
    return x_adv
```

---

#### Targeted Attacks (p. 112)

**Goal:** push $x$ toward a _specific incorrect_ target label $y_{\ne}$ (different from true $y$) — **minimize** the loss w.r.t. the target label:

$$x_{adv} = \arg\min_{x' \in B_\epsilon(x)} l\big(f(x';\theta), y_{\ne}\big)$$

**FGSM (targeted)** — note the **minus sign** (opposite direction from untargeted):

$$x_{adv} = x - \epsilon\cdot\text{sign}\big(\nabla_x l(f(x;\theta), y_{\ne})\big)$$

**PGD (targeted):**

$$x_0 = x + \text{Uniform}([-\epsilon,\epsilon])$$

$$\tilde{x}_{t+1} = x_t - \eta\cdot\text{sign}\big(\nabla_x l(f(x_t;\theta), y_{\ne})\big)$$

$$x_{t+1} = \text{Proj}_{B_\epsilon(x)}(\tilde{x}_{t+1})$$

- Run for $k$ steps ($k=20$), $\eta > 0$ learning rate
- $x_{adv} = x_k$

```python
def fgsm_targeted(x, y_target, model, loss_fn, epsilon):
    x = x.clone().detach().requires_grad_(True)
    loss = loss_fn(model(x), y_target)
    loss.backward()
    x_adv = x - epsilon * x.grad.sign()   # note: minus sign for targeted attack
    return x_adv.detach()
```

> **Untargeted vs. targeted — key sign difference:** untargeted attacks **ascend** the loss w.r.t. the true label (push away from correct class); targeted attacks **descend** the loss w.r.t. the chosen wrong label (pull toward a specific incorrect class).

---

### 2.6 Adversarial Training (p. 114)

**Goal:** train a model whose loss surface is **smooth (flat)** around each data point $x$, so small perturbations don't drastically change predictions.

**Ideal (min-max) optimization problem:**

$$\min_{\theta} \ \mathbb{E}_{(x,y)\sim p(x,y)} \left[\max_{x'\in B_\epsilon(x)} l\big(f(x';\theta), y\big)\right]$$

- $p(x,y)$ is the true data distribution generating $(x,y)$
- **Inner maximization**: find the worst-case (most violated) adversarial example within the $\epsilon$-ball
- **Outer minimization**: train $\theta$ to minimize loss even on these worst-case examples

**PGD Adversarial Training Algorithm:**

```
for epoch in n_epochs:
    for iter in range(n_iter_per_epoch):
        Sample mini-batch (x_1,y_1), ..., (x_b,y_b) from training set
        Find PGD untargeted adversarial examples x_1^adv, ..., x_b^adv
            for x_1,...,x_b w.r.t. labels y_1,...,y_b
        batch_loss = (1/b) * sum_i l(f(x_i;theta), y_i)
                   + (1/b) * sum_i l(f(x_i^adv;theta), y_i)
        theta = theta - eta * d(batch_loss)/d(theta)
```

Formally:

$$\text{batch_loss} = \frac{1}{b}\sum_{i=1}^{b} l\big(f(x_i;\theta), y_i\big) + \frac{1}{b}\sum_{i=1}^{b} l\big(f(x_i^{adv};\theta), y_i\big)$$

$$\theta = \theta - \eta\frac{\partial, \text{batch_loss}}{\partial \theta}$$

```python
def pgd_adversarial_training_step(model, optimizer, loss_fn, x_batch, y_batch, epsilon, eta_pgd, k=20):
    # 1. Generate PGD adversarial examples for the batch
    x_adv_batch = pgd_untargeted(x_batch, y_batch, model, loss_fn, epsilon, eta_pgd, k)

    # 2. Compute combined loss on clean + adversarial examples
    optimizer.zero_grad()
    clean_loss = loss_fn(model(x_batch), y_batch)
    adv_loss = loss_fn(model(x_adv_batch), y_batch)
    batch_loss = clean_loss + adv_loss

    # 3. Backprop and update
    batch_loss.backward()
    optimizer.step()
    return batch_loss.item()
```

---

## 3. Recurrent Neural Networks — Introduction

### 3.1 Simplest RNN: Two Time-Slices, No Output (p. 116)

**Input:** $x_0, x_1 \in \mathbb{R}^{1\times\text{input_size}}$

**Hidden state at $t=0$:**

$$h_0 = \tanh(x_0 U + b) \in \mathbb{R}^{1\times\text{hidden_size}}$$

where $U \in \mathbb{R}^{\text{input_size}\times\text{hidden_size}}$

**Hidden state at $t=1$** (function of $h_0$ and $x_1$):

$$h_1 = \tanh(h_0 W + x_1 U + b) \in \mathbb{R}^{1\times\text{hidden_size}}$$

where $W \in \mathbb{R}^{\text{hidden_size}\times\text{hidden_size}}$

> **Key structural idea:** $U$ (input-to-hidden) and $b$ (bias) are **shared/reused** at every time step; $W$ (hidden-to-hidden) connects consecutive hidden states.

---

### 3.2 Batched RNN Implementation (p. 117)

**Input (batched):** $X_0, X_1 \in \mathbb{R}^{bs \times \text{input_size}}$

$$h_0 = \tanh(X_0 U + b) \in \mathbb{R}^{bs\times\text{hidden_size}}$$

$$h_1 = \tanh(h_0 W + X_1 U + b) \in \mathbb{R}^{bs\times\text{hidden_size}}$$

**Tensor shape conventions:**

- $[batch_size, seq_len, input_size]$ — "batch-first" convention
- $[seq_len, batch_size, input_size]$ — "sequence-first" convention (obtained via `torch.transpose(X, 0, 1)`)

```python
import numpy as np
import torch

hidden_size = 5
input_size = 3

# Creating the parameters
U = torch.nn.Parameter(torch.randn(input_size, hidden_size, dtype=torch.float32))
W = torch.nn.Parameter(torch.randn(hidden_size, hidden_size, dtype=torch.float32))
b = torch.nn.Parameter(torch.zeros(1, hidden_size, dtype=torch.float32))

X0 = np.array([[0.0, 1.0, -2.0],
               [-3.0, 4.0, 5.0],
               [6.0, 7.0, -8.0],
               [6.0, -1.0, 2.0]], dtype=np.float32)   # t = 0, shape [batch_size=4, input_size=3]

X1 = np.array([[9.0, 8.0, 7.0],
               [0.0, 0.0, 0.0],
               [6.0, 5.0, 4.0],
               [1.0, 2.0, 3.0]], dtype=np.float32)    # t = 1

X0 = torch.tensor(X0)
X1 = torch.tensor(X1)

# Implementing the operations
h0 = torch.tanh(torch.matmul(X0, U) + b)
h1 = torch.tanh(torch.matmul(X1, U) + torch.matmul(h0, W) + b)

print("h0= {}".format(h0.detach().numpy()))
print("h1= {}".format(h1.detach().numpy()))

# Stacking two time steps into a single 3D tensor:
# X = np.stack((X0, X1), axis=0)              -> [seq_len, batch_size, input_size]
# X = np.transpose(X, (1, 0, 2))               -> [batch_size, seq_len, input_size]
```

> **Shape note from the slide:** with `batch_size=4`, `seq_len=2`, `input_size=3`, stacking `X0` and `X1` along a new axis 0 gives `[seq_len, batch_size, input_size]`; transposing axes (1,0,2) converts this to the more common `[batch_size, seq_len, input_size]` layout (equivalent to `torch.transpose(X, 0, 1)` on the batch and sequence dimensions).

---

### 3.3 RNN as a Dynamic System (p. 118)

**Core idea:** share parameters across every time step of the input sequence.

Given a sequence $x_1, x_2, \dots, x_T$, the RNN models a dynamic system driven by an external signal $x_t$:

$$h_t = f(h_{t-1}, x_t) = f\big(f(h_{t-2}, x_{t-1}), x_t\big) = \dots = \text{summary}(x_{1:t}, h_0)$$

> $h_t$ can be viewed as a **lossy summary** of the history $x_{1:t}$ — it compresses all past inputs into a fixed-size vector, discarding information not needed for the task.

**Unrolled RNN (no output):** the same parameters $U$ (input-to-hidden) and $W$ (hidden-to-hidden) are reused at every time step when the recurrence is "unrolled" into a chain: $x_{t-1}!\to! h_{t-1} \to h_t \to h_{t+1}$, each transition using the same $U, W$.

---

### 3.4 RNN with Output — Two Time-Slices (p. 119)

**Input:** $x_0, x_1 \in \mathbb{R}^{1\times\text{input_size}}$, targets $y_0, y_1 \in Y$

**Time step 0:**

$$h_0 = \tanh(x_0 U + b)$$

$$ \hat{y}_0 = \begin{cases} h_0 V + c & \text{(regression)} \ \text{softmax}(h_0 V + c) & \text{(classification)} \end{cases} $$

- Suffer loss $l(\hat{y}_0, y_0)$

**Time step 1:**

$$h_1 = \tanh(h_0 W + x_1 U + b)$$

$$ \hat{y}_1 = \begin{cases} h_1 V + c & \text{(regression)} \ \text{softmax}(h_1 V + c) & \text{(classification)} \end{cases} $$

- Suffer loss $l(\hat{y}_1, y_1)$

**Total loss over both time steps:**

$$\text{Total_loss} = l(\hat{y}_0,y_0) + l(\hat{y}_1,y_1)$$

> New parameter here: $V$ (hidden-to-output), $c$ (output bias) — also **shared across time steps**.

---

### 3.5 RNN Generalized to Multiple Time Slices (p. 120)

**Input:** $x_0, x_1, \dots, x_t, \dots \in \mathbb{R}^{1\times\text{input_size}}$, targets $y_0, y_1, \dots \in Y$

**Initialization:**

$$h_0 = \tanh(x_0 U + b)$$

$$ \hat{y}_0 = \begin{cases} h_0 V + c & \text{(regression)} \ \text{softmax}(h_0 V + c) & \text{(classification)} \end{cases} $$

- Suffer loss $l(\hat{y}_0, y_0)$

**Recurrence, for $t = 1, 2, \dots, T$:**

$$h_t = \tanh(h_{t-1}W + x_t U + b)$$

$$ \hat{y}_t = \begin{cases} h_t V + c & \text{(regression)} \ \text{softmax}(h_t V + c) & \text{(classification)} \end{cases} $$

- Suffer loss $l(\hat{y}_t, y_t)$

**Total loss over the full sequence:**

$$\text{Total_loss} = \sum_{t=0}^{T} l(\hat{y}_t, y_t)$$

```python
import torch
import torch.nn as nn

class SimpleRNNCell(nn.Module):
    """Manual RNN matching the slide's h_t = tanh(h_{t-1} W + x_t U + b) recurrence,
       plus an output head y_t = h_t V + c (or softmax(h_t V + c))."""
    def __init__(self, input_size, hidden_size, output_size, classification=True):
        super().__init__()
        self.U = nn.Parameter(torch.randn(input_size, hidden_size) * 0.01)
        self.W = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b = nn.Parameter(torch.zeros(1, hidden_size))
        self.V = nn.Parameter(torch.randn(hidden_size, output_size) * 0.01)
        self.c = nn.Parameter(torch.zeros(1, output_size))
        self.classification = classification

    def forward(self, x_seq):
        # x_seq: [seq_len, batch_size, input_size]
        seq_len, batch_size, _ = x_seq.shape
        h_t = torch.zeros(batch_size, self.W.shape[0])
        outputs = []
        for t in range(seq_len):
            h_t = torch.tanh(x_seq[t] @ self.U + h_t @ self.W + self.b)
            y_t = h_t @ self.V + self.c
            if self.classification:
                y_t = torch.softmax(y_t, dim=-1)
            outputs.append(y_t)
        return torch.stack(outputs), h_t  # outputs: [seq_len, batch_size, output_size]

# Equivalent built-in PyTorch layer for comparison:
rnn = nn.RNN(input_size=3, hidden_size=5, nonlinearity='tanh', batch_first=False)
```

---

## Cross-Topic Connections

|Concept|Connects to|Relationship|
|---|---|---|
|Mixup / CutMix (p.101–102)|Data augmentation, L1/L2 regularization (pages 81–100)|Both are regularization strategies that reduce overfitting by expanding effective training diversity|
|Softmax + CE notation (p.104)|Softmax/CE loss (pages 1–80), Label smoothing (p.100)|Same $CE(\mathbf{1}_y, f(x;\theta))$ framework reused as the foundation for defining adversarial loss|
|Image distance metrics (p.105)|—|Provides the formal basis for defining the $\epsilon$-ball $B_\epsilon(x)$ used throughout adversarial ML|
|Gradient direction $g$ (p.108–110)|Backpropagation, gradient descent (pages 1–80)|Same $\nabla_x$ machinery as training, but now differentiating loss **w.r.t. input** $x$ instead of parameters $\theta$|
|FGSM/PGD attacks (p.111–112)|Gradient descent/SGD (pages 1–80)|PGD is literally gradient ascent/descent in _input_ space, mirroring parameter-space optimization mechanics|
|Adversarial training (p.114)|Regularization & overfitting (pages 81–100)|Both aim for smoother, more generalizable functions — adversarial training smooths the loss surface w.r.t. $x$, while L1/L2/dropout constrain complexity w.r.t. $\theta$|
|RNN parameter sharing (p.116–120)|Weight sharing in CNNs (pages 1–80)|Same core deep learning principle — reusing parameters ($U, W, V$) across positions (time steps here, spatial locations in CNNs) for parameter efficiency and generalization|

---

## Quick-Reference Formula Sheet

$$\text{Mixup: } \tilde{x}=\lambda x_1+(1-\lambda)x_2,\ \ \tilde{y}=\lambda\mathbf{1}_{y_1}+(1-\lambda)\mathbf{1}_{y_2},\ \ \lambda\sim\text{Beta}(\alpha,\alpha)$$

$$\text{CutMix: } \tilde{x}=M\odot x_1+(1-M)\odot x_2,\ \ \frac{\text{area}(B)}{\text{area(image)}}=1-\lambda$$

$$B_\epsilon(x) = {x': |x'-x|\le\epsilon}$$

$$\text{FGSM (untargeted): } x_{adv}=x+\epsilon,\text{sign}(\nabla_x l(f(x;\theta),y))$$

$$\text{FGSM (targeted): } x_{adv}=x-\epsilon,\text{sign}(\nabla_x l(f(x;\theta),y_{\ne}))$$

$$\text{Adversarial training: } \min_\theta \mathbb{E}_{(x,y)}\Big[\max_{x'\in B_\epsilon(x)} l(f(x';\theta),y)\Big]$$

$$\text{RNN recurrence: } h_t = \tanh(h_{t-1}W + x_t U + b), \qquad \hat{y}_t = \text{softmax}(h_t V + c)$$