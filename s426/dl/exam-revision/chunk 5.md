# FIT5215 — Deep Learning: Pages 81–100

## 1. He and Xavier Weight Initialization (p. 81)

**Goal:** control the variance of activations and gradients as they flow through layers, so training doesn't suffer from vanishing/exploding signals.

### Xavier Initialization

- Ensures the **variance of outputs** of each layer equals the **variance of its inputs**
- Also keeps gradient variance stable **before and after** flowing backward through a layer
- Works well for **sigmoid** and **tanh** — poor fit for **ReLU**

For a layer with $n_{in}$ input units and $n_{out}$ output units:

$$w_{Xa} \sim \mathcal{N}\left(0, \ \frac{2}{n_{in}+n_{out}}\right)$$

### He Initialization

- A variant of Xavier where a scale factor $\alpha$ is introduced
- Works better for **ReLU** activations

$$w_{He} \sim \mathcal{N}\left(0, \ \alpha \times \frac{2}{n_{in}+n_{out}}\right)$$

$$ \alpha = \begin{cases} 1 & \text{if sigmoid} \ 4 & \text{if tanh} \ 2 & \text{if ReLU} \end{cases} $$

> **Example:** if $n_{in}=3$, $n_{out}=2$:
> 
> - Xavier variance: $\dfrac{2}{3+2} = 0.4$
> - He variance (ReLU, $\alpha=2$): $2 \times 0.4 = 0.8$

**Paper:** [He et al., ICCV 2015 — Delving Deep into Rectifiers](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf)

---

## 2. Deep Learning Pipeline (p. 82)

We train a DL model on a **training set** such that the trained model generalizes well to **unseen data** in a separate **testing set**.

**Key hyperparameters to tune:**

- Learning rate
- Number of layers
- Number of neurons per layer

---

## 3. Regularization and Overfitting — Framework (p. 83)

### Three Elements of ML/DL

1. **Data** — training data, testing data, validation data
2. **Model** — a mathematical function $f(x;\theta)$ mapping input $x$ to output $y$
3. **Evaluation** — a performance metric quantifying how well $f(x;\theta)$ performs

### ML as Optimization

- Define a per-instance loss: $l\big(f(x;\theta), y\big)$
- Total loss over training data:

$$J(\theta) = N^{-1}\sum_{i=1}^{N} l\big(f(x_i,\theta), y_i\big)$$

- Learning = finding $\theta^*$ that minimizes the loss:

$$\theta^{*}=\arg\min_{\theta} J(\theta)$$

### What Can Go Wrong?

- The learning function $f(x;\theta)$ is too hard to learn (poor architecture choice)
- The loss function $l(f(x),y)$ is inadequate for the task
- **Overfitting**: model does well on training data but poorly on unseen test data

---

## 4. Overfitting & Underfitting (p. 84–85)

### Underfitting

- Model is **too simple** to characterize the training set
- Example: using a linear model to learn non-linear data

### Overfitting

- Model performs **very well on training data** but **fails to generalize** to test data
- **Most common problem in DL** since deep networks are highly expressive
- Overfitting = model "memorizes" training samples instead of learning generalizable patterns

**Causes:** too many layers, too many hidden nodes, over-training (too many epochs)

**Key insight:** performance must be measured by error on _unseen_ (validation/test) data — minimizing training error alone is not enough.

---

## 5. Model Capacity (p. 86)

Hyperparameters and their effect on capacity:

|Hyperparameter|Increases capacity when...|Reason|
|---|---|---|
|Number of hidden units|increased|Increases representational capacity|
|Learning rate|tuned optimally|Poor LR (too high/low) → optimization failure → effectively low capacity|
|Convolution kernel width|increased|More parameters (but narrows output unless zero-padded)|
|Implicit zero padding|increased|Keeps representation size large|
|Weight decay coefficient|decreased|Frees parameters to grow larger|
|Dropout rate|decreased|Units get more chances to "conspire" and fit training data|

**Capacity vs. loss curve:** training error decreases monotonically with capacity; generalization error follows a U-shape — there exists an **optimal capacity** minimizing the generalization gap.

_(Source: Goodfellow, Bengio, Courville — Deep Learning, Ch. 11, Table 11.1)_

---

## 6. Early Stopping (p. 87)

- Early on: both train and validation losses drop — model is **underfitting**
- At some point: train loss **keeps decreasing**, but validation loss **starts increasing** → **overfitting begins** — this is the **early stopping point**
- The gap between validation and training loss at convergence is the **generalization gap**

**Practical rule:** stop training when validation loss starts to rise, even though training loss continues to fall.

---

## 7. Regularization Techniques

### 7.1 L1 / L2 Regularization (p. 88)

**Regularized optimization problem:**

$$\min_{\theta} J(\theta) = \Omega(\theta) + \frac{1}{N}\sum_{i=1}^{N} l\big(y_i, f(x_i;\theta)\big)$$

#### L2 Regularization (weight decay)

$$\Omega(\theta) = \lambda \sum_{k}\sum_{i,j} \big(W_{i,j}^{k}\big)^2 = \lambda\sum_k |W^k|_F^2$$

- $\lambda > 0$ is the regularization parameter
- Gradient: $\nabla_{W^k}\Omega(\theta) = 2W^k$
- Applied to **weights only**, never to biases

#### L1 Regularization

$$\Omega(\theta) = \lambda\sum_k\sum_{i,j} \left|W_{i,j}^k\right|$$

- Optimization is harder — requires **subgradients** (L1 is non-differentiable at 0)
- Also applied to weights only

**Effect:** as $\lambda$ increases, training error rises but test/generalization error can be reduced (up to a point) — this is the classic bias–variance tradeoff visualized in the training-vs-test-error curve.

---

### 7.2 Dropout (p. 89–90)

**Idea:** cheaply reduce model capacity to combat overfitting.

- At each iteration, at each layer: **randomly select neurons** and **drop all their connections**
- $\text{dropout_rate} = 1 - \text{keep_prob}$

**Why it works:**

- Computationally efficient
- Can be interpreted as a **bagged ensemble** of an exponential number ($2^N$) of sub-networks sharing weights

#### Training Phase

$$r \sim \text{Bernoulli}(\mu)$$

$$\tilde{h}^{l} = h^{l} \odot r$$

$$h^{l+1} = \sigma!\left(W^{(l)\top}\tilde{h}^{l} + b^{l}\right)$$

Here $r$ is a binary mask sampled per unit, $\odot$ is elementwise multiplication.

#### Testing Phase

- **No dropout** applied (dropout_rate = 0) — all units and connections are active
- (Implicitly, activations are typically scaled to match expected value seen during training — "inverted dropout" convention used in practice, though slides just state dropout_rate=0 at test time)

```python
import torch.nn as nn

class MLPWithDropout(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, p=0.5):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.drop = nn.Dropout(p=p)   # p = dropout_rate
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        h = self.drop(h)              # active only in .train() mode
        return self.fc2(h)
```

> `nn.Dropout` automatically disables itself when `model.eval()` is called — matching the "no dropout at test time" rule above.

---

## 8. Internal Covariate Shift & Batch Normalization

### 8.1 The Problem (p. 91–92)

- As mini-batches of inputs pass through a network's layers, the **statistical distribution** of both the **input batches** and the **intermediate representation batches** can differ significantly from one mini-batch to the next
- These statistical differences make it **harder for the classifier** (downstream layers) to learn stable patterns from the data
- This is called the **internal covariate shift problem**

**Paper:** [Ioffe & Szegedy, 2015 — Batch Normalization](http://proceedings.mlr.press/v37/ioffe15.pdf)

### 8.2 Batch Normalization Mechanism (p. 93–95)

**Setup:** Let $z = W^k h^k + b^k$ be the pre-activation values for a mini-batch (before applying $\sigma$).

**Step 1 — Normalize:**

$$\hat{z} = \frac{z - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$

where $\epsilon$ is a small constant (e.g., $1\text{e}{-7}$) for numerical stability, and:

$$\mu_B = \frac{1}{b}\sum_{i=1}^{b} z_i \quad \text{(empirical batch mean)}$$

$$\sigma_B^2 = \frac{1}{b}\sum_{i=1}^{b} (z_i - \mu_B)^2 \quad \text{(empirical batch variance)}$$

> Normalization is done **column-wise** (i.e., per-feature/per-neuron, across the mini-batch dimension).

**Step 2 — Scale and shift:**

$$z_{BN} = \gamma\hat{z} + \beta$$

where $\gamma, \beta$ are **learnable parameters** (scale and shift), allowing the network to undo normalization if that's optimal.

**Step 3 — Apply activation:**

$$h^{k+1} = \sigma(z_{BN})$$

#### Why Use BN? (5 benefits)

1. Copes with internal covariate shift
2. Reduces gradient vanishing/exploding
3. Reduces overfitting (has a slight regularizing effect)
4. Makes training more stable
5. Converges faster — and allows training with a **bigger learning rate**

### 8.3 Batch Norm at Test Time (p. 96)

**Problem:** at test time we may only have a **single input** $x$ — there's no mini-batch to compute $\mu_B, \sigma_B$ from.

**Solution:** use **running averages** accumulated during training:

$$\tilde{\mu}_B = \theta\tilde{\mu}_B + (1-\theta)\mu_B$$

$$\tilde{\sigma}_B = \theta\tilde{\sigma}_B + (1-\theta)\sigma_B$$

where $0 < \theta < 1$ is the **momentum decay** for the running statistics.

At test time, these running averages $\tilde{\mu}_B, \tilde{\sigma}_B$ replace the batch statistics $\mu_B, \sigma_B$ in the normalization formula.

```python
import torch.nn as nn

# PyTorch handles the running mean/var and train/eval switching automatically
bn_layer = nn.BatchNorm1d(num_features=128, eps=1e-7, momentum=0.1)
# momentum here corresponds to (1 - theta) in the slide notation

# During model.train(): uses batch statistics (mu_B, sigma_B^2)
# During model.eval():  uses running statistics (running_mean, running_var)
```

---

## 9. Data Augmentation (p. 97–99)

### Motivation

- Real-world datasets are often small (e.g., 10K images) — insufficient to train a good deep net from scratch
- Key question: is a "qualified" training set about **quantity** (collect more data) or **quality** (more diverse / better-labeled data)?

### Technique

Apply simple transformations to create new training variants, exposing the model to diverse examples similar to what it might see at test time:

- Rotation
- Width shifting / Height shifting
- Brightness adjustment
- Shear intensity
- Zoom
- Channel shift
- Horizontal flip / Vertical flip

**Effect:** acts as a **regularization technique** — reduces overfitting by expanding the effective diversity of the training set. The trick is generating **realistic** training instances (i.e., augmentations that plausibly resemble real variations, not distortions the model would never see).

### PyTorch Implementation (p. 99)

```python
from torchvision import transforms
import torchvision
import torch

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # normalize R,G,B with mean=0.5, std=0.5
    transforms.Resize((32, 32)),
])

train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),          # flip w.r.t. horizontal axis
    transforms.RandomRotation(4),                # rotate by a specified angle
    # transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),  # zoom, shear
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

full_train_set = torchvision.datasets.CIFAR10("./data", download=True, transform=train_transform)
full_valid_set  = torchvision.datasets.CIFAR10("./data", download=True, transform=test_transform)
full_test_set   = torchvision.datasets.CIFAR10("./data", download=True, train=False, transform=test_transform)

n_train, n_valid, n_test = 5000, 5000, 5000

total_num_train = len(full_train_set)
total_num_test  = len(full_test_set)

train_valid_idx = torch.randperm(total_num_train)
train_set = torch.utils.data.Subset(full_train_set, train_valid_idx[:n_train])
valid_set = torch.utils.data.Subset(full_valid_set, train_valid_idx[n_train:n_train+n_valid])

test_idx = torch.randperm(total_num_test)
test_set = torch.utils.data.Subset(full_test_set, test_idx[:n_test])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_set, batch_size=32)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=32)
```

> **Important:** note that `test_transform` (no augmentation, only resize/normalize) is applied to validation and test sets, while `train_transform` (with random flips/rotations/color jitter) is applied only to the training set.

---

## 10. Label Smoothing (p. 100)

**Setup:** data instance $(x, y)$ with label $y \in {1,\dots,M}$ (one of $M$ classes).

**Standard (hard) one-hot label:**

$$\mathbf{1}_y = [0, \dots, 1, \dots, 0] \quad (\text{1 at the } y\text{-th position})$$

**Standard CE loss (no smoothing):**

$$l(y,\hat{y}) = CE(\mathbf{1}_y, \mathbf{p}(x))$$

where $\mathbf{p}(x) = \text{softmax}(h^L(x))$ are the predicted class probabilities from the logits $h^L(x)$, and:

$$\hat{y} = \arg\max_{1\le m \le M} p_m(x)$$

### Label Smoothing

Instead of a hard one-hot target, construct a **smoothed label** that redistributes a small probability mass $\alpha$ across all classes:

$$\tilde{\mathbf{1}}_y = (1-\alpha)\times \mathbf{1}_y + \frac{\alpha}{M}\times \mathbf{1}$$

where $\mathbf{1}$ is the all-ones vector and $0 < \alpha < 1$.

Explicitly, the smoothed label vector is:

$$\tilde{\mathbf{1}}_y = \left[\frac{\alpha}{M}, \dots, \underbrace{1-\alpha+\frac{\alpha}{M}}_{y\text{-th position}}, \dots, \frac{\alpha}{M}\right]$$

**Label-smoothed CE loss:**

$$l(y,\hat{y}) = CE\big(\tilde{\mathbf{1}}_y, \mathbf{p}(x)\big)$$

> **Intuition:** label smoothing prevents the model from becoming overconfident (assigning probability arbitrarily close to 1 for the correct class and 0 elsewhere), which improves calibration and can reduce overfitting to noisy labels.

**Paper:** [Müller, Kornblith, Hinton (Google Brain), 2019 — When Does Label Smoothing Help?](https://papers.nips.cc/paper/2019/file/f1748d6b0fd9d439f71450117eba2725-Paper.pdf)

```python
import torch.nn as nn

# PyTorch has built-in label smoothing support
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # alpha = 0.1
loss = criterion(logits, targets)  # targets are class indices, not one-hot
```

---

## Cross-Topic Connections

|Concept|Connects to|Relationship|
|---|---|---|
|He/Xavier init (p.81)|Gradient vanishing/exploding (pages 1–80)|Proper init controls activation/gradient variance, complementing careful architecture design|
|Overfitting (p.83–86)|Regularization techniques (p.88–90)|L1/L2, dropout, and data augmentation are all _solutions_ to the overfitting problem framed on p.83–86|
|Early stopping (p.87)|Model capacity table (p.86)|Both are practical capacity-control tools; early stopping controls _effective_ capacity via training duration|
|Batch Norm (p.91–96)|Gradient vanishing/exploding, optimization (pages 1–80)|BN addresses covariate shift _and_ directly helps stabilize gradients — ties initialization and optimization together|
|Data augmentation (p.97–99)|Regularization (p.88–90)|Framed explicitly as a regularization technique — increases effective data diversity instead of penalizing weights|
|Label smoothing (p.100)|Softmax + Cross-Entropy loss (pages 1–80)|Modifies the target distribution in the same CE loss framework covered earlier, reducing overconfidence|

---

## Quick-Reference Formula Sheet

$$w_{He} \sim \mathcal{N}\left(0,\ \alpha\cdot\frac{2}{n_{in}+n_{out}}\right), \quad \alpha=2 \text{ (ReLU)}$$

$$\Omega_{L2}(\theta) = \lambda\sum_k |W^k|_F^2, \qquad \Omega_{L1}(\theta) = \lambda\sum_k\sum_{i,j}|W_{i,j}^k|$$

$$\tilde{h}^l = h^l \odot r, \quad r\sim\text{Bernoulli}(\mu) \quad \text{(dropout)}$$

$$\hat{z} = \frac{z-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}, \quad z_{BN}=\gamma\hat{z}+\beta \quad \text{(batch norm)}$$

$$\tilde{\mathbf{1}}_y = (1-\alpha)\mathbf{1}_y + \frac{\alpha}{M}\mathbf{1} \quad \text{(label smoothing)}$$