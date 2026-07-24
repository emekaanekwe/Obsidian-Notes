# FIT5215 — Deep Learning Revision Notes (Part 4: Pooling Deep-Dive, ResNet, Practical Skills)

_Source: Final Exam Revision slides, pp. 61–80_

## 1. Pooling Layer — Why It's Needed

**Core properties**:

- Max/average-pooling is **locally invariant** and applied **independently to each feature map/channel**
- Reduces the size of the feature map → makes flattening produce a smaller, more manageable vector
- Reduces the **difference between output tensors** for inputs that are similar but not identical

### The intuition (face-part example)

Consider slightly different images of an eye, nose, and mouth (drawn by different people, or the same feature at slightly different positions). Before any conv+pooling, these small differences show up as **large differences** in the raw feature maps ("More Difference").

After passing through **Conv2D + Pooling** layers (which down-sample and effectively sharpen/generalize the representation), the resulting feature maps for the _same conceptual feature_ (e.g. "an eye") become much more similar to each other ("Less Difference"), even though the original inputs differed. This continues to shrink further after flattening, right up to the final prediction.

**Why this matters**: pooling gives the network a degree of **robustness to small spatial shifts or variations** in where/how a feature appears — two photos of eyes that are slightly shifted, rotated, or drawn differently will still produce similar downstream feature representations, which is what allows the network to generalize.

---

## 2. Hinton's Critique of CNNs

> "CNNs cannot capture the spatial relationships among objects, hence it wrongly predicts every face with two eyes, one nose, and one mouth as a human face."

This is the same fundamental limitation raised earlier (see Part 3 notes, Section 10): a CNN's flattened feature vector records **which features were detected**, not their spatial arrangement — so a scrambled arrangement of the right parts can still be misclassified as valid.

This observation directly motivated **Capsule Networks** (Sabour, Frosst, Hinton — _"Dynamic Routing Between Capsules"_, [arXiv:1710.09829](https://arxiv.org/pdf/1710.09829.pdf)), which attempt to explicitly encode part-whole spatial relationships.

---

## 3. Global Pooling Layer

**Definition**: like a regular pooling layer, but the **kernel size is set to the entire spatial size of the input** — so each feature map collapses to a **single number**.

- **Global Average Pooling (GAP)**: average of all values in a feature map
- **Global Max Pooling (GMP)**: maximum of all values in a feature map

**Shape transformation**: $$\text{4D input tensor } [\text{batch_size}, \text{in_channel}, \text{in_height}, \text{in_width}] ;\rightarrow; \text{2D output tensor } [\text{batch_size}, \text{in_channel}]$$

### Worked example: replacing flatten+FC with GAP

```
Input Layer [3,32,32]
   ↓ Conv2D(3→4 filters, kernel=4, stride=2, padding=0)
Feature volume [4,15,15]
   ↓ MaxPool2D(kernel=2, stride=2, padding=1)
Feature volume [4,8,8]
   ↓ GAP applied to EACH of the 4 feature maps independently
FC layer: 4 neurons        ← compare to flatten's 4×8×8=256 neurons!
   ↓
Output layer: 10 neurons (10 classes) → softmax
```

```python
Conv2d(3, 4, kernel_size=4, stride=2, padding=0)
MaxPool2d(kernel_size=2, stride=2, padding=1)
AdaptiveAvgPool2d((1, 1))   # this IS global average pooling
Flatten(1)
```

**Key benefit**: GAP drastically reduces the number of parameters feeding into the final FC layer (here: 4 instead of 256), which reduces overfitting risk and total parameter count — this is why GAP is standard in many modern CNN architectures (including ResNet, seen next).

---

## 4. ResNet (Residual Networks)

**Historical note**: ResNet won the ImageNet competition in 2015.

**High-level structure**:

- A ResNet is composed of multiple **ResNet blocks**
- Each ResNet block contains multiple **residual blocks**
- Design choices per ResNet block:
    - Number of residual blocks it contains
    - Number of channels used by those residual blocks
- Special case: from the **second** ResNet block onward, the **first** residual block in that group uses a **1×1 conv skip connection** (to handle the change in spatial size/channels); all other residual blocks use a standard (identity) skip connection

### The Residual Learning Trick

**Core idea**: instead of directly learning the target mapping $f(x)$, learn: $$f(x) = g(x) + x, \quad \text{where } g(x) = f(x) - x$$

**Why this helps**:

1. **Model expressiveness is unchanged** — since $f(x) = g(x)+x$ is just a reparameterization, the network can still represent anything it could before
2. **The gradient looks better**: $$\nabla f(x) = \nabla g(x) + \mathbf{1}$$ where $\mathbf{1}$ is the all-ones vector with the same shape as $x$. This means even if $\nabla g(x)$ is small (e.g., due to vanishing gradient through many layers), the gradient of $f$ still has a **guaranteed contribution of 1** flowing through the skip connection — preventing gradients from vanishing entirely as they propagate back through very deep networks.

### Residual Block Architecture

Follows **VGG's full 3×3 convolutional layer design**:

- Two 3×3 convolutional layers with the **same number of output channels**
- Each conv layer is followed by **batch normalization** and a **ReLU activation**
- The **input is added directly** (skip connection) to the output of the two conv layers, **before** the final ReLU activation
- **Shape requirement**: the output of the two conv layers must match the input's shape exactly, so they can be added
    - If the desired output has a **different number of channels** (or spatial size, via stride), an additional **1×1 convolutional layer** transforms the input into the correct shape before the addition

**Two variants**:

||Not use 1×1 Conv|Use 1×1 Conv|
|---|---|---|
|**When**|Input/output shapes already match|Need to change channels and/or downsample spatial size|
|**Skip path**|Identity (just pass $x$ through)|$x$ passes through a 1×1 conv first|

### Worked shape examples

**Example 1 — same shape throughout** (no 1×1 conv needed):

```
Input (8,64,64)
   ↓ 3×3 Conv, padding=1, stride=(1,1), filters=8
(8,64,64)
   ↓ 3×3 Conv, padding=1, stride=(1,1), filters=8
(8,64,64)
   ↓ Add input (8,64,64) directly
   ↓ ReLU
(8,64,64)
```

**Example 2 — downsampling + channel change** (1×1 conv needed):

```
Input (8,64,64)
   ↓ 3×3 Conv, padding=1, stride=(2,2), filters=16
(16,32,32)
   ↓ 3×3 Conv, padding=1, stride=(1,1), filters=16
(16,32,32)
   ↓ Add [skip path: Input (8,64,64) → 1×1 Conv, padding=0, stride=(2,2), filters=16 → (16,32,32)]
   ↓ ReLU
(16,32,32)
```

### PyTorch Implementation: Residual Block

```python
class Residual(nn.Module):
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.LazyConv2d(num_channels, kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.LazyConv2d(num_channels, kernel_size=3, padding=1)
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = nn.LazyConv2d(num_channels, kernel_size=1, stride=strides)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()

    def forward(self, X):
        Y = self.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return self.relu(Y)
```

**Shape verification examples**:

```python
blk = Residual(3)
X = torch.rand((4, 3, 6, 6))
Y = blk(X)
print(Y.shape)  # torch.Size([4, 3, 6, 6])  -- unchanged, identity skip

blk = Residual(num_channels=3, use_1x1conv=True, strides=1)
X = torch.rand((10, 3, 32, 32))
Y = blk(X)
print(Y.shape)  # torch.Size([10, 3, 32, 32])

blk = Residual(6, use_1x1conv=True, strides=2)
print(blk(X).shape)  # torch.Size([4, 6, 3, 3])  -- downsampled + channels changed
```

### PyTorch Implementation: Full ResNet

```python
class ResnetBlock(nn.Module):
    def __init__(self, num_channels, num_residuals, first_block=False, **kwargs):
        super(ResnetBlock, self).__init__(**kwargs)
        self.residual_layers = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                self.residual_layers.append(
                    Residual(num_channels, use_1x1conv=True, strides=2))
            else:
                self.residual_layers.append(Residual(num_channels))
        self.residual_blk = nn.ModuleList(self.residual_layers)

    def forward(self, X):
        for layer in self.residual_blk:
            X = layer(X)
        return X


class create_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResnetBlock(64, 2, first_block=True),
            ResnetBlock(128, 2),
            ResnetBlock(256, 2),
            nn.AdaptiveAvgPool2d((1, 1)),   # Global Average Pooling
            nn.Flatten(1),
            nn.LazyLinear(10),
            # nn.Softmax(dim=-1)
        ])

    def forward(self, X):
        for _, layer in enumerate(self.layers):
            X = layer(X)
        return X
```

### Full ResNet Shape Trace (input `(1,3,H,W)`)

```
Input
   ↓ LazyConv2d(64, kernel=7, stride=2, padding=3)
(1, 64, 112, 112)
   ↓ BatchNorm2d + ReLU
(1, 64, 112, 112)
   ↓ MaxPool2d(kernel=3, stride=2, padding=1)
(1, 64, 56, 56)
   ↓ ResnetBlock(64, 2, first_block=True)
(1, 64, 56, 56)
   ↓ ResnetBlock(128, 2)     -- downsamples via first residual block's 1x1 conv
(1, 128, 28, 28)
   ↓ ResnetBlock(256, 2)
(1, 256, 14, 14)
   ↓ AdaptiveAvgPool2d((1,1))   -- Global Average Pooling
(1, 256, 1, 1)
   ↓ Flatten(1)
(1, 256)
   ↓ LazyLinear(10)
(1, 10)
```

**Architectural pattern to remember**: `7×7 Conv (stride 2) → BatchNorm → ReLU → 3×3 MaxPool (stride 2) → [ResNet blocks, each halving spatial size and doubling channels except the first] → Global Average Pool → Flatten → FC → (softmax)`.

---

## 5. Gradient Vanishing

**Definition**: gradients get **smaller and smaller** as backpropagation proceeds down to the lower (earlier) layers of the network.

**Consequence**: SGD updates leave the **lower layer weights virtually unchanged**, and training never converges to a good solution — the network effectively stops learning in its early layers.

**Root cause**: certain activation functions (sigmoid, tanh) **saturate** — for large positive or negative inputs, their output flattens out near 0/1 (sigmoid) or -1/1 (tanh), and their **derivative approaches zero** in those saturated regions.

$$\sigma(z) = s(z) = \frac{1}{1+\exp(-z)}, \qquad \sigma'(z) = \sigma(z)(1-\sigma(z))$$

The derivative $\sigma'(z)$ peaks at $z=0$ (value $0.25$) and decays toward $0$ as $|z|$ grows — this is the "saturated" region.

### Why this compounds through layers (the math)

For a 3-layer network with sigmoid activations, the gradient of the loss w.r.t. the **first** layer's weights $W^1$ involves a **chain of derivatives**:

$$\frac{\partial l}{\partial W^1} = \frac{\partial l}{\partial h^3}\cdot\frac{\partial h^3}{\partial h^2}\cdot\frac{\partial h^2}{\partial \bar h^2}\cdot\frac{\partial \bar h^2}{\partial h^1}\cdot\frac{\partial h^1}{\partial \bar h^1}\cdot\frac{\partial \bar h^1}{\partial W^1}$$

$$= \left[(p^T-\mathbf{1}_y)W^3, \text{diag}(\sigma'(\bar h^2)), W^2, \text{diag}(\sigma'(\bar h^1))\right]^T (h^0)^T$$

**The problem**: each activation layer contributes a $\text{diag}(\sigma'(\bar h))$ factor. If the pre-activations $\bar h^1, \bar h^2$ fall in a **saturated** region of sigmoid, each $\sigma'$ term is close to $0$ (recall the peak value is only $0.25$, and it's often much smaller). Multiplying **several such small factors together** (one per layer) makes the overall gradient for early layers **shrink exponentially with depth** — this is gradient vanishing.

### The Recipe (Practical Fix)

1. **Choose activation functions carefully**:
    - **Avoid** sigmoid or other easily-saturated activations in deep hidden layers
    - **ReLU is a common good choice** — its derivative is exactly $1$ for all positive inputs (no shrinking in the "active" region)
2. **Good weight initialization is critical** — see Xavier initialization below

---

## 6. Gradient Exploding

**Definition**: the opposite problem — gradients can grow **bigger and bigger**, causing many layers to receive **insanely large weight updates**, and training **diverges**.

**Where this mostly happens**: **recurrent models**, e.g. Recurrent Neural Networks (RNN), Bidirectional RNN — because these architectures **reuse the same weight matrix $W$ repeatedly across time steps**.

### Why RNNs are especially vulnerable

For an RNN unrolled over $T$ time steps, computing the gradient of the loss at the final time step w.r.t. the very first hidden state involves a **product of many copies of the same weight matrix $W$**:

$$\frac{\partial l_T}{\partial h_0} = \frac{\partial l_T}{\partial h_T}\times\frac{\partial h_T}{\partial h_{T-1}}\times\cdots\times\frac{\partial h_1}{\partial h_0}$$

Each of the $T$ factors in this product contributes a copy of $W$ (or a function of $W$).

**Simplified intuition** (treating $W$ as a scalar): $$W^m \to 0 \quad \text{if } |W| < 1$$ $$W^m \to \infty \quad \text{if } |W| > 1$$

So depending on whether the "effective" weight magnitude is below or above 1, repeated multiplication across many time steps drives the gradient toward **zero** (vanishing) or **infinity** (exploding) — the exact same underlying mechanism as gradient vanishing in deep feedforward nets, but amplified because RNNs reuse the _same_ matrix $W$ at every step (rather than different matrices per layer).

---

## 7. Weight Initialization: Why It's Crucial

**The problem**: unlike some optimizers that are theoretically guaranteed to converge regardless of initialization (for convex problems), **deep learning optimization is not yet well understood** theoretically:

- Training is iterative, but there's no strong guarantee about convergence behavior
- The **initial point matters enormously** — a bad initialization can cause the algorithm to become unstable and fail entirely, or converge to a poor solution

**What makes a good weight/filter initialization?**

1. **Break symmetry**: two hidden nodes receiving the same input must have **different** weights — otherwise they'll always compute the same function and update identically (redundant units, wasted capacity)
2. **Support healthy gradient flow in both directions** — signal shouldn't die out (vanish) or explode/saturate as it flows forward or backward
3. **Trade-off with initialization scale**:
    - **Large initial weights**: better symmetry breaking, helps avoid losing signal / redundant units
    - **But** can cause **exploding values** during forward and backward passes — especially problematic in RNNs (see Section 6)

---

## 8. Xavier (Glorot) Initialization

**Goal**: choose initial weights so that:

1. The **variance of each layer's output** matches the **variance of its input** (forward pass stability)
2. The **variance of the gradients** is equal before and after flowing through a layer in the **reverse** direction (backward pass stability)

**Setup example**: layer with $n_{in}=3$ input units, $n_{out}=2$ output units.

### Gaussian version

$$w_{Xa} \sim \mathcal{N}\left(0,\ \frac{2}{n_{in}+n_{out}}\right)$$

### Uniform version (alternative)

$$w_{Xa} \sim \text{Uniform}\left(-\sqrt{\frac{6}{n_{in}+n_{out}}},\ \sqrt{\frac{6}{n_{in}+n_{out}}}\right)$$

**Why $\frac{2}{n_{in}+n_{out}}$ specifically?** This choice balances two competing constraints — keeping forward-pass output variance stable (which alone would suggest scaling by $1/n_{in}$) and keeping backward-pass gradient variance stable (which alone would suggest scaling by $1/n_{out}$). Averaging these two considerations (via the harmonic-mean-like combination $n_{in}+n_{out}$ in the denominator) gives a single initialization scheme that reasonably satisfies both simultaneously.

**When to use it**:

- ✅ **Good** for **sigmoid** and **tanh** activation functions
- ❌ **Not good** for **ReLU** (ReLU has different variance-preservation properties since it zeroes out negative inputs; this motivated the later development of **He initialization**, specifically designed for ReLU-based networks — referenced by the slide title but not detailed on this particular slide)

**Paper reference**: Glorot & Bengio, _"Understanding the difficulty of training deep feedforward neural networks"_, [PMLR v9](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)

---

## Summary Table: Key Formulas (Part 4)

|Concept|Formula|
|---|---|
|Global pooling shape|$[batch, C, H, W] \to [batch, C]$|
|Residual learning|$f(x) = g(x) + x$|
|Residual gradient|$\nabla f(x) = \nabla g(x) + \mathbf{1}$|
|Sigmoid|$\sigma(z) = \frac{1}{1+e^{-z}}$|
|Sigmoid derivative|$\sigma'(z) = \sigma(z)(1-\sigma(z))$|
|Gradient vanishing (chain)|product of many $\text{diag}(\sigma'(\cdot))$ terms → shrinks toward 0|
|Gradient exploding (RNN)|$\frac{\partial l_T}{\partial h_0} = \prod_{t} \frac{\partial h_t}{\partial h_{t-1}}$, scales like $W^T$|
|Exploding/vanishing condition|$W^m\to0$ if $\lvert W\rvert<1$; $W^m\to\infty$ if $\lvert W\rvert>1$|
|Xavier init (Gaussian)|$w \sim \mathcal{N}\left(0, \frac{2}{n_{in}+n_{out}}\right)$|
|Xavier init (Uniform)|$w \sim \text{Uniform}\left(-\sqrt{\frac{6}{n_{in}+n_{out}}}, \sqrt{\frac{6}{n_{in}+n_{out}}}\right)$|

---

## Quick Reference: Vanishing vs Exploding Gradients

||Gradient Vanishing|Gradient Exploding|
|---|---|---|
|**Symptom**|Gradients shrink toward 0 in early/lower layers|Gradients grow unboundedly large|
|**Effect on training**|Lower layer weights barely update; training stalls|Massive weight updates; training diverges|
|**Typical cause**|Saturating activations (sigmoid, tanh) across many layers|Repeated multiplication of the same weight matrix (RNNs)|
|**Typical fix**|Use ReLU; good weight init|Gradient clipping (mentioned contextually); good weight init; architectures like LSTM/GRU (covered in later weeks)|