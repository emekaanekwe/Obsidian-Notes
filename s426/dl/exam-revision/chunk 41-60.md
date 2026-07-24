# FIT5215 — Deep Learning Revision Notes (Part 3: Optimizers II & CNNs)

_Source: Final Exam Revision slides, pp. 41–60_

## 1. SGD with Momentum

**Problem with plain SGD**: fast in early epochs, becomes much slower later.

### Plain SGD algorithm

```
Input: η > 0, initial model θ
while stopping criterion not met:
    Sample mini-batch {(x¹,y¹),...,(xᵇ,yᵇ)}
    Compute g = (1/b) Σᵢ ∇θ l(f(xᵢ,θ), yᵢ)
    Update θ = θ - ηg
```

### SGD with momentum

Introduces a **velocity vector** $v$ that combines past gradients with the current gradient to accelerate convergence.

```
Input: η > 0, α ∈ [0,1], initial model θ
while stopping criterion not met:
    Sample mini-batch {(x¹,y¹),...,(xᵇ,yᵇ)}
    Compute g = (1/b) Σᵢ ∇θ l(f(xᵢ,θ), yᵢ)
    Compute v = αg + (1-α)v      // velocity v
    Update θ = θ - ηv
```

**Hyperparameter $\alpha$**: controls how quickly contributions of previous gradients decay. Common practical values: $0.5, 0.9, 0.99$.

**What momentum solves** — two specific problems:

1. **Poor conditioning of the Hessian matrix** (loss surface curves much more steeply in some directions than others, causing zig-zag updates)
2. **Variance in the stochastic gradient** (mini-batch gradients are noisy estimates of the true gradient; averaging with past gradients smooths this out)

**Visual intuition**: without momentum, SGD zig-zags across a narrow valley; with momentum, the path smooths into a more direct route toward the minimum (since consistent-direction gradients accumulate while oscillating components cancel out).

---

## 2. AdaGrad

**Idea**: scale the learning rate for each parameter individually, based on the **cumulative sum of squared gradients** seen so far for that parameter.

```
Input: η > 0, ε > 0 (e.g. 10⁻⁶), initial model θ
while stopping criterion not met:
    Sample mini-batch {(x¹,y¹),...,(xᵇ,yᵇ)}
    Compute g = (1/b) Σᵢ ∇θ l(f(xᵢ,θ), yᵢ)
    Accumulate squared gradient: γ = γ + g⊙g
    Update θ = θ - (η / (ε+√γ)) ⊙ g
```

(⊙ denotes element-wise product)

**Effect**:

- **Directions with large partial derivatives** → $\gamma$ grows quickly → learning rate shrinks rapidly for those parameters
- **Directions with small partial derivatives** → $\gamma$ grows slowly → learning rate shrinks only slightly

**Weakness**: the learning rate can only ever **decrease** (since $\gamma$ only accumulates, never resets). This works well on convex problems, where you generally want to slow down as you approach the minimum, but performs poorly on non-convex deep learning problems, where you may need to keep exploring at a healthy rate even late in training.

---

## 3. Convolution Operation — Formal Definition

**General rule**: given two tensors $W$ (filter) and $x$ (input patch) of the **same shape**, the convolution operation is: $$W * x = \text{sum}(W \otimes x)$$ where $\otimes$ is element-wise product and $\text{sum}$ adds every element of the resulting tensor.

### 1D convolution ($W, x \in \mathbb{R}^{m\times1}$)

$$W*x = \sum_{i=1}^{m} W_i x_i$$

### 2D convolution ($W, x \in \mathbb{R}^{m\times n}$)

$$W*x = \sum_{i=1}^{m}\sum_{j=1}^{n} W_{ij} x_{ij}$$

### 3D convolution ($W, x \in \mathbb{R}^{m\times n\times p}$)

$$W*x = \sum_{i=1}^{m}\sum_{j=1}^{n}\sum_{k=1}^{p} W_{ijk} x_{ijk}$$

**Key insight**: convolution as used in CNNs is really just "multiply element-wise, then add everything up" — applied repeatedly as a filter slides across the input.

---

## 4. General CNN Architecture

Three core building blocks, applied in sequence (often repeated multiple times before a final classifier):

1. **Convolutional layer** — extracts local features via learned filters
2. **Pooling layer** — downsamples/subsamples feature maps
3. **Fully connected layer** — combines extracted features to make a final prediction

```
Input Layer → Convolutional Layer → Pooling Layer → ... → Fully Connected Layer → Output Layer
```

---

## 5. Convolution Layer Mechanics

### Output size formula

Given:

- $W_i, H_i$: width/height of the input
- $f_w, f_h$: filter/kernel width/height
- $s_w, s_h$: stride width/height
- $p$: zero padding (applied symmetrically on each side)

$$W_o = \left\lfloor\frac{W_i + 2p - f_w}{s_w}\right\rfloor + 1, \qquad H_o = \left\lfloor\frac{H_i + 2p - f_h}{s_h}\right\rfloor + 1$$

### Worked example 1: with zero padding

- Input: $7\times7$, kernel $3\times3$, stride $(2,2)$, padding $p=1$

$$W_o = \left\lfloor\frac{7+2(1)-3}{2}\right\rfloor+1 = 4, \qquad H_o = \left\lfloor\frac{7+2(1)-3}{2}\right\rfloor+1 = 4$$

Output feature map: $4\times4$.

**Process**: the padded input (now effectively $9\times9$ with the zero border) is scanned left→right, top→bottom by the $3\times3$ filter, with each window's convolution value becoming one entry in the $4\times4$ feature map.

### Worked example 2: without padding, rectangular input

- Input: $7\times8$ ($H_i=7, W_i=8$), kernel $3\times3$, stride $(2,2)$, padding $p=0$

$$W_o = \left\lfloor\frac{8+0-3}{2}\right\rfloor+1 = 3, \qquad H_o = \left\lfloor\frac{7+0-3}{2}\right\rfloor+1 = 3$$

Output feature map: $3\times3$.

### Multiple filters → multiple feature maps ("feature volume")

Given input $x$ with shape $(4,7,8)$ (i.e. **depth 4**, height 7, width 8):

- Filter $W_1$ has shape $(4,3,3)$ — note it must match the input's **depth** (4), convolving across all input channels simultaneously to produce **one** 2D feature map
- Filter $W_2$, another $(4,3,3)$ filter, produces a second feature map
- Stacking the outputs of $W_1$ and $W_2$ gives a **feature volume** of shape $(2,3,3)$ — 2 feature maps, each $3\times3$

**General pattern**: for $F$ filters each of shape $(C_{in}, f_h, f_w)$ applied to input of shape $(C_{in}, H_i, W_i)$, the output feature volume has shape $(F, H_o, W_o)$.

**With a batch dimension**: input shape $(\text{batch}, C_{in}, H_i, W_i)$, filters shape $(F, C_{in}, f_h, f_w)$, output feature volume shape $(\text{batch}, F, H_o, W_o)$.

```python
output = torch.nn.functional.conv2d(
    input=batch_tensor,      # (batch_size, in_channels, height, width)
    weight=filters_tensor,   # (out_channels, in_channels, fil_height, fil_width)
    stride=(2,2),
    padding=3
)
```

---

## 6. Pooling Layer

**Purpose**:

- Makes representations smaller and more manageable
- Subsamples the image (reduces spatial resolution)
- Operates on **each feature map/channel independently**

### Max/Average pooling

Both are **locally invariant** (small shifts in the input don't change the pooled output much) and applied per-channel.

**Standard SOTA configuration** (used in ResNet, GoogleNet, VGG):

- Kernel size $(2,2)$, stride $(2,2)$, padding $0$

**Output size** (same formula as convolution, with pooling's kernel/stride/padding): $$W_o = \left\lfloor\frac{224+0-2}{2}\right\rfloor+1 = 112, \qquad H_o = \left\lfloor\frac{224+0-2}{2}\right\rfloor+1 = 112$$

With this setting, the input is **down-sampled by a factor of 2** in each spatial dimension.

**Max pooling** takes the maximum value in each pooling window; **average pooling** takes the mean. Both reduce a 2D grid of values (e.g. a $4\times4$ block) to a smaller grid (e.g. $2\times2$) by summarizing each local window.

---

## 7. Fully Connected Layer & Full CNN Pipeline

**Flattening**: after the convolutional/pooling stages produce a final 3D tensor (feature volume), it's **flattened** into a 1D vector so it can feed into standard fully-connected (dense) layers for classification.

**Example**: last 3D tensor $[5,5,10]$ (height, width, channels) → flattened into 1 layer with $5\times5\times10 = 250$ neurons.

### Complete worked pipeline (single sample)

```
Input [3,32,32]
   ↓ Conv2D(3→4 filters, kernel=4, stride=2, padding=0)
   ↓   out_size = ⌊(32+0-4)/2⌋+1 = 15
Feature volume [4,15,15]
   ↓ MaxPool2D(kernel=2, stride=2, padding=1)
   ↓   out_size = ⌊(15+2×1-2)/2⌋+1 = 8
Feature volume [4,8,8]
   ↓ Flatten
FC layer: 4×8×8 = 256 neurons
   ↓
Output layer: 10 neurons (10 classes)
   ↓ softmax
```

### With a batch dimension

```
Input [batch_size, 3, 32, 32]
   ↓ Conv2D
Feature volume [batch_size, 4, 15, 15]
   ↓ MaxPool2D
Feature volume [batch_size, 4, 8, 8]
   ↓ Flatten
FC layer: [batch_size, 256]
   ↓
Output: [batch_size, 10]
   ↓ softmax
```

**PyTorch layer definitions matching this example**:

```python
Conv2d(3, 4, kernel_size=4, stride=2, padding=0)
MaxPool2d(kernel_size=2, stride=2, padding=1)
```

---

## 8. Automatic Feature Extraction & Receptive Fields

**Feature hierarchy** (deep learning for visual data):

|Level|Representation|Examples|
|---|---|---|
|Low-level|edges, pixels, corners|first conv layers|
|Mid-level|circles, triangles, boxes|middle conv layers|
|High-level|objects|later conv layers, before classifier|

**Receptive field**: the region of the _original input image_ that a given neuron's activation is influenced by. As you go deeper into the network (Layer 1 → Layer 2 → Layer 3), each neuron's receptive field **grows larger**, because each layer's neuron aggregates information from a local patch of the _previous_ layer, which itself already aggregated a patch of the layer before that — so the effect compounds with depth.

**Why this matters**: deeper neurons can "see" more of the original image (larger context), which is what allows later layers to represent whole objects rather than just small local textures.

---

## 9. How CNNs Work: Building Abstraction Through Depth

**Mechanism**: higher-level filters combine lower-level filters to represent more abstract concepts. The context is "locally expanded" at each layer.

**Example (Human face, simplified shapes)**:

- **Filter 1 (depth=1)** and **Filter 2 (depth=1)** each detect simple diagonal line segments → produce **Feature map 1**, **Feature map 2** (each showing where those line orientations appear)
- **Filter 1 (depth=2)** — operating on the _previous layer's_ feature maps (not the raw image) — combines patterns across Feature maps 1 and 2 to detect more complex shapes (e.g. combinations of lines forming parts of a rectangle or triangle)

This illustrates the general CNN principle: **early layers detect simple local patterns (edges/lines); later layers combine those patterns into increasingly complex, abstract shapes**, eventually assembling recognizable objects (in the face example: eyes, nose, mouth as combinations of simpler shapes).

---

## 10. Key Limitation: CNNs Cannot Capture Spatial Relationships

**The problem**: after a series of conv → pool operations across multiple filters, the network produces several feature maps, each indicating "this shape/pattern was detected somewhere in the image." These feature maps are then **flattened** into a single vector before the final classification (dense + softmax) layers.

**What's lost in flattening**: the **relative spatial arrangement** between detected features. The network can confidently say "an eye-shaped pattern was detected" and "a nose-shaped pattern was detected" and "a mouth-shaped pattern was detected" — but it has no explicit mechanism to check _how these detected features are arranged relative to each other_ (e.g., are the eyes above the nose? Is the mouth below the nose? Are the two eyes at the same height?).

**Concrete illustration** (face recognition example):

```
Filter 1 → Feature map 1 (detects diagonal-line-like patterns, e.g. potential "eye" regions)
Filter 2 → Feature map 2 (detects other rectangle/line patterns, e.g. potential "nose" regions)
Filter 3 → Feature map 3 (detects patterns near a triangle shape)
Filter 4 → Feature map 4 (detects rectangle/box patterns, e.g. potential "mouth" regions)
       ↓ Flatten
       ↓ Prediction
```

Because flattening discards 2D spatial structure and just lists "which features were detected, how strongly" as a 1D vector, a CNN can be fooled by an image where all the "correct" features (eye-shapes, nose-shapes, mouth-shapes) are present but **scrambled into an anatomically incorrect arrangement** — the network may still confidently classify it as a face, since it primarily reasons over "which patterns fired," not "where, relative to each other, did they fire."

**This is a foundational motivation for later architectures** (covered in later parts of the unit) that attempt to preserve or explicitly model spatial/positional relationships — e.g. attention mechanisms, Capsule Networks, or Vision Transformers with positional encoding.

---

## Summary Table: Key Formulas (Part 3)

|Concept|Formula|
|---|---|
|SGD with momentum velocity|$v = \alpha g + (1-\alpha) v$|
|SGD with momentum update|$\theta = \theta - \eta v$|
|AdaGrad accumulator|$\gamma = \gamma + g\odot g$|
|AdaGrad update|$\theta = \theta - \dfrac{\eta}{\epsilon+\sqrt{\gamma}}\odot g$|
|Convolution (general)|$W*x = \text{sum}(W\otimes x)$|
|1D convolution|$\sum_{i=1}^m W_i x_i$|
|2D convolution|$\sum_{i=1}^m\sum_{j=1}^n W_{ij}x_{ij}$|
|3D convolution|$\sum_{i=1}^m\sum_{j=1}^n\sum_{k=1}^p W_{ijk}x_{ijk}$|
|Conv/pool output size|$W_o=\lfloor\frac{W_i+2p-f_w}{s_w}\rfloor+1,\ H_o=\lfloor\frac{H_i+2p-f_h}{s_h}\rfloor+1$|
|Flatten (3D → 1D)|$H\times W\times C$ neurons|

---

## Quick Comparison: Optimizers Covered So Far

|Optimizer|Key mechanism|Main strength|Main weakness|
|---|---|---|---|
|SGD|mini-batch gradient only|simple, cheap per-step|slows down significantly over training|
|SGD + Momentum|velocity accumulates past gradients|smooths oscillation, addresses Hessian conditioning & gradient variance|extra hyperparameter $\alpha$ to tune|
|AdaGrad|per-parameter learning rate scaled by cumulative squared gradient|adapts step size per-dimension automatically|learning rate only shrinks — poor for non-convex DL over long training|