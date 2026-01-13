# study list (math-focused)

1. **Prereqs:** logs, softmax, dot products, gradients, chain rule
    
2. **Core NN math:** logits → probabilities → loss
    
3. **Key gradients:** p−yp-yp−y results, backprop intuition
    
4. **Optimization:** GD/SGD/(momentum/Adam if in slides)
    
5. **Regularization & stability:** L2, dropout (if covered), gradient clipping, vanishing/exploding
    
6. **Normalization:** BatchNorm (and LayerNorm if transformers)
    
7. **Architectures:**
    
    - Feed-forward classification
        
    - CNNs (shapes + conv math + pooling)
        
    - RNN/GRU/LSTM (sequence recurrence + gating)
        
    - Word2Vec (neg sampling objective)
        
    - Seq2Seq + attention
        
    - Transformers / ViT / Swin (attention math + patch tokens)
        
    - GANs (objectives)

---

## The correct logical order (matches the knowledge graph)

1. **How training works (core math):** objective J(θ)J(\theta)J(θ), forward pass, backprop, gradients
    
2. **Activation functions:** ReLU/tanh/sigmoid + their derivatives
    
3. **Loss functions + probabilities:** logits → sigmoid/softmax → cross-entropy (and MSE for regression)
    
4. **Optimization basics:** GD / SGD update rules (not Adam memorization)
    
5. **Stability/regularization:** L2, dropout idea, batch norm, gradient clipping
    
6. **Supervised model math:**
    
    - Feed-forward NN (vector data)
        
    - CNNs (vision data: shapes/conv/pool; plus attack/defense)
        
    - RNNs (sequential data)
        
    - Advanced sequential: Seq2Seq + (self-)Attention
        
7. **Self-supervised:** Word2Vec (Skip-gram/CBOW, negative sampling)
    
8. **Generative:** GAN objective
    
9. **Fine-tuning:** prompt-tuning, LoRA, adapters (parameter-efficient updates)








---

if ||gradient|| > threshold:
       gradient = threshold * (gradient / ||gradient||)

✅ **Lower learning rate**  
✅ **Weight regularization** (L2 penalty)  
✅ **Batch normalization**  
✅ Better weight initialization

---

### Gradient Clipping (Key Technique)

**Formula:**
$$\tilde{\mathbf{g}} = \begin{cases} 
\mathbf{g} & \text{if } ||\mathbf{g}|| \leq \text{threshold} \\
\frac{\text{threshold}}{||\mathbf{g}||} \mathbf{g} & \text{if } ||\mathbf{g}|| > \text{threshold}
\end{cases}$$

where:
- $\mathbf{g}$ = gradient vector
- $||\mathbf{g}||$ = gradient norm (magnitude)
- $\tilde{\mathbf{g}}$ = clipped gradient

**Types:**
1. **Clip by value:** Clip each element to [-threshold, threshold]
2. **Clip by norm:** Scale entire gradient if norm exceeds threshold

---

## 6. Overfitting and Underfitting

### Underfitting

**Definition:** Model is too simple to capture underlying patterns in data.

**Symptoms:**
- High training error
- High validation error
- Poor performance on both train and test sets
- Model cannot fit training data

**Visual:**
```
     Data points: o o o o o
Model prediction: ————————— (straight line through curved data)
```

**Causes:**
- Model too simple (not enough parameters)
- Too much regularization
- Insufficient training time
- Poor feature representation

**Solutions:**
✅ Increase model capacity (more layers, more units)  
✅ Reduce regularization  
✅ Train longer  
✅ Better feature engineering  
✅ Use more complex architecture

---

### Overfitting

**Definition:** Model learns training data too well, including noise and outliers, failing to generalize.

**Symptoms:**
- Very low training error
- High validation error
- **Large gap** between train and validation performance
- Model memorizes rather than learns patterns

**Visual:**
```
     Data points: o o o o o
Model prediction: ~~~∿~∿~~~ (wiggly line through every point)
```

**Causes:**
- Model too complex (too many parameters)
- Insufficient training data
- Training too long
- No regularization

**Solutions:**
✅ More training data  
✅ Data augmentation  
✅ Regularization (L1, L2, dropout)  
✅ Early stopping  
✅ Reduce model complexity  
✅ Cross-validation

---

### Bias-Variance Tradeoff

**Formula:**
$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

**Components:**

**Bias:** Error from wrong assumptions
- High bias = underfitting
- Model is too simple

**Variance:** Error from sensitivity to training data
- High variance = overfitting
- Model is too complex

**Irreducible Error:** Noise in data (can't be reduced)

| Model Complexity | Bias | Variance | Total Error |
|------------------|------|----------|-------------|
| Too simple | High | Low | High (underfit) |
| Just right | Low | Low | **Minimum** |
| Too complex | Low | High | High (overfit) |

---

## 7. Deep Learning Regularization

### What is Regularization?
Techniques to prevent overfitting by constraining or adding penalty to the model.

---

### 7.1 L1 and L2 Regularization

**L2 Regularization (Weight Decay):**

**Loss function:**
$$J(\theta) = \text{Loss}(\theta) + \frac{\lambda}{2} \sum_{i} w_i^2$$

where:
- $\lambda$ = regularization strength
- $w_i$ = model weights
- Higher $\lambda$ = more regularization

**Effect:**
- Penalizes large weights
- Encourages small, distributed weights
- Smooth decision boundaries

**Gradient update:**
$$w_i \leftarrow w_i - \eta \frac{\partial \text{Loss}}{\partial w_i} - \eta \lambda w_i$$
$$= (1 - \eta\lambda) w_i - \eta \frac{\partial \text{Loss}}{\partial w_i}$$

The $(1 - \eta\lambda)$ term causes "weight decay"

**L1 Regularization:**

**Loss function:**
$$J(\theta) = \text{Loss}(\theta) + \lambda \sum_{i} |w_i|$$

**Effect:**
- Encourages **sparse** weights (many exactly zero)
- Feature selection
- Interpretable models

---

### 7.2 Dropout

**What it does:** Randomly "drop" (set to zero) neurons during training with probability $p$.

**Formula:**

**Training:**
$$h_i = \begin{cases}
0 & \text{with probability } p \\
\frac{a_i}{1-p} & \text{with probability } (1-p)
\end{cases}$$

where:
- $a_i$ = activation before dropout
- $h_i$ = activation after dropout
- $p$ = dropout rate (typically 0.5)

**Testing:** Use all neurons, no dropout

**Why it works:**
- Prevents co-adaptation of neurons
- Equivalent to training ensemble of networks
- Forces redundancy in representations

**Common values:**
- Hidden layers: $p = 0.5$
- Input layer: $p = 0.1$ to $0.2$

**Advantages:**
✅ Very effective for large networks  
✅ Acts like ensemble learning  
✅ Simple to implement

**Disadvantages:**
❌ Increases training time  
❌ Requires tuning dropout rate  
❌ Can slow convergence

---

### 7.3 Early Stopping

**What it does:** Stop training when validation error stops improving.

**Algorithm:**
```
1. Train model on training set
2. Evaluate on validation set after each epoch
3. Keep track of best validation performance
4. If no improvement for N epochs (patience):
   - Stop training
   - Return model with best validation performance
```

**Advantages:**
✅ Simple and effective  
✅ No hyperparameters to tune (besides patience)  
✅ Computational efficiency

**Disadvantages:**
❌ Requires validation set  
❌ Might stop too early  
❌ Doesn't reduce model size

---

### 7.4 Data Augmentation

**What it does:** Artificially increase training data by applying transformations.

**For images:**
- Rotation, flipping, cropping
- Color jittering
- Random erasing
- Mixup (blend two images)

**For text:**
- Synonym replacement
- Back-translation
- Random insertion/deletion

**Advantages:**
✅ Increases effective dataset size  
✅ Improves generalization  
✅ Reduces overfitting

---

## 8. Batch Normalization

### What It Is
**Batch Normalization (BatchNorm)** normalizes activations within each mini-batch to have zero mean and unit variance.

### The Problem It Solves
**Internal Covariate Shift:** Distribution of layer inputs changes during training, slowing down learning.

### Formula

**For a mini-batch** $\mathcal{B} = \{x_1, \ldots, x_m\}$:

**Step 1: Compute mean and variance**
 $$\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i$$
$$\sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2$$

**Step 2: Normalize**
$$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$

where $\epsilon$ (e.g., $10^{-5}$) prevents division by zero

**Step 3: Scale and shift (learnable parameters)**
$$y_i = \gamma \hat{x}_i + \beta$$

where:
- $\gamma$ = learnable scale parameter
- $\beta$ = learnable shift parameter
- Allows model to undo normalization if needed!

### Where to Apply

**Typical placement:**
```
Input → Linear/Conv → BatchNorm → Activation → Next Layer
```

Or:
```
Input → Linear/Conv → Activation → BatchNorm → Next Layer
```

(Order debated; both work)

### Training vs Testing

**Training:** Use batch statistics ($\mu_{\mathcal{B}}$, $\sigma_{\mathcal{B}}^2$)

**Testing:** Use running averages from training:
$$\mu_{\text{test}} = \mathbb{E}[\mu_{\mathcal{B}}]$$
$$\sigma_{\text{test}}^2 = \mathbb{E}[\sigma_{\mathcal{B}}^2]$$

### Advantages
✅ Faster training (can use higher learning rates)  
✅ Reduces internal covariate shift  
✅ Acts as regularization  
✅ Less sensitive to initialization  
✅ Reduces gradient vanishing

### Disadvantages
❌ Adds computational cost  
❌ Behaves differently in training vs testing  
❌ Doesn't work well with small batch sizes  
❌ Doesn't work well for RNNs (different sequence lengths)  
❌ Can hurt performance in some cases (GANs, RL)

---

# Part 3: Recurrent Neural Networks

## 9. Recurrent Neural Networks (RNNs)

### What They Are
**RNNs** are neural networks designed to process **sequential data** by maintaining a hidden state that captures information from previous time steps.

### Key Idea
Unlike feedforward networks, RNNs have **loops** that allow information to persist.

### Architecture Diagram
```
    x₁        x₂        x₃        x₄
    ↓         ↓         ↓         ↓
   [RNN] → [RNN] → [RNN] → [RNN]
    ↓         ↓         ↓         ↓
    h₁        h₂        h₃        h₄
    ↓         ↓         ↓         ↓
    y₁        y₂        y₃        y₄
```

### Formula

**Hidden state update:**
$$\mathbf{h}_t = \tanh(\mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{W}_{xh} \mathbf{x}_t + \mathbf{b}_h)$$

**Output:**
$$\mathbf{y}_t = \mathbf{W}_{hy} \mathbf{h}_t + \mathbf{b}_y$$

where:
- $\mathbf{x}_t$ = input at time $t$
- $\mathbf{h}_t$ = hidden state at time $t$
- $\mathbf{h}_{t-1}$ = previous hidden state
- $\mathbf{y}_t$ = output at time $t$
- $\mathbf{W}_{hh}$ = hidden-to-hidden weights (recurrent weights)
- $\mathbf{W}_{xh}$ = input-to-hidden weights
- $\mathbf{W}_{hy}$ = hidden-to-output weights

**Key insight:** Same weights $(\mathbf{W}_{hh}, \mathbf{W}_{xh}, \mathbf{W}_{hy})$ used at every time step!

### Unrolled View
```
Input:  x₁ → x₂ → x₃ → x₄
        ↓    ↓    ↓    ↓
Hidden: h₁ → h₂ → h₃ → h₄  (arrows show information flow)
        ↓    ↓    ↓    ↓
Output: y₁   y₂   y₃   y₄
```

### Advantages
✅ Can process variable-length sequences  
✅ Model size doesn't grow with sequence length  
✅ Computation considers historical information  
✅ Weights shared across time steps

### Disadvantages
❌ **Vanishing/exploding gradients** (major problem!)  
❌ Difficult to capture long-range dependencies  
❌ Sequential processing (can't parallelize)  
❌ Training is slow

---

## 10. RNN Architecture Types

### Based on Input/Output Structure

**1. One-to-One**
```
x → [RNN] → y
```
- Standard neural network
- Not really an RNN
- Example: Image classification

**2. One-to-Many**
```
x → [RNN] → [RNN] → [RNN] → y₁, y₂, y₃
```
- One input, sequence output
- Example: Image captioning (image → sentence)

**3. Many-to-One**
```
x₁, x₂, x₃ → [RNN] → [RNN] → [RNN] → y
```
- Sequence input, one output
- Example: Sentiment analysis (sentence → positive/negative)

**4. Many-to-Many (same length)**
```
x₁, x₂, x₃ → [RNN] → [RNN] → [RNN] → y₁, y₂, y₃
```
- Sequence input, sequence output (aligned)
- Example: Video classification (frame-by-frame labeling)

**5. Many-to-Many (different lengths)**
```
x₁, x₂, x₃ → [Encoder] → [Decoder] → y₁, y₂, y₃, y₄
```
- Sequence input, different length sequence output
- Example: Machine translation (English → French)

---

### Based on Depth

**1. Vanilla RNN (Single Layer)**
```
x₁ → x₂ → x₃
↓    ↓    ↓
h₁ → h₂ → h₃
```

**2. Stacked/Deep RNN (Multiple Layers)**
```
Layer 2:  h₁² → h₂² → h₃²
          ↑    ↑    ↑
Layer 1:  h₁¹ → h₂¹ → h₃¹
          ↑    ↑    ↑
Input:    x₁   x₂   x₃
```

Each layer's output becomes the next layer's input.

**3. Bidirectional RNN**
```
Forward:  h₁→ → h₂→ → h₃→
          ↑    ↑    ↑
Input:    x₁   x₂   x₃
          ↓    ↓    ↓
Backward: h₁← ← h₂← ← h₃←

Final: h₁ = [h₁→; h₁←]
```

Processes sequence in both directions, concatenates hidden states.

**Use case:** When future context helps (e.g., POS tagging, named entity recognition)

---

## 11. LSTM (Long Short-Term Memory)

### The Problem with Vanilla RNNs
- Vanishing gradients make learning long-range dependencies impossible
- Information from early time steps gets lost

### The LSTM Solution
Introduces a **cell state** that acts as a "memory highway" and **gates** that control information flow.

### Architecture Diagram
```
        ┌─────────────────────────┐
        │     Cell State Cₜ       │  ← Memory highway
        └─────────────────────────┘
              ↑           ↓
         Forget Gate   Output Gate
              ↑           ↓
         Input Gate
              ↑
        Hidden State hₜ₋₁, Input xₜ