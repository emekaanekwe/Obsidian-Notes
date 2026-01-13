# Feed-Forward Neural Networks

---

## 1. The Single Neuron (Building Block)

### Basic Structure

A **neuron** is a computational unit that:

- Takes inputs: $\mathbf{x} := (x_1, x_2, x_3)$ (and a +1 intercept term)
- Computes a weighted sum plus bias
- Applies an activation function
- Produces an output

### Mathematical Formulation

$$h_{W,b}(\mathbf{x}) := f(W^T \mathbf{x}) = f\left(\sum_{i=1}^{3} W_i x_i + b\right)$$

where:

- $W_i$ are the **weights** (parameters)
- $b$ is the **bias** (intercept term)
- $f: \mathbb{R} \to \mathbb{R}$ is the **activation function**

**Diagram:**

```
x₁ ─────╲
x₂ ─────→ ● ───→ h_{w,b}(x)
x₃ ─────╱
+1 ─────
```

---

## 2. Activation Functions

### Why Activation Functions?

Activation functions introduce **non-linearity** into the network. Without them, multiple layers would collapse into a single linear transformation (no matter how deep!).

### Common Activation Functions

#### 2.1 Sigmoid Function

$$\sigma(z) := \frac{1}{1 + \exp(-z)}$$

**Properties:**

- **Output range:** $[0, 1]$
- **Interpretation:** Can be viewed as a probability
- **Shape:** S-curve (smooth, bounded)
- **Derivative:** $\frac{\partial}{\partial z}\sigma(z) = \sigma(z)(1 - \sigma(z))$

**When to use:** Output layer for binary classification

#### 2.2 Hyperbolic Tangent (tanh)

$$\tanh(z) := \frac{e^z - e^{-z}}{e^z + e^{-z}}$$

**Properties:**

- **Output range:** $[-1, 1]$
- **Relationship to sigmoid:** tanh is a rescaled version of sigmoid
- **Shape:** S-curve, but centered at zero
- **Derivative:** $\frac{\partial}{\partial z}\tanh(z) = 1 - (\tanh(z))^2$

**When to use:** Hidden layers (zero-centered outputs help with learning)

#### 2.3 Rectified Linear Unit (ReLU)

$$\text{ReLU}(z) := \max(0, z)$$

**Properties:**

- **Output range:** $[0, \infty)$
- **Shape:** Linear for positive values, zero for negative
- **Derivative:** 1 if $z > 0$, 0 if $z < 0$ (undefined at 0)
- **Advantages:** Computationally efficient, helps with vanishing gradient problem

**When to use:** Most common choice for hidden layers in modern deep learning

### Activation Function Derivatives (Critical for Training!)

These derivatives are essential for backpropagation (next chapter):

$$\frac{\partial}{\partial z}\sigma(z) = \sigma(z)(1 - \sigma(z))$$

$$\frac{\partial}{\partial z}\tanh(z) = 1 - (\tanh(z))^2$$

**Memory Aid for Sigmoid Derivative:**

- At $\sigma(z) = 0.5$ (when $z = 0$): derivative is $0.5 \times 0.5 = 0.25$ (maximum)
- At extremes ($\sigma \approx 0$ or $\sigma \approx 1$): derivative $\approx 0$ (vanishing gradient!)

---

## 3. Neural Network Architecture

### From Single Neuron to Network

A **neural network** is created by **hooking together many neurons**, where the output of one neuron can be the input to another.

### Terminology

**Layers:**

- **Input layer** ($L_1$): The leftmost layer, receives the raw input data
- **Hidden layers** ($L_2, L_3, \ldots, L_{n_l-1}$): Middle layers, not observed in training
- **Output layer** ($L_{n_l}$): The rightmost layer, produces the final prediction

**Bias Units:**

- Circles labeled "+1"
- Correspond to the **intercept term**
- Always output the value +1
- Present in input layer and all hidden layers (not in output layer)

**Network Size:**

- Let $n_l$ = number of layers (thus $n_l = 3$ in the example)
- Layer $\ell$ is denoted as $L_{\ell}$
- $L_1$ is the input layer, $L_{n_l}$ is the output layer
- Let $s_{\ell}$ = number of nodes in layer $\ell$ (not counting bias unit)

### Example: 3-Layer Network

```
      x₁ ─────╲                        ╱─→ a₁⁽²⁾
               ╲                      ╱
      x₂ ─────→●───→●───→●──────────●─────→ h_{W,b}(x)
               ╱     ╱     ╲        ╲
      x₃ ─────╱     ╱       ╲        ╲─→ a₃⁽²⁾
                   ╱         ╲
      +1 ────────+1           +1
      
   Layer L₁    Layer L₂    Layer L₃
   (Input)     (Hidden)    (Output)
```

**In this example:**

- 3 input units (not counting bias)
- 3 hidden units (not counting bias)
- 1 output unit
- Network has $n_l = 3$ layers

---

## 4. Network Parameters

### Notation for Weights and Biases

The neural network has parameters: $$\boldsymbol{\theta} = (\mathbf{W}^{(1)}, \mathbf{b}^{(1)}, \mathbf{W}^{(2)}, \mathbf{b}^{(2)})$$

**Weight notation:**

- $W_{ij}^{(\ell)}$ = weight associated with connection from unit $j$ in layer $\ell$ to unit $i$ in layer $\ell + 1$
- Note the order of indices: **destination first, source second**

**Bias notation:**

- $b_i^{(\ell)}$ = bias associated with unit $i$ in layer $\ell + 1$

**Matrix dimensions:**

- $\mathbf{W}^{(1)} \in \mathbb{R}^{3 \times 3}$ (in our example)
- $\mathbf{W}^{(2)} \in \mathbb{R}^{1 \times 3}$ (in our example)

**Important:** Bias units don't have inputs or connections going into them; they always output +1.

---

## 5. Forward Propagation (Computing Network Output)

### Single Unit Computation

For unit $i$ in layer $\ell$:

**Step 1: Compute weighted input** (pre-activation) $$z_i^{(\ell)} := \sum_{j=1}^{s_{\ell-1}} W_{ij}^{(\ell-1)} x_j + b_i^{(\ell-1)}$$

**Step 2: Apply activation function** (activation) $$a_i^{(\ell)} := f(z_i^{(\ell)})$$

### Example Network Computation (Detailed)

For our 3-layer network:

**Layer 2 (Hidden Layer) Computations:** $$a_1^{(2)} := f(W_{11}^{(1)} x_1 + W_{12}^{(1)} x_2 + W_{13}^{(1)} x_3 + b_1^{(1)})$$

$$a_2^{(2)} := f(W_{21}^{(1)} x_1 + W_{22}^{(1)} x_2 + W_{23}^{(1)} x_3 + b_2^{(1)})$$

$$a_3^{(2)} := f(W_{31}^{(1)} x_1 + W_{32}^{(1)} x_2 + W_{33}^{(1)} x_3 + b_3^{(1)})$$

**Layer 3 (Output Layer) Computation:** $$h_{\boldsymbol{\theta}}(\mathbf{x}) := f(W_{11}^{(2)} a_1^{(2)} + W_{12}^{(2)} a_2^{(2)} + W_{13}^{(2)} a_3^{(2)} + b_1^{(2)})$$

---

## 6. Vectorized Forward Propagation

### Compact Notation

We can extend the activation function $f(\cdot)$ to apply **element-wise** to vectors: $$f([z_1, z_2, z_3]) = [f(z_1), f(z_2), f(z_3)]$$

This allows us to write forward propagation compactly:

**General form for layer $\ell + 1$:** $$\mathbf{z}^{(\ell+1)} := \mathbf{W}^{(\ell)} \mathbf{a}^{(\ell)} + \mathbf{b}^{(\ell)}$$

$$\mathbf{a}^{(\ell+1)} := f(\mathbf{z}^{(\ell+1)})$$

**For our example (starting from input):** $$\mathbf{z}^{(2)} := \mathbf{W}^{(1)} \mathbf{x} + \mathbf{b}^{(1)}$$ $$\mathbf{a}^{(2)} := f(\mathbf{z}^{(2)})$$ $$\mathbf{z}^{(3)} := \mathbf{W}^{(2)} \mathbf{a}^{(2)} + \mathbf{b}^{(2)}$$ $$h_{\boldsymbol{\theta}}(\mathbf{x}) := f(\mathbf{z}^{(3)})$$

### Notation Convention

We use $\mathbf{a}^{(1)} = \mathbf{x}$ to denote the input layer activations.

**The Forward Propagation Algorithm:** Given layer $\ell$'s activations $\mathbf{a}^{(\ell)}$, we can compute layer $\ell + 1$'s activations $\mathbf{a}^{(\ell+1)}$ using the formulas above.

**Why "Forward"?** Because we're moving information from input → hidden → output (left to right in the diagram).

---

## 7. General Feed-Forward Architecture

### Definition

A **feed-forward neural network** is one where the connectivity graph has **no directed loops or cycles**.

Most common choice: **$n_l$-layered network** where:

- Layer $L_1$ is the input layer
- Layer $L_{n_l}$ is the output layer
- Each layer $\ell$ is **densely connected** to layer $\ell + 1$
- All connections go forward (no feedback loops)

### Computing Network Output

To compute the output, we successively compute all activations in:

- Layer $L_2$, then
- Layer $L_3$, and so on, up to
- Layer $L_{n_l}$

Using the vectorized equations above that describe forward propagation.

### Advantage of Matrix Operations

By organizing parameters in matrices and using matrix-vector operations (provided by GPUs), we can take advantage of fast linear algebra routines to quickly perform calculations.

---

## 8. Multiple Output Units

### Extension to Multi-Output Networks

Neural networks can have **multiple output units**. For example, a 4-layer network with 2 output units:

```
      x₁ ─────╲                             ╱─→ output 1
               ╲                           ╱
      x₂ ─────→●───→●───→●──────────────→●
               ╱     ╱     ╲               ╲
      x₃ ─────╱     ╱       ╲               ╲─→ output 2
                   ╱         ╲
      +1 ────────+1           +1
      
   Layer L₁    Layer L₂  Layer L₃   Layer L₄
```

### Training Data Format

To train this network, we need training examples $(\mathbf{x}_i, \mathbf{y}_i)$ where $\mathbf{y}_i \in \mathbb{R}^2$.

**Use Cases:**

- **Multi-class classification:** Output layer has one unit per class
- **Multi-task learning:** Different outputs predict different properties
- **Medical diagnosis:** Different outputs for different diseases

**Example:**

- Input $\mathbf{x}$: patient features
- Outputs $\mathbf{y}$: presence/absence of different diseases

---

## 9. Neural Networks vs Perceptron

### Key Differences

|Aspect|Perceptron|Neural Networks|
|---|---|---|
|**Stages**|Multiple stages of processing|Multiple stages of processing|
|**Nonlinearities**|Step-function (discontinuous)|Continuous (e.g., $\sigma$, $\tanh$)|
|**Differentiability**|Not differentiable|**Differentiable** w.r.t. parameters|
|**Training**|Perceptron algorithm|Gradient-based (backpropagation)|
|**Power**|Linear decision boundaries|Can approximate any function|

### Why Continuous Nonlinearities Matter

The neural network function is **differentiable with respect to the network parameters**.

This crucial property enables:

- **Gradient descent** optimization
- **Backpropagation** algorithm (next chapter)
- Efficient training on large datasets

The perceptron's step function is not differentiable, limiting optimization options.

---

## 10. The Power of Neural Networks (Universal Approximation)

### Theoretical Results

Feed-forward networks have been proven to have very general approximation properties:

**Universal Approximation Theorem:** Neural networks are **universal approximators**. Specifically:

1. A **2-layer network** with linear outputs can uniformly approximate any **continuous function** on an input domain to arbitrary accuracy, provided the network has a sufficiently large number of hidden units.
    
2. This holds for a wide range of hidden unit activation functions (but excluding polynomials).
    

### What This Means

**For Regression:**

- Neural networks can approximate the target function to any precision
- Given enough hidden units

**For Classification:**

- Neural networks can approximate the target decision boundary to any precision
- Can represent almost any classification rule

### Important Caveats

1. **"Sufficient hidden units"** might mean a very large number in practice
2. The theorem says approximation is **possible**, not that we can **find** it (training may be difficult)
3. Doesn't tell us the **architecture** needed or **how to train** it
4. **Generalization** to unseen data is a separate concern

---

## 11. Key Concepts and Intuitions

### Understanding Layers

**Input Layer ($L_1$):**

- Simply holds the input features
- No computation happens here
- $\mathbf{a}^{(1)} = \mathbf{x}$

**Hidden Layers ($L_2, \ldots, L_{n_l-1}$):**

- Extract increasingly abstract features
- Early layers: low-level features (edges, textures)
- Later layers: high-level features (objects, concepts)
- **Why "hidden"?** We don't observe these values during training (only inputs and outputs)

**Output Layer ($L_{n_l}$):**

- Produces final prediction
- Number of units depends on task:
    - **1 unit:** Binary classification or regression
    - **K units:** K-class classification

### The Role of Depth

**Why Multiple Layers?**

- Single hidden layer is universal approximator, but might need exponentially many units
- Multiple layers can represent same function with exponentially fewer parameters
- Depth creates **hierarchical representations**

**Deep Learning:**

- Networks with many hidden layers (typically 10-100+ layers)
- Each layer builds on previous layer's features
- Enables learning of complex, hierarchical patterns

### Computational Flow

Think of forward propagation as:

1. **Input layer:** Raw features
2. **Hidden layer 1:** Combinations of raw features
3. **Hidden layer 2:** Combinations of combinations
4. **Output layer:** Final decision based on high-level features

Each layer transforms the representation to be more suitable for the task.

---

## 12. Practical Considerations

### Choosing Network Architecture

**Number of layers:**

- Start with 1-2 hidden layers
- Increase if underfitting
- Modern deep networks: 10-100+ layers

**Number of units per layer:**

- Rule of thumb: between input and output size
- Often use same size for all hidden layers
- More units = more capacity, but risk of overfitting

**Activation functions:**

- **Hidden layers:** ReLU (most common), tanh
- **Output layer:**
    - Sigmoid for binary classification
    - Softmax for multi-class classification
    - Linear for regression

### Network Notation Summary

For a network with $n_l$ layers:

- $s_{\ell}$ = number of units in layer $\ell$ (excluding bias)
- $\mathbf{W}^{(\ell)} \in \mathbb{R}^{s_{\ell+1} \times s_{\ell}}$ = weight matrix from layer $\ell$ to $\ell+1$
- $\mathbf{b}^{(\ell)} \in \mathbb{R}^{s_{\ell+1}}$ = bias vector for layer $\ell+1$
- $\mathbf{z}^{(\ell)} \in \mathbb{R}^{s_{\ell}}$ = pre-activation values in layer $\ell$
- $\mathbf{a}^{(\ell)} \in \mathbb{R}^{s_{\ell}}$ = activation values in layer $\ell$

---

## 13. Mathematical Prerequisites Refresher

### To Understand Neural Networks, You Need:

#### Matrix-Vector Multiplication

$$\mathbf{y} = \mathbf{W}\mathbf{x}$$

where if $\mathbf{W} \in \mathbb{R}^{m \times n}$ and $\mathbf{x} \in \mathbb{R}^n$, then $\mathbf{y} \in \mathbb{R}^m$.

Each element: $y_i = \sum_{j=1}^{n} W_{ij} x_j$

#### Element-wise Function Application

For a function $f$ and vector $\mathbf{z} = [z_1, z_2, z_3]$: $$f(\mathbf{z}) = [f(z_1), f(z_2), f(z_3)]$$

This is sometimes denoted with $\odot$ for element-wise operations.

#### Chain Rule (for backpropagation in next chapter)

If $y = f(g(x))$, then: $$\frac{dy}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

This extends to multivariate functions via the Jacobian.

---

## Summary: Key Takeaways

1. **Single neuron** computes: weighted sum → activation function → output
    
2. **Activation functions** introduce nonlinearity (sigmoid, tanh, ReLU are most common)
    
3. **Neural networks** are built by stacking layers of neurons
    
4. **Forward propagation** computes output by passing information through layers: $\mathbf{a}^{(\ell+1)} = f(\mathbf{W}^{(\ell)} \mathbf{a}^{(\ell)} + \mathbf{b}^{(\ell)})$
    
5. **Differentiability** of activation functions enables gradient-based training (next chapter!)
    
6. **Universal approximation** theorem: 2-layer networks can approximate any continuous function
    
7. **Depth** provides efficiency: deep networks can represent complex functions with fewer parameters than shallow wide networks
    
8. **Feed-forward** means no cycles: information flows only in one direction (input → output)
    

**Next Up:** Backpropagation algorithm for training these networks!