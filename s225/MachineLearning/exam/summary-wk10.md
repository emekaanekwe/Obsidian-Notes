# Study Sheet: Network Training (Backpropagation & Optimization)

---

## 1. The Training Objective

### Problem Setup

**Goal:** Determine the network parameters to minimize prediction error.

**Given:**

- Training set of input vectors: ${\mathbf{x}^{(n)}}$ where $n = 1, \ldots, N$
- Corresponding target vectors: ${\mathbf{y}^{(n)}}$

Neural networks map from input vector $\mathbf{x}$ to output vector $\mathbf{y}$ using a parametric nonlinear function $h_{\boldsymbol{\theta}}(\mathbf{x})$.

---

## 2. Error Functions for Different Tasks

### 2.1 Regression

**Error Function (Sum-of-Squares):**

$$E(\boldsymbol{\theta}) := \frac{1}{N} \sum_{n=1}^{N} \frac{1}{2} ||h_{\boldsymbol{\theta}}(\mathbf{x}^{(n)}) - \mathbf{y}^{(n)}||_2^2$$

**Probabilistic Interpretation:**

Assume the target is distributed according to a Gaussian: $$p(\mathbf{y}|\mathbf{x}, \boldsymbol{\theta}) := \mathcal{N}(\mathbf{y}|h_{\boldsymbol{\theta}}(\mathbf{x}), \mathbf{I})$$

where $\mathbf{I}$ is the identity covariance matrix.

**Log-Likelihood:**

$$\mathcal{L}(\boldsymbol{\theta}) := -\sum_{n=1}^{N} \ln p(\mathbf{y}^{(n)}|\mathbf{x}^{(n)}, \boldsymbol{\theta}) = \sum_{n=1}^{N} ||h_{\boldsymbol{\theta}}(\mathbf{x}^{(n)}) - \mathbf{y}^{(n)}||_2^2 + \text{constant}$$

**Key Insight:** Maximizing log-likelihood $\mathcal{L}(\boldsymbol{\theta})$ is equivalent to minimizing error function $E(\boldsymbol{\theta})$.

**Why Sum-of-Squares for Regression?**

- Corresponds to maximum likelihood under Gaussian noise assumption
- Penalizes large errors more heavily (quadratic penalty)
- Differentiable everywhere (needed for gradient descent)

---

### 2.2 Binary Classification

**Setup:**

- Single target variable $y$ where $y = 1$ denotes class $C_1$ and $y = 0$ denotes class $C_2$
- Network has single output with sigmoid activation: $\sigma \in [0,1]$

**Probabilistic Interpretation:**

The network output $h_{\boldsymbol{\theta}}(\mathbf{x})$ represents the conditional probability: $$p(C_1|\mathbf{x}) = h_{\boldsymbol{\theta}}(\mathbf{x})$$ $$p(C_2|\mathbf{x}) = 1 - h_{\boldsymbol{\theta}}(\mathbf{x})$$

**Conditional Probability of Training Pair:**

$$p(y|\mathbf{x}, \boldsymbol{\theta}) = [h_{\boldsymbol{\theta}}(\mathbf{x})]^y [1 - h_{\boldsymbol{\theta}}(\mathbf{x})]^{1-y}$$

This elegantly captures both cases:

- If $y = 1$: $p = h_{\boldsymbol{\theta}}(\mathbf{x})$
- If $y = 0$: $p = 1 - h_{\boldsymbol{\theta}}(\mathbf{x})$

**Error Function (Cross-Entropy):**

Given independent training observations, the error function (negative log-likelihood) is:

$$E(\boldsymbol{\theta}) := -\sum_{n=1}^{N} y^{(n)} \ln h_{\boldsymbol{\theta}}(\mathbf{x}^{(n)}) + (1 - y^{(n)}) \ln(1 - h_{\boldsymbol{\theta}}(\mathbf{x}^{(n)}))$$

This is called **binary cross-entropy** or **log loss**.

---

### 2.3 Multiclass Classification

**Setup:**

- Each input assigned to one of $K$ mutually exclusive classes
- Binary target variables $y_k \in {0, 1}$ using 1-of-$K$ coding (one-hot encoding)
- Network outputs interpreted as: $h_{k,\boldsymbol{\theta}}(\mathbf{x}) = p(y_k = 1|\mathbf{x})$

**Error Function (Categorical Cross-Entropy):**

$$E(\boldsymbol{\theta}) := -\sum_{n=1}^{N} \sum_{k=1}^{K} y_k^{(n)} \ln h_{k,\boldsymbol{\theta}}(\mathbf{x}^{(n)})$$

**Output Layer Activation (Softmax):**

$$h_{k,\boldsymbol{\theta}}(\mathbf{x}) := \frac{\exp(z_k^{(n_l)})}{\sum_{j=1}^{K} \exp(z_j^{(n_l)})}$$

where $z_j^{(n_l)}$ is the total weighted sum of inputs to the $j$-th neuron of the last layer.

**Properties of Softmax:** $$\sum_k h_k = 1$$ $$0 \leq h_k \leq 1$$

These ensure outputs can be interpreted as probabilities!

---

### Summary: Matching Activations and Loss Functions

|Task|Output Activation|Loss Function|Why?|
|---|---|---|---|
|**Regression**|Linear (identity)|Sum-of-Squares (MSE)|MLE under Gaussian noise|
|**Binary Classification**|Sigmoid|Binary Cross-Entropy|MLE for Bernoulli distribution|
|**Multiclass Classification**|Softmax|Categorical Cross-Entropy|MLE for categorical distribution|

**Key Principle:** Natural choice of output activation and error function comes from the probabilistic interpretation of the problem!

---

## 3. Parameter Optimization

### The Error Surface

The error function $E(\boldsymbol{\theta})$ can be viewed as a **surface sitting over weight space**.

**Geometric Picture (Figure 5.2.1):**

- Horizontal axes: weight parameters ($w_1, w_2$)
- Vertical axis: error $E(\mathbf{w})$
- Surface is **nonlinear and non-convex** with multiple local minima

**Key Observations:**

- $\mathbf{w}_A$: a local minimum
- $\mathbf{w}_B$: the global minimum
- $\mathbf{w}_C$: current position, gradient $\nabla E$ points toward steepest ascent

**Challenge:** No analytical solution exists! We must use iterative optimization.

---

## 4. The Cost Function with Regularization

### Adding Weight Decay (L2 Regularization)

For regression, we minimize:

$$J(\boldsymbol{\theta}) := \frac{1}{N} \sum_{n=1}^{N} \frac{1}{2} ||h_{\boldsymbol{\theta}}(\mathbf{x}^{(n)}) - \mathbf{y}^{(n)}||^2 + \frac{\lambda}{2} \sum_{\ell=1}^{n_l-1} \sum_{i=1}^{s_\ell} \sum_{j=1}^{s_{\ell+1}} (W_{ji}^{(\ell)})^2$$

$$= \underbrace{\frac{1}{N} \sum_{n=1}^{N} \frac{1}{2} ||h_{\boldsymbol{\theta}}(\mathbf{x}^{(n)}) - \mathbf{y}^{(n)}||^2}_{E(\boldsymbol{\theta})} + \underbrace{\frac{\lambda}{2} \sum_{\ell,i,j} (W_{ji}^{(\ell)})^2}_{\text{L2 regularization}}$$

### Understanding Regularization

**First term:** Average sum-of-squares error (data fit)

**Second term:** $\ell_2$-regularization or **weight decay**

- Penalizes large weights
- Helps prevent overfitting
- Usually NOT applied to bias terms $b_i^{(\ell)}$

**Weight decay parameter $\lambda$:** Controls the trade-off between:

- Fitting the training data (small $\lambda$)
- Keeping weights small (large $\lambda$)

**Why "weight decay"?** During gradient descent, this term causes weights to shrink (decay) toward zero.

---

## 5. Gradient Descent Algorithm

### The Update Rule

Since $J(\boldsymbol{\theta})$ is non-convex, we use **(stochastic) gradient descent**.

**Initialization:**

- Initialize each parameter $W_{ij}^{(\ell)}$ and $b_i^{(\ell)}$ to small random values near zero
- Draw from $\mathcal{N}(0, \epsilon^2)$ for small $\epsilon$ (e.g., 0.01)

**Why random initialization?**

- If all parameters start at identical values, all hidden units in a layer will learn the same function
- This is called **symmetry breaking**
- Example: if $W_{ij}^{(1)}$ are all the same, then $a_1^{(2)} = a_2^{(2)} = a_3^{(2)} = \cdots$ for any input $\mathbf{x}$

**One iteration updates parameters:**

$$W_{ij}^{(\ell)} = W_{ij}^{(\ell)} - \eta \frac{\partial}{\partial W_{ij}^{(\ell)}} J(\boldsymbol{\theta})$$

$$b_i^{(\ell)} = b_i^{(\ell)} - \eta \frac{\partial}{\partial b_i^{(\ell)}} J(\boldsymbol{\theta})$$

where $\eta$ is the **learning rate**.

**The Challenge:** How do we compute the partial derivatives efficiently?

**The Answer:** The **backpropagation algorithm**!

---

## 6. The Backpropagation Algorithm

### Overview

**Backpropagation** is an efficient algorithm to compute partial derivatives of the cost function with respect to network parameters.

**Key Idea:** Use the chain rule to propagate error gradients backward through the network.

### 6.1 Single Training Example

For a single training example, define: $$J(\boldsymbol{\theta}; \mathbf{x}, \mathbf{y}) := \frac{1}{2}||h_{\boldsymbol{\theta}}(\mathbf{x}) - \mathbf{y}||_2^2$$

We need to compute: $$\frac{\partial}{\partial W_{ij}^{(\ell)}} J(\boldsymbol{\theta}; \mathbf{x}, \mathbf{y}) \quad \text{and} \quad \frac{\partial}{\partial b_i^{(\ell)}} J(\boldsymbol{\theta}; \mathbf{x}, \mathbf{y})$$

Once we can compute these for a single example, the overall gradient is:

$$\frac{\partial}{\partial W_{ij}^{(\ell)}} J(\boldsymbol{\theta}) = \frac{1}{N} \sum_{n=1}^{N} \frac{\partial}{\partial W_{ij}^{(\ell)}} J(\boldsymbol{\theta}; \mathbf{x}^{(n)}, \mathbf{y}^{(n)}) + \lambda W_{ij}^{(\ell)}$$

$$\frac{\partial}{\partial b_i^{(\ell)}} J(\boldsymbol{\theta}) = \frac{1}{N} \sum_{n=1}^{N} \frac{\partial}{\partial b_i^{(\ell)}} J(\boldsymbol{\theta}; \mathbf{x}^{(n)}, \mathbf{y}^{(n)})$$

---

### 6.2 The Error Term $\delta$

**Intuition:** The backpropagation algorithm computes an **error term** $\delta_i^{(\ell)}$ for each node $i$ in layer $\ell$ that measures how much that node was _responsible_ for errors in the output.

**For output nodes:** Directly measure difference between network's activation and true target.

**For hidden nodes:** Compute based on weighted average of error terms of nodes that use $a_i^{(\ell)}$ as input.

### 6.3 Backpropagation Algorithm (Scalar Notation)

Given a training datum $(\mathbf{x}, \mathbf{y})$:

**Step 1: Forward Pass** Compute activations for layers $L_2, \ldots, L_{n_l}$

**Step 2: Output Layer Error** For each output unit $i$ in layer $n_l$ (output layer): $$\delta_i^{(n_l)} = \frac{\partial}{\partial z_i^{(n_l)}} \frac{1}{2} ||\mathbf{y} - h_{\boldsymbol{\theta}}(\mathbf{x})||_2^2 = -(y_i - a_i^{(n_l)}) f'(z_i^{(n_l)})$$

**Step 3: Backpropagate Error** For $\ell = n_l - 1, n_l - 2, \ldots, 2$:

- For each node $i$ in layer $\ell$: $$\delta_i^{(\ell)} = \left(\sum_{j=1}^{s_{\ell+1}} W_{ji}^{(\ell)} \delta_j^{(\ell+1)}\right) f'(z_i^{(\ell)})$$

**Step 4: Compute Partial Derivatives** $$\frac{\partial}{\partial W_{ij}^{(\ell)}} J(\boldsymbol{\theta}; \mathbf{x}, \mathbf{y}) = a_j^{(\ell)} \delta_i^{\ell+1}$$

$$\frac{\partial}{\partial b_i^{(\ell)}} J(\boldsymbol{\theta}; \mathbf{x}, \mathbf{y}) = \delta_i^{\ell+1}$$

---

### 6.4 Backpropagation Algorithm (Matrix-Vector Notation)

**Step 1: Forward Pass** Compute activations for layers $L_2, \ldots, L_{n_l}$

**Step 2: Output Layer Error** $$\boldsymbol{\delta}^{(n_l)} = -(\mathbf{y} - \mathbf{a}^{n_l}) \odot f'(\mathbf{z}^{n_l})$$

**Step 3: Backpropagate Error** For $\ell = n_l - 1, n_l - 2, \ldots, 2$:

- For each node $i$ in layer $\ell$: $$\boldsymbol{\delta}^{(\ell)} = \left((\mathbf{W}^{(\ell)})^T \cdot \boldsymbol{\delta}^{(\ell+1)}\right) \odot f'(\mathbf{z}^{(\ell)})$$

**Step 4: Compute Partial Gradients** $$\nabla_{\mathbf{W}^{(\ell)}} J(\boldsymbol{\theta}; \mathbf{x}, \mathbf{y}) = \boldsymbol{\delta}^{(\ell+1)} \cdot (\mathbf{a}^{(\ell)})^T$$

$$\nabla_{\mathbf{b}^{(\ell)}} J(\boldsymbol{\theta}; \mathbf{x}, \mathbf{y}) = \boldsymbol{\delta}^{(\ell+1)}$$

where $\odot$ denotes element-wise (Hadamard) multiplication: $\mathbf{a} = \mathbf{b} \odot \mathbf{c}$ means $a_i = b_i \cdot c_i$.

---

## 7. Complete Training Algorithm

### Batch Gradient Descent

**Initialize:**

1. Randomly initialize parameters $\mathbf{W}^{(\ell)}$ and $\mathbf{b}^{(\ell)}$ for all $\ell$

**While stopping condition not met:**

2. Set $\Delta \mathbf{W}^{(\ell)} = \mathbf{0}$ and $\Delta \mathbf{b}^{(\ell)} = \mathbf{0}$ for all $\ell$
    
3. **For** $n = 1, \ldots, N$ **do:**
    
    - Use backpropagation to compute $\nabla_{\mathbf{W}^{(\ell)}} J(\boldsymbol{\theta}; \mathbf{x}^{(n)}, \mathbf{y}^{(n)})$ and $\nabla_{\mathbf{b}^{(\ell)}} J(\boldsymbol{\theta}; \mathbf{x}^{(n)}, \mathbf{y}^{(n)})$
    - Set $\Delta \mathbf{W}^{(\ell)} := \Delta \mathbf{W}^{(\ell)} + \nabla_{\mathbf{W}^{(\ell)}} J(\boldsymbol{\theta}; \mathbf{x}^{(n)}, \mathbf{y}^{(n)})$
    - Set $\Delta \mathbf{b}^{(\ell)} := \Delta \mathbf{b}^{(\ell)} + \nabla_{\mathbf{b}^{(\ell)}} J(\boldsymbol{\theta}; \mathbf{x}^{(n)}, \mathbf{y}^{(n)})$
4. **Update parameters:**
    
    - $\mathbf{W}^{(\ell)} \leftarrow \mathbf{W}^{(\ell)} - \eta\left[\frac{1}{N}\Delta \mathbf{W}^{(\ell)} + \lambda \mathbf{W}^{(\ell)}\right]$
    - $\mathbf{b}^{(\ell)} \leftarrow \mathbf{b}^{(\ell)} - \eta\left[\frac{1}{N}\Delta \mathbf{b}^{(\ell)}\right]$

**Note:** This is **batch** gradient descent—parameters updated once after computing gradient over whole training dataset.

---

### Stochastic Gradient Descent (SGD)

**Modification:** Update parameters immediately after computing gradient for each individual training datum.

**Advantages:**

- Faster convergence (especially for large datasets)
- Can escape shallow local minima
- Computationally efficient for big data

**In Practice:** Most deep learning uses variants of SGD with mini-batches (small subsets of data).

**The notes mention:** "There are a lot of studies investigating the convergence behaviour of these two flavours of the gradient descent algorithm, leaning towards the superiority of the stochastic gradient descent (particularly for big data)."

---

## 8. Understanding Backpropagation: The Math

### To Understand Backpropagation, You Need:

#### 8.1 The Chain Rule

**Single variable:** If $y = f(u)$ and $u = g(x)$, then: $$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

**Multiple variables:** If $z = f(x, y)$, $x = g(t)$, and $y = h(t)$, then: $$\frac{dz}{dt} = \frac{\partial z}{\partial x} \frac{dx}{dt} + \frac{\partial z}{\partial y} \frac{dy}{dt}$$

#### 8.2 Understanding the Error Term Formula

For hidden layer $\ell$: $$\delta_i^{(\ell)} = \left(\sum_{j=1}^{s_{\ell+1}} W_{ji}^{(\ell)} \delta_j^{(\ell+1)}\right) f'(z_i^{(\ell)})$$

**Intuition:**

1. $\delta_j^{(\ell+1)}$: error in layer $\ell+1$
2. $W_{ji}^{(\ell)}$: how much unit $i$ in layer $\ell$ contributes to unit $j$ in layer $\ell+1$
3. Sum: total error contribution from unit $i$ to all units in next layer
4. $f'(z_i^{(\ell)})$: how sensitive the activation is to changes in weighted input

**Why it works:** This follows directly from applying the chain rule to compute $\frac{\partial J}{\partial z_i^{(\ell)}}$.

#### 8.3 Why Gradient = Activation × Error

$$\frac{\partial}{\partial W_{ij}^{(\ell)}} J = a_j^{(\ell)} \delta_i^{\ell+1}$$

**Derivation:** $$z_i^{(\ell+1)} = \sum_j W_{ij}^{(\ell)} a_j^{(\ell)} + b_i^{(\ell)}$$

Taking partial derivative: $$\frac{\partial z_i^{(\ell+1)}}{\partial W_{ij}^{(\ell)}} = a_j^{(\ell)}$$

By chain rule: $$\frac{\partial J}{\partial W_{ij}^{(\ell)}} = \frac{\partial J}{\partial z_i^{(\ell+1)}} \cdot \frac{\partial z_i^{(\ell+1)}}{\partial W_{ij}^{(\ell)}} = \delta_i^{(\ell+1)} \cdot a_j^{(\ell)}$$

**Intuition:** The gradient depends on:

- How active the source neuron was ($a_j^{(\ell)}$)
- How much error the destination neuron has ($\delta_i^{(\ell+1)}$)

---

## 9. Key Insights and Intuitions

### Why "Backpropagation"?

1. **Forward pass:** Compute activations from input → output
2. **Backward pass:** Compute error gradients from output → input

The error is "propagated back" through the network layer by layer.

### Computational Efficiency

**Naive approach:** Computing gradients by finite differences would require:

- Forward pass for each parameter
- If network has 1,000,000 parameters → 1,000,000 forward passes!

**Backpropagation:** Computes all gradients with:

- 1 forward pass
- 1 backward pass

**Time complexity:** $O(\text{number of edges in network})$ for both passes.

This is why backpropagation was revolutionary!

### The Role of Activation Function Derivatives

Recall from earlier: $$\frac{\partial}{\partial z}\sigma(z) = \sigma(z)(1 - \sigma(z))$$ $$\frac{\partial}{\partial z}\tanh(z) = 1 - (\tanh(z))^2$$

These appear in the error term computation. They determine:

- How much the neuron's output changes with its input
- How quickly error gradients propagate through the network

**Vanishing gradient problem:** For sigmoid/tanh, derivatives approach 0 at extremes → gradients vanish in deep networks. This is why ReLU is often preferred.

---

## 10. Practical Considerations

### Hyperparameters to Tune

1. **Learning rate ($\eta$):**
    
    - Too small: slow convergence
    - Too large: overshooting, instability, divergence
    - Typical values: 0.001 - 0.1
2. **Regularization parameter ($\lambda$):**
    
    - Controls overfitting
    - Typical values: 0.0001 - 0.1
3. **Network architecture:**
    
    - Number of hidden layers
    - Number of units per layer
4. **Initialization scale ($\epsilon$):**
    
    - Too small: slow learning
    - Too large: instability
    - Typical: 0.01

### Convergence and Stopping Criteria

**Monitor:**

- Training error (should decrease)
- Validation error (for early stopping)

**Stop when:**

- Error change falls below threshold
- Maximum iterations reached
- Validation error starts increasing (overfitting)

### Batch vs Stochastic vs Mini-batch

|Method|Update Frequency|Memory|Convergence|
|---|---|---|---|
|**Batch GD**|After full dataset|High|Smooth, slow|
|**Stochastic GD**|After each example|Low|Noisy, fast|
|**Mini-batch GD**|After small batch|Medium|Best trade-off|

**Modern practice:** Mini-batch (e.g., 32-256 examples) with GPUs for parallel computation.

---

## Summary: Key Takeaways

1. **Loss functions** should match the probabilistic interpretation of the task:
    
    - Regression → MSE (Gaussian likelihood)
    - Binary classification → Binary cross-entropy (Bernoulli likelihood)
    - Multiclass → Categorical cross-entropy (Categorical likelihood)
2. **Regularization** (weight decay) prevents overfitting by penalizing large weights
    
3. **Random initialization** is crucial for symmetry breaking
    
4. **Backpropagation** efficiently computes gradients using:
    
    - Forward pass: compute activations
    - Backward pass: compute error terms via chain rule
    - Gradients: error × activation
5. **Error terms** ($\delta$) represent how much each neuron is responsible for output errors
    
6. **Gradient descent** iteratively updates parameters in direction of steepest decrease
    
7. **Stochastic GD** is often superior to batch GD for large datasets
    
8. **The algorithm** is:
    
    - Initialize randomly
    - Repeat: forward pass → compute loss → backpropagation → update weights
    - Until convergence

This is the foundation of training all neural networks! 🎯