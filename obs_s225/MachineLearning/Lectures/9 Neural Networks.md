### ML method of formal structure of neuron:

$$h_{W,b}=f(W^Tx)=f(\sum_{i=1}^{n}W_ix_i+b)$$
output of sigmoid is the inteval [0,1]
hyperbolic output is the interval [-1,1]
ReLU output is a non-neg minimized num from 0 to z

---

### **Analysis of Machine Learning Lecture: Fully Connected Neural Networks**

This lecture transitions from simpler models (like the Perceptron) to the powerful, non-linear world of Fully Connected Neural Networks (FCNNs), covering their architecture, components, training, and the crucial Backpropagation algorithm.

---

### **# 1. The Single Neuron (The Building Block)**

At the heart of any neural network is the artificial neuron. Its operation can be summarized by two steps:

1.  **Compute the weighted sum:** `Z = (W1 * X1) + (W2 * X2) + ... + (Wn * Xn) + B`
2.  **Apply an activation function:** `Output = f(Z)`

#### **## Prerequisite Math Knowledge & Refresher**

**### 1. Weighted Sum (Linear Combination)**
This is a simple linear algebra concept. Each input `Xi` is multiplied by a corresponding weight `Wi`, and a bias `B` is added. The bias allows the neuron to shift its activation function left or right, providing more flexibility.

**### 2. Activation Functions `f(Z)`**
These functions are the source of a neural network's ability to model non-linear relationships. They are applied element-wise to the output `Z` of a layer.

#### **## Common Activation Functions**

| Function | Formula (MathJax) | Range | Why it's used |
| :--- | :--- | :--- | :--- |
| **Sigmoid** | $f(Z) = \sigma(Z) = \frac{1}{1 + e^{-Z}}$ | (0, 1) | Squashes outputs to a probability range. Historically popular. |
| **Tanh** | $f(Z) = \tanh(Z) = \frac{e^{Z} - e^{-Z}}{e^{Z} + e^{-Z}}$ | (-1, 1) | Zero-centered, often performs better than sigmoid in hidden layers. |
| **ReLU** | $f(Z) = \max(0, Z)$ | [0, ∞) | Default choice for hidden layers. Computationally cheap and avoids vanishing gradient for positive Z. |

**Handwritten Example:**
Let's say for a single neuron: `X = [2, 3]`, `W = [0.5, -1.0]`, `B = 1`.
1.  `Z = (2 * 0.5) + (3 * -1.0) + 1 = 1 - 3 + 1 = -1`
2.  If `f` is **ReLU**: `Output = max(0, -1) = 0`
3.  If `f` is **Sigmoid**: `Output = 1 / (1 + e^(-(-1))) ≈ 1 / (1 + 2.718) ≈ 0.269`

---

### **# 2. Key Difference: Perceptron vs. Neural Network**

*   **Perceptron:** Uses a **step function** (binary output). It's a **linear classifier**. It can only solve linearly separable problems.
*   **Neural Network:** Uses **continuous, differentiable activation functions** (Sigmoid, Tanh, ReLU). It's a **non-linear classifier**. By stacking layers of these neurons, it can learn highly complex, non-linear decision boundaries.

**Universal Approximation Theorem:** The lecture hints at this powerful idea: a neural network with even a single hidden layer containing a sufficient number of neurons can approximate *any* continuous function to any desired precision. This is the theoretical foundation for their power.

---

### **# 3. Architecture of a Fully Connected Neural Network**

An FCNN is composed of layers:
1.  **Input Layer:** The layer that receives your feature vector. (Number of units = number of features).
2.  **Hidden Layer(s):** Layers between input and output where computation happens. A network with >1 hidden layer is considered "deep".
3.  **Output Layer:** The final layer, which produces the prediction. Its design is task-dependent.

**"Fully Connected"** means every neuron in layer `l` is connected to every neuron in layer `l+1`.

**Notation (Crucial for Backpropagation):**
*   $n_l$: Number of units in layer $l$.
*   $w^{l}_{ij}$: Weight connecting unit $j$ in layer $(l-1)$ to unit $i$ in layer $l$.
*   $b^{l}_i$: Bias for unit $i$ in layer $l$.
*   $z^{l}_i$: Weighted sum for unit $i$ in layer $l$ (before activation). $z^{l}_i = \sum_j w^{l}_{ij} a^{l-1}_j + b^{l}_i$.
*   $a^{l}_i$: Activation (output) of unit $i$ in layer $l$. $a^{l}_i = f(z^{l}_i)$.

---

### **# 4. Output Layers, Activation Functions, and Loss Functions**

The choice of output activation and loss function is dictated by the problem type. This is a critical exam concept.

| Problem Type | Output Units | Output Activation | Loss Function | Intuition |
| :--- | :--- | :--- | :--- | :--- |
| **Regression** | # of target dimensions | **Identity:** $a = z$ | **Mean Squared Error (MSE):** $\frac{1}{N}\sum (y - \hat{y})^2$ | Directly predicts a continuous value. Penalizes large errors heavily. |
| **Binary Classification** | 1 | **Sigmoid:** $a = \sigma(z)$ | **Binary Cross-Entropy:** $-[y \log(a) + (1-y)\log(1-a)]$ | Treats output `a` as probability of Class 1. Maximizes log-probability of the correct class. |
| **Multi-Class Classification (C classes)** | C | **Softmax:** $a_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$ | **Categorical Cross-Entropy:** $-\sum_{i=1}^{C} y_i \log(a_i)$ | Squashes outputs to a probability distribution over `C` classes. |

#### **## The Softmax Function and Cross-Entropy Derivation**

**# The Mathematical Formula**
For a network output vector $\mathbf{z} = [z_1, z_2, ..., z_C]$, the Softmax function is:
$$a_i = \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}$$

The Categorical Cross-Entropy loss for a single example with true label (one-hot encoded) $\mathbf{y}$ and prediction $\mathbf{a}$ is:
$$J = -\sum_{i=1}^{C} y_i \log(a_i)$$

**## Prerequisite Math Knowledge & Refresher**

**### 1. One-Hot Encoding**
A way to represent class labels as binary vectors. If an example belongs to class $k$, then $y_k = 1$ and all other $y_j = 0$.
*Example:* For 3 classes, class 2 is represented as $\mathbf{y} = [0, 1, 0]$.

**### 2. Softmax Function**
It exponentiates each element (to make them positive) and then normalizes by the sum of all exponentiated elements. This converts the raw logits $\mathbf{z}$ into a probability distribution $\mathbf{a}$ where all $a_i$ sum to 1.

**### 3. Cross-Entropy**
Measures the "distance" between two probability distributions: the true distribution $\mathbf{y}$ and the predicted distribution $\mathbf{a}$.

**## Step-by-Step Intuition of the Loss**

Let's say we have $C=3$ classes, and our true class is $k=2$, so $\mathbf{y} = [0, 1, 0]$.
Our network's softmax output is $\mathbf{a} = [a_1, a_2, a_3]$.

1.  **Plug into the loss formula:**
    $J = -[y_1 \log(a_1) + y_2 \log(a_2) + y_3 \log(a_3)]$
2.  **Substitute the one-hot values:**
    Since $y_1=0$ and $y_3=0$, those terms become zero.
    $J = -[0 \cdot \log(a_1) + 1 \cdot \log(a_2) + 0 \cdot \log(a_3)] = -\log(a_2)$
3.  **The Goal of Training:**
    To **minimize** $J$, we must **maximize** $\log(a_2)$, which means we must **maximize** $a_2$, the predicted probability for the correct class.

This is the core intuition: **Cross-Entropy loss directly rewards the model for putting high probability on the correct class.**

---

### **# 5. Training Neural Networks: Gradient Descent & Backpropagation**

The error function $E(\theta)$ for a neural network (where $\theta$ represents all weights and biases) is **non-convex** with many local minima. We use **Gradient Descent** to navigate this complex landscape.

**Parameter Update Rule:**
$$\theta_{\text{new}} = \theta_{\text{old}} - \eta \cdot \nabla E(\theta_{\text{old}})$$
Where $\eta$ is the **learning rate**.

**The Central Challenge:** How do we compute the gradient $\nabla E(\theta)$, i.e., the partial derivatives $\frac{\partial E}{\partial w}$ and $\frac{\partial E}{\partial b}$ for every parameter in the network? The answer is **Backpropagation**.

#### **## The Backpropagation Algorithm (Intuition & Steps)**

Backpropagation is an application of the **chain rule** from calculus to efficiently compute gradients from the output layer back to the input layer.

**### Prerequisite Math Knowledge: The Chain Rule**
If a variable `z` depends on `y`, and `y` depends on `x`, then `z` depends on `x` via:
$$\frac{dz}{dx} = \frac{dz}{dy} \cdot \frac{dy}{dx}$$
In the context of neural networks, the "error" flows backwards, and we chain together the derivatives at each layer.

**## The Two-Pass Algorithm**

1.  **Forward Pass:**
    *   **What:** Feed an input $\mathbf{x}$ through the network.
    *   **Compute:** All activations $a^l_i$ and weighted sums $z^l_i$ for every layer, all the way to the output, where the loss $J$ is calculated.
    *   **Why:** You need these intermediate values for the backward pass.

2.  **Backward Pass:**
    *   **What:** Propagate the error backwards through the network to compute the gradients.
    *   **Key Concept: Error Term $\delta$:** We define an "error term" for each neuron, which represents how much that neuron is "to blame" for the final loss.
        *   $\delta^l_i = \frac{\partial J}{\partial z^l_i}$

    **Step-by-Step:**
    a.  **Output Layer (l = L):** Compute the error for the output units. This is direct.
        *   *Example for MSE Loss:* $\delta^L_i = (a^L_i - y_i) \cdot f'(z^L_i)$
        The derivative of the activation function $f'$ is crucial here.

    b.  **Backpropagate the Error:** For any hidden layer `l`, the error is calculated from the layer ahead of it `l+1`.
        $$\delta^l = ( (\mathbf{W}^{l+1})^T \delta^{l+1} ) \odot f'(z^l)$$
        Where $\odot$ is the element-wise product (Hadamard product). This is the core of Backpropagation: the error from layer `l+1` is passed backwards through the weights `W` to compute the error for layer `l`.

    c.  **Compute the Gradients:** Once you have the error term $\delta^l_i$ for a layer, the gradients for its weights and biases are simple:
        $$\frac{\partial J}{\partial w^l_{ij}} = a^{l-1}_j \cdot \delta^l_i$$
        $$\frac{\partial J}{\partial b^l_i} = \delta^l_i$$

The lecture's final calculation for $\frac{\partial J}{\partial w^{L-1}_{ij}}$ is an example of applying this exact rule.

### **Key Exam Takeaways:**

1.  **Understand the Neuron:** Know how to compute the output of a single neuron given inputs, weights, a bias, and an activation function.
2.  **Architecture & Notation:** Be comfortable with the concepts of layers (input, hidden, output) and the notation ($w^{l}_{ij}$, $a^l_i$, $z^l_i$).
3.  **Problem-Dependent Outputs:** Memorize the table linking problem types (Regression, Binary/Multi-Class Classification) to their correct output activation and loss functions. Be able to explain *why* (e.g., why we use Cross-Entropy for classification).
4.  **Backpropagation Intuition:** You don't necessarily need to derive the entire algorithm from scratch for an exam, but you must understand the two-pass process (forward and backward) and the role of the error term $\delta$. Be able to compute a simple gradient given the $\delta$ from the next layer.