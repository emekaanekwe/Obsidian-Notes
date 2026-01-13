
---

# ✅ **Example 1 — Softmax Probability Computation (Numerical Example)**

### **Objective:**

Convert logits to a valid probability distribution.

---

### **Requires knowledge of:**

- **Exponential function**  
    $$e^x = \sum_{n=0}^{\infty}\frac{x^n}{n!}$$
    
- **Softmax formula**  
    $$  
    \text{softmax}(z_k)=\frac{e^{z_k}}{\sum_{j=1}^{K}e^{z_j}}  
    $$
    
- **Logits**  
    Unnormalized scores before softmax.
    

---

### **Numerical Example**

Logits:

- $z_1 = 1$,
    
- $z_2 = 0$,
    
- $z_3 = -1$
    

**Step 1: Compute exponentials**

- $e^1 = 2.718$
    
- $e^0 = 1$
    
- $e^{-1} \approx 0.3679$
    

**Step 2: Compute denominator**  
$$  
Z = e^1 + e^0 + e^{-1}  
= 2.718 + 1 + 0.3679  
= 4.0859  
$$

**Step 3: Compute probabilities**

$$  
p_1 = \frac{2.718}{4.0859} \approx 0.665  
$$

$$  
p_2 = \frac{1}{4.0859} \approx 0.245  
$$

$$  
p_3 = \frac{0.3679}{4.0859} \approx 0.090  
$$

✔ This forms a correct probability distribution:  
$0.665 + 0.245 + 0.090 = 1$

---

# ✅ **Example 2 — Gradient Descent on a Quadratic Function**

### **Objective:**

Perform one update step of gradient descent.

---

### **Requires knowledge of:**

- **Derivative rule**  
    $$  
    \frac{d}{dw}(w - a)^2 = 2(w - a)  
    $$
    
- **Gradient descent update rule**  
    $$  
    w_{\text{new}} = w_{\text{old}} - \eta ,\frac{dE}{dw}  
    $$
    

---

### **Numerical Example**

Let the loss be:  
$$  
E(w) = (w - 4)^2  
$$

- Initial weight: $w_{\text{old}} = 10$
    
- Learning rate: $\eta = 0.05$
    

**Step 1: Compute gradient**

$$  
\frac{dE}{dw} = 2(w - 4)  
$$

At $w = 10$:  
$$  
\frac{dE}{dw} = 2(10 - 4) = 12  
$$

**Step 2: Gradient descent update**

$$  
w_{\text{new}} = 10 - 0.05 \cdot 12  
= 10 - 0.6  
= 9.4  
$$

✔ After one step, the weight moves closer to the optimum $w=4$.

---

# ✅ **Example 3 — EM Algorithm Soft Assignment (Document Clustering)**

### **Objective:**

Compute soft cluster memberships (responsibilities) for a document.

---

### **Requires knowledge of:**

- **Categorical mixture model likelihood**  
    $$  
    p(d_i \mid z=k) = \prod_{w=1}^V \mu_{wk}^{n_{iw}}  
    $$
    
- **Log transformation**  
    $$  
    \ln p(d_i \mid z=k) = \sum_{w=1}^V n_{iw} \ln \mu_{wk}  
    $$
    
- **Soft-EM responsibility**  
    $$  
    \gamma_{ik} = \frac{\pi_k,p(d_i \mid z=k)}{\sum_{j=1}^K \pi_j,p(d_i \mid z=j)}  
    $$
    

---

### **Numerical Example**

Two clusters ($K=2$):

Priors:

- $\pi_1 = 0.6$
    
- $\pi_2 = 0.4$
    

Document word counts:

- Word A: 3
    
- Word B: 1
    

Cluster word distributions:

- $\mu_{A1} = 0.7$, $\mu_{B1} = 0.3$
    
- $\mu_{A2} = 0.4$, $\mu_{B2} = 0.6$
    

---

### **Step 1: Likelihood under cluster 1**

$$  
p(d\mid 1) = 0.7^3 \cdot 0.3^1  
= (0.343) \cdot 0.3  
= 0.1029  
$$

### **Step 2: Likelihood under cluster 2**

$$  
p(d\mid 2) = 0.4^3 \cdot 0.6^1  
= 0.064 \cdot 0.6  
= 0.0384  
$$

### **Step 3: Multiply by priors**

- Cluster 1:  
    $0.6 \cdot 0.1029 = 0.06174$
    
- Cluster 2:  
    $0.4 \cdot 0.0384 = 0.01536$
    

### **Step 4: Normalize**

Denominator:  
$$  
0.06174 + 0.01536 = 0.0771  
$$

Responsibilities:  
$$  
\gamma_{i1} = \frac{0.06174}{0.0771} \approx 0.801  
$$

$$  
\gamma_{i2} = \frac{0.01536}{0.0771} \approx 0.199  
$$

✔ The document belongs **80% to cluster 1** and **20% to cluster 2**.

---

# ✅ **Example 4 — Perceptron Update Step (Binary Classification)**

### **Objective:**

Update perceptron weights after a misclassification.

---

### **Requires knowledge of:**

- **Perceptron prediction rule**  
    $$  
    \hat{y} = \text{sign}(w^\top x)  
    $$
    
- **Update rule**  
    $$  
    w_{\text{new}} = w_{\text{old}} + \eta(y - \hat{y})x  
    $$
    
- **Labels must be in ${-1, +1}$**
    

---

### **Numerical Example**

Weights:  
$$  
w = [1,\ -2]  
$$

Learning rate:  
$$  
\eta = 1  
$$

Training example:

- Input: $x = [2,\ 1]$
    
- True label: $y = +1$
    

**Step 1: Compute prediction**

$$  
w^\top x = 1\cdot 2 + (-2)\cdot 1 = 2 - 2 = 0  
$$

Convention: $\text{sign}(0) = -1$

So:  
$$  
\hat{y} = -1  
$$

**Step 2: Update weights** (misclassification)

$$  
w_{\text{new}} = w + \eta(y - \hat{y})x  
$$

Since $(y - \hat{y}) = 2$:

$$  
w_{\text{new}} = [1, -2] + 2[2, 1]  
= [1 + 4,\ -2 + 2]  
= [5,\ 0]  
$$

✔ Weight vector shifts to classify the example correctly.

---

# ✅ **Example 5 — Autoencoder Reconstruction Error**

### **Objective:**

Compute Euclidean reconstruction error for one example.

---

### **Requires knowledge of:**

- **Euclidean distance**  
    $$  
    |x - \hat{x}|_2 = \sqrt{\sum_i (x_i - \hat{x}_i)^2}  
    $$
    

---

### **Numerical Example**

Original vector:  
$$  
x = [1,\ 0,\ 1]  
$$

Reconstruction:  
$$  
\hat{x} = [0.8,\ 0.1,\ 0.6]  
$$

Compute squared errors:

- $(1 - 0.8)^2 = 0.04$
    
- $(0 - 0.1)^2 = 0.01$
    
- $(1 - 0.6)^2 = 0.16$
    

Sum:  
$$  
0.04 + 0.01 + 0.16 = 0.21  
$$

Reconstruction error:  
$$  
\sqrt{0.21} \approx 0.458  
$$

---

# ✅ **Example 6 — PCA Projection Onto First Principal Component**

### **Objective:**

Project a data point onto the first principal component.

---

### **Requires knowledge of:**

- **Projection formula**  
    $$  
    \text{proj}_{u}(x) = (x^\top u),u  
    $$
    
- **Unit vectors**  
    $$  
    |u| = 1  
    $$
    

---

### **Numerical Example**

Data point:  
$$  
x = [3,\ 4]  
$$

First principal component (unit vector):  
$$  
u = \left[\frac{1}{\sqrt{2}},\ \frac{1}{\sqrt{2}}\right]  
$$

**Step 1: Dot product**

$$  
x^\top u = 3\left(\frac{1}{\sqrt{2}}\right) +  
4\left(\frac{1}{\sqrt{2}}\right)  
= \frac{7}{\sqrt{2}}  
$$

**Step 2: Projection**

$$  
\text{proj}_u(x) = \frac{7}{\sqrt{2}} \cdot  
\left[\frac{1}{\sqrt{2}},\ \frac{1}{\sqrt{2}}\right]  
= \left[\frac{7}{2},\ \frac{7}{2}\right]  
$$

✔ PCA maps the point onto the main direction of variance.

Got it — I’ll rewrite from **Example 7** onward with:

- Correct MathJax formatting (`$...$` and `$$...$$` only, no weird brackets)
    
- The exact structure you asked for:
    
    - **objective**
        
    - **numerical example**
        
    - **solution & interpretation**
        
    - **requires knowledge of** (with proper formulas)
        

I’ll also make the autoencoder example clearer.

---

## Example 7 — Autoencoder Reconstruction Error (Euclidean Distance)

### objective

Compute how well an autoencoder reconstructs an input by using the **squared Euclidean reconstruction error**:

$$  
|x - \hat{x}|_2^2 = \sum_i (x_i - \hat{x}_i)^2  
$$

This measures how far the reconstruction $\hat{x}$ is from the original input $x$.

---

### numerical example

Suppose we have a **3-dimensional input** and its reconstruction:

- Original input:  
    $$  
    x = (1.0,; 2.0,; 3.0)  
    $$
    
- Autoencoder reconstruction:  
    $$  
    \hat{x} = (0.9,; 2.2,; 2.5)  
    $$
    

Compute the **squared reconstruction error** $|x - \hat{x}|_2^2$.

---

### solution & interpretation

1. **Compute the difference vector** $x - \hat{x}$:
    
    $$  
    x - \hat{x}  
    = (1.0 - 0.9,; 2.0 - 2.2,; 3.0 - 2.5)  
    = (0.1,; -0.2,; 0.5)  
    $$
    
2. **Square each component**:
    
    $$  
    (0.1)^2 = 0.01,\quad  
    (-0.2)^2 = 0.04,\quad  
    (0.5)^2 = 0.25  
    $$
    
3. **Sum the squared differences**:
    
    $$  
    |x - \hat{x}|_2^2  
    = 0.01 + 0.04 + 0.25  
    = 0.30  
    $$
    

So the **squared reconstruction error** is:

$$  
|x - \hat{x}|_2^2 = 0.30  
$$

**Interpretation:**

- A **smaller** value (closer to $0$) means the autoencoder is reconstructing $x$ **more accurately**.
    
- A **larger** value means the reconstruction is **less faithful**, so the autoencoder hasn’t captured the input structure well.
    
- In practice, you average this over many samples to get a mean reconstruction error and compare different architectures or hidden sizes.
    

---

### requires knowledge of

- **Euclidean norm (L2 norm)**  
    The Euclidean norm of a vector $v = (v_1,\dots,v_d)$ is:
    
    $$  
    |v|_2 = \sqrt{\sum_{i=1}^d v_i^2}  
    $$
    
- **Squared Euclidean distance between two vectors**  
    For vectors $x$ and $\hat{x}$:
    
    $$  
    |x - \hat{x}|_2^2 = \sum_{i=1}^d (x_i - \hat{x}_i)^2  
    $$
    
- **Coordinate-wise subtraction and squaring**  
    Given $x = (x_1,\dots,x_d)$ and $\hat{x} = (\hat{x}_1,\dots,\hat{x}_d)$, the difference is:
    
    $$  
    x - \hat{x} = (x_1 - \hat{x}_1,\dots,x_d - \hat{x}_d)  
    $$
    

---

## Example 8 — Autoencoder with Different Hidden Sizes (Effect on Reconstruction Error)

### objective

Understand how changing the **hidden layer size** of an autoencoder affects the **average reconstruction error** on a dataset.

We compare two autoencoders:

- Autoencoder A: hidden size = 10
    
- Autoencoder B: hidden size = 50
    

We compute the **mean squared reconstruction error** on the same dataset:

$$  
\text{MSE} = \frac{1}{N} \sum_{n=1}^N |x^{(n)} - \hat{x}^{(n)}|_2^2  
$$

---

### numerical example

Assume we have **3 data points** (1D for simplicity) and two autoencoders:

- Data:  
    $$  
    x^{(1)} = 1,\quad x^{(2)} = 2,\quad x^{(3)} = 3  
    $$
    
- Autoencoder A (small hidden layer) reconstructions:  
    $$  
    \hat{x}_A^{(1)} = 0.5,\quad \hat{x}_A^{(2)} = 1.5,\quad \hat{x}_A^{(3)} = 2.0  
    $$
    
- Autoencoder B (larger hidden layer) reconstructions:  
    $$  
    \hat{x}_B^{(1)} = 0.9,\quad \hat{x}_B^{(2)} = 2.1,\quad \hat{x}_B^{(3)} = 2.9  
    $$
    

Compute the **mean squared reconstruction error** for each autoencoder and compare.

---

### solution & interpretation

#### Autoencoder A

For each data point:

1. $n=1$:
    
    $$  
    (x^{(1)} - \hat{x}_A^{(1)})^2 = (1 - 0.5)^2 = 0.5^2 = 0.25  
    $$
    
2. $n=2$:
    
    $$  
    (x^{(2)} - \hat{x}_A^{(2)})^2 = (2 - 1.5)^2 = 0.5^2 = 0.25  
    $$
    
3. $n=3$:
    
    $$  
    (x^{(3)} - \hat{x}_A^{(3)})^2 = (3 - 2.0)^2 = 1.0^2 = 1.0  
    $$
    

Now compute the mean:

$$  
\text{MSE}_A = \frac{1}{3}(0.25 + 0.25 + 1.0) = \frac{1.5}{3} = 0.5  
$$

#### Autoencoder B

1. $n=1$:
    
    $$  
    (x^{(1)} - \hat{x}_B^{(1)})^2 = (1 - 0.9)^2 = 0.1^2 = 0.01  
    $$
    
2. $n=2$:
    
    $$  
    (x^{(2)} - \hat{x}_B^{(2)})^2 = (2 - 2.1)^2 = (-0.1)^2 = 0.01  
    $$
    
3. $n=3$:
    
    $$  
    (x^{(3)} - \hat{x}_B^{(3)})^2 = (3 - 2.9)^2 = 0.1^2 = 0.01  
    $$
    

Now compute the mean:

$$  
\text{MSE}_B = \frac{1}{3}(0.01 + 0.01 + 0.01) = \frac{0.03}{3} = 0.01  
$$

#### Interpretation

- Autoencoder A: $\text{MSE}_A = 0.5$ (worse reconstruction)
    
- Autoencoder B: $\text{MSE}_B = 0.01$ (better reconstruction)
    

A larger hidden layer (more capacity) allowed the model to better capture the structure of the data and **reduce reconstruction error**.  
In practice, too large a hidden layer can cause **overfitting**, but this simple example shows the reconstruction-error trend.

---

### requires knowledge of

- **Mean Squared Error (MSE)**
    
    For predictions $\hat{y}^{(n)}$ and targets $y^{(n)}$:
    
    $$  
    \text{MSE} = \frac{1}{N} \sum_{n=1}^N (y^{(n)} - \hat{y}^{(n)})^2  
    $$
    
- **Squared reconstruction error for vectors**
    
    For each sample:
    
    $$  
    |x^{(n)} - \hat{x}^{(n)}|_2^2 = \sum_{i=1}^d (x^{(n)}_i - \hat{x}^{(n)}_i)^2  
    $$
    
- **Averaging over samples**
    
    Given values $a_1,\dots,a_N$, the average is:
    
    $$  
    \frac{1}{N}\sum_{n=1}^N a_n  
    $$
    

---

## Example 9 — 3-Layer Neural Network Classification Error

### objective

Compute the **classification error** of a simple 3-layer neural network on a small test set.

We define classification error as:

$$  
\text{Classification Error} = \frac{\text{Number of misclassified samples}}{\text{Total number of samples}}  
$$

---

### numerical example

Suppose we have a 3-layer neural network (input → hidden → softmax output) and a tiny test set of **5** samples.

- True labels:  
    $$  
    y = [0,; 1,; 1,; 2,; 0]  
    $$
    
- Network predictions (argmax of softmax outputs):  
    $$  
    \hat{y} = [0,; 2,; 1,; 2,; 1]  
    $$
    

Compute the classification error on this test set.

---

### solution & interpretation

1. Compare each prediction with the true label:
    
    - Sample 1: $y_1 = 0$, $\hat{y}_1 = 0$ → correct
        
    - Sample 2: $y_2 = 1$, $\hat{y}_2 = 2$ → incorrect
        
    - Sample 3: $y_3 = 1$, $\hat{y}_3 = 1$ → correct
        
    - Sample 4: $y_4 = 2$, $\hat{y}_4 = 2$ → correct
        
    - Sample 5: $y_5 = 0$, $\hat{y}_5 = 1$ → incorrect
        
2. Count misclassified samples:
    
    - Misclassified: samples 2 and 5 → total $= 2$
        
    - Total samples: $5$
        
3. Compute classification error:
    
    $$  
    \text{Classification Error}  
    = \frac{2}{5} = 0.4  
    $$
    

So the classification error is $0.4$ (or $40%$).  
Equivalently, the accuracy is $1 - 0.4 = 0.6$ (or $60%$).

**Interpretation:**

- On this tiny test set, the model gets 3/5 correct.
    
- High error suggests more training, better hyperparameters, or different model architecture might be needed.
    

---

### requires knowledge of

- **Indicator of misclassification**
    
    For predicted label $\hat{y}^{(n)}$ and true label $y^{(n)}$, the misclassification indicator is:
    
    $$  
    \mathbb{1}[\hat{y}^{(n)} \neq y^{(n)}] =  
    \begin{cases}  
    1, & \text{if } \hat{y}^{(n)} \neq y^{(n)} \  
    0, & \text{if } \hat{y}^{(n)} = y^{(n)}  
    \end{cases}  
    $$
    
- **Classification error as an average**
    
    Over $N$ samples:
    
    $$  
    \text{Classification Error}  
    = \frac{1}{N} \sum_{n=1}^N \mathbb{1}[\hat{y}^{(n)} \neq y^{(n)}]  
    $$
    
- **Basic fraction to percentage conversion**
    
    For error $e$:
    
    $$  
    \text{Error (%)} = 100 \times e  
    $$
    

---

## Example 10 — Self-Taught Learning: Augmented Feature Dimension

### objective

Understand how **self-taught learning** augments the feature space by concatenating:

- Original features: $x \in \mathbb{R}^d$
    
- Autoencoder hidden features: $h \in \mathbb{R}^k$
    

The augmented feature vector has dimension:

$$  
\tilde{x} = [x;, h] \in \mathbb{R}^{d + k}  
$$

---

### numerical example

Suppose:

- Original input features: $x \in \mathbb{R}^{784}$ (e.g., flattened $28 \times 28$ image)
    
- Autoencoder’s hidden layer: $k = 100$ units
    

We build the augmented feature vector by concatenating $x$ and $h$.

What is the dimension of the augmented feature vector $\tilde{x}$?  
And what does it look like conceptually?

---

### solution & interpretation

1. Original feature dimension:
    
    $$  
    x \in \mathbb{R}^{784}  
    $$
    
2. Hidden representation dimension:
    
    $$  
    h \in \mathbb{R}^{100}  
    $$
    
3. Concatenate:
    
    $$  
    \tilde{x} = [x;, h] \in \mathbb{R}^{784 + 100} = \mathbb{R}^{884}  
    $$
    

So the augmented feature vector has dimension $884$.

**Interpretation:**

- The classifier now sees **both**:
    
    - The raw pixel inputs ($784$-dim vector).
        
    - Learned high-level features from the autoencoder ($100$-dim vector).
        
- This often improves performance if the autoencoder learns useful structure in the data (e.g., edges, strokes, shapes).
    

---

### requires knowledge of

- **Vector concatenation**
    
    If $x \in \mathbb{R}^d$ and $h \in \mathbb{R}^k$, then the concatenated vector is:
    
    $$  
    \tilde{x} =  
    \begin{bmatrix}  
    x \  
    h  
    \end{bmatrix}  
    \in \mathbb{R}^{d + k}  
    $$
    
- **Dimensionality of real vector spaces**
    
    A vector in $\mathbb{R}^n$ has $n$ components:
    
    $$  
    x = (x_1,\dots,x_n)^\top,\quad x_i \in \mathbb{R}  
    $$
    
- **Basic linear algebra notation**
    
    Knowing that:
    
    $$  
    x \in \mathbb{R}^d,\quad h \in \mathbb{R}^k \implies [x;,h] \in \mathbb{R}^{d+k}  
    $$
    

---
