
- Every formula includes:  
    **(1) Explanation**  
    **(2) Required math prerequisites**  
    **(3) A tiny numerical example** _(when useful)_  
    *_(4) Intuition that is interview-ready_


---

# 📌 MASTER CHEAT SHEET FOR MACHINE LEARNING

# **1. Data, Weights, Biases, Parameters, Model Type**

### **Model Parameters**

Parameters are tunable values (learned from data) that define the model.
- In linear regression:  
 $\mathbf{w} = (w_1, \dots, w_d)$`, bias `$b$
    
- In neural networks:  
    Weights = connections, Biases = shifts before activation.
    

### **Mathematical Form**

$$  
y = \mathbf{w}^\top \mathbf{x} + b  
$$

### **Refresher: Vector Dot Product**

$$  
\mathbf{w}^\top \mathbf{x} = \sum_{i=1}^d w_i x_i  
$$

**Example:**  
If $\mathbf{w} = (2,3)$ and $\mathbf{x}=(1,4)$,  
$\mathbf{w}^\top\mathbf{x} = 2\cdot1 + 3\cdot4 = 14$.

---

# **2. Error Function (Loss) & Model Complexity**

### **Squared Error Loss**

$$  
E = \frac12 \sum_n (t_n - y_n)^2  
$$

**Why:** Comes from assuming Gaussian noise.

### **Model Complexity**

- Too simple → **underfit**
    
- Too flexible → **overfit**
    

---

# **3. Generalization, Overfitting, Underfitting**

### **Training vs Test Error**

- Training error ↓ as model becomes more complex
    
- Test error forms a **U-shape**
    

### **Bias–Variance Tradeoff**

$$  
\text{Test Error} = \text{Bias}^2 + \text{Variance} + \text{Noise}  
$$

---

# **4. Regularization & Validation**

### **L2 Regularization (Ridge)**

$$  
E(\mathbf{w}) = \frac12 \sum_n (t_n - y_n)^2 + \frac{\lambda}{2}|\mathbf{w}|^2  
$$

**Prerequisite:**  
Norm squared:  
$|\mathbf{w}|^2 = \sum_i w_i^2$

### **Why:** Reduces overfitting by shrinking weights.

---

# **5. Cross-Validation, k-fold, LOOCV**

### **k-Fold CV**

Split data into `$k$` folds, train `$k$` times, average error.

### **Leave-One-Out CV (LOOCV)**

$N$ folds with exactly 1 test point each.

**Why:** Nearly unbiased but computationally expensive.

---

# **6. Bootstrap**

Sampling _with_ replacement to estimate variability (variance of estimator).

---

# **7. Bayesian Concepts**

- Prior: `$p(\theta)$`
    
- Likelihood: `$p(D \mid \theta)$`
    
- Posterior:  
    $$  
    p(\theta \mid D) = \frac{p(D\mid\theta)p(\theta)}{p(D)}  
    $$
    

**Tiny Example:**  
If prior = 0.6, likelihood = 0.5, evidence = 0.5 → posterior = 0.6.

---

# **8. Variance & Covariance**

### **Variance**

$$  
\mathrm{Var}(X) = \mathbb{E}[(X - \mu)^2]  
$$

### **Covariance**

$$  
\mathrm{Cov}(X,Y) = \mathbb{E}[(X-\mu_X)(Y-\mu_Y)]  
$$

---

# **9. Basis Functions (Feature Engineering)**

Transform data:  
$$  
\phi(\mathbf{x}) = [1, x, x^2, \sin x, \dots]  
$$

Model becomes:  
$$  
y = \mathbf{w}^\top \phi(\mathbf{x})  
$$

Basis → increases model flexibility.

---

# **10. Linear Regression**

### **Normal Equation**

$$  
\mathbf{w} = (X^\top X)^{-1} X^\top \mathbf{t}  
$$

**Prerequisite:**  
Matrix inverse & multiplication.

---

# **11. Logistic Regression**

### **Sigmoid Function**

$$  
\sigma(z) = \frac{1}{1+e^{-z}}  
$$

### **Model**

$$  
p(t=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^\top\mathbf{x})  
$$

### **Binary Cross-Entropy Loss**

$$  
E = -\sum_{n=1}^N \big[ t_n \log y_n + (1-t_n)\log(1-y_n) \big]  
$$

---

# **12. Gradient & Gradient Descent**

### **Gradient Descent Update**

$$  
\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_{\mathbf{w}}E  
$$

Where η = learning rate.

**Example:**  
If `$w=3$`, gradient `$=2$`, η `$=0.1$`  
→ new `$w = 3 - 0.1\cdot2 = 2.8$`.

---

# **13. Perceptron**

### **Decision Function**

$$  
t = \text{sign}(\mathbf{w}^\top\mathbf{x})  
$$

### **Update Rule**

$$  
\mathbf{w} \leftarrow \mathbf{w} + \eta (t_n - y_n)\mathbf{x}_n  
$$

---

# **14. k-Means Clustering**

### **Objective Function**

$$  
J = \sum_{n=1}^N | x_n - \mu_{z_n} |^2  
$$

Alternates between:

- E-step: assign to nearest centroid
    
- M-step: recompute centroids
    

---

# **15. Mixture Models & Gaussian Mixture Models (GMM)**

### **Mixture Distribution**

$$  
p(x) = \sum_{k=1}^K \pi_k , \mathcal{N}(x \mid \mu_k, \Sigma_k)  
$$

---

# **16. Expectation–Maximization (EM) Algorithm**

### **E-step (Soft Assignments)**

$$  
\gamma_{nk} =  
\frac{  
\pi_k, p(x_n \mid \theta_k)  
}{  
\sum_{j=1}^K \pi_j, p(x_n \mid \theta_j)  
}  
$$

### **M-step**

Update parameters using responsibilities.

### **Hard EM**

Replace $\gamma_{nk}$ with 1 for cluster with max probability.

---

# **17. Complete & Incomplete Data Log-Likelihood**

### **Complete Data Log-Likelihood**

For mixture models:  
$$  
\ln p(X,Z\mid\theta)  
= \sum_{n=1}^N \sum_{k=1}^K  
z_{nk}\big( \ln\pi_k + \ln p(x_n\mid\theta_k) \big)  
$$

### **Q-Function**

Expectation under old parameters:  
$$  
Q(\theta,\theta^{old}) =  
\mathbb{E}_{Z \mid X,\theta^{old}}[\ln p(X,Z\mid\theta)]  
$$

---

# **18. Neural Networks**

## **Forward Propagation**

For hidden unit:  
$$  
h = \sigma(\mathbf{w}_h^\top \mathbf{x} + b_h)  
$$

Output:  
$$  
y = \mathbf{w}_o^\top \mathbf{h} + b_o  
$$

## **Backpropagation (Core Derivative)**

Example of chain rule for $w_1$:  
$$  
\frac{\partial E}{\partial w_1}  
= (y - t), w_{ho}, x_1  
$$

---

# **19. Hidden Units & Generalization**

- Too few → underfit
    
- Too many → overfit
    
- Regularization, early stopping, dropout prevent this
    

---

# **20. Autoencoders**

### **Structure**

$$  
x \to h = \sigma(W_e x + b_e)  
$$  
$$  
\hat{x} = \sigma(W_d h + b_d)  
$$

### **Loss**

$$  
E = \frac12 |x - \hat{x}|^2  
$$

Autoencoders learn **compressed hidden features**.

---

# **21. PCA (Principal Component Analysis)**

### **Goal:**

Find directions of maximum variance.

### **Covariance Matrix**

$$  
C = \frac{1}{N}X^\top X  
$$

### **Eigenvalue Problem**

$$  
C v = \lambda v  
$$

Top eigenvectors = principal components.

---

# ✔️ This is your complete exam cheat sheet.

If you'd like:

### **Option A:** Turn this into structured **Obsidian vault pages**

### **Option B:** Convert all sections into **flashcards**

### **Option C:** Add worked numerical examples for every major topic

### **Option D:** Produce a **1-page ultra-compressed exam summary**

### **Option E:** Make a _final exam mock test_ covering every topic

Just tell me which one you want next.