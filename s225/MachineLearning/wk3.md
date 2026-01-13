# 1. Basis Functions: Making Linear Models Nonlinear
**Slides 1-6**  
## Core Idea 
- **Problem**: Linear regression *is too rigid for complex data* 
	- **Solution**: Use *basis functions* 

### Basis Functions

	These functions are used for nonlinear parameters, which gives linear models simple analytical properties, and yet can be nonlinear with respect to the input variables.

#### Defining Basis Functions

**first we define a linear function**

A function is linear  iff: $$f(\alpha x_1+\beta x_2)=f(\alpha x_1)+f(\beta x_2) \ and \ \ x_1, x_2 \in \ \mathbb{R}^d \ \alpha, \beta \in \mathbb{R}$$
- Where $\mathbb{R}^d$ is the set of reals with *vector length d* 
-  $\alpha$ and $\beta$ are *scalar* coefficients that multiplies the vectors
- Note that the function is *additive* (sum of two inputs = sum of whole) and *homogenous* (the function is proportional to the scalar multiplication)

**Now for linear regression functions**
$$y(x,w):=w_0+x_1w_1+,...,+x_nw_n$$
- with w as the *functions parameters* 
- y(w.x) as *target values, which must be linear*

**Putting them together** $$y(w,x):=w_0+w_1\phi_1(x)+...+w_{M-1}\phi_{M-1}(x) = \sum^{M-1}_{j=0}w_j\phi_j(x)$$
- $\phi_0(x):=1$ as a *dummy basis function*
- $w:=(w_0, ..., w_{M-1})$ are the *parameters*
- what changes are the *feature functions*



![[Pasted image 20250811173432.png]]
  
##### Example Basis Functions
  
- Gaussian: $$\phi_j(x) = \exp\left(-\frac{(x-\mu_j)^2}{2s^2}\right)$$
- Sigmoidal: $$\phi_j(x) = \frac{1}{1 + \exp(-(x-\mu_j)/s)}$$ 
- Hyperbolic Tangent $$tanh(a) = \frac{1-\epsilon^{-2a}}{1+\epsilon^{-2a}}$$

### Why This Matters  
- Models can fit curves (e.g., polynomials, periodic functions) without losing the **convex optimization** benefits of linear-in-parameters.  

**Analogy**:  
Think of basis functions as lenses that distort the input space so a linear plane in the new space can fit complex patterns in the original space.  

**Python Example**:  
```python
import numpy as np
# Gaussian basis functions
def gaussian_basis(x, mu, s):
    return np.exp(-(x - mu)**2 / (2 * s**2))

X = np.linspace(0, 1, 100)
Phi = np.column_stack([gaussian_basis(X, mu=0.3, s=0.1), 
                       gaussian_basis(X, mu=0.7, s=0.1)])
```

---

## Optimizing Error Functions

	the receipes of learning w
### Key Concepts 


#### Sum-of-Squares Error
   $$
   E(w) = \frac{1}{2} \sum_{n=1}^N [t_n - w^T \phi(x_n)]^2
   $$  
- Derived from **maximum likelihood** under Gaussian noise assumption.  
- notice that the w is the core of what we re trying to approximate to 

**Set up**
Data D = $${(x_n,t_n)}^{N}_{n=1}$$
function: $y(w,x)$
Noise: $\epsilon = N(0, \sigma^2)$
Coefficient:  $t_n=y(w,x)+\epsilon$
Assume training data points are independent from $P(t|x,w,\sigma^2)=N(t|y(x,w),\sigma^2$)

this is how we assume the target is the sum of the regression model + noise from guass distro

**Wih Likelihood function**


**And log-Likelihood Function**
$$L(w):=logP(t|x,w,\sigma)$$

2. **Closed-Form Solution (Normal Equations)**:  
   $$
   w = (\Phi^T \Phi)^{-1} \Phi^T t
   $$  
   - **Pros**: Exact, one-step solution.  
   - **Cons**: Inefficient for large \( N \) (matrix inversion is \( O(N^3) \)).  

## More Specific Optimization with Partial Derivatives

get all the partial derivatives of the loss function and then find the stat points by  setting them to 0

#### Final Form of Loss Function

$$\frac{1}{2\sigma^2}\sum^{N}_{n=1}[t_n-w*\phi(x_n)]^2$$

when get all partial derivatives, and stack them as a vector, we call that a *gradient*

**Gradient** ***fix this!**
$$gradient \ L(w)=[\frac{dL(w)}{dw_0}] = [0]$$

##### the problem is that the inverse is computationally high!

##### Solution:
## Iterative Optimization Algorithms

### Gradient Descent (also called batch gradient descent)
   - Iterative update: $w^{(t)} = w^{(t-1)} - \eta \nabla E(w^{(t-1)})$ 
   - **Learning rate \( \eta \)**: Too small → slow; too large → divergence.  

**Exam Tip**:  
- Expect derivations of $\nabla E(w)$  (e.g., for sigmoid basis).  

**Code Snippet (GD)**:  
```python
def gradient_descent(Phi, t, eta=0.01, epochs=1000):
    w = np.zeros(Phi.shape[1])
    for _ in range(epochs):
        grad = Phi.T @ (Phi @ w - t)  # ∇E(w)
        w -= eta * grad
    return w
```

---

### Stochastic Gradient Descent
**Slides 22-26**  
#### **Why SGD?**  
- GD processes all \( N \) points per step → computationally expensive.  
- SGD updates \( w \) using **one random data point** per iteration:  
  $$
  w^{(t)} = w^{(t-1)} + \eta (t_n - w^{(t-1)T} \phi(x_n)) \phi(x_n)
  $$ 
- **Pros**: Faster per iteration, escapes local minima.  
- **Cons**: Noisy updates; requires careful tuning of \( \eta \).  

**Analogy**:  
GD is like carefully surveying an entire mountain before stepping; SGD is a hiker taking small, random steps downhill.  

**When to Use**:  
- Large datasets (e.g., deep learning).  

---

### **4. Regularization: Preventing Overfitting**  
**Slides 28-38**  
#### **Core Idea**:  
Add a penalty term \( \lambda \Omega(w) \) to the error function to control model complexity:  
$$
E(w) = \text{Error} + \lambda \Omega(w)
$$ #### **Types**:  
1. **Ridge (L2)**: $\Omega(w) = \frac{1}{2} \|w\|_2^2$  
   - **Effect**: Shrinks weights smoothly; preserves all features.  
   - **Solution**: $w = (\lambda I + \Phi^T \Phi)^{-1} \Phi^T t$  

1. **Lasso (L1)**: $\Omega(w) = \|w\|_1$  
   - **Effect**: Forces some weights to **zero** (feature selection).  
   - **Optimization**: Sub-gradient methods (non-differentiable at 0).  

**Example**:  
- **Lasso** might discard irrelevant features (e.g., shoe size vs. weight for height prediction).  

**Visualization**:  
<img src="https://miro.medium.com/v2/resize:fit:1400/1*HQug3p5yHZtGg6MORu6WQQ.png" width="400" alt="Ridge vs Lasso">  

**Python**:  
```python
from sklearn.linear_model import Ridge, Lasso
ridge = Ridge(alpha=0.1).fit(Phi, t)  # alpha = λ
lasso = Lasso(alpha=0.1).fit(Phi, t)  # Sparse weights
```

---

### **5. Key Takeaways**  
1. **Basis Functions**: Enable nonlinearity while keeping optimization linear.  
2. **Optimization**:  
   - Closed-form for small datasets.  
   - GD/SGD for large datasets.  
3. **Regularization**:  
   - Ridge for gentle shrinkage.  
   - Lasso for feature selection.  

---

### **6. Exam/Assignment Focus Areas**  
1. **Derive** the gradient of the error function for a given basis.  
2. **Compare** GD vs. SGD in terms of convergence and computation.  
3. **Explain** why Lasso leads to sparsity (hint: diamond vs. circle constraint).  

**Practice Problem**:  
Given $\phi(x) = [1, x, x^2]  and  D = {(1, 2), (2, 5)}$, compute  w  using Ridge regression (\( \lambda = 0.5 \)).  

---

### **7. Further Resources**  
1. **Basis Functions**: [Bishop’s PRML, Chapter 3](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/).  
2. **Optimization**: [Stanford CS229 Notes on GD/SGD](http://cs229.stanford.edu/notes2021fall/cs229-notes1.pdf).  
3. **Regularization**: [Scikit-learn Ridge/Lasso Guide](https://scikit-learn.org/stable/modules/linear_model.html).  

---

### **Next Steps**  
1. **Clarify**: Any confusion about basis functions or regularization?  
2. **Code**: Want to implement SGD from scratch?  
3. **Math**: Need help deriving the Ridge closed-form solution?  
