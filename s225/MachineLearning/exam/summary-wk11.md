# Study Sheet: Regularization in Neural Networks

---

## 1. The Regularization Problem

### Model Complexity and Generalization

The number of input and output units in a neural network is determined by the dimensionality of the dataset, whereas the number $M$ of hidden units is a **free parameter** that can be adjusted for best predictive performance.

**Key Insight:** $M$ controls the number of parameters (weights and biases) in the network, and we might expect that in a maximum likelihood setting there will be an optimum value of $M$ that gives the best generalization performance, corresponding to the **optimum balance between under-fitting and over-fitting**.

---

## 2. Effect of Network Size on Performance

### Visual Example: Sinusoidal Regression

**Figure 5.3.1** shows examples of two-layer networks trained on 10 data points from sinusoidal data with different numbers of hidden units:

- **$M = 1$** (left): Under-fitting
    - Too simple, cannot capture the sinusoidal pattern
    - High bias, low variance
- **$M = 3$** (middle): Good fit
    - Captures the underlying pattern well
    - Optimal balance
- **$M = 10$** (right): Over-fitting
    - Fits training data perfectly but exhibits wild oscillations
    - High variance, low bias
    - Poor generalization to new data

### The Challenge with Model Selection

The generalization error is **not a simple function of** $M$ due to the presence of **local minima** in the error function (Figure 5.3.2).

**Figure 5.3.2** shows:

- Plot of sum-of-squares test-set error vs. number of hidden units
- 30 random starts for each network size
- Scatter shows effect of local minima
- Each point represents a different random initialization
- Overall best validation performance occurred for $M = 8$

**Practical Approach:**

1. Plot a graph of validation error vs. $M$ (as in Figure 5.3.2)
2. Choose the specific solution having the **smallest validation set error**
3. Train multiple times with different random initializations to avoid poor local minima

---

## 3. Regularization Techniques

### 3.1 Weight Decay (L2 Regularization)

**Motivation:** Alternative approach to control model complexity beyond just choosing $M$. From polynomial curve fitting (Module 1), we can choose a relatively large value for $M$ and then control complexity by adding a regularization term.

**Regularized Error Function:**

The simplest regularizer is the **quadratic (L2) weight decay**:

$$J(\boldsymbol{\theta}) = E(\boldsymbol{\theta}) + \frac{\lambda}{2} \sum_{\ell=1}^{n_l-1} \sum_{i=1}^{s_\ell} \sum_{j=1}^{s_{\ell+1}} (W_{ji}^{(\ell)})^2$$

where:

- $E(\boldsymbol{\theta})$ is the original error function (e.g., sum-of-squares)
- $\lambda$ is the **regularization coefficient**
- The sum is over all weights in the network

**Key Properties:**

- The effective model complexity is determined by $\lambda$
- Larger $\lambda$ → stronger regularization → simpler effective model
- Usually **NOT** applied to bias terms
- Also called **weight decay** because it causes weights to shrink toward zero during training

**Why It Works:**

- Penalizes large weights
- Encourages smooth functions (small weights → less sensitive to inputs)
- Prevents overfitting to noise in training data

---

### 3.2 Early Stopping

**Motivation:** Alternative to regularization for controlling effective complexity without explicitly adding a penalty term.

**The Procedure:**

Training corresponds to an **iterative reduction** of the error function with respect to training data. The key insight:

1. Error on training data: **monotonically decreasing** (nonincreasing function of iteration index)
2. Error on validation data: **initially decreases**, then **increases** as network overfits

**Algorithm:**

1. Split data into: training set, validation set, test set
2. Train the network on training set
3. Monitor error on validation set after each epoch
4. **Stop training** at the point of smallest validation error
5. Evaluate final performance on test set

**Figure 5.3.3** illustrates:

- Left plot: Training error continues to decrease
- Right plot: Validation error reaches minimum then increases
- Vertical dashed line: Optimal stopping point (minimum validation error)
- Goal: Stop before overfitting begins

**Why It Works:**

- Early in training: network learns general patterns (low complexity)
- Later in training: network memorizes training data (high complexity)
- Stopping early = limiting effective network complexity

---

### 3.3 Relationship Between Early Stopping and Weight Decay

**Theoretical Connection:**

For **quadratic error functions** (regression), early stopping and weight decay are closely related and can give similar results!

**Mathematical Insight (Figure 5.3.4):**

In weight decay training (without early stopping), the weight vector path:

1. Starts at origin $\tilde{\mathbf{w}}$ (small random initialization)
2. Follows gradient descent along a path through weight space
3. Eventually reaches minimum $\mathbf{w}_{ML}$ (maximum likelihood solution)

**With weight decay:**

- Minimum is shifted by the regularization term
- Optimal weights are "pulled back" from $\mathbf{w}_{ML}$ toward origin
- Red ellipse shows contour of equal error

**With early stopping:**

1. Start at origin $\tilde{\mathbf{w}}$
2. Follow blue path toward $\mathbf{w}_{ML}$
3. Stop before reaching $\mathbf{w}_{ML}$ at point $\bar{\mathbf{w}}$
4. Stopping early = not moving too far from origin

**Quantitative Relationship:**

The quantity $\tau \eta$ (where $\tau$ is iteration index and $\eta$ is learning rate) plays the role of the reciprocal of the regularization parameter $\lambda$.

$$\tau \eta \propto \frac{1}{\lambda}$$

**Intuition:**

- More training iterations ($\tau$ large) ↔ Less regularization ($\lambda$ small)
- Fewer training iterations ($\tau$ small) ↔ More regularization ($\lambda$ large)

**Effective Number of Parameters:**

The effective number of parameters in the network **grows during training**:

- Early: Few effective parameters (simple model)
- Late: Many effective parameters (complex model)

This explains why both methods control complexity similarly!

---

## 4. Understanding Effective Complexity

### Degrees of Freedom Perspective

The behavior of neural networks during training can be explained qualitatively in terms of the **effective number of degrees of freedom**:

1. **Start of training:** Small effective degrees of freedom
    
    - Weights are small (near initialization)
    - Network behaves almost linearly
    - Simple model
2. **During training:** Effective degrees of freedom **grows**
    
    - Weights get larger
    - Nonlinearities become more pronounced
    - Model complexity increases
3. **End of training:** Maximum degrees of freedom
    
    - Full network capacity utilized
    - Risk of overfitting

**Connection to Regularization:**

**Early stopping:** Halts training before reaching maximum complexity

- Directly limits degrees of freedom by stopping iteration

**Weight decay:** Limits weight magnitudes throughout training

- Indirectly limits degrees of freedom by constraining weights

Both achieve similar effect: **preventing the network from becoming too complex** for the given training data.

---

## 5. Practical Guidelines

### 5.1 Choosing the Number of Hidden Units ($M$)

**Strategy 1: Model Selection**

1. Try multiple values of $M$ (e.g., $M = 1, 2, 3, \ldots, 20$)
2. For each $M$, train with multiple random initializations (e.g., 30 runs)
3. Evaluate validation error for all combinations
4. Select architecture and initialization with best validation performance

**Strategy 2: Use Regularization**

1. Choose relatively large $M$ (to ensure sufficient capacity)
2. Use weight decay or early stopping to control complexity
3. Tune regularization parameter $\lambda$ instead of $M$

**In Practice:** Often combine both strategies!

---

### 5.2 Choosing the Regularization Parameter ($\lambda$)

**Typical Approach:**

1. Start with no regularization ($\lambda = 0$)
2. Gradually increase $\lambda$: try $\lambda = 0.001, 0.01, 0.1, 1.0, \ldots$
3. Monitor validation error
4. Choose $\lambda$ that minimizes validation error

**Grid Search:** Systematically try combinations of hyperparameters

**Cross-Validation:** Use k-fold CV to estimate generalization performance for each $\lambda$

---

### 5.3 Implementing Early Stopping

**Best Practices:**

1. **Validation Set Size:**
    
    - Typically 10-20% of available data
    - Balance: larger → better estimate, smaller → more training data
2. **Patience Parameter:**
    
    - Don't stop at first increase in validation error
    - Wait for several epochs of no improvement (e.g., 10-20 epochs)
    - Prevents premature stopping due to noise
3. **Save Best Model:**
    
    - Keep track of model with lowest validation error
    - Final model = best checkpoint, not last checkpoint
4. **Learning Rate Schedule:**
    
    - Can combine with learning rate decay
    - Reduce learning rate when validation error plateaus

---

## 6. Comparing Regularization Methods

### Advantages and Disadvantages

|Method|Advantages|Disadvantages|
|---|---|---|
|**Weight Decay**|• Simple to implement<br>• Smooth control via $\lambda$<br>• Theoretical foundation|• Requires tuning $\lambda$<br>• Adds computational cost|
|**Early Stopping**|• No extra hyperparameters<br>• Computationally efficient<br>• Intuitive|• Requires validation set<br>• Less smooth control<br>• May need patience tuning|
|**Model Selection**|• Direct control of capacity<br>• Interpretable|• Computationally expensive<br>• Discrete choices only<br>• Sensitive to initialization|

---

## 7. Mathematical Prerequisites Refresher

### Understanding Weight Decay Gradient

When we add L2 regularization: $$J(\boldsymbol{\theta}) = E(\boldsymbol{\theta}) + \frac{\lambda}{2} \sum (W_{ji}^{(\ell)})^2$$

The gradient becomes: $$\frac{\partial J}{\partial W_{ji}^{(\ell)}} = \frac{\partial E}{\partial W_{ji}^{(\ell)}} + \lambda W_{ji}^{(\ell)}$$

**Update rule:** $$W_{ji}^{(\ell)} \leftarrow W_{ji}^{(\ell)} - \eta \left(\frac{\partial E}{\partial W_{ji}^{(\ell)}} + \lambda W_{ji}^{(\ell)}\right)$$

**Rewrite:** $$W_{ji}^{(\ell)} \leftarrow (1 - \eta\lambda) W_{ji}^{(\ell)} - \eta \frac{\partial E}{\partial W_{ji}^{(\ell)}}$$

**Interpretation:** The factor $(1 - \eta\lambda)$ causes weights to **decay** (shrink) toward zero each iteration!

**Why "Weight Decay"?**

- If $\eta\lambda = 0.01$, then weights are multiplied by $0.99$ each step
- This is a **multiplicative decay** toward zero
- Balances against gradient updates that try to increase weights

---

## 8. Key Insights and Intuitions

### The Bias-Variance Tradeoff

**Without Regularization:**

- Simple model ($M$ small): High bias (underfitting), low variance
- Complex model ($M$ large): Low bias, high variance (overfitting)

**With Regularization:**

- Can use large $M$ (sufficient capacity)
- Regularization controls effective complexity
- Achieve good balance between bias and variance

### Why Multiple Random Initializations?

From Figure 5.3.2, we see **huge variance** in performance for same architecture due to:

1. **Local minima:** Different initializations converge to different local optima
2. **Random variation:** Stochasticity in optimization process

**Solution:** Train multiple models with different initializations and select the best!

### The Training Dynamics

**Typical Training Curve (Figure 5.3.3):**

**Phase 1: Learning general patterns**

- Both training and validation error decrease
- Network learns useful features
- Good generalization

**Phase 2: Overfitting**

- Training error continues decreasing
- Validation error starts increasing
- Network memorizes training data
- Poor generalization

**Optimal Point:** Minimum of validation error (transition between phases)

---

## 9. Advanced Considerations

### Other Regularization Techniques (Not in These Notes)

**Dropout:**

- Randomly "drop" neurons during training
- Prevents co-adaptation of features
- Very popular in modern deep learning

**Data Augmentation:**

- Artificially increase training set size
- Apply transformations to existing data
- Especially effective for images

**Batch Normalization:**

- Normalize activations within mini-batches
- Has regularizing effect
- Also speeds up training

**L1 Regularization:**

- Penalize $\sum |W_{ji}|$ instead of $\sum W_{ji}^2$
- Encourages sparse weights (many exactly zero)
- Feature selection property

---

## 10. Practical Example: Complete Workflow

### Step-by-Step Guide

1. **Split Data:**
    
    ```
    Training: 60-70%
    Validation: 15-20%
    Test: 15-20%
    ```
    
2. **Choose Architecture:**
    
    - Start with 1 hidden layer, moderate size (e.g., $M = 10$)
    - Or try several architectures
3. **Set Hyperparameters:**
    
    - Learning rate: $\eta = 0.01$ (adjust as needed)
    - Weight decay: $\lambda = 0.001$ (if using)
    - Mini-batch size: 32-256
4. **Train:**
    
    - Initialize weights randomly (small values)
    - Run gradient descent with backpropagation
    - Monitor training and validation error each epoch
    - Save model when validation error improves
5. **Early Stopping:**
    
    - Stop if validation error doesn't improve for 10-20 epochs
    - Load best saved model
6. **Evaluate:**
    
    - Compute error on test set (final generalization estimate)
    - Never use test set for any training decisions!
7. **Iterate:**
    
    - If performance insufficient, try different $M$, $\lambda$, or architecture
    - Repeat process with different hyperparameters

---

## Summary: Key Takeaways

1. **Model complexity** is controlled by:
    
    - Number of hidden units ($M$)
    - Regularization strength ($\lambda$)
    - Training duration (early stopping)
2. **Weight decay (L2 regularization)** penalizes large weights:
    
    - Added term: $\frac{\lambda}{2} \sum W_{ji}^2$
    - Causes weights to "decay" toward zero
    - Controlled by $\lambda$ hyperparameter
3. **Early stopping** halts training at minimum validation error:
    
    - Simpler than weight decay (no extra hyperparameter)
    - Requires validation set
    - Similar effect to weight decay for quadratic loss
4. **Local minima** cause variation in performance:
    
    - Train with multiple random initializations
    - Select best performing model
    - Figure 5.3.2 shows huge variance
5. **Bias-variance tradeoff:**
    
    - Too simple → underfitting (high bias)
    - Too complex → overfitting (high variance)
    - Regularization helps find sweet spot
6. **Validation set** is crucial:
    
    - Used for model selection (choosing $M$, $\lambda$)
    - Used for early stopping
    - Never use test set for these decisions!
7. **Effective complexity grows during training:**
    
    - Early: simple model (few effective parameters)
    - Late: complex model (many effective parameters)
    - Early stopping = limiting this growth

**Bottom Line:** Regularization is essential for preventing overfitting and achieving good generalization in neural networks! 🎯