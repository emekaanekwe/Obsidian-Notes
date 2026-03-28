# FIT5217 Week 4 Study Sheet: Bias-Variance Tradeoff & Regularisation

## 1. Linear Models for Regression — Quick Recap

### The Core Idea

A **linear regression model** predicts an output by computing a weighted sum of **basis functions** φ(x):

$$\hat{y}(\mathbf{x}) = \sum_{j} w_j \phi_j(\mathbf{x})$$

- The model is **linear in the weights** $w_j$, even if the basis functions φ(x) are non-linear (e.g., polynomial, Gaussian)
- This is why complex polynomial models are still called "linear regression" — linearity refers to the _parameters_, not the input features

### Why This Matters for the Exam

> ⚠️ **Common pitfall:** Students confuse "linear in the inputs" with "linear in the parameters." A 6th-order polynomial basis function is still a linear regression model. Markers will test this distinction.

### Basis Functions (Common Examples)

|Type|Form|Notes|
|---|---|---|
|Polynomial|$\phi_j(x) = x^j$|Order controls complexity|
|Gaussian|$\phi_j(x) = \exp\left(-\frac{(x-\mu_j)^2}{2s^2}\right)$|Localised bumps|
|Linear (trivial)|$\phi_j(x) = x$|Simplest case|

### Loss Function

The standard **sum of squared errors (SSE)** loss:

$$\mathcal{L}(\mathbf{w}) = \sum_{n=1}^{N} (t_n - \hat{y}(\mathbf{x}_n))^2$$

- Under a **Gaussian noise assumption**, minimising SSE is equivalent to Maximum Likelihood Estimation (MLE) — this connection is covered formally in Module 5
- Setting $\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 0$ gives the closed-form **normal equations**
- For large datasets, use **Stochastic Gradient Descent (SGD)** — samples mini-batches each step, follows a zigzag path to the optimum

---

## 2. Regularisation (L1 & L2)

### Why Regularise?

Without regularisation, a sufficiently complex model will **overfit** — it memorises training noise rather than learning the true underlying pattern. The model performs well on training data but fails on unseen data.

### The Regularised Loss Function

$$\mathcal{L}_{reg}(\mathbf{w}) = \underbrace{\sum_{n=1}^{N}(t_n - \hat{y}_n)^2}_{\text{data fit term}} + \underbrace{\lambda \Omega(\mathbf{w})}_{\text{regularisation term}}$$

- $\lambda \geq 0$ is a **hyperparameter** you choose (not learned from data)
- $\Omega(\mathbf{w})$ penalises large weights

### L2 Regularisation — Ridge Regression

$$\Omega(\mathbf{w}) = |\mathbf{w}|_2^2 = \sum_j w_j^2$$

- Geometric interpretation: constraint region is a **circle** (2D) or hypersphere (high-D)
- Shrinks all weights toward zero but rarely makes them exactly zero
- Differentiable everywhere → easy to optimise analytically

### L1 Regularisation — Lasso Regression

$$\Omega(\mathbf{w}) = |\mathbf{w}|_1 = \sum_j |w_j|$$

- Geometric interpretation: constraint region is a **diamond** with sharp corners at axes
- The optimum often lands exactly on a corner → **sparse solutions** (some weights become exactly zero)
- This performs automatic **feature selection** — irrelevant features get zeroed out
- Not differentiable at zero → requires special optimisation techniques

### L1 vs L2 — Key Comparison Table

|Property|L2 (Ridge)|L1 (Lasso)|
|---|---|---|
|Penalty form|$\sum w_j^2$|$\sum \|w_j\|$|
|Solutions|Dense (all weights small)|Sparse (some weights = 0)|
|Feature selection|✗ No|✓ Yes|
|Differentiable|✓ Always|✗ Not at zero|
|Geometry|Circle / sphere|Diamond / polytope|

> ⚠️ **Common pitfall:** Students say L1 "removes" features. More precisely — it drives irrelevant weights to _exactly zero_, effectively removing those features from the model. L2 only makes them _very small_ but non-zero.

### Controlling Model Complexity via λ

- **Large λ** → heavy penalty → weights shrink → model simplifies → **underfitting risk** (high bias)
- **Small λ** → light penalty → weights grow freely → model complexifies → **overfitting risk** (high variance)
- **λ = 0** → no regularisation → pure empirical risk minimisation

> 🎯 **Exam tip:** You must be able to explain _why_ increasing λ increases bias and decreases variance (and vice versa). This is the direct bridge to the bias-variance tradeoff.

---

## 3. The Bias-Variance Tradeoff — Mathematical Derivation

This is the most examinable section. Know the derivation, not just the conclusion.

### Setup & Notation

|Symbol|Meaning|
|---|---|
|$f(\mathbf{x})$|True underlying function (unknown in practice)|
|$t = f(\mathbf{x}) + \epsilon$|Observed target with noise $\epsilon \sim \mathcal{N}(0, \sigma^2)$|
|$\hat{y}(\mathbf{x}; \mathcal{D})$|Model prediction trained on dataset $\mathcal{D}$|
|$\bar{y}(\mathbf{x})$|Expected prediction: $\mathbb{E}_\mathcal{D}[\hat{y}(\mathbf{x}; \mathcal{D})]$|

### Step-by-Step Derivation

**Start with the expected squared loss:**

$$\mathbb{E}\left[(t - \hat{y})^2\right]$$

**Step 1:** Substitute $t = f(\mathbf{x}) + \epsilon$:

$$= \mathbb{E}\left[(f + \epsilon - \hat{y})^2\right]$$

**Step 2:** Expand the square:

$$= \mathbb{E}\left[(f - \hat{y})^2\right] + 2\mathbb{E}\left[(f-\hat{y})\epsilon\right] + \mathbb{E}\left[\epsilon^2\right]$$

**Step 3:** The cross-term vanishes because $\hat{y}$ and $\epsilon$ are **independent** and $\mathbb{E}[\epsilon] = 0$:

$$= \mathbb{E}\left[(f - \hat{y})^2\right] + \sigma^2$$

**Step 4:** Add and subtract $\bar{y}$ inside the first term:

$$\mathbb{E}\left[(f - \hat{y})^2\right] = \mathbb{E}\left[(\hat{y} - \bar{y})^2\right] + (\bar{y} - f)^2$$

_(The cross-term here also vanishes — $\mathbb{E}[\hat{y} - \bar{y}] = 0$ by definition of $\bar{y}$)_

**Final Result:**

$$\boxed{\mathbb{E}\left[(t - \hat{y})^2\right] = \underbrace{(\bar{y} - f)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\left[(\hat{y} - \bar{y})^2\right]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible noise}}}$$

### Interpreting Each Term

**Bias²** — $(\bar{y} - f)^2$

- How far your _average_ prediction is from the true function
- Reflects systematic error due to model form or underfitting
- You can reduce this by using a more complex model

**Variance** — $\mathbb{E}[(\hat{y} - \bar{y})^2]$

- How much your prediction fluctuates across different training datasets
- Reflects sensitivity to training data
- You can reduce this via regularisation or more data

**Irreducible Noise** — $\sigma^2$

- Inherent measurement error in the data
- **Cannot be reduced** regardless of model quality
- This is why perfect prediction is impossible with noisy data

> ⚠️ **The most common exam mistake:** Claiming you can reduce irreducible noise. You cannot. It is a property of the data, not the model.

---

## 4. The Tradeoff — Intuition & Diagnostics

### The Dartboard Analogy

||Low Variance|High Variance|
|---|---|---|
|**Low Bias**|✅ Ideal — consistent & accurate|Accurate on average but erratic|
|**High Bias**|Consistently wrong (systematic error)|❌ Worst case — wrong and erratic|

### Overfitting vs Underfitting

|Condition|Bias|Variance|Training Error|Test Error|
|---|---|---|---|---|
|**Underfitting**|High|Low|High|High|
|**Just right**|Low-ish|Low-ish|Moderate|Close to training|
|**Overfitting**|Low|High|Very low|Much higher than training|

> 🎯 **Key diagnostic:** If test error >> training error → overfitting (high variance). If both errors are high → underfitting (high bias).

### Effect of Regularisation on Bias-Variance

```
Increasing λ:
  Variance ↓  (model less sensitive to data fluctuations)
  Bias ↑      (model constrained, may miss true function)

Decreasing λ:
  Variance ↑  (model fits noise in training data)
  Bias ↓      (model can approximate true function well)
```

The **sweet spot** is the λ that minimises total expected test error — found via cross-validation.

---

## 5. Key Conceptual Points for the Exam

1. **Why does test error > training error?** The model is trained to minimise training error, so it is optimistically biased toward training data. Unseen data may have different noise realisations.
    
2. **Why can't we just minimise training error?** Minimising training error alone risks overfitting. The goal is to minimise _generalisation error_ (test error).
    
3. **Why is bias-variance universal?** The derivation only assumes squared loss and additive noise. It applies to _any_ regression model — linear, tree-based, neural network, etc.
    
4. **What does more data do?** More data typically **reduces variance** (your model is less sensitive to any single dataset's quirks) but does not change bias — that depends on model form.
    
5. **What does model complexity do?**
    
    - More complex model → lower bias, higher variance
    - Simpler model → higher bias, lower variance

---

## 6. Quick Formula Reference

|Concept|Formula|
|---|---|
|Ridge loss|$\mathcal{L} = \sum(t_n - \hat{y}_n)^2 + \lambda\|\mathbf{w}\|_2^2$|
|Lasso loss|$\mathcal{L} = \sum(t_n - \hat{y}_n)^2 + \lambda\|\mathbf{w}\|_1$|
|Expected loss decomposition|$\text{Bias}^2 + \text{Variance} + \sigma^2$|
|Bias|$(\mathbb{E}_\mathcal{D}[\hat{y}] - f(\mathbf{x}))^2$|
|Variance|$\mathbb{E}_\mathcal{D}[(\hat{y} - \mathbb{E}[\hat{y}])^2]$|

---

## ⚠️ Top Pitfalls to Avoid in the Exam

1. Confusing "linear in inputs" vs "linear in parameters"
2. Saying L2 regularisation performs feature selection (it doesn't — L1 does)
3. Claiming irreducible noise can be reduced by improving the model
4. Forgetting the cross-term cancellation steps in the bias-variance derivation — markers look for this reasoning
5. Saying "complex model = always bad" — it has _low bias_, which is desirable; the problem is high variance
6. Confusing training error with bias — they are related but not the same thing