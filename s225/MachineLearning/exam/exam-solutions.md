## stion 2: Goal of Supervised Learning [3 marks]

### **What You Need to Know**

**Key Concepts**:

1. **Ultimate Goal**: Minimize **generalization error** (test error on unseen data)
2. **Training Error**: $$E_{train} = \frac{1}{N} \sum_{i=1}^{N} L(y_i, f(x_i)) $$Error on the training dataset
3. **Test Error** (Generalization Error): $$E_{test} = \mathbb{E}_{(x,y) \sim P_{data}}[L(y, f(x))] $$Expected error on new, unseen data from the true distribution
4. **The Tension**:
    - Easy to minimize training error (just memorize!)
    - Hard to minimize test error (need to generalize)
    - **Overfitting**: Low training error, high test error
    - **Underfitting**: High training error, high test error

$$\frac{\partial \sigma(w^T x)}{\partial w_i} = \frac{\partial \sigma(a)}{\partial a} \cdot \frac{\partial a}{\partial w_i}$$ where $a = w^T x$

$$\log p(X, Z | \theta) = \sum_{n=1}^{N} \sum_{k=1}^{K} z_{nk} \left[\log \pi_k + \log p(x_
n | \mu_k)\right]$$

$$\frac{\partial E}{\partial w_1} = 37.4 \times 6 \times 0.5 = 112.2$$
