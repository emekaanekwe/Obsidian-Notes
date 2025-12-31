# FIT5201 - Bias and Variance Study Guide

## 📚 **Lecture Overview**
This lecture covers the mathematical foundations of bias and variance decomposition in machine learning, providing tools to diagnose model performance and address overfitting/underfitting issues.

---

## 🎯 **Key Learning Objectives**
- Understand what bias and variance mean in machine learning contexts
- Learn to decompose generalization error into bias and variance components
- Apply bias-variance analysis to diagnose model performance
- Use bias-variance tradeoff to guide model selection decisions

---

## 🔍 **Core Concepts**

### **Bias**
- **Definition**: The difference between the expected prediction of a model and the correct target value
- **Indicator of**: Model accuracy - higher bias = lower accuracy
- **Mathematical**: `Bias = E[ŷ(x)] - h(x)`
- **Intuition**: How far off are your predictions on average?

### **Variance**
- **Definition**: The variability of model predictions for the same input across different training sets
- **Indicator of**: Model consistency - higher variance = less consistent predictions
- **Mathematical**: `Var = E[(ŷ(x) - E[ŷ(x)])²]`
- **Intuition**: How much do your predictions vary with different training data?

### **Generalization Error Decomposition**
```
Error = Bias² + Variance
```
This fundamental equation shows that total error can be broken down into these two components.

---

## 🎯 **Dartboard Analogy**

| Configuration | Bias | Variance | Description |
|---------------|------|----------|-------------|
| **Top Left** (🎯 IDEAL) | Low | Low | Accurate and consistent - best model |
| **Top Right** | Low | High | Accurate on average but inconsistent |
| **Bottom Left** | High | Low | Consistent but inaccurate (underfitting) |
| **Bottom Right** (❌ WORST) | High | High | Inaccurate and inconsistent |

**Key Insight**: The center (bull's eye) represents the true target value, and dots represent individual model predictions.

---

## 🧮 **Mathematical Framework**

### **Setup**
- Training dataset: `D = {(x₁,t₁), (x₂,t₂), ..., (xₙ,tₙ)}`
- Input distribution: `p(x)`
- Target function: `h(x)` (unknown ground truth)
- Model prediction: `y(x|D)` (depends on training data D)

### **Generalization Error**
```
E = ∫ [h(x) - y(x|D)]² p(x) dx
```

### **Bootstrap Approach**
1. Create multiple datasets via bootstrap sampling
2. Train separate models on each dataset
3. For any input x, get multiple predictions
4. Analyze the distribution of these predictions

### **Bias-Variance Decomposition Proof**
Starting from: `E[(y(x|D) - h(x))²]`

Adding and subtracting `E[y(x|D)]`:
```
E[(y(x|D) - E[y(x|D)] + E[y(x|D)] - h(x))²]
```

Expanding and using expectation properties:
```
= E[(y(x|D) - E[y(x|D)])²] + (E[y(x|D)] - h(x))²
= Variance + Bias²
```

---

## 📊 **Experimental Examples**

### **Example 1: Ridge Regression with Different λ Values**
- **Function**: `sin(2πx)`
- **Models**: Linear regression with 24 Gaussian basis functions
- **Regularization**: L2 (Ridge) with varying λ values

**Results**:
- **High λ (Simple models)**: High bias, Low variance
- **Low λ (Complex models)**: Low bias, High variance
- **Optimal λ**: Balanced bias-variance tradeoff

### **Example 2: Polynomial Regression with Different Degrees**
- **Function**: `sin(2πx)` with noise
- **Models**: Polynomials of degrees 0, 1, 3, 15

**Results**:
| Degree | Bias | Variance | Overall Performance |
|--------|------|----------|-------------------|
| 0 | Highest | Lowest | Poor (underfitting) |
| 1 | High | Low | Poor |
| 3 | Low | Low | **Best** |
| 15 | Lowest | Highest | Poor (overfitting) |

---

## ⚖️ **Bias-Variance Tradeoff**

### **Key Principles**
- **Flexible models** (many parameters): Low bias, High variance
- **Rigid models** (few parameters): High bias, Low variance
- **Goal**: Find the sweet spot that minimizes total error

### **Model Complexity Effects**
```
Increasing Model Complexity
    ↓
Bias decreases ←→ Variance increases
```

---

## 🛠️ **Practical Decision Making Guide**

### **When to Add More Training Data?**
- ✅ **Helpful**: When model has **high variance**
- ❌ **Not helpful**: When model has **high bias**
- **Reason**: More data reduces variance but doesn't fix underfitting

### **When to Remove Features?**
- ✅ **Helpful**: When model has **high variance**
- ❌ **Not helpful**: When model has **high bias**
- **Reason**: Fewer features reduce complexity and variance

### **When to Add Features?**
- ✅ **Helpful**: When model has **high bias**
- ❌ **Not helpful**: When model has **high variance**
- **Reason**: More features increase model flexibility

### **When to Use Regularization?**
- ✅ **Helpful**: When model has **high variance**
- **Examples**: Ridge regression (L2), Lasso (L1)
- **Effect**: Reduces model complexity and variance

---

## 📈 **Diagnostic Process**

### **Step 1: Identify the Problem**
1. Train model and measure test error
2. If error is high, proceed to bias-variance analysis

### **Step 2: Decompose the Error**
1. Use bootstrap sampling or cross-validation
2. Calculate bias and variance components
3. Identify which component is dominant

### **Step 3: Apply Appropriate Solution**
- **High Bias** → Increase model complexity, add features
- **High Variance** → Reduce complexity, add regularization, get more data
- **Both High** → Start with addressing bias, then variance

---

## 🔑 **Key Exam Points**

### **Must Remember**
1. **Error = Bias² + Variance**
2. **Bias**: Distance between average prediction and truth
3. **Variance**: Spread of predictions across different training sets
4. **Tradeoff**: Improving one often worsens the other
5. **Optimal complexity**: Minimizes total error, not individual components

### **Common Mistakes to Avoid**
- Confusing bias with error (bias is specifically about the average prediction)
- Thinking more complex models are always better
- Not considering the variance cost when adding model complexity
- Forgetting that bias is squared in the decomposition formula

### **Application Examples**
- **Underfitting**: High bias, low variance
- **Overfitting**: Low bias, high variance
- **Good fit**: Both bias and variance are reasonably low

---

## 💡 **Study Tips for Exam**

1. **Practice with the dartboard analogy** - it makes bias/variance intuitive
2. **Work through the mathematical derivation** - understand each step
3. **Remember practical guidelines** for when to add data, features, or regularization
4. **Connect to previous lectures** on overfitting, underfitting, and regularization
5. **Practice interpreting bias-variance plots** like those shown in the examples

---

## 🔗 **Connections to Other Topics**
- **Regularization**: Tool to control bias-variance tradeoff
- **Cross-validation**: Method to estimate bias and variance
- **Model selection**: Use bias-variance analysis to choose optimal complexity
- **Ensemble methods**: Can reduce variance while maintaining low bias

---

*This study guide synthesizes the key concepts from FIT5201 Lecture 4 on Bias and Variance. Focus on understanding the fundamental tradeoff and how to apply it practically in model diagnosis and improvement.*