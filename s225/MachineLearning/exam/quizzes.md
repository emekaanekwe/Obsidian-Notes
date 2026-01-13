## **Quiz 1: Foundations & Basics**

### 1. What Is the Objective of Machine Learning?
- **Correct**: Predict target values for new inputs; generalize well to unseen data.
- **Incorrect**: Solely mapping inputs to targets accurately *on the training set* (that’s training, not generalization).

### 2. What Are the Main Problem Types & Learning Methods?
- **Supervised learning**: Target variables given in training data.
- **Unsupervised learning**: No labels (e.g., clustering).
- **Classification vs. Regression**:
  - Classification: Discrete output (e.g., spam detection).
  - Regression: Continuous output (e.g., temperature prediction → **not classification**).

### 3. What Are Key Properties of Models & Sources of Uncertainty?
- **K-NN regression**: Non-parametric.
- **Test data**: Used for evaluation, **not** for parameter optimization.
- **Uncertainty**: Caused by finite data and noise.

### 4. What Is Underfitting & Overfitting?
- **Overfitting**: Low training error, high test error.
- **Underfitting**: High training and test error.
- **How to recognize overfitting**: Requires monitoring *test* error, not just training error.

### 5. How Does Regularization Affect a Model?
- Large regularization → **underfitting** (high bias, low variance).
- Small regularization → risk of overfitting.

### 6. What Affects Model Complexity in Polynomial Regression?
- **Affects complexity**: Polynomial order, regularization parameter.
- **Does not affect**: Values of input variables.

### 7. What Are the Basics of Probability?
- **Probabilities**: Between 0 and 1.
- **Probability densities**: Can be >1 (only integrals = 1).

### 8. What Are the Key Components of Bayesian Learning?
- **Likelihood**: \( p(D \mid w) \) (probability of data given parameters).
- **Prior**: \( p(w) \) (belief about parameters before seeing data).
- **Posterior**: \( p(w \mid D) \) (after seeing data).

---

## **Quiz 2: Optimization, Bias-Variance, Linear Models**

### 1. What Are Key Properties of Optimization Algorithms?
- **Iterative methods**: Gradient descent, stochastic gradient descent (SGD).
- **Sensitive to initial values** (not insensitive).
- Can get stuck in local minima.

### 2. What Is the Bias & Variance Trade-off?
- **Bias**: Systematic error (high bias = underfitting).
- **Variance**: Sensitivity to training data (high variance = overfitting).
- **Trade-off**: Flexible models → low bias, high variance.
- **Best model**: Balances bias and variance (not simply lowest bias).

### 3. What Are Basis Functions & How Does Linear Regression Work?
- **Basis functions**: Can be nonlinear (e.g., polynomial, radial).
- **Linear regression**: Linear in parameters, not necessarily in inputs.

### 4. What Are the Advantages of SGD Over Gradient Descent?
- **SGD advantages**:
  - Handles large datasets efficiently.
  - Works with streaming/out-of-memory data.
- **Not an advantage**: Insensitivity to initialization (still sensitive).

### 5. How Do You Diagnose & Fix High Variance vs. High Bias?
- **High variance**: Reduce features, add regularization, get more data.
- **High bias**: Add features, use more complex model, reduce regularization.

### 6. How Does the Gradient Descent Update Rule Work?
- Parameters updated in **opposite direction** of gradient (to minimize error).

---

## **Quiz 3: Classification & Generative Models**

### 1. What Are Examples of Regression vs. Classification Problems?
- Regression: Predicting house prices.
- Classification: Digit recognition, spam detection, credit rating.

### 2. What Is the Difference Between Discriminative & Generative Models?
- **Discriminative**: Models \( p(y \mid x) \) directly (e.g., logistic regression).
- **Generative**: Models \( p(x \mid y) \) and \( p(y) \) (e.g., Naive Bayes, Gaussian models).
- Generative models can **generate new data** given a class.

### 3. How Do You Generalize Binary Classification to Multi-class?
- **One-vs-One**: \( K(K-1)/2 \) classifiers.
- **One-vs-All**: \( K \) classifiers.
- **K discriminant functions**: Choose class with highest score.

### 4. What Are Key Properties of the Perceptron Algorithm?
- Converges if data is **linearly separable**.
- Sensitive to initialization.
- Does **not** converge for non-separable data.

### 5. What Do Probabilistic Generative Models Actually Model?
- **Model \( p(x \mid C_k) \)**: Gaussian-based classifiers, Bayes classifier.
- **Decision boundary**: Linear if shared covariance matrix in multivariate Gaussian.

### 6. Which Classifiers Have Analytical Solutions?
- **No analytical solution**: Logistic regression (iterative optimization needed).
- **Analytical solution possible**: Gaussian generative models.

---

## **Quiz 4: Clustering & Neural Networks**

### 1. What Are Key Properties of K-Means Clustering?
- **Iterative, sensitive to initialization**.
- Each point assigned to **one cluster only** (hard assignment).

### 2. What Are Latent Variable Models (e.g., GMM)?
- **Latent variables**: Unobserved cluster assignments.
- **Optimization**: EM algorithm (estimates parameters and latent variables).
- **No analytical solution** for GMM (use EM).

### 3. How Does the Expectation-Maximization (EM) Algorithm Work?
- **E-step**: Compute expected latent variables.
- **M-step**: Maximize Q function (lower bound on likelihood).
- **Guarantees** increase in log-likelihood each iteration.

### 4. What Is the Difference Between a Perceptron and a Neuron?
- **Perceptron**: Binary output (step function), for linearly separable data.
- **Neuron**: Continuous output (activation function).

### 5. What Are the Fundamental Properties of Neural Networks?
- **Can have multiple outputs**.
- **Universal approximation**: With enough hidden units, can approximate any function.
- **More hidden units** → higher capacity, but risk of overfitting (not automatically higher test accuracy).

### 6. How Do You Prevent Overfitting in Neural Networks?
- Regularization, dropout, early stopping, more training data.
- **Shallow architecture** → reduces capacity, may increase bias (not a direct overfitting fix).

---

## **Key Takeaways for Exam Prep**
- **Generalization** is the ultimate goal, not just fitting training data.
- **Bias-variance trade-off** is central to model selection and diagnosis.
- **Know your algorithms**: GD vs. SGD, EM, K-Means, Perceptron.
- **Probability is fundamental** in Bayesian methods and generative models.
- **Neural networks** are powerful but require careful regularization.

---

Let me know if you'd like a **condensed cheat sheet** or **flashcards** from this material.




Q1![[Quiz 1 2025 S2_ Attempt review _ MonashELMS1.pdf]]
Q2
![[Quiz 2 2025 S2_ Attempt review _ MonashELMS1.pdf]]
Q3
![[Quiz 3 2025 S2_ Attempt review _ MonashELMS1.pdf]]
Q4
![[Quiz 4 2025 S1_ Attempt review _ MonashELMS1.pdf]]