# Books
1. Pattern Rcognition & ML (Bishop 2006)
	1. PRML
# Wk 1

	1. PRML pg 4-57
	2. PRML exercises pg 58 
	3. PRML pg 137-140

---

### **1. Core Problem: Regression with Noisy Data**  
**Slide 1-4**  
![[Pasted image 20250728164350.png]]
- **Scenario**: Predict a continuous target \( t \) (e.g., temperature) from input \( x \) (e.g., time of day).  
- **Training Data**: 10 noisy points sampled from  $t = \sin(2\pi x)$ \) (green curve).  
- **Goal**: Learn a model that generalizes to new \( x \) values (test set: 100 points).  
- **Key Idea**: Avoid "cheating" by never touching the test set during training.  

**Analogy**:  
Think of studying past exams (training data) to predict future questions (test data). Peeking at the actual exam (test set) defeats the purpose!  

---

### **2. Model Selection: Polynomial Curve Fitting**  
**Slide 5-12**  
- **Model Choice**: Polynomial of degree \( M \):  
$$
  y(x, w) = w_0 + w_1x + \cdots + w_Mx^M   
  $$
- **Linear in Parameters**: Though nonlinear in \( x \), it’s linear in \( w \) (simplifies optimization).  
**Error Function**: Sum of squared errors (SSE):  

$$E(w) = \frac{1}{2} \sum_{n=1}^N [y(x_n, w) - t_n]^2$$
**Why SSE?**: Penalizes large errors quadratically (like how a harsh penalty discourages big mistakes).  

**Optimization**: Find \( w^* = \arg\min_w E(w) \). For linear models, this has a **unique closed-form solution** (unlike deep learning!).  

---

### **3. Overfitting vs. Underfitting**  
**Slide 13-19**  
- **Underfitting (M=0,1)**: Model too simple (flat line/linear). High error on *both* train and test sets.  
  - Like using a ruler to fit a curvy road.  
- **Overfitting (M=9)**: Model too complex (wiggly curve). Zero train error but high test error.  
  - Like memorizing exam answers instead of learning concepts.  

**Visual Clue**:  
- \( M=9 \) fits all training points perfectly but oscillates wildly between them (Slide 19’s huge \( w \) values).  

**Root Cause**:  
- Limited data (\( N=10 \)) + high flexibility (\( M=9 \)) → Model "hallucinates" patterns from noise.  

---

### **4. Regularization: Taming Overfitting**  
**Slide 21-23**  
**Idea**: Penalize large weights to keep the model simple.  

$E(w) = \text{SSE} + \frac{\lambda}{2} \|w\|^2 \quad (\text{L2 regularization})$

- **\( \lambda \) (lambda)**: Controls "how much to punish complexity."  
  - Large \( \lambda \): Model becomes too simple (underfitting).  
  - Small \( \lambda \): Model stays complex (overfitting).  

**Analogy**:  
Think of \( \lambda \) as a "parent" controlling how strictly a child (model) follows rules. Too strict? Child is rigid. Too lenient? Child misbehaves.  

**Effect**:  
- Regularized \( M=9 \) behaves like a lower-degree polynomial (Slide 22-23).  

---

### **5. Model Selection: Cross-Validation**  
**Slide 25-27**  
**Problem**: How to choose \( M \) or \( \lambda \) without touching the test set?  
**Solution**: **K-Fold Cross-Validation** (e.g., \( K=10 \)):  
1. Split training data into \( K \) folds.  
2. Train on \( K-1 \) folds, validate on the remaining fold.  
3. Repeat \( K \) times and average validation errors.  

**Why?**: Mimics test error without using the actual test data.  

**Leave-One-Out (LOO)**: Extreme case where \( K = N \) (each point is a validation set). Accurate but computationally expensive.  

---

### **6. Key Takeaways**  
1. **Bias-Variance Tradeoff**:  
   - Simple models → High bias (underfit).  
   - Complex models → High variance (overfit).  
2. **Regularization**: Balances this tradeoff via \( \lambda \).  
3. **Validation**: Use cross-validation to tune hyperparameters (\( M \), \( \lambda \)).  

---

### **7. Common Exam Questions**  
1. **"Why does \( M=9 \) overfit?"**  
   - Answer: It has too many parameters tuned to noise in a small dataset.  
2. **"How does L2 regularization help?"**  
   - Answer: It shrinks weights, reducing model flexibility.  
3. **"Why can’t we use test error to choose \( M \)?"**  
   - Answer: It leaks information, breaking generalization.  

---

### **8. Python Snippet: Polynomial Regression**  
```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score

# Generate data (like slides)
x = np.linspace(0, 1, 10)
t = np.sin(2 * np.pi * x) + np.random.normal(0, 0.1, 10)

# Fit polynomial (M=9)
model = make_pipeline(PolynomialFeatures(9), LinearRegression())
model.fit(x[:, None], t)

# Cross-validation (K=5)
scores = cross_val_score(model, x[:, None], t, cv=5, scoring='neg_mean_squared_error')
print("CV MSE:", -scores.mean())
```

---

### **Next Steps**  
1. **Clarify**: Which concept needs deeper explanation? (e.g., gradient descent for \( \min E(w) \))  
2. **Practice**: Want to derive the optimal \( w \) for SSE? Or code regularized regression?  
3. **Exam Prep**: Need mock questions on bias-variance tradeoff?  

# Wk 2

### **1. Key Theme: Uncertainty in Machine Learning**  
**Slides 3-4**  
- **Why Uncertainty Matters**:  
  - Real-world data is noisy (e.g., sensor errors, human variability).  
  - Finite datasets → models can’t capture true data distribution perfectly.  
- **Example**:  
  Polynomial regression $y = w_0 + w_1x + \dots + w_9x^9$  
  - Different training sets \( D \) → different \( w \) → uncertain predictions.  

**Analogy**:  
Predicting coin flips with a biased coin. Even after 10 flips, can we be *certain* about the true bias?  

---

### **2. Probability Theory Basics**  
**Slides 6-12**  
#### **Random Variables & Distributions**  
- **Coin Toss Example**:  
  - Let \( X \) be a random variable: $( X \in \{H, T\} )$.  
  - Probability distribution: $( p(X=H) = w ), ( p(X=T) = 1-w ).$  
  - Rules: \( 0 \leq w \leq 1 \), \( p(H) + p(T) = 1 \).  

#### **Joint & Conditional Probability**  
- **Joint Probability**: \( p(X=H, Y=T) \) = probability of Head first *and* Tail second.  
- **Conditional Probability**: \( p(Y=T|X=H) \) = probability of Tail second *given* Head first.  

#### **Bayes’ Theorem**  
$$
p(y|x) = \frac{p(x|y)p(y)}{p(x)}
$$
**Use Case**:  
- Update beliefs (posterior $p(y|x)$ after observing data (likelihood $p(x|y)$  

**Example**:  
- **Prior \( p(w) \)**: Believe coin is fair (\( w=0.5 \)).  
- **Likelihood \( p(D|w) \)**: Observe 3 Heads in 10 flips.  
- **Posterior \( p(w|D) \)**: Revise belief about \( w \) (more weight to \( w \approx 0.3 \)).  

---

### **3. Maximum Likelihood Estimation (MLE)**  
**Slides 17-24**  
#### **Goal**: Find parameter \( w \) that maximizes the likelihood of observed data \( D \).  
- **Likelihood Function**: \( p(D|w) = w^3(1-w)^7 \) (for 3 Heads, 7 Tails).  
- **Log-Likelihood Trick**: Maximize $\ln p(D|w) = 3\ln w + 7\ln(1-w)$ (converts products to sums).  

#### **Derivation**:  
1. Take derivative w.r.t. \( w \):  
   $$
   \frac{d}{dw} [3\ln w + 7\ln(1-w)] = \frac{3}{w} - \frac{7}{1-w} = 0$$
2. Solve: $w = 0.3$.  

**Intuition**:  
MLE picks the coin bias \( w \) that makes the observed data *most probable*.  

**Python Code**:  
```python
import numpy as np
w = np.linspace(0, 1, 100)
likelihood = (w**3) * ((1-w)**7)
best_w = w[np.argmax(likelihood)]  # Output: 0.3
```

---

### **4. Bootstrap: Quantifying Uncertainty**  
**Slides 25-26**  
#### **Problem**: How does \( w \) vary if we resample data?  
- **Bootstrap Method**:  
  1. Resample \( D \) with replacement (e.g., create 1000 fake datasets \( D'_1, \dots, D'_{1000} \)).  
  2. Compute \( w \) for each \( D'_i \).  
  3. Plot histogram of \( w \) → reveals uncertainty.  

**Example**:  
- Original \( D \): [H, T, T, T, H, T, T, H, T] → \( w = 0.3 \).  
- Bootstrap sample \( D'_1 \): [H, H, T, T, T, H, T, T, T] → \( w_1 = 0.33 \).  

**Why Use It?**:  
- No need for more data; reuse existing data to estimate confidence intervals.  

**Resource**:  
- [StatQuest Bootstrap Video](https://www.youtube.com/watch?v=Xz0x-8-cgaQ).  

---

### **5. Bayesian vs. Frequentist Views**  
**Slide 2, 5**  
- **Frequentist (MLE)**:  
  - \( w \) is fixed; estimate it from data.  
  - Example: \( w = 0.3 \) from MLE.  
- **Bayesian**:  
  - \( w \) is random; assign it a prior (e.g., \( p(w) = \text{Beta}(2,2) \)), update to posterior \( p(w|D) \).  
  - Predict using posterior mean: $\mathbb{E}[w|D]$.  

**When to Use Which?**:  
- MLE: Simpler, works well with large data.  
- Bayesian: Incorporates prior knowledge, handles small data better.  

---

### **6. Exam/Assignment Focus Areas**  
1. **Derive MLE for Bernoulli/Binomial** (e.g., coin flips).  
2. **Interpret Bayes’ Theorem** in a classification problem.  
3. **Bootstrap Application**: How to estimate confidence intervals for model parameters?  

**Practice Problems**:  
1. Given \( D = [H, H, T, H] \), find MLE for \( w \).  
   - *Solution*: $w = \frac{3}{4}$  
1. Compute $p(w|D) \ \ \text{if prior is}  \ \\text{Beta}(1,1)$ (uniform).  

---

### **7. Further Learning Resources**  
1. **Probability Theory**:  
   - [MIT 6.431 Probability](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/)  
2. **Bayesian Methods**:  
   - [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)  
3. **Coding MLE/Bootstrap**:  
   - [Scikit-learn Bootstrap](https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html)  

---

### **Key Takeaways**  
- **Uncertainty** is fundamental (noise, finite data).  
- **MLE**: Finds "best" parameters given data.  
- **Bayesian**: Updates beliefs with data.  
- **Bootstrap**: Estimates uncertainty by resampling.  

**Next Steps**:  
1. Want to derive MLE for Gaussian distributions?  
2. Need a Python demo for Bayesian coin flips?  
3. Clarify any slide (e.g., joint vs. conditional probability)?  

---
---

### **1. Core Idea: Bayesian vs. Frequentist Approaches**  
**Slide 1-4**  
- **Frequentist (MLE)**:  
  - Treats parameters (e.g., coin bias \( w \)) as fixed.  
  - Estimates \( w \) by maximizing likelihood \( p(D|w) \).  
  - **Limitation**: Fails with small data (e.g., \( D = \{T, T\} \) → \( w_{MLE} = 0 \), implying "coin never lands Heads!").

- **Bayesian**:  
  - Treats \( w \) as random with a **prior distribution** \( p(w) \) (e.g., "coins are usually fair").  
  - Updates prior to **posterior** \( p(w|D) \) using Bayes’ Theorem:  
    $$
    p(w|D) = \frac{p(D|w)p(w)}{p(D)} \propto \text{Likelihood} \times \text{Prior}
    $$  
  - **Key Advantage**: Incorporates prior knowledge and quantifies uncertainty.

**Analogy**:  
- **Frequentist**: Guessing a coin’s bias *only* from observed flips.  
- **Bayesian**: Starting with a belief ("probably fair"), then updating it with data.

---

### **2. Bayesian Workflow**  
**Slide 4, 9-10**  
1. **Choose Prior**: \( p(w) \) (e.g., Beta distribution for coin flips).  
2. **Compute Likelihood**: \( p(D|w) \) (same as MLE).  
3. **Calculate Posterior**:  
   $$
   p(w|D) \propto w^{|H|+a-1}(1-w)^{|T|+b-1} \quad \text{(Conjugate prior: Beta)}
   $$
4. **Predict**: Integrate over all \( w \) (Slide 11-13):  
   $$
   p(H|D) = \int w \cdot p(w|D) \, dw = \mathbb{E}[w|D] = \frac{|H| + a}{|H| + |T| + a + b}
   $$  

**Example**:  
- Prior: \( \text{Beta}(a=2, b=2) \) (weak belief in fairness).  
- Data: \( D = \{H, T, T\} \) → \( |H|=1, |T|=2 \).  
- Posterior: \( \text{Beta}(a'=3, b'=4) \).  
- Prediction: \( p(H|D) = \frac{3}{7} \approx 0.43 \) (vs. MLE’s \( \frac{1}{3} \)).  

**Python Code**:  
```python
from scipy.stats import beta
a, b = 2, 2  # Prior
heads, tails = 1, 2  # Data
posterior_mean = (heads + a) / (heads + tails + a + b)  # Output: 0.428
```

---

### **3. Choosing the Prior: Beta Distribution**  
**Slide 7-8**  
- **Beta Distribution**: \( \text{Beta}(w|a, b) \propto w^{a-1}(1-w)^{b-1} \).  
  - \( a-1 \): "Pseudo-counts" of Heads.  
  - \( b-1 \): "Pseudo-counts" of Tails.  
- **Interpreting Parameters**:  
  - \( a=b=1 \): Uniform prior ("no idea").  
  - \( a=b=100 \): Strong prior ("coin is very likely fair").  

**Example**:  
- Prior: \( \text{Beta}(a=100, b=100) \), Data: \( D = \{T, T\} \).  
- \( p(H|D) = \frac{0+100}{2+200} \approx 0.495 \) (still near 0.5, unlike MLE’s 0).  

**Visualization**:  
<img src="https://miro.medium.com/v2/resize:fit:1400/1*FZW1Z5yGln8H0k7RQIqQ8A.png" width="400" alt="Beta distributions">  

---

### **4. Predictive Distribution**  
**Slide 11-13**  
- **Goal**: Predict next outcome \( H \) given data \( D \).  
- **Bayesian Approach**: Average over all possible \( w \):  
$$
  p(H|D) = \int p(H|w)p(w|D) \, dw = \mathbb{E}[w|D]
  $$
- **Result**: Posterior mean \( \frac{|H|+a}{|H|+|T|+a+b} \).  

**Why This Matters**:  
- Avoids overconfidence (e.g., MLE’s \( p(H) = 0 \) after \( \{T, T\} \)).  
- Balances data and prior belief.  

**Exam Tip**:  
- Expect questions like: *"Compute \( p(H|D) \) given prior \( \text{Beta}(a,b) \) and data \( D \)."*  

---

### **5. Key Takeaways**  
1. **Bayesian vs. MLE**:  
   - MLE: \( w = \frac{|H|}{|H|+|T|} \).  
   - Bayesian: \( p(H|D) = \frac{|H|+a}{|H|+|T|+a+b} \).  
2. **Prior Strength**:  
   - Large \( a+b \) → Prior dominates.  
   - Small \( a+b \) → Data dominates.  
3. **Conjugate Priors**: Beta for binomial likelihood (coin flips), Gaussian for Gaussian.  

---

### **6. Practice Problems**  
1. **Problem**: Given prior \( \text{Beta}(a=5, b=5) \) and data \( D = \{H, H, T\} \), find \( p(H|D) \).  
   - *Solution*: \( \frac{2+5}{3+10} = \frac{7}{13} \approx 0.54 \).  

2. **Derivation**: Show that \( p(w|D) \) is Beta if prior is Beta and likelihood is binomial.  

---

### **7. Further Resources**  
1. **Textbooks**:  
   - *Pattern Recognition and Machine Learning* (Bishop), Chapter 2.  
2. **Visual Guides**:  
   - [Bayesian Coin Flips Interactive](https://seeing-theory.brown.edu/bayesian-inference/index.html).  
3. **Coding**:  
   - [PyMC3 Tutorial](https://docs.pymc.io/en/v3/) (for Bayesian modeling in Python).  

---

### **Next Steps**  
1. **Clarify**: Any confusion about priors/posteriors?  
2. **Code**: Want a PyTorch example for Bayesian logistic regression?  
3. **Exam Prep**: Need more practice problems?  

---
---
# Wk 3

---


---
---

# Wk 4

![[Pasted image 20250720112344.png]]

![[Pasted image 20250720112420.png]]Understood, Emeka! Let’s break down **Week 5’s Bias-Variance Analysis** into intuitive concepts, practical implications, and actionable strategies for improving your models. I’ll use analogies, examples, and connect this to your assignments/exams.

---

### **1. Core Concepts: Bias & Variance**  
**Slides 1-11**  
#### **Definitions**:  
- **Bias**: Average prediction error across models trained on different datasets.  
  - *High Bias*: Model consistently underfits (e.g., linear model for `sin(x)`).  
  - *Low Bias*: Complex models (e.g., high-degree polynomials) fit training data closely.  
- **Variance**: Variability in predictions for a given input across different models.  
  - *High Variance*: Small data changes cause large prediction swings (overfitting).  

**Analogy**:  
- **Bias**: Like an archer consistently missing the bullseye in the same direction.  
- **Variance**: Like an archer whose arrows are scattered wildly around the target.  

**Graphical Representation (Slide 9)**:  
<img src="https://scott.fortmann-roe.com/docs/docs/BiasVariance/biasvariance.png" width="400" alt="Bias-Variance Tradeoff">  

---

### **2. Mathematical Formulation**  
**Slides 12-20**  
#### **Generalization Error Decomposition**:  
\[
\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}
\]  
- **Bias²**: \( \mathbb{E}[\bar{y}(x) - h(x)]^2 \) (average prediction vs true value).  
- **Variance**: \( \mathbb{E}[(y(x) - \bar{y}(x))^2] \) (prediction variability).  

**Example (Slide 20)**:  
For Ridge regression with regularization parameter \( \lambda \):  
- Small \( \lambda \) → Low Bias, High Variance (overfitting).  
- Large \( \lambda \) → High Bias, Low Variance (underfitting).  

**Key Insight**:  
The "sweet spot" minimizes total error (Slide 23):  
<img src="https://i.imgur.com/JZkRk7H.png" width="400" alt="Bias-Variance Tradeoff Curve">  

---

### **3. Practical Implications**  
**Slides 25-36**  
#### **Model Complexity & Regularization**:  
- **Underfitting (High Bias)**:  
  - Symptoms: High training *and* test error.  
  - Fixes: Use more complex models (e.g., higher-degree polynomials), reduce regularization.  
- **Overfitting (High Variance)**:  
  - Symptoms: Low training error but high test error.  
  - Fixes: Simplify models, increase regularization, get more data.  

**Quantitative Example (Slide 36)**:  
| Model          | Bias   | Variance | MSE    |  
|----------------|--------|----------|--------|  
| 0-order Poly   | 0.9117 | 0.0052   | 1.0361 |  
| 3-order Poly   | 0.0039 | 0.0028   | 0.1032 |  
| 15-order Poly  | 0.0008 | 0.0121   | 0.1069 |  

**Takeaway**: 3rd-order polynomial strikes the best balance.  

---

### **4. Bootstrap for Estimating Variance**  
**Slides 4-6**  
#### **Method**:  
1. Resample data with replacement to create "fake" datasets.  
2. Train models on each resampled dataset.  
3. Compute variance of predictions.  

**Why It Matters**:  
- Simulates having multiple datasets to estimate model stability.  

**Python Example**:  
```python
from sklearn.utils import resample
import numpy as np

# Original data
X = np.linspace(0, 1, 100)
y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.1, 100)

# Bootstrap
predictions = []
for _ in range(100):
    X_resampled, y_resampled = resample(X, y)
    model.fit(X_resampled, y_resampled)
    predictions.append(model.predict(X_test))

variance = np.var(predictions, axis=0)
```

---

### **5. Actionable Strategies**  
**Slide 37**  
#### **Diagnosing Your Model**:  
- **High Bias?** (Poor training performance)  
  - Increase model complexity.  
  - Add more features.  
  - Decrease regularization (\( \lambda \)).  
- **High Variance?** (Poor test performance)  
  - Use simpler models.  
  - Increase regularization.  
  - Get more training data.  

**Exam Tip**:  
- Expect questions like: *"Given a learning curve (training vs test error), identify bias/variance issues."*  

---

### **6. Key Takeaways**  
1. **Tradeoff**: Simpler models → High Bias, Low Variance; Complex models → Low Bias, High Variance.  
2. **Regularization**: Controls this tradeoff (Slide 31-34).  
3. **Bootstrap**: Quantifies variance without needing new data.  

---

### **7. Further Resources**  
1. **Visualization**: [Bias-Variance Tradeoff Interactive](https://mlu-explain.github.io/bias-variance/).  
2. **Textbook**: *ESL* (Hastie et al.), Chapter 7.  
3. **Coding**: [Scikit-learn Learning Curves](https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html).  

---

### **Next Steps**  
1. **Clarify**: Any confusion about the bias-variance decomposition?  
2. **Practice**: Want to simulate bias/variance with Python?  
3. **Exam Prep**: Need more examples of diagnosing model issues?  

# wk 5

# Machine Learning Lecture: Classification and Perceptrons  
  
Let me break down these lecture slides into clear, understandable concepts about classification and perceptrons in machine learning.  
  
## 1. Regression vs. Classification  
  
### Regression  
- **Purpose**: Predict continuous values  
- **Examples**:  
  - House price prediction  
  - Exam score prediction  
  - Movie ticket sales prediction  
- **Model output**: A real number (e.g., $425,000 for a house)  
  
### Classification  
- **Purpose**: Assign inputs to discrete categories  
- **Examples**:  
  - Spam vs. not spam emails  
  - Credit risk assessment (good/bad/grey)  
  - Handwritten digit recognition (0-9)  
- **Model output**: A class label (e.g., "spam")  
  
## 2. Linear Classifiers  
  
### Key Components  
1. **Decision Boundary**: The surface that separates classes  
   - For 2D: A line (e.g., `3x₁ + 2x₂ - 250 = 0`)  
   - For 3D: A plane  
   - For higher dimensions: A hyperplane  
  
2. **Weight Vector (w)**: Determines the orientation of the boundary  
3. **Bias Term (w₀)**: Shifts the boundary away from the origin  
  
### Example: Fish Classifier  
Given features:  
- x₁: Length  
- x₂: Width  
  
Decision function:  
```  
f(x) = 3x₁ + 2x₂ - 250  
```  
Classification rule:  
- If f(x) ≥ 0 → Tuna  
- Else → Bass  
  
## 3. Multiclass Classification Strategies  
  
### 1. One-vs-Rest (OvR)  
- Train K binary classifiers (one per class)  
- Each classifier separates one class from all others  
- **Problem**: Ambiguous regions where multiple classifiers claim ownership  
  
### 2. One-vs-One (OvO)  
- Train K(K-1)/2 binary classifiers (all pairs)  
- Classify by majority vote  
- **Problem**: Still has ambiguous regions  
  
### 3. Multiclass Discriminant (Recommended)  
- Train K discriminant functions: yₖ(x) = wₖᵀx + wₖ₀  
- Assign to class with highest yₖ(x) value  
- Decision boundary between classes k and j: yₖ(x) = yⱼ(x)  
  
## 4. The Perceptron Algorithm  
  
### What is a Perceptron?  
- The simplest neural network unit  
- A linear binary classifier with step activation:  
  ```  
  f(a) = +1 if a ≥ 0  
         -1 otherwise  
  ```  
  
### Learning Algorithm  
1. Initialize weights randomly  
2. For each training example (x, t):  
   - Compute prediction: y = sign(wᵀx)  
   - If y ≠ t (misclassified):  
     - Update weights: w ← w + η·t·x  
     (η = learning rate, typically η=1)  
  
### Example Walkthrough  
Initial weights: w = [-0.3, 0.6]  
  
| Point | x₁  | x₂  | True (t) | wᵀx | Prediction (y) | Update? | New w |  
|-------|-----|-----|----------|-----|----------------|---------|-------|  
| A     | 0.6 | 0.5 | -1       | 0.12| +1 (wrong)      | Yes     | [-0.9, 0.1] |  
| B     | -0.3| -0.2| +1       | 0.25| +1 (correct)    | No      | - |  
| C     | -0.1| -0.6| -1       | 0.03| +1 (wrong)      | Yes     | [-0.8, 0.7] |  
| D     | -0.5| 0.3 | +1       | 0.61| +1 (correct)    | No      | - |  
| A     | 0.6 | 0.5 | -1       | -0.13| -1 (correct)   | No      | - |  
  
**Converged** after one full pass with all correct!  
  
### Important Properties  
- **Guaranteed convergence** if data is linearly separable  
- **May not converge** for non-separable data  
- **Sensitive to initialization** and example order  
  
## 5. Multiclass Perceptron  
  
Extension to K classes:  
1. Maintain K weight vectors (one per class)  
2. For each example (xₙ, tₙ):  
   - Predict: yₙ = argmaxₖ(wₖᵀxₙ)  
   - If yₙ ≠ tₙ (wrong):  
     - Decrease wrong class: w_{yₙ} ← w_{yₙ} - ηxₙ  
     - Increase true class: w_{tₙ} ← w_{tₙ} + ηxₙ  
  
## Key Takeaways  
  
1. **Linear classifiers** make decisions based on whether wᵀx + w₀ is positive/negative  
2. The **perceptron** is a simple but powerful learning algorithm for finding these weights  
3. For multiclass problems, we can either:  
   - Use multiple binary classifiers (OvR/OvO)  
   - Directly learn K discriminant functions (recommended)  
4. Perceptrons are **foundational** - they lead to more complex neural networks