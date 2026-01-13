# *Key Theme: Uncertainty in Machine Learning*
**Slides 3-4**  
# Why Uncertainty Matters 
  - Real-world data is noisy (e.g., sensor errors, human variability).  
  - We are unsure if models have captured true data distribution correctly.  
- **Example**:  
  Polynomial regression $y = w_0 + w_1x + \dots + w_9x^9$  
  - Different training sets \( D \) → different \( w \) → uncertain predictions.  

---

# 2. Probability Theory Basics  
**Slides 6-12**  
## Random Variables & Distributions
- **Coin Toss Example**:  
  - Let \( X \) be a random variable: \( X \in \{H, T\} \).  
  - Probability distribution: \( p(X=H) = w \), \( p(X=T) = 1-w \).  
  - Rules: \( 0 \leq w \leq 1 \), \( p(H) + p(T) = 1 \).  

## Definition of Probability

***Machine Learning Definition***
$$\sum_{a\in D_{X}} Pr(X=a)=1\  and\ \forall a \in D_x:Pr(X=a) \ge 0 $$

## Joint & Conditional Probability
- **Joint Probability**: \( p(X=H, Y=T) \) = probability of Head first *and* Tail second.:
$$
\sum{_x}\sum{_y}\ \ P(x,y)=1
$$

- **Conditional Probability**: \( p(Y=T|X=H) \) = probability of Tail second *given* Head first.  

- **Independence** 
$$
P(X=H,Y=H) = P(X=H)P(Y=H)
$$
		***Why is it important??***
			1. *Training and test data must be independent samples from the same distribution.*	
			2. reduces computational complexity
			ex: spam filter treats word occurrences as independent (they have their own vals) to make them easier to calculate.
			3. Give non-redundant information
			ex: feature selection *relies* on independ fetures
			4. Core math theorems *assume* it
			Naive Bayes
			Gaussian processes
			Hidden Markov Chains

## Bayes’ Theorem
$$
p(y|x) = \frac{p(x|y)p(y)}{p(x)}
$$
#### When to use:  
- Update beliefs (posterior $p(y|x)$ after observing data (likelihood $p(x|y)$  

**Example**:  
- **Prior \( p(w) \)**: Believe coin is fair (\( w=0.5 \)).  
- **Likelihood \( p(D|w) \)**: Observe 3 Heads in 10 flips.  
- **Posterior \( p(w|D) \)**: Revise belief about \( w \) (more weight to \( w \approx 0.3 \)).  

---

### **3. Maximum Likelihood Estimation (MLE)**  
**Slides 17-24**  
#### Our Goal is to find parameter \( w \) that maximizes the likelihood of observed data \( D \).  
- **Likelihood Function**:  $$
( p(D|w) = w^3(1-w)^7 ) (for\ 3 \  Heads, \ 7 \ Tails).  
 $$
- **Finding w using Log Transformation**: 
	find a w that Maximizes: 
		***Converts products to sums***
	$$
	\ln p(D|w) = 3\ln w + 7\ln(1-w)  
$$
### Derivation
##### ==QUESTION: Are the slides suggesting that the method is not an effective one in finding maximum likelihood of w?== 

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
  2. Compute \( w \) for each D'.  
  3. Plot histogram of \( w \) → reveals uncertainty.  

**Example**:  
- Original \( D \): [H, T, T, T, H, T, T, H, T] → \( w = 0.3 \).  
- Bootstrap sample \( D'_1 \): [H, H, T, T, T, H, T, T, T] → \( w_1 = 0.33 \).  

**Why Use It?**:  
- No need for more data; reuse existing data to estimate confidence intervals:

 ![[Screenshot from 2025-08-04 16-32-32.png]]![[Pasted image 20250804163355.png]]

---

## 5. Bayesian vs. Frequentist Views
**Slide 2, 5**  
### Frequentist (MLE)
  - \( w \) is fixed; estimate it from data.  
  - Example: \( w = 0.3 \) from MLE.  
### Bayesian  
  - \( w \) is random; assign it a prior (e.g., \( p(w) = \text{Beta}(2,2) \)), update to posterior \( p(w|D) \).  
  - Predict using posterior mean: $\mathbb{E}[w|D]$.  

#### When to Use Which?
- *MLE*: Simpler, works well with *large data*.  
- *Bayesian*: Incorporates prior knowledge, handles *small data* better.  

---

### **6. Exam/Assignment Focus Areas**  
1. **Derive MLE for Bernoulli/Binomial** (e.g., coin flips).  
2. **Interpret Bayes’ Theorem** in a classification problem.  
3. **Bootstrap Application**: How to estimate confidence intervals for model parameters?  

**Practice Problems**:  
1. Given \( D = (H, H, T, H) \), find MLE for \( w \).  
   - *Solution*: $w = \frac{3}{4}$  
1. Compute $p(w|D) \ \ \text{if prior is}  \ \\text{Beta}(1,1)$ (uniform).  
##### 7. Further Learning Resources  
1. Probability Theory:  
   - [MIT 6.431 Probability](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-041-probabilistic-systems-analysis-and-applied-probability-fall-2010/)  
1. Bayesian Methods:  
   - [Bayesian Methods for Hackers](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers)  
1. Coding MLE/Bootstrap:  
   - [Scikit-learn Bootstrap](https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html)  


---
---

# 3. Bayesian vs. Frequentist Approaches
**Slide 1-4**  
## Frequentist (MLE)
  - Treats parameters (e.g., coin bias \( w \)) as fixed.  
  - Estimates \( w \) by maximizing likelihood \( p(D|w) \).  
  - **Limitation**: Fails with small data (e.g., \( D = \{T, T\} \) → \( w_{MLE} = 0 \), implying "coin never lands Heads!").

## Bayesian
  - Treats \( w \) as random with a **prior distribution** \( p(w) \) (e.g., "coins are usually fair").  
  - Updates prior to **posterior** \( p(w|D) \) using Bayes’ Theorem:  
    $$
    p(w|D) = \frac{p(D|w)p(w)}{p(D)} \propto \text{Likelihood} \times \text{Prior}
    $$  
  - ***Key Advantage***: Incorporates prior knowledge and quantifies uncertainty.

#### Analogy:  
- **Frequentist**: Guessing a coin’s bias *only* from observed flips.  
- **Bayesian**: Starting with a belief ("probably fair"), then updating it with data.

---

## Bayesian Workflow
**Slide 4, 9-10**  
### Step 1: Choose Prior 
\( p(w) \) (e.g., Beta distribution for coin flips).  
### Step 2: Compute Likelihood
\( p(D|w) \) (same as MLE).  
### Step 3: Calculate Posterior  
   $$
   p(w|D) \propto w^{|H|+a-1}(1-w)^{|T|+b-1} \quad \text{(Conjugate prior: Beta)}
   $$
### Step 4: Predict
Integrate over all \( w \) (Slide 11-13):  

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

## Encoding Prior Knowledge: Beta Distribution
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
## 6. Practice Problems
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
