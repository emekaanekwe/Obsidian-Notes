# Machine Learning Week 2: Probabilistic Machine Learning - Detailed Notes

## Part 1: Introduction to Probabilistic Machine Learning

---

## 1. Why Probabilistic Machine Learning?

### **Core Concept**

Probabilistic ML deals with **uncertainty** and uses **probability** to make predictions and decisions.

### **Real-World Applications**

**Example 1: Search Autocomplete**

- When you type "Monash" in Google search
- System predicts likely completions: "Monash University", "Monash Clayton campus"
- Based on **probability distributions** of what users typically search

**Example 2: Large Language Models (LLMs)**

- Generative AI is fundamentally **probabilistic**
- Predicts next word based on probability distributions
- Given "deep learning is", predicts likely next words

### **Why Uncertainty Exists**

**Key Insight**: In reality, we face uncertainty every day

- If we knew everything → no need for ML (could find exact underlying functions)
- Real data contains:
    - **Bias**: Systematic errors in data collection
    - **Noise**: Random variations/anomalies
    - **Missing information**: Incomplete observations

**Goal of ML**: Find patterns despite noise and uncertainty

---

## 2. Describing Uncertainty with Probability

### **Probability as a Tool**

**Definition**: Probability quantifies how likely an event is to occur

**Weather Example**:

- "50% chance of rain today"
- "5% chance of rain tomorrow"

Higher percentage = more likely to occur

### **Probability Range**

$$P(X) \in [0, 1]$$

Where:

- $P(X) = 0$: Event will definitely NOT occur
- $P(X) = 1$: Event will definitely occur
- $P(X) = 0.5$: Event has equal chance of occurring or not

**Important**: Probability can NEVER exceed 1 or be negative

---

## 3. Uncertainty and Model Generalization

### **Small Dataset Problem**

**Overfitting Risk**:

- With small dataset → model tries to fit every single data point
- Captures noise as if it were signal
- Poor generalization to new data

**Visualization**:

```
Small dataset: Model fits every point perfectly (overfitting)
    •
   /|\    Model curves through every noise point
  • • •
```

### **Larger Dataset with Noise**

**Benefits**:

- More diverse examples (use cases)
- Model learns to generalize despite noise
- Better balance → avoids overfitting
- Performs better on unseen data

**Key Takeaway**: Sometimes noise/uncertainty actually **helps** create better generalization!

---

## 4. Probability Theory Fundamentals

### Why Probability Theory?
***Describes uncertainty**:* 
Natural language for unknown outcomes
 ***Foundation for models***: 
 Many ML models (e.g., Naive Bayes) are purely probabilistic

---

#### Use of Bayes' Theorem (Preview)

**Formula**: $$P(\text{class}|\text{data}) = \frac{P(\text{data}|\text{class}) \times P(\text{class})}{P(\text{data})}$$

**Components**:

- **Posterior probability**: $P(\text{class}|\text{data})$ - what we want to find
- **Likelihood**: $P(\text{data}|\text{class})$ - similarity measure
- **Prior probability**: $P(\text{class})$ - initial belief before seeing data
- **Evidence**: $P(\text{data})$ - probability of observing the data

**All components are probabilities!**

---

### Basic Probability Example: The Coin Toss

#### Prior Knowledge (Before Experiment)

 *What's the probability of getting heads when tossing a fair coin?*

**Common sense answer**: 
P(H) = 0.5 = 50% P(T) = 0.5 = 50%

This is our ***prior probability*** - belief before any experiments

---

#### When Prior Knowledge Might Be Wrong

**What if the coin is damaged or biased?**

Wrong prior → wrong model predictions
If Prior is wrong → Posterior will likely be wrong
- Prior affects posterior calculation in Bayes' theorem

***Prior probability is crucial for model training***

**Formula relationship**: $$\text{Posterior} = f(\text{Likelihood}, \text{Prior}, \text{Evidence})$$



---

### Key Terminology and Notation

#### ⚠️ IMPORTANT: Notation Matters!

Pay close attention to capitalization and symbols - they have specific meanings!

---

##### Random Variable: $X$

Denotes the *outcome* of a random process, where the *Domain* of $X$ is The *set of all possible outcomes*

**Coin toss exa
$$\text{Domain}(X) = {H, T}$$
mple**: $$X \in {H, T}$$
Where:
- $X$ = random variable (the outcome)
- $H$ = heads
- $T$ = tails
---

##### Dataset: $D$ (capital D)

Collection of *observed outcomes*

**Coin toss example** (10 tosses): $$D = {H, T, H, H, T, T, H, T, H, T}$$

**Properties**:

- Capital $D$ = entire dataset
- Contains actual experimental results
- Used for training the model

---

##### Probability Distribution: $p$ (lowercase p)

*Function* that *describes the probability of each outcome*

**Notation**: $$p(X = H)$$

**Meaning**: "What is the probability that $X$ equals heads?"

**Example**: $$p(X = H) = 0.5$$ $$p(X = T) = 0.5$$

---

### Fundamental Probability Rules

#### Rule 1: Probability Bounds

$$0 \leq p(X) \leq 1 \quad \forall X$$

---

#### Rule 2: The 100% Rule (Normalization)

set of all possible outcomes must equal 1:
$$\sum_{\text{all possible outcomes}} p(X) = 1$$



**Coin toss example**: $$p(X=H) + p(X=T) = 1$$ $$0.5 + 0.5 = 1 \quad \checkmark$$

---

#### Why Normalization Matters

Raw model outputs might not sum to 1, so we need to *convert the scores/values into proportions of a whole*

**Example** (3-class classification):

- Class 1: 0.6
- Class 2: 0.8
- Class 3: 0.4
- **Sum**: 1.8 (violates 100% rule!)

##### Normalization Formula
class 1 + class 2 + class 3 = N

**Visual representation**: $$p(C_1) = \frac{P_{\text{raw}}(C_1)}{P_{\text{raw}}(C_1) + P_{\text{raw}}(C_2) + P_{\text{raw}}(C_3)}$$

**Example calculation**: $$p(C_1) = \frac{0.6}{0.6 + 0.8 + 0.4} = \frac{0.6}{1.8} = 0.333$$ $$p(C_2) = \frac{0.8}{1.8} = 0.444$$ $$p(C_3) = \frac{0.4}{1.8} = 0.222$$

**Verify**: $0.333 + 0.444 + 0.222 = 0.999 \approx 1$ ✓

---

### Events vs Random Variables (Notation)

**Examples**:

- $X$ = "outcome of first coin toss"
- $Y$ = "outcome of second coin toss"
- $Z$ = "outcome of first two tosses"
### Computing Probabilities from Data

**Formula**: $$p(X = x) = \frac{\text{count}(x)}{\text{total observations}}$$

**Example 1**: 10 tosses, 5 heads, 5 tails 
$p(X=H) = \frac{5}{10} = 0.5$
**Example 2:** 10 tosses, 4 heads, 6 tails 
$p(X=H) = \frac{4}{10} = 0.4$

---

### Conditional Probability Notation

**Computing $Z$ given $X$**:

$$p(Z|X)$$

**Meaning**: "Probability of $Z$ given that we already know $X$"

**Why this matters**:

- During training, we have historical data (experiments already done)
- Use known outcomes to predict future probabilities
- Based on observed data, not just theory

---

### Joint Probability

Probability that *multiple events ALL occur*
$$p(X = H \text{ AND } Y = H)$$
Or more compactly: $$p(X=H, Y=H)$$
### **Coin Toss Example**

$Z$ = event of first two tosses

**Specific outcome**: Both heads $$p(Z = HH) = p(X=H \text{ AND } Y=H)$$

**This is a joint distribution!**

---

### Chain Rule for Joint Probability

**Formula**: $$p(X, Y) = p(Y|X) \times p(X)$$
**In words**: $$P(\text{both events}) = P(\text{second}|\text{first}) \times P(\text{first})$$

**Example**: $$p(X=H, Y=H) = p(Y=H|X=H) \times p(X=H)$$
If tosses are independent: $$= 0.5 \times 0.5 = 0.25$$

---

#### Summing Joint Probabilities

**100% Rule still applies**: $$\sum_{\text{all outcomes}} p(X, Y) = 1$$

**All possible outcomes** for two tosses:

- $p(HH) + p(HT) + p(TH) + p(TT) = 1$
- $0.25 + 0.25 + 0.25 + 0.25 = 1$ ✓

---

### Independent Events

Events are independent if *the outcome of one doesn't affect the other*

---

#### Simplification for Independent Events

**If $X$ and $Y$ are independent**:

$$p(Y|X) = p(Y)$$

**Therefore**: $$p(X, Y) = p(Y|X) \times p(X) = p(Y) \times p(X)$$

***Joint probability of independent events = product of individual probabilities***

---

### **Example Calculation**

First toss is heads ($X=H$), what's probability second toss is heads?

**If independent**: $$p(Y=H|X=H) = p(Y=H) = 0.5$$

The first toss result is **irrelevant** - we just need $p(Y)$

**Joint probability**: $$p(Z=HH) = p(Y=H) \times p(X=H) = 0.5 \times 0.5 = 0.25$$

---

### Important Probability Rules Summary

#### Marginal Probability (Sum Rule)

$$p(X) = \sum_{y} p(X, Y=y)$$

**Sum over all possible values of $Y$ to get probability of $X$**

---

#### Product Rule

$$p(X, Y) = p(Y|X) \times p(X)$$

**Joint distribution** = conditional × marginal

---

#### Bayes' Theorem

$$p(Y|X) = \frac{p(X|Y) \times p(Y)}{p(X)}$$

**Components**:

|Symbol|Name|Meaning|
|---|---|---|
|$p(Y\|X)$|**Posterior**|What we want (after seeing data)|
|$p(X\|Y)$|**Likelihood**|Similarity between $X$ and $Y$|
|$p(Y)$|**Prior**|Initial belief (before data)|
|$p(X)$|**Evidence**|Probability of observing $X$|

---

### ⚠️ Don't Forget to Compute Normalization AFTER Posterior!

***After computing posterior probabilities, always normalize***:

$$p_{\text{normalized}}(C_i) = \frac{p(C_i)}{\sum_j p(C_j)}$$

***This ensures the 100% rule is satisfied!***

---

## Training a Probability Model

### Problem Setup

**Experiment**: Tossed coin 10 times independently

**Result**: Only 3 heads out of 10 (30%)

**Observation**: This is lower than expected 50%!

**Possible explanations**:

1. Coin is damaged/biased
2. Just bad luck (random variation)

**Goal**: Build a model to predict probability of heads in next toss

---

#### Model Parameters

**What we're looking for**:

$$w = p(H)$$

Where:

- $w$ = **parameter** we want to learn (probability of heads)
- This is our **training target**

**Why not just use 30%?**

- 30% is from limited experiment (10 tosses)
- Model tries to find **best estimate** of true probability
- Might be different from observed frequency

---

#### Probability of Tails

**Using 100% rule**:

$$p(T) = 1 - w$$

**Why?**: Only two outcomes, must sum to 1

---

#### Training Objective

**Notation**: $$p(D|w)$$

**Meaning**: "Probability of observing dataset $D$ given parameter $w$"

**What is $D$?**

- $D$ = all 10 observations
- $D = {H, T, H, T, T, T, T, H, T, H}$ (example)

**We want to find $w$ that maximizes** $p(D|w)$

***This is called Maximum Likelihood Estimation***

---

#### Formula for Dataset Probability

For 10 independent tosses:

$$p(D|w) = \prod_{i=1}^{10} p(x_i|w)$$

Where $x_i$ is the outcome of toss $i$

**Each toss contributes**:

- If $x_i = H$: contributes $w$
- If $x_i = T$: contributes $(1-w)$


---

## Maximum Likelihood Estimation (MLE) - Detailed Derivation

**Problem Recap**

We have:

- 10 coin tosses (dataset $D$)
- 3 heads, 7 tails observed
- Unknown parameter: $w = p(H)$
- Goal: Find best value of $w$

---

### Likelihood Function

**Probability of entire dataset given parameter $w$**:

$$p(D|w) = \prod_{i=1}^{10} p(x_i|w)$$

**For each toss**:

- If outcome is H: contributes $w$
- If outcome is T: contributes $(1-w)$

**Since we have 3 heads and 7 tails**:

$$p(D|w) = w \times w \times w \times (1-w) \times (1-w) \times (1-w) \times (1-w) \times (1-w) \times (1-w) \times (1-w)$$

**Simplified**: $$\boxed{p(D|w) = w^3 \times (1-w)^7}$$

Where:

- $w^3$: probability of getting 3 heads
- $(1-w)^7$: probability of getting 7 tails

---

### General Formula

For $n_H$ heads and $n_T$ tails in $N$ total tosses:

$$p(D|w) = w^{n_H} \times (1-w)^{n_T}$$

Where $n_H + n_T = N$

---

## Why Use Log-Likelihood?

### Problem with Direct Likelihood

Computing $p(D|w) = w^3(1-w)^7$ is difficult because:

- **Powers make derivatives messy**
- **Products of many terms** (imagine 100 tosses!)
- **Numerical underflow** (multiplying many small probabilities)

**Example**: What if we had 100 tosses? $$p(D|w) = w^{30}(1-w)^{70}$$

This is extremely hard to differentiate and compute!

---

### Solution: Log-Likelihood

**Apply logarithm** to the likelihood function:

$$\log p(D|w) = \log[w^3(1-w)^7]$$

**Using logarithm properties**:

$$\log(a \times b) = \log(a) + \log(b)$$ $$\log(a^n) = n\log(a)$$

**Apply these rules**:

$$\log p(D|w) = \log(w^3) + \log[(1-w)^7]$$

$$= 3\log(w) + 7\log(1-w)$$

**This is a LINEAR MODEL!**

$$\boxed{\log p(D|w) = 3\ln(w) + 7\ln(1-w)}$$

---

### Why Log-Likelihood Is Better

**Before (likelihood)**: $$p(D|w) = w^3(1-w)^7$$
- Polynomial/power function
- Hard to differentiate
- Numerical issues

**After (log-likelihood)**: $$\log p(D|w) = 3\ln(w) + 7\ln(1-w)$$

- **Linear in the logarithms**
- Much easier to differentiate
- Numerically stable

---

## Finding Maximum Likelihood Estimate

### Optimization Goal

Find $w$ that **maximizes** log-likelihood:

$$w_{ML} = \arg\max_w \log p(D|w)$$

$$= \arg\max_w [3\ln(w) + 7\ln(1-w)]$$

---

### **Step 1: Take Derivative**

**Partial derivative with respect to $w$**:

$$\frac{\partial}{\partial w}\log p(D|w) = \frac{\partial}{\partial w}[3\ln(w) + 7\ln(1-w)]$$

**Apply derivative rules**:

- $\frac{d}{dw}\ln(w) = \frac{1}{w}$
- $\frac{d}{dw}\ln(1-w) = -\frac{1}{1-w}$ (chain rule)

$$\frac{\partial \log p(D|w)}{\partial w} = 3 \cdot \frac{1}{w} + 7 \cdot \left(-\frac{1}{1-w}\right)$$

$$= \frac{3}{w} - \frac{7}{1-w}$$

---

### **Step 2: Set Derivative to Zero**

For maximum, set derivative equal to zero:

$$\frac{3}{w} - \frac{7}{1-w} = 0$$

---

### **Step 3: Solve for $w$**

$$\frac{3}{w} = \frac{7}{1-w}$$

**Cross multiply**: $$3(1-w) = 7w$$

$$3 - 3w = 7w$$

$$3 = 10w$$

$$\boxed{w_{ML} = \frac{3}{10} = 0.3}$$

---

### **The "Problem"**

**Result**: w = 0.3 = 30%

**But wait!** This is exactly what we observed in the experiment (3 heads out of 10)!

**Why did we do all this work?**

Student reaction: "I spent 10 minutes using advanced mathematics (partial derivatives) to get the same answer as just counting!"

---

### **What Went Wrong?**

**Answer**: **Training dataset is too small!**

**Key insight**:

- ***With small dataset (10 tosses), MLE just gives you the observed frequency***
- ***No generalization benefit***
- ***Risk of overfitting to limited data***

**Questions to ask**:

1. Do I have enough data?
2. Is my model overfitting?
3. How can I get more data?

---

## 16. Getting More Data

### Option 1: Generate Synthetic Data (Gaussian Distribution)

**Method**: Use Gaussian distribution to generate new data points

**Formula**: $$x \sim \mathcal{N}(\mu, \Sigma)$$
**Cons**:

- Need to define **mean** $\mu$ (don't know it!)
- Need to define **covariance matrix** $\Sigma$ (don't know it!)
- **Not guaranteed** to produce good quality data
- Quality depends heavily on initial values

**Problem**: If we don't know true parameters, generated data might be wrong!

---

### Option 2: Bootstrapping ✓

**Best solution for small datasets!**

**Key advantages**:

- Works with existing limited data
- No need to know underlying distribution
- Simulates reality through randomization
- Introduces natural uncertainty

---

## 17. Bootstrapping - Detailed Explanation

***Resampling technique that generates a larger dataset from a small existing dataset through random sampling with replacement***

**Goal**: Simulate reality and create more training data

---

### **Key Concept: Sampling WITH Replacement**

**Critical principle**: After selecting a data point, **put it back** before next selection

**Why this matters**:

**Without replacement** ❌:

```
Start: [d1, d2, d3, d4, d5] (5 data points)
Pick d3 → Remove it
Now:   [d1, d2, d4, d5] (only 4 left)
Pick d1 → Remove it
Now:   [d2, d4, d5] (only 3 left)
...
After 5 picks: [] (nothing left!)
```

**With replacement** ✓:

```
Start: [d1, d2, d3, d4, d5] (5 data points)
Pick d3 → Put it back
Still: [d1, d2, d3, d4, d5] (still 5!)
Pick d1 → Put it back
Still: [d1, d2, d3, d4, d5] (still 5!)
...
Can pick forever!
```

---

### **Bootstrapping Process**

**Step-by-step algorithm**:

1. **Start with original dataset**: $D = {d_1, d_2, \ldots, d_N}$
    
2. **Define parameters**:
    
    - **Sample size** ($n$): How many data points per iteration?
    - **Number of iterations** ($B$): How many bootstrap samples?
3. **For each iteration** $b = 1, 2, \ldots, B$:
    
    - Randomly select $n$ data points from $D$ **with replacement**
    - Store as bootstrap sample $D_b^*$
4. **Combine all bootstrap samples**
    
5. **CRITICAL: Shuffle the entire combined dataset**
    

---

### **Example: Bootstrapping with 100 Data Points**

**Original dataset**: 100 data points

**Choose parameters**:

- Sample size: $n = 100$ (same as original)
- Iterations: $B = 1000$

**Process**:

**Iteration 1**:

- Randomly pick 100 points from original 100 (with replacement)
- Might get: [d₇₉, d₁₂, d₇₉, d₄₅, ..., d₇₉] (note: d₇₉ appears 3 times!)
- Store as $D_1^*$

**Iteration 2**:

- Again pick 100 points from original 100 (with replacement)
- Might get: [d₃, d₈₈, d₃, d₁₀, ..., d₅₀]
- Store as $D_2^*$

...continue for 1000 iterations

**Result**: 1000 bootstrap samples, each with 100 data points

---

### **Final Dataset Size**

$$\text{Total size} = \text{iterations} \times \text{sample size}$$

$$= B \times n$$

**Example**: $$= 1000 \times 100 = 100{,}000 \text{ data points}$$

**From 100 original points → 100,000 bootstrapped points!**

---

### **Possible Duplicates**

**Important observation**: Same data point can appear multiple times!

**Example scenario** (unlikely but possible):

- Iteration 1: Pick d₇₉
- Iteration 2: Pick d₇₉ again
- Iteration 3: Pick d₇₉ again

**This is OK!** It's part of the randomization that simulates reality

**But**: This is why we need **shuffling** afterward!

---

## 18. Why Shuffling Is Critical

### **Problem Without Shuffling**

**Scenario**: After 1000 iterations without shuffling

```
First 10 iterations: Very good quality data
Next 990 iterations: Poor quality data
```

**Risk**:

- Data is **imbalanced**
- Not randomly distributed
- Could hurt model training (especially with cross-validation)

---

### **Solution: Shuffle the Entire Dataset**

**When**: After completing ALL bootstrap iterations

**What**: Randomly reorder all data points in the combined dataset

**Why**:

1. Ensures random distribution
2. Prevents sequential patterns
3. Avoids clusters of duplicates
4. Makes data more realistic

---

### **Shuffling Process**

**Before shuffle** (dangerous pattern):

```
[Iteration 1 samples] [Iteration 2 samples] ... [Iteration 1000 samples]
    ↓                       ↓                           ↓
First 100 points      Next 100 points            Last 100 points
(might be similar)    (might be different)       (might be repeated)
```

**After shuffle** (random distribution):

```
[d₄₅₇, d₂₃, d₉₉₈, d₁₂, d₇₈₉, d₃₄, ...]
Completely randomized order
```

---

### **When to Shuffle**

**Correct** ✓:

```
1. Do all bootstrapping
2. Combine all samples
3. SHUFFLE entire dataset
4. Use for training
```

**Incorrect** ❌:

```
1. Do iteration 1
2. Shuffle
3. Do iteration 2
4. Shuffle
...
```

**Why incorrect method fails**: You're continuously changing what you haven't even finished creating yet!

---

## 19. Bootstrapping Parameters

### **Question: How Many Iterations?**

**Answer**: No fixed requirement - depends on your needs!

**Guidelines**:

- Start with 100 or 1,000 iterations
- Train model and evaluate performance
- If not enough → increase iterations
- If good enough → stop

---

### **Factors to Consider**

**Sample size ($n$)**:

- Often same as original dataset size
- Can be smaller or larger depending on use case

**Number of iterations ($B$)**:

- More iterations → more data → potentially better model
- But diminishing returns after a point
- Computational cost increases

**Trade-off**: $$\text{Quality vs. Computation time}$$

---

### **Example Decision Process**

**Scenario**: Original dataset has 500 points

**Experiment 1**:

- $n = 500$, $B = 100$
- Total: 50,000 points
- Model accuracy: 75%
- **Conclusion**: Maybe not enough

**Experiment 2**:

- $n = 500$, $B = 1{,}000$
- Total: 500,000 points
- Model accuracy: 88%
- **Conclusion**: Better! But can we do more?

**Experiment 3**:

- $n = 500$, $B = 10{,}000$
- Total: 5,000,000 points
- Model accuracy: 89%
- **Conclusion**: Marginal improvement, not worth 10× computation

**Decision**: Use $B = 1{,}000$ (good balance)

---

## 20. Important Questions About Shuffling

### **Q: Do you shuffle rows, columns, or what?**

**A**: Shuffle the **entire dataset** (all rows)

**Process**:

1. After bootstrapping: Have combined dataset with all samples
2. Shuffle: Randomly reorder the **rows** (data points)
3. Keep each row intact (don't shuffle within rows)

---

### **Q: Does ordering matter?**

**A**: No, the final ordering doesn't matter for information

**But**: Shuffling prevents **imbalanced distributions** that could occur from:

- Good data clustered at the beginning
- Bad data clustered at the end
- Repeated samples appearing together

**Goal**: Make dataset appear as **randomly distributed** as possible

---

### **Q: When does shuffling help with cross-validation?**

**Important scenario**:

**Without shuffling**:

```
Fold 1: [Samples from iterations 1-200]    (might be similar)
Fold 2: [Samples from iterations 201-400]  (might be similar)
Fold 3: [Samples from iterations 401-600]  (might be similar)
...
```

**Problem**: Each fold might have similar characteristics!

**With shuffling**:

```
Fold 1: [Random mix from all iterations]
Fold 2: [Random mix from all iterations]
Fold 3: [Random mix from all iterations]
...
```

**Benefit**: Each fold is truly representative of full dataset

---

## 21. Frequency and Randomization

### **Key Concepts Covered**

**Bootstrapping achieves**:

1. ✓ Simulates **infinity** (unlimited sampling)
2. ✓ Based on **limited dataset** (works with small data)
3. ✓ Introduces **uncertainty** (through randomization)
4. ✓ Creates **realistic** data distribution

---

### **Connection to Earlier Topics**

**Maximum Likelihood**:

- Used to find optimal parameters
- Works better with more data
- Bootstrapping provides that data!

**Evaluation**:

- More data → better evaluation
- Can use cross-validation effectively
- Reduces overfitting risk

---

## 22. Bayes' Theorem Revisited

### **The Formula** (Review)

$$p(Y|X) = \frac{p(X|Y) \times p(Y)}{p(X)}$$

**Terminology reminder**:

|Symbol|Name|Meaning|
|---|---|---|
|$p(Y\|X)$|**Posterior**|Probability after seeing data|
|$p(X\|Y)$|**Likelihood**|How well data fits hypothesis|
|$p(Y)$|**Prior**|Initial belief|
|$p(X)$|**Evidence**|Normalizing constant|

---

### **The Proportional Symbol: $\propto$**

**What it means**: "Proportional to"

$$p(Y|X) \propto p(X|Y) \times p(Y)$$

**Translation**: $$p(Y|X) = C \times p(X|Y) \times p(Y)$$

Where $C$ is a normalizing constant

---

### **Why Use $\propto$?**

**Problem**: $$p(Y|X) = \frac{p(X|Y) \times p(Y)}{p(X)}$$

Computing $p(X)$ (evidence) is often very hard!

**Solution**: $$p(Y|X) \propto p(X|Y) \times p(Y)$$

Then **normalize** afterward to ensure probabilities sum to 1

---

### **Normalization Process**

**Step 1**: Compute unnormalized posteriors $$\tilde{p}(Y=y_1|X) = p(X|Y=y_1) \times p(Y=y_1)$$ $$\tilde{p}(Y=y_2|X) = p(X|Y=y_2) \times p(Y=y_2)$$ $$\vdots$$

**Step 2**: Sum all unnormalized values $$Z = \sum_i \tilde{p}(Y=y_i|X)$$

**Step 3**: Normalize $$p(Y=y_i|X) = \frac{\tilde{p}(Y=y_i|X)}{Z}$$

**This guarantees** $\sum_i p(Y=y_i|X) = 1$ ✓

---

## 23. Bayesian Model Training Process

### **Algorithm Steps**

**Step 1: Initialize**

- Set parameter $w$ based on **prior knowledge**
- Example: For fair coin, initialize $w = 0.5$

**Step 2: Compute Likelihood**

- Calculate $p(D|w)$ for observed data $D$
- Use likelihood function we derived earlier

**Step 3: Compute Posterior**

- Apply Bayes' theorem: $$p(w|D) = \frac{p(D|w) \times p(w)}{p(D)}$$
- Or use proportional form with normalization

**Step 4: Update**

- Use posterior as new prior
- Repeat steps 2-4 until convergence

---

### **Iterative Training Loop**

```
Initialize w based on prior
WHILE not converged:
    Compute likelihood p(D|w)
    Compute posterior p(w|D) using Bayes' theorem
    Update w ← w_new
    Check convergence
END WHILE
Return final w
```

**Convergence criteria**:

- Change in $w$ is very small
- Change in likelihood is very small
- Maximum iterations reached

---

## Summary: Key Formulas for Obsidian Notes

### **Maximum Likelihood Estimation**

$$p(D|w) = w^{n_H}(1-w)^{n_T}$$ $$\log p(D|w) = n_H \ln(w) + n_T \ln(1-w)$$ $$w_{ML} = \frac{n_H}{n_H + n_T}$$

### **Bootstrapping**

$$\text{Total bootstrap samples} = B \times n$$

- $B$ = number of iterations
- $n$ = sample size per iteration
- **Always shuffle after combining all samples!**

### **Bayes' Theorem**

$$p(Y|X) = \frac{p(X|Y) \times p(Y)}{p(X)} \propto p(X|Y) \times p(Y)$$

### **Normalization**

$$p_{\text{norm}}(Y=y_i) = \frac{p(Y=y_i)}{\sum_j p(Y=y_j)}$$

---
# Machine Learning Week 2: Probabilistic Machine Learning - Part 3

## Continuing Bayesian Training and Beta Distribution...

---

## 24. Limitations of Maximum Likelihood Estimation (MLE)

### **The Zero Probability Problem**

**Critical Issue**: What if you have **extremely limited data**?

**Example**: Only 2 observations, both tails $$D = {T, T}$$

**Computing probabilities**: $$p(H) = \frac{0}{2} = 0$$ $$p(T) = \frac{2}{2} = 1$$

---

### **Consequences of Zero Probability**

**Problem 1**: Model always predicts tail

- $p(H) = 0$ → will NEVER predict heads
- Lost all generalization ability

**Problem 2**: Likelihood becomes zero $$p(D|w) = w^{n_H}(1-w)^{n_T}$$

If we ever observe a head in testing but $w = 0$: $$p(\text{new head}|w=0) = 0^1 = 0$$

**Entire model collapses!**

---

### **Why This Is Catastrophic**

**Remember Bayes' Theorem**: $$p(\text{class}|x) = \frac{p(x|\text{class}) \times p(\text{class})}{p(x)}$$

**If any term in likelihood is zero**: $$p(x|\text{class}) = 0 \times \text{other terms} = 0$$

**Then posterior is zero**: $$p(\text{class}|x) = 0$$

**One zero probability kills the entire model!**

---

## 25. Solution 1: Epsilon Smoothing (Laplace Smoothing)

### **Add Small Constant**

**Modified probability formula**: $$p(H) = \frac{n_H + \epsilon}{n_H + n_T + 2\epsilon}$$

Where $\epsilon$ is a small positive number (often $\epsilon = 1$)

---

### **Example with $\epsilon = 1$**

**Data**: 0 heads, 2 tails

**Without smoothing**: $$p(H) = \frac{0}{2} = 0$$ ❌

**With smoothing** ($\epsilon = 1$): $$p(H) = \frac{0 + 1}{2 + 2(1)} = \frac{1}{4} = 0.25$$ ✓

**Now probability is small but NOT zero!**

---

### **General Formula**

For $K$ possible outcomes: $$p(\text{outcome}_i) = \frac{\text{count}_i + \epsilon}{\text{total} + K\epsilon}$$

**Benefits**:

- ✓ Prevents zero probabilities
- ✓ Simple to implement
- ✓ Provides "pseudo-counts" for unseen events

**Limitation**:

- Arbitrary choice of $\epsilon$
- Not based on principled probability theory

---

## 26. Solution 2: Better Prior with Beta Distribution

### **The Problem with Common Sense Prior**

**Previous approach**: Just use $p(H) = 0.5$ as prior (common sense)

**Problems**:

- Not statistically rigorous
- Doesn't account for uncertainty in prior
- Fixed value, no flexibility

**Better approach**: Use a **probability distribution** for the prior!

---

### **Beta Distribution - Definition**

**Instead of**: $p(w) = 0.5$ (single fixed value)

**Use**: $p(w) = \text{Beta}(w|a, b)$ (distribution over possible values)

**Formula**: $$\boxed{p(w|a,b) \propto w^{a-1}(1-w)^{b-1}}$$

Where:

- $w \in [0, 1]$ (probability parameter)
- $a, b > 0$ (hyperparameters)
- Symbol $\propto$ means "proportional to" (needs normalization)

---

### **Full Beta Distribution Formula**

$$p(w|a,b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} w^{a-1}(1-w)^{b-1}$$

Where $\Gamma(\cdot)$ is the gamma function

**For our purposes**, we can work with the proportional form: $$p(w|a,b) \propto w^{a-1}(1-w)^{b-1}$$

---

### **Hyperparameters $a$ and $b$**

**Definition**: Parameters that control other parameters

**Key insight**: $a$ and $b$ are NOT fixed!

- They are **tunable**
- Different values → different prior distributions
- Must be found through experimentation

**Relationship to data**:

- $a - 1$ ≈ pseudo-count of heads
- $b - 1$ ≈ pseudo-count of tails

---

### **Connection to Likelihood**

**Remember our likelihood**: $$p(D|w) = w^{n_H}(1-w)^{n_T}$$

**Beta prior**: $$p(w|a,b) \propto w^{a-1}(1-w)^{b-1}$$

**Notice the similarity!**

- Likelihood: powers are data counts ($n_H$, $n_T$)
- Prior: powers are $(a-1)$, $(b-1)$

**This is NOT a coincidence** - the Beta distribution is the **conjugate prior** for the Binomial likelihood!

---

## 27. Different Beta Distributions (Visual Understanding)

### **Case 1: $a = b$ (Symmetric)**

**$a = 1, b = 1$** (Uniform): $$p(w|1,1) \propto w^0(1-w)^0 = 1$$

```
p(w)
  |
1 |____________
  |            
0 |___________w
  0          1
```

**Interpretation**: No prior knowledge - all values of $w$ equally likely

---

**$a = 3, b = 3$** (Bell curve): $$p(w|3,3) \propto w^2(1-w)^2$$

```
p(w)
  |    ╱‾‾╲
  |   ╱    ╲
  |  ╱      ╲
  |_╱________╲_w
  0    0.5    1
```

**Interpretation**: Weak belief that $w \approx 0.5$ (fair coin), but uncertain

---

### **Case 2: $a \neq b$ (Asymmetric)**

**$a = 100, b = 1$** (Heavily biased toward heads): $$p(w|100,1) \propto w^{99}(1-w)^{0}$$

```
p(w)
  |          ╱
  |        ╱
  |      ╱
  |    ╱
  |__╱_______w
  0    0.9  1
```

**Interpretation**: Strong belief that coin is heavily biased toward heads

---

### **Case 3: Strength of Belief**

**$a = 100, b = 100$** (Strong belief in fair coin):

- Narrow distribution centered at 0.5
- High confidence in prior

**$a = 2, b = 2$** (Weak belief in fair coin):

- Wide distribution around 0.5
- Low confidence in prior

**Key insight**: Sum $a + b$ represents **strength** of prior belief

---

## 28. Bayesian Inference with Beta Prior

### **Combining Beta Prior with Binomial Likelihood**

**Bayes' Theorem**: $$p(w|D) \propto p(D|w) \times p(w)$$

**Likelihood** (from data): $$p(D|w) = w^{n_H}(1-w)^{n_T}$$

**Prior** (Beta distribution): $$p(w) \propto w^{a-1}(1-w)^{b-1}$$

---

### **Posterior Derivation**

**Step 1: Multiply likelihood and prior**

$$p(w|D) \propto w^{n_H}(1-w)^{n_T} \times w^{a-1}(1-w)^{b-1}$$

**Step 2: Combine powers** (using $x^m \cdot x^n = x^{m+n}$)

$$p(w|D) \propto w^{n_H + a - 1}(1-w)^{n_T + b - 1}$$

**Step 3: Recognize as Beta distribution**

$$\boxed{p(w|D) = \text{Beta}(w | n_H + a, n_T + b)}$$

---

### **Beautiful Result!**

**Posterior is also a Beta distribution!**

**New parameters**:

- New $a'$ = $n_H + a$ (old pseudo-heads + observed heads)
- New $b'$ = $n_T + b$ (old pseudo-tails + observed tails)

**This is called conjugacy**:

- Beta prior + Binomial likelihood → Beta posterior
- Makes math much simpler!

---

## 29. Computing Predictions with Beta Posterior

### **Goal: Predict Probability of Heads**

**Using integral** (marginalizing over $w$):

$$p(H|D) = \int_0^1 p(H|w) \times p(w|D) , dw$$

Where:

- $p(H|w) = w$ (if we know $w$, probability of heads IS $w$)
- $p(w|D) = \text{Beta}(w | n_H + a, n_T + b)$ (posterior)

---

### **Simplification**

$$p(H|D) = \int_0^1 w \times p(w|D) , dw$$

This is the **expected value** of $w$ under the posterior distribution!

$$p(H|D) = \mathbb{E}[w|D]$$

---

### **Expected Value of Beta Distribution**

**Formula**: $$\boxed{\mathbb{E}[w] = \frac{a}{a + b}}$$

For our posterior with $a' = n_H + a$ and $b' = n_T + b$:

$$\boxed{p(H|D) = \frac{n_H + a}{n_H + n_T + a + b}}$$

---

### **Interpretation**

$$p(H|D) = \frac{\text{observed heads} + \text{pseudo-heads}}{\text{total observations} + \text{total pseudo-counts}}$$

**This is MLE with smoothing!**

- $a, b$ act as pseudo-counts
- Prevents zero probabilities
- Statistically principled (from Bayesian framework)

---

## 30. Examples with Different Hyperparameters

### **Example 1: $a = 100, b = 100$ (Strong Prior)**

**Data**: 3 heads, 7 tails

**Posterior**: $$p(H|D) = \frac{3 + 100}{10 + 200} = \frac{103}{210} \approx 0.490$$

**Interpretation**:

- Prior belief was strong ($a + b = 200$)
- Prior believed $p(H) = \frac{100}{200} = 0.5$
- Only 10 observations → prior dominates
- Posterior ≈ 0.49 (close to prior 0.5)

---

### **Example 2: $a = 2, b = 2$ (Weak Prior)**

**Data**: 3 heads, 7 tails

**Posterior**: $$p(H|D) = \frac{3 + 2}{10 + 4} = \frac{5}{14} \approx 0.357$$

**Interpretation**:

- Prior belief was weak ($a + b = 4$)
- 10 observations → data dominates
- Posterior ≈ 0.36 (closer to MLE of 0.3)

---

### **Example 3: $a = 100, b = 1$ (Biased Prior)**

**Data**: 3 heads, 7 tails

**Posterior**: $$p(H|D) = \frac{3 + 100}{10 + 101} = \frac{103}{111} \approx 0.928$$

**Interpretation**:

- Strong prior belief coin is biased toward heads
- Even with 70% tails observed, posterior predicts 93% heads!
- **Prior is too strong** - dominates the data
- This would be a BAD choice of hyperparameters!

---

## 31. How to Choose Hyperparameters $a$ and $b$

### **No Simple Answer**

**Finding best $a$ and $b$ requires**:

1. Experimentation
2. Cross-validation
3. Testing on held-out data

---

### **General Guidelines**

**For fair events** (like coin toss):

- Start with $a = b$ (symmetric)
- Higher values → stronger belief
- Lower values → weaker belief

**For biased events**:

- Use domain knowledge to set ratio $a:b$
- Example: If you know coin is 70% heads, try $a=7, b=3$

**Iterative process**:

1. Choose initial $a, b$
2. Train model
3. Evaluate on validation set
4. Adjust $a, b$ based on performance
5. Repeat until satisfactory

---

### **Evaluation Metric**

**For classification**:

- Use **mismatch rate** (how many predictions are wrong)
- Or **cross-entropy loss**

$$\text{Loss} = -\sum_{i=1}^N [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

**NOT using error functions** like in regression!

---

## 32. Summary: Key Concepts

### **Probability Theory Foundations**

**Sum Rule** (100% rule): $$\sum_{\text{all outcomes}} p(x) = 1$$

**Product Rule** (joint probability): $$p(X, Y) = p(Y|X) \times p(X)$$

**Bayes' Theorem**: $$p(Y|X) = \frac{p(X|Y) \times p(Y)}{p(X)}$$

---

### **Maximum Likelihood Estimation**

**Likelihood**: $$p(D|w) = w^{n_H}(1-w)^{n_T}$$

**Log-Likelihood** (easier to optimize): $$\log p(D|w) = n_H \log(w) + n_T \log(1-w)$$

**MLE Solution**: $$w_{ML} = \frac{n_H}{n_H + n_T}$$

---

### **Bootstrapping**

**Purpose**: Generate more data from limited dataset

**Method**:

1. Sample with replacement
2. Multiple iterations
3. **Always shuffle final dataset**

**Parameters**:

- Sample size $n$
- Number of iterations $B$
- Total size = $B \times n$

---

### **Beta Distribution for Priors**

**Beta PDF**: $$p(w|a,b) \propto w^{a-1}(1-w)^{b-1}$$

**Posterior** (after seeing data): $$p(w|D) = \text{Beta}(w | n_H + a, n_T + b)$$

**Prediction**: $$p(H|D) = \frac{n_H + a}{n_H + n_T + a + b}$$

---

## 33. This Week's Tutorial

### **Topic: KNN with Bootstrapping**

**What you'll do**:

1. Apply bootstrapping to create larger dataset
2. Use bootstrapped data to train KNN model
3. Find best $K$ value
4. Compare performance with/without bootstrapping

**Parameters to experiment with**:

- Number of iterations
- Sample size per iteration
- Impact on model performance

**Goal**: See how bootstrapping improves model with limited data!

---

## 34. Assignment Information

### **Assignment 1 Details**

**Release**: Available now **Due**: Week 7, Friday **Weight**: 25% of total unit grade

**Components**:

1. **Coding** (50 marks = 12.5% of unit)
2. **Oral interview** (50 marks = 12.5% of unit)

---

### **Scope**: Modules 1, 2, 3 only

**Module 1**: KNN, Cross-Validation **Module 2**: Probability, MLE, Bootstrapping  
**Module 3**: Classification, Logistic Regression, Naive Bayes

---

### **Questions Overview**

**Question 1**: Build KNN model (custom implementation)

- Follow specific restrictions
- NOT the lab version - this is YOUR own implementation

**Question 2**: Implement cross-validation with KNN

**Question 3**: Implement nested cross-validation

- Advanced version of cross-validation

**Question 4**: Ridge regression (L1/L2 regularization)

- Covered in next module

**Question 5**: Classification models

- Logistic regression
- Naive Bayes classifier

---

### **Important Notes**

**Packages**:

- ✓ Allowed packages are listed
- ✗ Restricted packages are listed
- **Using restricted packages → mark deduction**

**Academic Integrity**:

- ⚠️ **NO ChatGPT/AI tools for code generation**
- This is taken VERY seriously
- Oral interview will test your understanding
- Final exam is closed book (no AI help there!)

**Students who use AI**:

- May get 95+ on assignment
- But score 7-8 out of 100 on final exam
- **Learn the material properly!**

---

### **Oral Interview Details**

**When**: Week 8, during lab session **Where**: On campus (in-person required) **Duration**: ~10 minutes **Format**: Face-to-face, looking at your code on screen

**Question types**:

- Why did you use this approach?
- Explain this part of your code
- Why split training/testing data?
- What technology did you use and why?

**NOT asked**:

- Unrelated theory (e.g., "What is LLM?")
- Code from other assignments
- Material outside assignment scope

---

### **Final Exam Information**

**Format**: Closed book, handwritten

- **NO coding questions**
- Short theory questions
- Mathematical derivations
- Algorithm calculations

**Example questions**:

- Derive MLE for Gaussian distribution
- Compute Bayes' theorem by hand
- Explain bias-variance tradeoff

---

## Complete Formula Reference for Obsidian

### **Probability Basics**

$$P(X) \in [0, 1]$$ $$\sum_{\text{all } X} p(X) = 1$$ $$p(X, Y) = p(Y|X) \times p(X)$$

### **Bayes' Theorem**

$$p(Y|X) = \frac{p(X|Y) \times p(Y)}{p(X)} \propto p(X|Y) \times p(Y)$$

### **Normalization**

$$p_{\text{norm}}(C_i) = \frac{p(C_i)}{\sum_j p(C_j)}$$

### **Maximum Likelihood**

$$p(D|w) = w^{n_H}(1-w)^{n_T}$$ $$\log p(D|w) = n_H \ln(w) + n_T \ln(1-w)$$ $$\frac{\partial \log p(D|w)}{\partial w} = \frac{n_H}{w} - \frac{n_T}{1-w}$$ $$w_{ML} = \frac{n_H}{n_H + n_T}$$

### **Laplace Smoothing**

$$p(H) = \frac{n_H + \epsilon}{n_H + n_T + 2\epsilon}$$

### **Beta Distribution**

$$p(w|a,b) \propto w^{a-1}(1-w)^{b-1}$$ $$\text{Full: } p(w|a,b) = \frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)} w^{a-1}(1-w)^{b-1}$$

### **Bayesian Update**

$$p(w|D) = \text{Beta}(w | n_H + a, n_T + b)$$ $$p(H|D) = \mathbb{E}[w|D] = \frac{n_H + a}{n_H + n_T + a + b}$$

### **Bootstrapping**

$$\text{Total samples} = B \times n$$

- $B$ = iterations
- $n$ = sample size
- **Always with replacement**
- **Always shuffle after combining**

---
