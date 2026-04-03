# FIT5217 — Week 5: Text Classification, Naïve Bayes & Logistic Regression

**Unit:** FIT5217 Natural Language Processing · Monash University  
**Topics:** Text Classification · Bag-of-Words · Naïve Bayes · Logistic Regression · Gradient Descent · Evaluation

---

## Table of Contents

1. [[#1. The Big Picture — What is Text Classification?]]
2. [[#2. Feature Representation — Bag of Words]]
3. [[#3. Classification Problem Types]]
4. [[#4. Naïve Bayes Classifier]]
5. [[#5. Log-Space Naïve Bayes]]
6. [[#6. Naïve Bayes — Strengths and Limitations]]
7. [[#7. Logistic Regression]]
8. [[#8. Training with Gradient Descent]]
9. [[#9. Cross-Entropy Loss]]
10. [[#10. Generalisation, Overfitting & Data Splits]]
11. [[#11. Evaluation — Confusion Matrix, Precision, Recall, F1]]
12. [[#12. Exam Preparation]]
13. [[#13. Connections to Other Topics]]

---

## 1. The Big Picture — What is Text Classification?

### Definition

> **Text classification** is the task of assigning a category label $c \in C$ to an input document $d$.

Unlike the tasks from Weeks 3–4, we are no longer labelling individual words or building tree structures. We are assigning **one label to an entire document**.

### Real-World Examples

|Task|Input|Output Classes|
|---|---|---|
|**Spam detection**|Email|Spam / Not Spam|
|**Sentiment analysis**|Product review|Positive / Negative / Neutral|
|**Authorship attribution**|Piece of text|Author A / Author B / …|
|**Topic classification**|News article|Politics / Sport / Economy / …|
|**AI-generated text detection**|Any text|Human / AI-generated|
|**Gender inference**|Text excerpt|Male / Female|

> ⚠️ **Fairness note (from the lecture):** Some tasks like gender inference risk learning and reinforcing social stereotypes from biased data. The goal is to identify statistical patterns in a dataset — not to generalise to all writers. Always critically assess the **bias and fairness** of training data and model outputs.

### General Pipeline

$$\text{Raw Text} \xrightarrow{\text{feature function } \phi} \text{Feature Vector } \mathbf{x} \xrightarrow{\text{classifier } f} \text{Class Label } c$$

---

## 2. Feature Representation — Bag of Words

### The Problem

Classifiers require **numerical inputs**. Raw text must be converted into a fixed-length feature vector $\mathbf{x} \in \mathbb{R}^n$.

### Bag-of-Words (BoW)

The simplest approach: **ignore word order** and represent a document by its word counts (or presence/absence).

**Terminology:**

- **Word token:** A specific occurrence of a word at a position (e.g., "good" appearing twice = 2 tokens)
- **Word type:** A unique word in the vocabulary (e.g., "good" = 1 type, regardless of count)

**Feature vector:** One dimension per word type in vocabulary $V$. If $|V| = 10{,}000$, the feature vector has length 10,000.

$$x_i = \begin{cases} 1 & \text{if word } v_i \text{ appears in document (binary)} \ \text{count}(v_i, d) & \text{if using word counts} \end{cases}$$

> **Analogy:** Imagine tipping a document into a bag and shaking it. You lose all order, but you can count how many times each word appears.

### Properties of BoW

- **Very sparse:** A 20-word document has at most 20 non-zero entries in a 10,000-dimensional vector
- **Simple but effective** for many classification tasks
- **Limitation:** Cannot capture word order or negation (_"not good"_ treated as just _"not"_ + _"good"_)

### Extending BoW to n-grams

To partially capture dependencies, use **bigrams or trigrams**:

- Unigram: _{"not", "good"}_
- Bigram: _{"not good"}_ — captures the negation

Trade-off: bigram/trigram counts are much sparser → needs interpolation (same technique as Week 2 n-gram LMs).

---

## 3. Classification Problem Types

|Type|Description|Example|
|---|---|---|
|**Binary**|Exactly 2 classes|Spam / Not Spam|
|**Multi-class**|More than 2 classes; each input gets exactly 1 label|Topic classification (politics, sport, economy…)|
|**Multi-label**|Each input can get multiple labels simultaneously|Article tagged as both _politics_ AND _economy_|

> This week focuses on **binary classification**. Multi-class is covered in Week 6.

---

## 4. Naïve Bayes Classifier

### Intuition

Naïve Bayes is a **probabilistic, generative** classifier based on Bayes' Rule. It is "naïve" because it makes a strong independence assumption: every feature (word) is assumed to be **conditionally independent** given the class.

> **Analogy:** Imagine judging a movie by reading a bag of words from its review. You ignore the order, ignore which words came before which. Each word independently "votes" for a class. That's Naïve Bayes.

### Derivation via Bayes' Rule

Our goal: find the most probable class $c^*$ given document $d$:

$$c^* = \underset{c \in C}{\arg\max}\ P(c \mid d)$$

Apply Bayes' Rule:

$$P(c \mid d) = \frac{P(d \mid c)\ P(c)}{P(d)}$$

Since $P(d)$ is the same for all classes, we can drop the denominator (it doesn't affect the ranking):

$$c^* = \underset{c \in C}{\arg\max}\ \underbrace{P(d \mid c)}_{\text{likelihood}} \cdot \underbrace{P(c)}_{\text{prior}}$$

This is called **Maximum A Posteriori (MAP)** classification.

> ⚠️ **Important:** Because we dropped the denominator $P(d)$, the resulting score $\hat{P}$ is **not a valid probability** — scores don't sum to 1. Do not compute the other class as $1 - \hat{P}(c \mid d)$.

### The Two Key Assumptions

**1. Bag-of-Words assumption:** Word position doesn't matter; only which words appear.

**2. Conditional independence assumption:** Given the class $c$, all words are independent of each other:

$$P(d \mid c) = P(w_1, w_2, \ldots, w_n \mid c) = \prod_{i=1}^{n} P(w_i \mid c)$$

This reduces an intractable joint probability to a product of simple unigram probabilities.

### Final Naïve Bayes Formula

$$c^* = \underset{c \in C}{\arg\max}\ P(c) \cdot \prod_{i=1}^{n} P(w_i \mid c)$$

### Estimating the Parameters (MLE)

**Prior class probability:**

$$\hat{P}(c) = \frac{\text{count of documents with class } c}{\text{total number of training documents}}$$

**Class-conditional word likelihood (unigram MLE):**

$$\hat{P}(w_i \mid c) = \frac{\text{count}(w_i,\ c)}{\sum_{w \in V} \text{count}(w,\ c)}$$

Count how often word $w_i$ appears in all documents of class $c$, divided by the total word count in all class-$c$ documents.

**Smoothing (Add-1 / Laplace):** Avoid zero probabilities for unseen words:

$$\hat{P}(w_i \mid c) = \frac{\text{count}(w_i,\ c) + 1}{\sum_{w \in V} \text{count}(w,\ c) + |V|}$$

### ✏️ Worked Example — Naïve Bayes Sentiment Classifier

**Training data:**

|Doc|Words|Class|
|---|---|---|
|D1|predictable, no fun|Negative|
|D2|no plot, funny|Negative|
|D3|fun, powerful|Negative|
|D4|powerful, fun, predictable|Positive|

**Test document D5:** _"predictable, fun"_ — predict class.

---

**Step 1 — Prior probabilities:**

$$\hat{P}(\text{Neg}) = \frac{3}{4} = 0.75 \qquad \hat{P}(\text{Pos}) = \frac{1}{4} = 0.25$$

Check: $0.75 + 0.25 = 1.0$ ✓

---

**Step 2 — Class-conditional likelihoods:**

Vocabulary $V = {\text{predictable, no, fun, plot, funny, powerful}}$, so $|V| = 6$

_Negative class_ — total words across D1, D2, D3: D1: predictable, no, fun (3) | D2: no, plot, funny (3) | D3: fun, powerful (2) → total = 8 words

|Word|Count in Neg|$\hat{P}(w \mid \text{Neg})$ with Add-1|
|---|---|---|
|predictable|1|$\frac{1+1}{8+6} = \frac{2}{14}$|
|no|2|$\frac{2+1}{8+6} = \frac{3}{14}$|
|fun|2|$\frac{2+1}{8+6} = \frac{3}{14}$|
|plot|1|$\frac{1+1}{8+6} = \frac{2}{14}$|
|funny|1|$\frac{1+1}{8+6} = \frac{2}{14}$|
|powerful|1|$\frac{1+1}{8+6} = \frac{2}{14}$|

_Positive class_ — total words in D4: powerful, fun, predictable → total = 3 words

|Word|Count in Pos|$\hat{P}(w \mid \text{Pos})$ with Add-1|
|---|---|---|
|predictable|1|$\frac{1+1}{3+6} = \frac{2}{9}$|
|fun|1|$\frac{1+1}{3+6} = \frac{2}{9}$|
|powerful|1|$\frac{1+1}{3+6} = \frac{2}{9}$|
|no, plot, funny|0 each|$\frac{0+1}{3+6} = \frac{1}{9}$ each|

---

**Step 3 — Score each class for D5 = (predictable, fun):**

$$\hat{P}(\text{Neg} \mid \text{D5}) \propto \hat{P}(\text{Neg}) \cdot \hat{P}(\text{predictable} \mid \text{Neg}) \cdot \hat{P}(\text{fun} \mid \text{Neg})$$

$$= 0.75 \times \frac{2}{14} \times \frac{3}{14} = 0.75 \times 0.1429 \times 0.2143 \approx 0.0230$$

$$\hat{P}(\text{Pos} \mid \text{D5}) \propto \hat{P}(\text{Pos}) \cdot \hat{P}(\text{predictable} \mid \text{Pos}) \cdot \hat{P}(\text{fun} \mid \text{Pos})$$

$$= 0.25 \times \frac{2}{9} \times \frac{2}{9} = 0.25 \times 0.2222 \times 0.2222 \approx 0.0123$$

**Prediction:** $\hat{P}(\text{Neg}) > \hat{P}(\text{Pos})$ → **D5 is classified as Negative** ✓

---

## 5. Log-Space Naïve Bayes

### The Problem with Multiplying Probabilities

With many features, products of many small probabilities cause **floating-point underflow** — numbers become too small for a computer to represent.

### Solution: Take the Logarithm

Since $\log$ is monotonically increasing, it **preserves the ranking** of classes:

$$c^* = \underset{c \in C}{\arg\max}\ \log P(c) + \sum_{i=1}^{n} \log P(w_i \mid c)$$

The **product becomes a sum** in log-space — much more numerically stable and computationally efficient.

> This also makes Naïve Bayes a **linear classifier**: the decision is a weighted sum of log-probabilities (one weight per word). Whichever class has the higher sum wins.

---

## 6. Naïve Bayes — Strengths and Limitations

### Strengths

- **Easy to implement:** Just count words in training data
- **Fast training:** Single pass over the data
- **Works well in practice** for many NLP tasks (spam, topic classification)
- **Generative model:** Explicitly models $P(c) \cdot P(d \mid c)$, which means it can also _generate_ text

### Limitations

|Limitation|Description|Fix|
|---|---|---|
|Bag-of-words|Cannot capture word order|Use bigrams/trigrams|
|Conditional independence|Words are not truly independent|— (this is the "naïve" part)|
|Negation|"not good" treated as two separate positive signals|Use bigrams to capture negation|
|Sparsity|Very high-dimensional, sparse vectors|Smoothing; dimensionality reduction|

---

## 7. Logistic Regression

### Generative vs. Discriminative Models

|Property|Naïve Bayes|Logistic Regression|
|---|---|---|
|Model type|**Generative** — models $P(c, d)$|**Discriminative** — models $P(c \mid d)$ directly|
|What it estimates|Joint probability $P(d \mid c) \cdot P(c)$|Conditional $P(c \mid d)$ directly|
|Data required|More (need to model full data distribution)|Less (only needs to learn the boundary)|
|Features|BoW (whole vocabulary)|Hand-crafted features (domain knowledge)|

> **Analogy — Cat vs. Dog classifier:**
> 
> - _Generative (Naïve Bayes):_ Learn everything about what cats look like and what dogs look like, then ask "which model better explains this animal?"
> - _Discriminative (Logistic Regression):_ Just learn the key differences (size, ears, tail…) enough to draw the boundary between them.

### Feature Engineering in Logistic Regression

Instead of using the entire vocabulary, logistic regression uses **hand-crafted features** defined by domain knowledge. Example for sentiment:

|Feature|Value for review _"I love this, but not recommended"_|
|---|---|
|Count of positive lexicon words|1 (_love_)|
|Count of negative lexicon words|1 (_not recommended_)|
|Is "no" present?|0|
|Count of 1st/2nd person pronouns|1 (_I_)|
|Is "!" present?|0|
|$\log(\text{sentence length})$|$\log(7)$|

This gives a compact feature vector $\phi(d) \in \mathbb{R}^k$ (e.g., $k = 6$).

### The Score Function (Dot Product)

Given feature vector $\phi(d)$ and weight vector $\mathbf{w}$ (same length $k$):

$$z = \mathbf{w} \cdot \phi(d) + b = \sum_{j=1}^{k} w_j \cdot \phi_j(d) + b$$

- $w_j > 0$: feature $j$ pushes prediction towards the positive class
- $w_j < 0$: feature $j$ pushes prediction towards the negative class
- $b$ = **bias term**: shifts the decision boundary away from the origin

This is a **linear classifier** — the decision boundary is a hyperplane in feature space.

### The Sigmoid Function

Convert the score $z \in (-\infty, +\infty)$ into a probability $\in [0, 1]$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

|$z$|$\sigma(z)$|Interpretation|
|---|---|---|
|Very large positive|$\approx 1$|Very confident positive|
|$0$|$0.5$|Uncertain|
|Very large negative|$\approx 0$|Very confident negative|

### Making a Prediction

$$\hat{P}(c = 1 \mid d) = \sigma(\mathbf{w} \cdot \phi(d) + b)$$

$$\hat{P}(c = 0 \mid d) = 1 - \sigma(\mathbf{w} \cdot \phi(d) + b)$$

> ✓ Unlike Naïve Bayes scores, logistic regression outputs **valid probabilities** that sum to 1 (for binary classification).

Predict label:

$$\hat{c} = \begin{cases} 1 & \text{if } \hat{P}(c=1 \mid d) \geq 0.5 \ 0 & \text{otherwise} \end{cases}$$

The threshold $0.5$ can be tuned (e.g., aggressive spam detection → lower threshold).

### ✏️ Worked Example — Logistic Regression Prediction

**Feature vector** (from the review): $\phi(d) = [3, 2, 0, 3, 1, 3.85]$

**Weight vector** (learned): $\mathbf{w} = [2.5, -5.0, -1.2, 0.5, 2.0, 0.7]$, bias $b = 0.1$

**Step 1 — Compute dot product:**

$$z = (2.5)(3) + (-5.0)(2) + (-1.2)(0) + (0.5)(3) + (2.0)(1) + (0.7)(3.85) + 0.1$$

$$z = 7.5 - 10.0 - 0 + 1.5 + 2.0 + 2.695 + 0.1 = 3.795$$

**Step 2 — Apply sigmoid:**

$$\hat{P}(c=1 \mid d) = \frac{1}{1 + e^{-3.795}} \approx \frac{1}{1 + 0.0224} \approx 0.978$$

**Prediction:** $\hat{P} = 0.978 > 0.5$ → **Positive sentiment** ✓

---

## 8. Training with Gradient Descent

### The Goal

Find weights $\mathbf{w}$ and bias $b$ that minimise the **training loss** $\mathcal{L}$ — a measure of how wrong the model is:

$$\mathbf{w}^* = \underset{\mathbf{w}}{\arg\min}\ \mathcal{L}(\mathbf{w}) = \frac{1}{N} \sum_{i=1}^{N} \ell(f(\mathbf{x}_i; \mathbf{w}),\ y_i)$$

### Gradient Descent

The **gradient** $\nabla_\mathbf{w} \mathcal{L}$ points in the direction of steepest increase in loss. Move in the **opposite direction** to reduce it:

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \cdot \nabla_\mathbf{w} \mathcal{L}$$

where $\eta > 0$ is the **learning rate (step size)** — a hyperparameter.

> **Analogy:** Imagine you're blindfolded on a hilly landscape and want to reach the lowest valley. At each step, feel which direction is steepest downhill, then take a step in that direction. The learning rate controls how big each step is.

### Learning Rate Trade-offs

|Learning Rate $\eta$|Effect|
|---|---|
|Too large|Overshoots minimum — unstable, diverges|
|Too small|Very slow convergence — takes many epochs|
|Just right|Stable, efficient convergence|

### Three Variants of Gradient Descent

|Variant|Update based on|Pros|Cons|
|---|---|---|---|
|**Batch GD**|All $N$ training examples|Stable, accurate gradient|Slow for large datasets|
|**Stochastic GD (SGD)**|1 training example|Very fast updates|Noisy gradient, unstable|
|**Mini-batch GD** ✓|Small batch (e.g., 32–64 examples)|Balanced: stable + fast|Need to choose batch size|

> **In practice:** Mini-batch GD is the standard. It is what _"gradient descent"_ means in modern deep learning.

### Training Loop (Pseudocode)

```
Initialise w, b (random or zero)
Repeat until convergence:
    Shuffle training data
    For each mini-batch B:
        1. Forward pass:  compute predictions ŷ and loss L
        2. Backward pass: compute gradient ∇L (via backpropagation)
        3. Update:        w ← w - η · ∇L
```

- **Epoch:** One full pass over the training data
- **Hyperparameter:** $\eta$ is chosen by the designer (not learned from data). Tune it on a validation set.

### ✏️ Gradient Calculation Intuition

For a regression loss $\mathcal{L} = (\hat{y} - y)^2$ where $\hat{y} = \mathbf{w} \cdot \phi(\mathbf{x})$:

Using the chain rule:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = 2(\hat{y} - y) \cdot \phi(\mathbf{x})$$

The update is proportional to the **error** scaled by the **input features**. Larger error or larger features → bigger update.

> You will not need to derive gradients in the exam. In practice (Week 6+), you will use **autograd** libraries (e.g., PyTorch) to compute gradients automatically.

---

## 9. Cross-Entropy Loss

### For Logistic Regression

The **cross-entropy loss** (also called negative log-likelihood) is used to train logistic regression:

$$\mathcal{L} = -\left[ c \log \hat{P}(c=1 \mid d) + (1 - c) \log \hat{P}(c=0 \mid d) \right]$$

where $c \in {0, 1}$ is the true label and $\hat{P}$ is the sigmoid output.

**Intuition:**

- If true label $c = 1$ and model predicts $\hat{P} \approx 1$: loss $\approx 0$ ✓
- If true label $c = 1$ and model predicts $\hat{P} \approx 0$: loss $\to \infty$ ✗

Minimising cross-entropy = maximising log-likelihood = learning the best weights.

> **You will see this loss again:** Cross-entropy appears in machine translation, neural LMs, and most deep learning tasks in later weeks.

---

## 10. Generalisation, Overfitting & Data Splits

### Three-Way Data Split

|Split|Purpose|
|---|---|
|**Training set**|Learn model parameters ($\mathbf{w}$, $b$, and Naïve Bayes probs)|
|**Validation set (dev set)**|Tune hyperparameters ($\eta$, model architecture); decide when to stop training|
|**Test set**|Final evaluation only — **never touch during training or tuning**|

> ⚠️ Evaluating on the test set during development is **data leakage** — it gives a falsely optimistic estimate of real-world performance.

### When Data is Scarce — $k$-Fold Cross Validation

Split data into $k$ equal folds. Train on $k{-}1$ folds, test on the remaining 1. Repeat $k$ times and average performance.

$$\text{Average accuracy} = \frac{1}{k} \sum_{i=1}^{k} \text{accuracy}_i$$

> **Note (from lecture):** In modern NLP with large datasets, $k$-fold cross-validation is rarely used because data is abundant. It remains important for small datasets (e.g., a few hundred samples).

---

## 11. Evaluation — Confusion Matrix, Precision, Recall, F1

### Confusion Matrix

For binary classification with classes Positive and Negative:

||Predicted Positive|Predicted Negative|
|---|---|---|
|**Actually Positive**|True Positive (TP)|False Negative (FN)|
|**Actually Negative**|False Positive (FP)|True Negative (TN)|

### Metrics

**Accuracy** — overall correctness:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

**Precision** — of all predicted positives, how many are actually positive?

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Recall (Sensitivity)** — of all actual positives, how many did we catch?

$$\text{Recall} = \frac{TP}{TP + FN}$$

**F1 Score** — harmonic mean of Precision and Recall:

$$F_1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Why F1 and Not Just Accuracy?

Accuracy can be misleading for **class-imbalanced** data. Example: if 95% of emails are not spam, a classifier that always predicts "not spam" achieves 95% accuracy but is useless.

|Strategy|Precision|Recall|F1|
|---|---|---|---|
|Predict everything as positive|Low|1.0|Low|
|Only predict when very confident|High|Low|Low|
|Balanced model|Moderate|Moderate|**Highest**|

### ✏️ Worked Example — Confusion Matrix

**Results:** TP = 80, FP = 10, FN = 20, TN = 90

$$\text{Accuracy} = \frac{80 + 90}{80 + 10 + 20 + 90} = \frac{170}{200} = 0.85$$

$$\text{Precision} = \frac{80}{80 + 10} = \frac{80}{90} \approx 0.889$$

$$\text{Recall} = \frac{80}{80 + 20} = \frac{80}{100} = 0.800$$

$$F_1 = \frac{2 \times 0.889 \times 0.800}{0.889 + 0.800} = \frac{1.422}{1.689} \approx \boxed{0.842}$$

---

## 12. Exam Preparation

### Likely Question Type 1 — Naïve Bayes Calculation

> _"Given a training corpus, compute the prior and likelihood probabilities, apply smoothing, and classify a test document."_

**Strategy:**

1. Count documents per class → compute $\hat{P}(c)$
2. Count word occurrences per class → compute $\hat{P}(w \mid c)$ with Add-1 smoothing
3. Multiply: $\hat{P}(c) \cdot \prod_i \hat{P}(w_i \mid c)$ for each class
4. Predict the class with the **higher product**
5. Use log-space to avoid underflow in practice: sum of log-probs

---

### Likely Question Type 2 — Conceptual: Naïve Bayes Assumptions

> _"What are the two key assumptions in Naïve Bayes, and what are the consequences of each?"_

**Model answer:**

1. **Bag-of-words:** Word position is ignored. Consequence: word order and syntactic structure are lost — _"not good"_ and _"good not"_ are treated identically.
2. **Conditional independence:** Given the class, words are independent of each other. Consequence: co-occurrence patterns between words (e.g., negation) cannot be modelled. Despite this, Naïve Bayes still works well in practice.

---

### Likely Question Type 3 — Generative vs. Discriminative

> _"Compare Naïve Bayes and Logistic Regression as classifiers."_

|Dimension|Naïve Bayes|Logistic Regression|
|---|---|---|
|Model type|Generative|Discriminative|
|What is modelled|$P(c) \cdot P(d \mid c)$|$P(c \mid d)$ directly|
|Features|Full BoW vocabulary|Hand-crafted features|
|Output|Score (not a valid probability)|Valid probability via sigmoid|
|Training|MLE counting (fast, 1 pass)|Gradient descent (iterative)|
|Strengths|Simple, fast, interpretable|More flexible, better calibrated probabilities|

---

### Likely Question Type 4 — Gradient Descent

> _"Explain gradient descent and the role of the learning rate."_

**Model answer:** Gradient descent minimises the training loss by iteratively moving the parameters in the direction opposite to the gradient. The gradient points toward steepest increase in loss, so moving against it reduces loss. The learning rate $\eta$ controls step size — too large causes instability, too small causes slow convergence. Mini-batch gradient descent is the practical standard, updating based on a small batch of examples rather than the full dataset or a single example.

---

### Likely Question Type 5 — Evaluation Metrics

> _"Given a confusion matrix, compute Precision, Recall, and F1."_

Know the formulas and the intuition:

- Precision = how precise are your positive predictions?
- Recall = how many positives did you catch?
- F1 = harmonic mean that penalises extreme imbalance between the two

---

## 13. Connections to Other Topics

|This Week|Connection|Linked Topic|
|---|---|---|
|MLE for Naïve Bayes priors/likelihoods|Same counting principle|n-gram LM (Wk 2), HMM (Wk 3), PCFG (Wk 4)|
|Add-1 smoothing for zero likelihoods|Same sparsity fix|Laplace smoothing (Wk 2)|
|BoW as unigram LM|Same unigram probability model|Language Models (Wk 2)|
|Bigrams to capture negation|Same interpolation idea|n-gram interpolation (Wk 2)|
|Naïve Bayes as generative model|Same generative framing|HMM as generative model (Wk 3)|
|Cross-entropy loss|Same loss function|Neural LMs, seq2seq, MT (Wk 6+)|
|Gradient descent training loop|Foundation of all neural training|Neural Networks (Wk 6)|
|POS tags as features for LR|Tags improve text features|POS Tagging (Wk 3)|

---

## Summary

**Text classification** assigns a label $c$ to a document $d$. The pipeline is:

$$\text{Raw text} \xrightarrow{\phi} \mathbf{x} \xrightarrow{f} c$$

**Bag-of-Words** represents text as a sparse vector of word counts, ignoring order.

**Naïve Bayes** (generative, MAP):

$$c^* = \underset{c}{\arg\max}\ \log P(c) + \sum_{i} \log P(w_i \mid c)$$

Trained with MLE counting. Assumes conditional independence (the "naïve" part). Apply Add-1 smoothing for zero probabilities.

**Logistic Regression** (discriminative):

$$\hat{P}(c=1 \mid d) = \sigma!\left(\mathbf{w} \cdot \phi(d) + b\right) = \frac{1}{1 + e^{-z}}$$

Trained with gradient descent to minimise cross-entropy loss. Uses hand-crafted features. Outputs valid probabilities.

**Evaluation:** Use Precision, Recall, and $F_1$ — not just accuracy — especially for imbalanced classes.

**Data splits:** Train / Validation / Test. Never touch the test set during development.

---

_FIT5217 · Week 5 Study Sheet · Monash University S1 2026_