# Study Sheet: Document Clustering with EM Algorithm

---

## 1. Document Clustering Problem

### Overview

Given a collection of documents ${d_1, \ldots, d_N}$, we want to partition them into $K$ clusters.

### Document Representation

**Bag of Words Representation:**

- Each document $d_n$ is treated as a **set of words** (irrespective of their position)
- Example: "Andi went to school to meet Bob" → ${\text{Andi, went, to, school, to, meet, Bob}}$
- Note: "Bob went to school to meet Andi" has the **same representation** (order doesn't matter)
- The words come from a dictionary denoted by $\mathcal{A}$

**Key Property:** This representation discards:

- Word order
- Grammar
- Sentence structure

But retains:

- Which words appear
- How often each word appears

---

## 2. The Generative Model for Document Clustering

### Generative Story

For each document $d_n$:

1. **Choose a cluster:** Toss a $K$-face dice (with parameter $\boldsymbol{\varphi}$) to choose the face (i.e., the cluster) $k$ that the $n$-th document belongs to
    
2. **Generate words:** For each word placeholder in the document $d_n$:
    
    - Generate the word by tossing a dice (with parameter $\boldsymbol{\mu}_k$) corresponding to cluster $k$

### Model Parameters

The model has two types of parameters:

1. **Cluster proportions:** $\boldsymbol{\varphi}$ which is a probability vector of size $K$ $$\sum_{k=1}^{K} \varphi_k = 1$$
    
2. **Word proportions:** $\boldsymbol{\mu}_k$ corresponding to the $k$-th cluster, where: $$\sum_{w \in \mathcal{A}} \mu_{k,w} = 1$$
    
    Note: We have $K$ such word proportion vectors, each corresponding to a face of the dice (or cluster).
    

**Intuition:**

- $\varphi_k$ = "What's the probability a random document belongs to cluster $k$?"
- $\mu_{k,w}$ = "If I'm in cluster $k$, what's the probability of seeing word $w$?"

---

## 3. Probability of Generating a Document

### Joint Probability

The probability of generating a pair of a document and its cluster $(k, d)$ according to our generative story is:

$$p(k, d) = p(k) \cdot p(d|k) = \varphi_k \prod_{w \in d} \mu_{k,w}$$

Using word counts, this can be rewritten as:

$$= \varphi_k \prod_{w \in \mathcal{A}} \mu_{k,w}^{c(w,d)}$$

where $c(w, d)$ is the **number of occurrences** of word $w$ in document $d$.

### Why the Exponent?

If word $w$ appears 3 times in document $d$, we multiply $\mu_{k,w}$ three times: $$\mu_{k,w} \cdot \mu_{k,w} \cdot \mu_{k,w} = \mu_{k,w}^{3} = \mu_{k,w}^{c(w,d)}$$

---

## 4. Complete Data Likelihood

### The Setup

**Complete Data:** Assume we know both:

- The documents: ${d_1, \ldots, d_N}$
- The cluster assignments: ${z_1, \ldots, z_N}$

where $\mathbf{z}_n := (z_{n1}, \ldots, z_{nk})$ is the cluster assignment vector for the $n$-th document.

**One-Hot Encoding:** $z_{nk} = 1$ if document $n$ belongs to cluster $k$, and zero otherwise.

### Complete Data Likelihood Formula

$$p(d_1, z_1, \ldots, d_N, z_N) = \prod_{n=1}^{N} \prod_{k=1}^{K} \left( \varphi_{k_n} \prod_{w \in \mathcal{A}} \mu_{k_n,w}^{c(w,d)} \right)^{z_{n,k}}$$

The exponent $z_{n,k}$ ensures only the term for the true cluster (where $z_{n,k} = 1$) is active.

### Complete Data Log-Likelihood

Taking the logarithm and using log rules:
$$\ln p(d_1, z_1, \ldots, d_N, z_N) = \sum_{n=1}^{N} \sum_{k=1}^{K} z_{n,k} \left( \ln \varphi_{k_n} + \sum_{w \in \mathcal{A}} c(w, d) \ln \mu_{k_n, w} \right)$$

---

## 5. Learning Parameters from Complete Data

If we had complete data (knew the cluster assignments), maximizing the complete data log-likelihood gives us:

### Mixing Components (Cluster Proportions)

$$\varphi_k = \frac{N_k}{N} \quad \text{where} \quad N_k := \sum_{n=1}^{N} z_{nk}$$

**Interpretation:** The proportion of documents assigned to cluster $k$.

### Word Proportion Parameters

$$\mu_{k,w} = \frac{\sum_{n=1}^{N} z_{nk} c(w, d_n)}{\sum_{w' \in \mathcal{A}} \sum_{n=1}^{N} z_{nk} c(w', d_n)}$$

**Interpretation (very intuitive!):**

- **Numerator:** Count the number of times word $w$ appears in documents belonging to cluster $k$
- **Denominator:** Count the total number of all words in documents belonging to cluster $k$
- **Result:** Normalize so that $\sum_{w} \mu_{k,w} = 1$

The notes emphasize: _"The best value for $\boldsymbol{\mu}_k$ is obtained by counting the number of times that each word of the dictionary has been seen in the documents belonging to cluster $k$, and then normalising this count vector so that it sums to 1."_

---

## 6. Incomplete Data Likelihood

### The Reality

In practice, document cluster IDs are **not given** to us, so $z_n$ is latent (hidden). We only observe the documents.

### Incomplete Data Likelihood

The probability of observing the documents (marginalizing out the latent cluster assignments):

$$p(d_1, \ldots, d_N) = \prod_{n=1}^{N} p(d_n) = \prod_{n=1}^{N} \sum_{k=1}^{K} p(z_{n,k} = 1, d_n)$$

$$= \prod_{n=1}^{N} \sum_{k=1}^{K} \left( \varphi_k \prod_{w \in \mathcal{A}} \mu_{k,w}^{c(w,d_n)} \right)$$

### Incomplete Data Log-Likelihood

$$\ln p(d_1, \ldots, d_N) = \sum_{n=1}^{N} \ln p(d_n) = \sum_{n=1}^{N} \ln \sum_{k=1}^{K} p(z_{n,k} = 1, d_n)$$

$$= \sum_{n=1}^{N} \ln \sum_{k=1}^{K} \left( \varphi_k \prod_{w \in \mathcal{A}} \mu_{k,w}^{c(w,d_n)} \right)$$

**The Problem:** We have a **log of a sum** again! This is hard to maximize directly.

**The Solution:** Resort to the EM Algorithm.

---

## 7. The EM Algorithm for Document Clustering

### The Q Function

The expected complete-data log likelihood under the posterior distribution:

$$Q(\boldsymbol{\theta}, \boldsymbol{\theta}^{\text{old}}) := \sum_{n=1}^{N} \sum_{k=1}^{K} p(z_{n,k} = 1 | d_n, \boldsymbol{\theta}^{\text{old}}) \ln p(z_{n,k} = 1, d_n | \boldsymbol{\theta})$$

Expanding this:

$$= \sum_{n=1}^{N} \sum_{k=1}^{K} p(z_{n,k} = 1 | d_n, \boldsymbol{\theta}^{\text{old}}) \left( \ln \varphi_k + \sum_{w \in \mathcal{A}} c(w, d_n) \ln \mu_{k,w} \right)$$

Using our responsibility notation:

$$= \sum_{n=1}^{N} \sum_{k=1}^{K} \gamma(z_{n,k}) \left( \ln \varphi_k + \sum_{w \in \mathcal{A}} c(w, d_n) \ln \mu_{k,w} \right)$$

where $\boldsymbol{\theta} := (\boldsymbol{\varphi}, \boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_K)$ is the collection of model parameters, and $\gamma(z_n, k) := p(z_{n,k} = 1 | d_n, \boldsymbol{\theta}^{\text{old}})$ are the responsibility factors.

---

## 8. Parameter Updates from EM

Maximizing the Q function (using Lagrangian to enforce constraints), we get:

### Mixing Components

$$\varphi_k = \frac{N_k}{N} \quad \text{where} \quad N_k := \sum_{n=1}^{N} \gamma(z_{n,k})$$

**Interpretation:** Same form as complete data, but now using **soft assignments** (responsibilities) instead of hard assignments.

### Word Proportion Parameters

$$\mu_{k,w} = \frac{\sum_{n=1}^{N} \gamma(z_{n,k}) c(w, d_n)}{\sum_{w' \in \mathcal{A}} \sum_{n=1}^{N} \gamma(z_{n,k}) c(w', d_n)}$$

**Interpretation:**

- Count word occurrences, but **weighted** by how much each document belongs to cluster $k$
- If document $n$ has responsibility 0.8 for cluster $k$, its word counts contribute 80% to that cluster's word distribution

---

## 9. The Complete EM Algorithm for Document Clustering

### Algorithm Steps

**Initialization:** Choose initial parameter values: $$\boldsymbol{\theta}^{\text{old}} = (\boldsymbol{\varphi}^{\text{old}}, \boldsymbol{\mu}_1^{\text{old}}, \ldots, \boldsymbol{\mu}_K^{\text{old}})$$

**While convergence is not met:**

**E-step:** Set $\forall n, \forall k: \gamma(z_{n,k})$ based on $\boldsymbol{\theta}^{\text{old}}$

Calculate responsibilities for each document-cluster pair: $$\gamma(z_{n,k}) = p(z_{n,k} = 1 | d_n, \boldsymbol{\theta}^{\text{old}})$$

Using Bayes' rule: $$= \frac{\varphi_k^{\text{old}} \prod_{w \in \mathcal{A}} (\mu_{k,w}^{\text{old}})^{c(w,d_n)}}{\sum_{j=1}^{K} \varphi_j^{\text{old}} \prod_{w \in \mathcal{A}} (\mu_{j,w}^{\text{old}})^{c(w,d_n)}}$$

**M-step:** Set $\boldsymbol{\theta}^{\text{new}}$ based on $\forall n, \forall k: \gamma(z_{n,k})$

Update parameters using the formulas above:

- Effective cluster size: $N_k = \sum_{n=1}^{N} \gamma(z_{n,k})$
- Mixing coefficients: $\varphi_k^{\text{new}} = \frac{N_k}{N}$
- Word proportions: $\mu_{k,w}^{\text{new}} = \frac{\sum_{n=1}^{N} \gamma(z_{n,k}) c(w, d_n)}{\sum_{w' \in \mathcal{A}} \sum_{n=1}^{N} \gamma(z_{n,k}) c(w', d_n)}$

**Update:** $\boldsymbol{\theta}^{\text{old}} \leftarrow \boldsymbol{\theta}^{\text{new}}$

---

## 10. Key Concepts and Intuitions

### Comparison: Complete vs Incomplete Data

|Aspect|Complete Data|Incomplete Data (EM)|
|---|---|---|
|**Assignment**|Hard: $z_{nk} \in {0,1}$|Soft: $\gamma(z_{nk}) \in [0,1]$|
|**Word counts**|Count words in assigned cluster|Weight word counts by responsibility|
|**Mixing coefficients**|Count documents in cluster|Sum of responsibilities|
|**Optimization**|Direct (closed-form)|Iterative (EM algorithm)|

### The Power of the Bag of Words Model

**Advantages:**

- Simple representation
- Computationally efficient
- Works well for topic modeling and clustering

**Limitations:**

- Loses word order information
- Loses grammatical structure
- "Dog bites man" vs "Man bites dog" are identical

### Multinomial Distribution Connection

The generative model uses a **multinomial distribution** twice:

1. **First multinomial:** Choose which cluster (dice with $K$ faces, probabilities $\boldsymbol{\varphi}$)
2. **Second multinomial:** Generate each word (dice with $|\mathcal{A}|$ faces, probabilities $\boldsymbol{\mu}_k$)

This is why it's called a **mixture of multinomials** model for document clustering.

---

## 11. Mathematical Details: Why c(w,d) Appears

### Understanding the Exponent

In the probability formula: $$p(d|k) = \prod_{w \in \mathcal{A}} \mu_{k,w}^{c(w,d)}$$

**Example:** If document $d$ = "the cat sat on the mat", then:

- $c(\text{"the"}, d) = 2$
- $c(\text{"cat"}, d) = 1$
- $c(\text{"sat"}, d) = 1$
- $c(\text{"on"}, d) = 1$
- $c(\text{"mat"}, d) = 1$

So: $$p(d|k) = \mu_{k,\text{"the"}}^2 \cdot \mu_{k,\text{"cat"}}^1 \cdot \mu_{k,\text{"sat"}}^1 \cdot \mu_{k,\text{"on"}}^1 \cdot \mu_{k,\text{"mat"}}^1$$

### In the Log-Likelihood

When we take the logarithm: $$\ln p(d|k) = \sum_{w \in \mathcal{A}} c(w,d) \ln \mu_{k,w}$$

The counts become **multiplicative weights** in the log-likelihood!

---

## 12. Practical Considerations

### Initialization

- Random initialization of $\boldsymbol{\mu}_k$ (ensuring they sum to 1)
- Could use K-Means-style initialization
- Could initialize from random document assignments

### Convergence

- Monitor change in log-likelihood
- Or monitor change in parameter values
- Or monitor change in responsibilities

### Applications

- Topic modeling
- Document categorization
- Content recommendation
- Spam detection
- News article clustering

---

## Summary: Key Takeaways

1. **Document clustering** extends the GMM framework to discrete data (words) using multinomial distributions
    
2. **Bag of words** representation treats documents as unordered collections of words
    
3. **Complete data** has known cluster assignments → simple counting and normalization
    
4. **Incomplete data** requires EM → responsibilities replace hard assignments
    
5. **The Q function** transforms the intractable log-of-sum into a tractable sum-of-logs
    
6. **Parameter updates** have intuitive interpretations: weighted counts normalized appropriately
    
7. **This framework** is the basis for more sophisticated models like Latent Dirichlet Allocation (LDA)