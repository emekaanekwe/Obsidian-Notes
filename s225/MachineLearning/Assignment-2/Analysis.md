# EM Derivation
## Section 1, Question 2: EM Algorithm Derivation

  

### 2.1 High-Level Idea of EM Algorithm

  

The **Expectation-Maximization (EM)** algorithm is an iterative method for finding maximum likelihood estimates when data has missing or hidden variables.

  

**Core Intuition:**

If we knew cluster assignments → easy to estimate parameters (complete data MLE), and if we knew parameters → easy to infer cluster assignments (posterior probabilities). However, we do not know either. This is where we can use iteration.

  
  

**Two-Step Iteration:**

1. **E-Step (Expectation):** Given current parameters, compute expected cluster assignments

- Calculate posterior probabilities $\gamma_{ik} = P(z_i=k | d_i, \Theta^{old})$

- These are "soft" assignments (probabilities instead of hard labels)

  

2. **M-Step (Maximization):** Given expected assignments, find parameters that maximize expected complete-data likelihood

- Update $\pi_k$ (cluster priors)

- Update $\mu_{wk}$ (word probabilities in each cluster)

  

**Why It Works:**

- Each iteration is guaranteed to increase (or not decrease) the likelihood

- Converges to a local maximum

- Transforms hard problem (log-sum) into easier problems (separate optimizations)

  

---

  

### 2.2 Derivation of Soft-EM for Document Clustering

  
#### Start with proof of Correctness
Suppose $p(\phi|\psi)$ as some conditional PDF. We know that,
$$\ln p(X|\theta)=\ln \int p(X, y | \theta) dy$$
$$= \ln \int \frac{p(X, y | \theta)}{p(y, X| \theta_{old})}p(y|X,\theta_{old})dy$$
based on the rule of conditional probability and the rules of random variables. We can then get,
$$\begin{matrix}= \ln E[ \frac{p(X, y | \theta)}{p(y, X| \theta_{old})}\ |\ (X ,\theta_{old})] \end{matrix}$$
Using *Jensen's Inequality*, 
$$\begin{matrix}= \ln E[ \frac{p(X, y | \theta)}{p(y, X| \theta_{old})}\ |\ (X ,\theta_{old})] \end{matrix}$$
#### Model Parameters to Learn:

- $\pi_k$: Prior probability of cluster $k$, where $\sum_{k=1}^{K} \pi_k = 1$

- $\mu_{wk}$: Probability of word $w$ in cluster $k$, where $\sum_{w \in A} \mu_{wk} = 1$ for each $k$

  

---

  

#### E-Step: Compute Posterior Probabilities

  

Calculate the responsibility (posterior probability) that cluster $k$ generated document $i$:

  

$$\gamma_{ik} = P(z_i = k | d_i, \Theta^{old}) = \frac{P(d_i | z_i=k, \Theta^{old}) P(z_i=k | \Theta^{old})}{P(d_i | \Theta^{old})}$$

  

Using Bayes' theorem:

  

$$\gamma_{ik} = \frac{\pi_k \prod_{w \in A} \mu_{wk}^{n_{iw}}}{\sum_{j=1}^{K} \pi_j \prod_{w \in A} \mu_{wj}^{n_{iw}}}$$

  

**Log-Space Computation (for numerical stability):**

  

$$\log \gamma_{ik} = \log \pi_k + \sum_{w \in A} n_{iw} \log \mu_{wk} - \log \sum_{j=1}^{K} \exp\left(\log \pi_j + \sum_{w \in A} n_{iw} \log \mu_{wj}\right)$$

  

**Properties:**

- $\sum_{k=1}^{K} \gamma_{ik} = 1$ (probabilities sum to 1)

- $\gamma_{ik} \in [0, 1]$ (valid probability)

  

---

  

#### M-Step: Update Parameters

  

Maximize the expected complete-data log-likelihood:

  

$$Q(\Theta | \Theta^{old}) = \sum_{i=1}^{N} \sum_{k=1}^{K} \gamma_{ik} \left[ \log \pi_k + \sum_{w \in A} n_{iw} \log \mu_{wk} \right]$$

  

**Update $\pi_k$ (cluster priors):**

  

Using Lagrange multipliers for constraint $\sum_k \pi_k = 1$:

  

$$\pi_k^{new} = \frac{\sum_{i=1}^{N} \gamma_{ik}}{N}$$

  

**Interpretation:** Average responsibility of cluster $k$ across all documents

  

**Update $\mu_{wk}$ (word probabilities):**

  

Using Lagrange multipliers for constraint $\sum_w \mu_{wk} = 1$:

  

$$\mu_{wk}^{new} = \frac{\sum_{i=1}^{N} \gamma_{ik} n_{iw}}{\sum_{i=1}^{N} \gamma_{ik} \sum_{w' \in A} n_{iw'}}$$

  

Simplifying:

  

$$\mu_{wk}^{new} = \frac{\sum_{i=1}^{N} \gamma_{ik} n_{iw}}{\sum_{i=1}^{N} \gamma_{ik} \cdot |d_i|}$$

  

where $|d_i| = \sum_w n_{iw}$ is the total word count in document $i$.

  

**Interpretation:** Weighted average of word $w$ counts, weighted by cluster responsibilities

  

---

  

#### Hard-EM vs Soft-EM

  

**Soft-EM (derived above):**

- Uses probabilistic assignments $\gamma_{ik} \in [0,1]$

- All clusters contribute to parameter updates

- More robust but slower convergence

  

**Hard-EM:**

- Uses hard assignments: $\gamma_{ik} = 1$ if $k = \arg\max_j \gamma_{ij}$, else $0$

- Only most probable cluster contributes

- Equivalent to K-means for document clustering

- Faster but can be less stable

  

**Algorithm Summary:**

  

1. **Initialize** $\pi_k, \mu_{wk}$ randomly (ensuring constraints)

2. **Repeat until convergence:**

- **E-step:** Compute $\gamma_{ik}$ for all documents and clusters

- **M-step:** Update $\pi_k$ and $\mu_{wk}$ using above formulas

- **Check:** Compute log-likelihood; stop if change < threshold

3. **Assign** documents to clusters: $z_i^* = \arg\max_k \gamma_{ik}$