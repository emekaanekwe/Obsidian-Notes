# Study Sheet: K-Means and Gaussian Mixture Models (GMMs)

---

## 1. The K-Means Algorithm

### Overview

K-Means is an **unsupervised clustering algorithm** that partitions $N$ unlabeled data points $\mathbf{x}_n$ into $K$ distinct clusters based on similarity (measured by distance, typically Euclidean distance).

### The K-Means Algorithm Steps

**Initialization:** Start with random guesses of $K$ cluster centers: $(\boldsymbol{\mu}_1^{(0)}, \ldots, \boldsymbol{\mu}_K^{(0)})$

**Iterate until convergence:**

1. **Update Assignment of Datapoints to Clusters (E-step):**
    
    - Calculate the distance of each data point $\mathbf{x}_n$ to all cluster centers
    - Assign each datapoint to the cluster with the **minimum distance**
    - Mathematically, the assignment indicator $r_{nk}$ is:
    
    $$r_{nk} = \begin{cases} 1 & \text{if } k = \arg\min_j d(\mathbf{x}_n, \boldsymbol{\mu}_j^{(r)}) \ 0 & \text{otherwise} \end{cases}$$
    
2. **Update Centers of the Clusters (M-step):**
    
    - For each cluster, calculate the new center as the **average** of all datapoints assigned to it:
    
    $$\boldsymbol{\mu}_k^{(r+1)} = \frac{\sum_n r_{nk} \mathbf{x}_n}{\sum_n r_{nk}}$$
    

### K-Means as an Optimization Problem

K-Means minimizes the following **objective function**:

$$J(\boldsymbol{\mu}, \mathbf{r}) := \sum_{n=1}^{N} \sum_{k=1}^{K} r_{nk} \times d(\mathbf{x}_n, \boldsymbol{\mu}_k)$$

**Constraints:**

- $r_{nk} \in {0, 1}$ (binary assignment)
- $\sum_{k=1}^{K} r_{nk} = 1$ for all $n$ (each point assigned to exactly one cluster)

**Variables being optimized:**

- Cluster centers: $\boldsymbol{\mu} := (\boldsymbol{\mu}_1, \ldots, \boldsymbol{\mu}_K)$
- Cluster assignments: $\mathbf{r} := (r_1, \ldots, r_N)$

### Key Properties of K-Means

1. **Sensitive to initialization:** Different initial cluster centers may lead to different solutions (local minima)
2. **Hard assignment:** Each data point belongs to **one and only one** cluster (non-probabilistic)
3. **Coordinate descent optimization:** Alternates between optimizing assignments (fixing centers) and optimizing centers (fixing assignments)

---

## 2. Gaussian Mixture Models (GMMs)

### The Generative Story

GMMs provide a **probabilistic** approach to clustering. The generative story for creating a data point $(k, \mathbf{x})$ is:

1. **First:** Generate a cluster label $k$ by tossing a dice with $K$ faces, where each face corresponds to a cluster. The probability of landing on face $k$ is $\varphi_k$.
    
2. **Second:** Generate the data point $\mathbf{x}$ by sampling from a Gaussian distribution $p_k(\cdot)$ corresponding to cluster label $k$.
    

### Key Concepts

**Latent Variable:** The cluster label $z \in {1, \ldots, K}$ is **hidden** from us by the "oracle". We only observe the data points $\mathbf{x}$, not which cluster they came from.

**Goal:**

- Find the best values for the latent variables (cluster assignments)
- Find the best estimates for the model parameters

### The Probabilistic Generative Model

**Multinomial Distribution for Cluster Selection:**

- Tossing a dice with $K$ faces is equivalent to sampling from a **multinomial distribution** on $K$ elements
- Parameters: probabilities $\varphi_k$ of selecting each cluster, where: $$\sum_{k=1}^{K} \varphi_k = 1 \quad \text{and} \quad \varphi_k \geq 0$$

**Gaussian Distribution for Data Generation:**

- For cluster $k$, data points are sampled from: $\mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$
- Parameters: mean $\boldsymbol{\mu}_k$ and covariance $\boldsymbol{\Sigma}_k$
- We have a **collection** of $K$ Gaussian distributions, one for each cluster

### Complete vs Incomplete Data

**Complete Data:** If we knew both the data point and its label $(k, \mathbf{x})$, the probability would be:

$$p(k, \mathbf{x}_n) = p(\text{face } k) \cdot p(\mathbf{x}_n | \text{face } k) = \varphi_k \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

**Incomplete (Observed) Data:** In practice, we only have the data $\mathbf{x}$, not the labels. We **marginalize out** the latent variable:

$$p(\mathbf{x}_n) = \sum_{z_n \in {1, \ldots, K}} p(z_n, \mathbf{x}_n) = \sum_{k=1}^{K} p(z_n = k) \cdot p(\mathbf{x} | \text{face } k)$$

$$= \sum_{k=1}^{K} \varphi_k \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

This is called the **Gaussian Mixture Model** - a mixture of Gaussian components with mixing coefficients $\boldsymbol{\varphi}$.

### Log-Likelihood of Observed Data

The log-likelihood we want to maximize is:

$$\mathcal{L}(\boldsymbol{\theta}) := \sum_{n=1}^{N} \ln p(\mathbf{x}_n) = \sum_{n=1}^{N} \ln \sum_{k=1}^{K} \varphi_k \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

where the model parameters are: $$\boldsymbol{\theta} := (\boldsymbol{\varphi}, \boldsymbol{\mu}_1, \boldsymbol{\Sigma}_1, \ldots, \boldsymbol{\mu}_K, \boldsymbol{\Sigma}_K)$$

**Challenge:** The log of a sum is **not** a sum of logs! This makes direct optimization difficult. (Unlike the complete data case where we had $\ln p(X, Z)$ which simplified nicely)

---

## 3. The Prediction Rule (Soft Assignment)

After estimating model parameters, we can predict which cluster an observed datapoint $\mathbf{x}_n$ belongs to using **Bayes' rule**:

$$\gamma(z_{nk}) := p(z_n = k | \mathbf{x}_n) = \frac{\varphi_k \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \varphi_j \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}$$

where $\sum_{k=1}^{K} \gamma(z_{nk}) = 1$

### Interpretation

- $\varphi_k$ is the **prior probability** of cluster $z_n = k$ (before seeing data)
- $\gamma(z_{nk})$ is the **posterior probability** of cluster $k$ (after observing $\mathbf{x}_n$)
- $\gamma(z_{nk})$ is the **responsibility** that cluster $k$ takes for explaining observation $\mathbf{x}_n$

**Key Difference from K-Means:**

- K-Means: **Hard assignment** - each point belongs to exactly one cluster ($r_{nk} \in {0,1}$)
- GMM: **Soft assignment** (partial assignment) - each point has a probability of belonging to each cluster ($\gamma(z_{nk}) \in [0,1]$)

---

## 4. Key Comparisons: K-Means vs GMMs

|Aspect|K-Means|GMMs|
|---|---|---|
|**Assignment Type**|Hard (binary: 0 or 1)|Soft (probabilistic: 0 to 1)|
|**Model Type**|Geometric/distance-based|Probabilistic/generative|
|**Cluster Shape**|Spherical (equal variance)|Elliptical (different variances)|
|**Objective**|Minimize distances|Maximize likelihood|
|**Parameters**|Cluster centers $\boldsymbol{\mu}_k$|Centers $\boldsymbol{\mu}_k$, covariances $\boldsymbol{\Sigma}_k$, mixing coefficients $\varphi_k$|
|**Uncertainty**|No uncertainty quantification|Provides probabilities (uncertainty)|

---

## 5. Important Notes for Next Steps

The material hints that:

1. **Cluster prediction** (finding assignments) and **parameter estimation** are intertwined
2. They are done **simultaneously** in an iterative algorithm
3. This algorithm is called the **Expectation-Maximization (EM) Algorithm**
4. The EM algorithm finds both the best latent variable assignments AND the best parameter estimates

This sets up perfectly for your next upload on the EM algorithm! 🎯

---

**Quick Memory Aid:**

- K-Means = "Which cluster?" (hard decision)
- GMM = "How much of each cluster?" (soft decision with probabilities)


# Study Sheet: Expectation-Maximization (EM) Algorithm for GMMs

---

## 1. The EM Algorithm Motivation

### The Challenge with GMMs

Recall the log-likelihood for GMMs (incomplete/observed data):

$$\mathcal{L}(\boldsymbol{\theta}) = \sum_{n=1}^{N} \ln \sum_{k=1}^{K} \varphi_k \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

**Problem:** The log of a sum makes this difficult to optimize directly!

**Solution:** The EM algorithm provides an iterative scheme to find the maximum likelihood solution.

---

## 2. Deriving the EM Updates for GMMs

The EM algorithm finds optimal parameters by setting gradients to zero. Let's derive each parameter update:

### 2.1 Updating the Mean Parameters $\boldsymbol{\mu}_k$

Setting the gradient of $\mathcal{L}(\boldsymbol{\theta})$ with respect to $\boldsymbol{\mu}_k$ to zero:

$$\frac{\partial}{\partial \boldsymbol{\mu}_k} \mathcal{L}(\boldsymbol{\theta}) = 0 \Rightarrow \sum_{n=1}^{N} \left( \frac{\varphi_k \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \varphi_j \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)} \sum_k (\mathbf{x}_n - \boldsymbol{\mu}_k) \right) = \mathbf{0}$$

Notice that the first fraction is exactly our **responsibility** $\gamma(z_{nk})$!

$$\underbrace{\frac{\varphi_k \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \varphi_j \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}}_{\gamma(z_{nk})}$$

Rearranging terms gives us:

$$\boldsymbol{\mu}_k = \frac{1}{N_k} \sum_{n=1}^{N} \gamma(z_{nk}) \mathbf{x}_n$$

where the **effective number of points** assigned to cluster $k$ is:

$$N_k := \sum_{n=1}^{N} \gamma(z_{nk})$$

**Interpretation:** The mean $\boldsymbol{\mu}_k$ for the $k$-th Gaussian is a **weighted average** of all points in the dataset, where the weighting factor for datapoint $\mathbf{x}_n$ is the posterior probability $\gamma(z_{nk})$ that component $k$ was responsible for generating $\mathbf{x}_n$.

---

### 2.2 Updating the Covariance Matrices $\boldsymbol{\Sigma}_k$

Similarly, setting the gradient with respect to $\boldsymbol{\Sigma}_k$ to zero and following a similar derivation:

$$\frac{\partial}{\partial \boldsymbol{\Sigma}_k} \mathcal{L}(\boldsymbol{\theta}) = 0 \Rightarrow \boldsymbol{\Sigma}_k = \frac{1}{N_k} \sum_{n=1}^{N} \gamma(z_{nk}) (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^T$$

**Interpretation:** Same form as the covariance for a single Gaussian fitted to the data, but with each data point weighted by its corresponding posterior probability and denominator given by the effective number of points.

---

### 2.3 Updating the Mixing Coefficients $\boldsymbol{\varphi}$

For the mixing coefficients, we must respect the constraint: $\sum_{k=1}^{K} \varphi_k = 1$.

We use a **Lagrange multiplier** $\lambda$ and maximize:

$$\mathcal{L}(\boldsymbol{\theta}) + \lambda \left( \sum_{k=1}^{K} \varphi_k - 1 \right)$$

Taking the gradient with respect to $\varphi_k$ and setting to zero:

$$\frac{\partial}{\partial \varphi_k} \left[ \mathcal{L}(\boldsymbol{\theta}) + \lambda \left( \sum_{k=1}^{K} \varphi_k - 1 \right) \right] = 0$$

This gives us:

$$\sum_{n=1}^{N} \frac{\mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \varphi_j \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)} + \lambda = 0$$

Multiply both sides by $\varphi_k$:

$$\sum_{n=1}^{N} \frac{\varphi_k \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_j \varphi_j \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)} + \lambda \varphi_k = 0$$

$$\Rightarrow \sum_{n=1}^{N} \gamma_{nk} + \lambda \varphi_k = 0 \Rightarrow \varphi_k = \frac{\sum_{n=1}^{N} \gamma_{nk}}{-\lambda}$$

Using the constraint $\sum_k \varphi_k = 1$, we find $\lambda = -N$. Hence:

$$\varphi_k = \frac{N_k}{N}$$

**Interpretation:** The mixing coefficient for the $k$-th component is the **average responsibility** that component takes for explaining the data points. It's simply the effective number of points in cluster $k$ divided by total points.

---

## 3. The Complete EM Algorithm for GMMs

### Key Insight

The optimal parameter estimates do **not** constitute a closed-form solution because the responsibilities $\gamma(z_{nk})$ depend on the parameters in a complex way. However, these results suggest a simple **iterative scheme**.

### Algorithm Steps

**Initialization:** Choose initial values for: $$\boldsymbol{\theta}^{\text{old}} = (\boldsymbol{\varphi}^{\text{old}}, \boldsymbol{\mu}_1^{\text{old}}, \ldots, \boldsymbol{\mu}_K^{\text{old}}, \boldsymbol{\Sigma}_1^{\text{old}}, \ldots, \boldsymbol{\Sigma}_K^{\text{old}})$$

**Iterate until convergence:**

**E-step (Expectation):** Use current parameter values to evaluate the posterior probabilities (responsibilities):

$$\gamma(z_{nk}) = \frac{\varphi_k^{\text{old}} \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k^{\text{old}}, \boldsymbol{\Sigma}_k^{\text{old}})}{\sum_j \varphi_j^{\text{old}} \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_j^{\text{old}}, \boldsymbol{\Sigma}_j^{\text{old}})}$$

Compute for all $n = 1, \ldots, N$ and $k = 1, \ldots, K$.

**M-step (Maximization):** Re-estimate the parameters using the current responsibilities:

$$N_k := \sum_{n=1}^{N} \gamma(z_{nk})$$

$$\boldsymbol{\mu}_k^{\text{new}} = \frac{1}{N_k} \sum_{n=1}^{N} \gamma(z_{nk}) \mathbf{x}_n$$

$$\boldsymbol{\Sigma}_k^{\text{new}} = \frac{1}{N_k} \sum_{n=1}^{N} \gamma(z_{nk}) (\mathbf{x}_n - \boldsymbol{\mu}_k)(\mathbf{x}_n - \boldsymbol{\mu}_k)^T$$

$$\varphi_k^{\text{new}} = \frac{N_k}{N}$$

**Update:** Set $\boldsymbol{\theta}^{\text{old}} \leftarrow \boldsymbol{\theta}^{\text{new}}$

---

## 4. Relationship to K-Means

The EM algorithm for GMMs is reminiscent of K-Means, with key differences:

|Aspect|K-Means|EM for GMMs|
|---|---|---|
|**Parameters**|Only means $\boldsymbol{\mu}_k$|Means $\boldsymbol{\mu}_k$, covariances $\boldsymbol{\Sigma}_k$, mixing coefficients $\boldsymbol{\varphi}$|
|**Assignment**|Hard: $r_{nk} \in {0, 1}$|Soft: $\gamma(z_{nk}) \in [0, 1]$|
|**E-step**|Assign to nearest cluster|Compute responsibilities (posterior probabilities)|
|**M-step**|Update means only|Update means, covariances, and mixing coefficients|
|**Distance metric**|Euclidean distance|Mahalanobis distance (accounts for covariance)|

**Key Insight:** K-Means can be viewed as a special case of EM where:

- Covariances are fixed to identity matrices
- Assignments are "hardened" to 0 or 1

---

## 5. Important Properties of EM

1. **Monotonic Increase:** Each update resulting from an E-step followed by an M-step is **guaranteed to increase** the log-likelihood function (we don't prove this here, but it's a fundamental property).
    
2. **Convergence Criterion:** In practice, the algorithm is deemed to have converged when the change in the log-likelihood function, or alternatively in the parameters, falls below some threshold.
    
3. **Computational Cost:** The EM algorithm takes many more iterations to reach convergence compared to K-Means, and each cycle requires significantly more computations.
    
4. **Local Optima:** Like K-Means, EM can converge to different solutions depending on initialization.
    

---

## 6. Visual Intuition: EM Steps Illustrated

From Figure 4.2.1 in your notes:

- **Plot (a):** Initial configuration with random Gaussian components (blue and red circles) far from the actual data clusters (green points)
    
- **Plot (b):** After first E-step, points are colored based on their posterior probabilities (purple = mixed responsibility between blue and red)
    
- **Plot (c):** After first M-step ($L=1$), the means have moved toward the data clusters, and covariances have been updated (ellipses show the shape/orientation)
    
- **Plots (d), (e), (f):** After 2, 5, and 20 complete EM cycles, the algorithm progressively improves the fit. By $L=20$, the algorithm is close to convergence with well-separated clusters.
    

**Key Observation:** The **elliptical** shape of the Gaussians (from covariance matrices) allows GMMs to capture clusters with different shapes and orientations, unlike K-Means which assumes spherical clusters.

---

## 7. The General EM Algorithm Framework

### Notation for General Case

- **Observed data:** $\mathbf{X}$
- **Latent variables:** $\mathbf{Z}$
- **Model parameters:** $\boldsymbol{\theta}$
- **Complete data:** ${\mathbf{X}, \mathbf{Z}}$ (if we knew the latent variables)
- **Incomplete data:** $\mathbf{X}$ (what we actually observe)

### The Training Objective

The goal is to find the maximum likelihood solution for models with latent variables:

$$\ln p(\mathbf{X} | \boldsymbol{\theta}) = \ln \sum_{\mathbf{Z}} p(\mathbf{X}, \mathbf{Z} | \boldsymbol{\theta})$$

### Why We Need EM

**The Problem:** We are not given the complete dataset ${\mathbf{X}, \mathbf{Z}}$, only the incomplete data $\mathbf{X}$.

**The Solution:** Our knowledge of the latent variables $\mathbf{Z}$ is given by the posterior distribution $p(\mathbf{Z} | \mathbf{X}, \boldsymbol{\theta})$.

Since we cannot use the complete-data log likelihood directly, we consider its **expected value** under the posterior distribution of the latent variables.

---

## 8. The General EM Algorithm

### The Q Function

Define the **Q function** as the expected value of the complete-data log likelihood:

$$Q(\boldsymbol{\theta}, \boldsymbol{\theta}^{\text{old}}) := \sum_{\mathbf{Z}} p(\mathbf{Z} | \mathbf{X}, \boldsymbol{\theta}^{\text{old}}) \ln p(\mathbf{X}, \mathbf{Z} | \boldsymbol{\theta})$$

**Note:** The logarithm acts directly on the joint distribution $p(\mathbf{X}, \mathbf{Z} | \boldsymbol{\theta})$, so the M-step maximization will be tractable!

### General EM Steps

**Choose initial parameters:** $\boldsymbol{\theta}^{\text{old}}$

**While convergence not met:**

**E-step:** Evaluate the posterior distribution of latent variables: $$p(\mathbf{Z} | \mathbf{X}, \boldsymbol{\theta}^{\text{old}})$$

**M-step:** Evaluate $\boldsymbol{\theta}^{\text{new}}$ by maximizing the Q function: $$\boldsymbol{\theta}^{\text{new}} \leftarrow \arg\max_{\boldsymbol{\theta}} \sum_{\mathbf{Z}} p(\mathbf{Z} | \mathbf{X}, \boldsymbol{\theta}^{\text{old}}) \ln p(\mathbf{X}, \mathbf{Z} | \boldsymbol{\theta})$$

**Update:** $\boldsymbol{\theta}^{\text{old}} \leftarrow \boldsymbol{\theta}^{\text{new}}$

**Guarantee:** The incomplete-data log likelihood is guaranteed to increase with each cycle (non-decreasing).

---

## 9. EM for GMMs: Connecting to the General Framework

### The Complete Data Log-Likelihood

If we knew the cluster assignments $\mathbf{Z} := {z_1, \ldots, z_N}$ where $\mathbf{z}_n := (z_{n1}, \ldots, z_{nk})$ is the one-hot encoded vector ($z_{nk} = 1$ if point $n$ belongs to cluster $k$), then:

$$\ln p(\mathbf{X}, \mathbf{Z} | \mu, \Sigma, \varphi) = \ln \prod_{n=1}^{N} \prod_{k=1}^{K} \left[ \varphi^{z_{nk}} \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)^{z_{nk}} \right]$$

$$= \sum_{n=1}^{N} \sum_{k=1}^{K} z_{nk} \ln \varphi_k + z_{nk} \ln \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

This is easy to maximize! (It's just a sum, not a log of sums)

### The Q Function for GMMs

$$Q(\boldsymbol{\theta}, \boldsymbol{\theta}^{\text{old}}) = \sum_{\mathbf{Z}} p(\mathbf{Z} | \mathbf{X}, \boldsymbol{\theta}^{\text{old}}) \ln p(\mathbf{X}, \mathbf{Z} | \boldsymbol{\theta})$$

$$= \sum_{n=1}^{N} \sum_{k=1}^{K} p(z_{nk} = 1 | \mathbf{x}_n, \boldsymbol{\theta}^{\text{old}}) \left[ z_{nk} \ln \varphi_k + z_{nk} \ln \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) \right]$$

$$= \sum_{n=1}^{N} \sum_{k=1}^{K} p(z_{nk} = 1 | \mathbf{x}_n, \boldsymbol{\theta}^{\text{old}}) \ln \varphi_k + p(z_{nk} = 1 | \mathbf{x}_n, \boldsymbol{\theta}^{\text{old}}) \ln \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$$

where $p(z_{nk} = 1 | \mathbf{x}_n, \boldsymbol{\theta}^{\text{old}})$ is exactly our responsibility:

$$\gamma(z_{nk}) := p(z_{nk} = 1 | \mathbf{x}_n, \boldsymbol{\theta}^{\text{old}}) = \frac{\varphi_k^{\text{old}} \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_k^{\text{old}}, \boldsymbol{\Sigma}_k^{\text{old}})}{\sum_j \varphi_j^{\text{old}} \mathcal{N}(\mathbf{x}_n | \boldsymbol{\mu}_j^{\text{old}}, \boldsymbol{\Sigma}_j^{\text{old}})}$$

Maximizing the Q function leads to the parameter updates we derived earlier!

---

## 10. Hard-EM Algorithm

A variant mentioned in the notes:

**Hard-EM:** Instead of taking an expectation over the latent variables, only the **most probable value** for the latent variable is chosen to define the Q function.

This is more similar to K-Means where we make hard assignments rather than soft assignments.

---

## Summary: Key Takeaways

1. **EM solves the "log of sum" problem** by introducing latent variables and working with the expected complete-data log likelihood
    
2. **E-step computes responsibilities** (soft assignments) based on current parameter estimates
    
3. **M-step updates parameters** using weighted statistics where weights are the responsibilities
    
4. **EM is guaranteed to increase (or maintain) the likelihood** at each iteration
    
5. **GMMs generalize K-Means** by adding covariances and probabilistic assignments
    
6. **The general EM framework** applies to any model with latent variables where the complete-data log likelihood is easier to maximize than the incomplete-data log likelihood

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