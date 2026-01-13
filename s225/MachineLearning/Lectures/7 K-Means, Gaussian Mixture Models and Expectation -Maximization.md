##### Vocab







# Core things to Understand
what is EM and what are the major steps?

(note: there are many that call it BKDM)

*sidenote: there is a clustering algorithm that is matrix-based




# What we have done so far

have done supervised machine learning. 
training model using log function and minimizin it
	all to learn the params of model

# What we are doing differently

## Unsupervised Learning

	the output not given, only the input
What we want to do is partition the clusters during training time, i.e. *clustering problem*
The unobserved labels are called *latent variables*

### Latent Variables
	label of training data are latent since cannot be directly observed

**Note**: the model params are NOT the same as the latent vars, but BOTH are present

# Clustering Problem

We need to find a feasible structure from unlabeled data. For example, what is the natural grouping of various species of cats, dogs, fish, bears, tigers?
	
	Ans: there are theoretically infinitely many groupings possible

So, **what similarity matrix can we use to draw a relation between 2 data points?**, i.e. what is a good cluster?

### There are thus a variety of clustering algorithms that could be applied
	Our focus will be K-Means

## Soft vs Hard Clusters

### Hard
	data points abelong to a single cluster, and classification possible

### Soft
	data points can be a member of more than one cluster, based on some Pr function

note: covariance matrix deifnes the shae and spread
***you can think of clusters as density estimations***

## Solution: K-Means Clustering
***basic assumption: clusters have same shape and spread but diff $\mu$ can result in ***
***why k-menas?	the centroid is the avg of all clusters***
**Note: the distance matrices are given, thus avoiding any circular problems of clustering**
	essentially an iterative algorithm
for and initial random sampling of K cluster centers ,, iterate:
1. update assignment of DP clusters *guess the latent variables*
2. Update centers of the clusters *update the parameters*
Until a stopping criterion is met
![[Pasted image 20250908183450.png]]
where $rnk$ are the latent variables
where $\frac{\sum n^r n_k^x n}{\sum_n rnk}$ is the average of center of all data points.
ex:
$r_{nk}: 1 x_n$ and 0 otherwise
then binary classify 
![[Pasted image 20250908183926.png]]

**K-Means is Hard clustering (non Pr)**



## Solution: Gaussian Mixture



***have two distro to represent tow clusters and the data points related to them vwhere each distro represents a cluster***

***But notice that the distros canconverge and stbailize:***


![[Pasted image 20250908190010.png]]

Goal is to *estimate the params* of the distro. Allows us to have partial membership for each DP as well.

### Generative Approach to GMMs
	suppose each DP-label pair (k,x) 
process being:
1. generate a cluster labeled k, by tossing a die (*random choice*) with k faces where each face corresponds to a label
2. generate the data point x by sampling from the distro $Pr_k(.)$ corresponding to the cluster label k, i.e. *estimate*

#### Needing the Pr Generative Model
for K faces:
$$
\phi_k => 0, \ \ \ \sum^{k}_{k=} \phi_k=1, \ \ \ Pr(z_n = k)=\phi_k$$
Without knowing:
$$\theta := (\phi, \mu_1,\sum_1,...,\mu_k, \sum_k)$$
And we find the best estimate for the model parameters, which is done by using the *maximum likelihood estimation*
	Here, something like a closed for solution is not possible

**to start**: assume the labels are given to use, how to use MLP: 

![[Pasted image 20250908191718.png]]
**This allows you to assign Pr based on your generalization**
when the cluster labels are observed, the estimation is aid forward (can maximize, get closed form, etc)
when labels are not observed, need to sum over the labels using baysian methods (no closed form solution can be found)
	you need to sum out k (latent vars) in the function, which makes it complex
	![[Pasted image 20250908192411.png]]
![[Pasted image 20250908192721.png]]


## Recap
![[Pasted image 20250908192428.png]]

**The Expected Maximization (EM) algorithm** is what you will use to **maximize** likelihood function

Notice that the first step in the EM is similar to the K-Means algo, and the second. But EM is a more "generalization" of the K-Means model.
	Specifically, the EM algo can be reduced to the K-Means algo

## GMMs Revisited


***log-likelihood - incomplete data???***

### The Objective Function
Goal: to get the log likelihood,
$$$$
![[Pasted image 20250908193429.png]]
***Em maximizes a lower bound.***

### The E Step For GMMs
***E step - compute the soft cluster assignments for each DP think of it as posterior Pr of cluster label, given data points and parameters of the model
***
***compute the soft cluster assignments for each DP***
***think of it as posterior Pr of cluster label, given data points and parameters of the model***

***after E-step, you already know membership AND we know the proportion*** 

![[Pasted image 20250908193750.png]]


### The M Step

***M step - from the E-Step, update the parameters `Θ` to maximize the expected complete-data log-likelihood. This step has beautiful, intuitive closed-form solutions that look very much like the standard MLE for Gaussians, but now **weighted by the responsibilities**.***

***Note that the $\gamma$ part are the partial clusters of the ditro This is also taken as the weight of the class membership***

## Visual Example of E Step and M Step
![[Pasted image 20250908194653.png]]
***Once this is updated, it ll reach a point of convergence where blue and yellow will not change***

in a way, the EM algo does hard clustering for you
***"the cluster with the higest Pr, we set to 1, and others to 0"***

***If you add up all the Em algos, and set the mu's to the identify matrix***




# Highe level overview

1. initialize values
2. run k-means until the centroid does not move 
3. E tep for Pr distribution
4. M step - recalculate the Pr








































# Clustering K-Means
1.  **Clustering & K-Means:** Introduces the concept of finding natural groupings in unlabeled data. K-Means is presented as a simple, iterative, center-based algorithm that performs *hard clustering*, where each point belongs to one cluster.



### Overall Theme: From Hard to Soft Clustering

The lecture progresses from a simple, intuitive, but limited algorithm (K-Means) to a more powerful, probabilistic framework (Gaussian Mixture Models) that requires a sophisticated algorithm to solve (Expectation-Maximization).

---

### Part 1: Clustering and K-Means Algorithm

#### 1.1 Core Concept
*   **Unsupervised Learning:** Finding hidden structure in data without pre-existing labels.
*   **Goal:** Group data points so that points in the same group (cluster) are more **similar** to each other than to those in other groups.
*   **Measures:** High **intra-cluster similarity**, low **inter-cluster similarity**.

#### 1.2 K-Means: The Algorithm (The Mathematics)
K-Means is an iterative algorithm that aims to partition `N` data points `{x₁, x₂, ..., x_N}` into `K` (≪ N) clusters.

**a) The Objective:** Minimize the **within-cluster sum of squares (WCSS)** or **inertia**.
`J = Σₙ Σₖ rₙₖ ||xₙ - μₖ||²`
*   `μₖ` is the center (centroid) of the `k-th` cluster.
*   `rₙₖ` is an **assignment variable**. It's 1 if point `xₙ` is in cluster `k`, and 0 otherwise. This is what makes it **hard clustering**.
*   `||xₙ - μₖ||` is the Euclidean distance.

**b) The Two-Step Iteration:**
1.  **Assignment Step (E-Step analog):** For each data point `xₙ`, find the closest centroid.
    `rₙₖ = { 1 if k = argmin_j ||xₙ - μ_j||²; 0 otherwise }`

2.  **Update Step (M-Step analog):** For each cluster `k`, recalculate the centroid `μₖ` as the mean of all points assigned to it.
    `μₖ = (Σₙ rₙₖ xₙ) / (Σₙ rₙₖ)`

You iterate until the assignments don't change (or change very little).

#### 1.3 Hand-Written Example: K-Means
Let's see this in action with a simple 1D example. We'll let K=2.

```plaintext
Initial Data: [1, 2, 3, 10, 11, 12]
Step 0: Randomly initialize centroids. Let's pick μ₁=2, μ₂=11.

Iteration 1:
- Assign points:
  Points near μ₁ (2): [1, 2, 3] -> Cluster 1
  Points near μ₂ (11): [10, 11, 12] -> Cluster 2
- Update centroids:
  New μ₁ = mean([1,2,3]) = 2
  New μ₂ = mean([10,11,12]) = 11

Centroids didn't change. Algorithm has converged.
Final Clusters: [1,2,3] and [10,11,12]
```
*(This is a simplified textual representation of the calculation you'd do by hand.)*

#### 1.4 Key Drawbacks of K-Means
*   **Hard Assignments:** A point is 100% in one cluster. What if it's in between?
*   **Sensitive to Initialization:** Bad initial centroids can lead to poor results.
*   **Assumes Circular Clusters:** Struggles with elongated or irregular shapes because it uses Euclidean distance.
*   **You must choose K:** The algorithm doesn't know how many clusters exist.

---

### Part 2: Gaussian Mixture Models (GMMs) & The Need for EM

#### 2.1 The Probabilistic Mindset
GMMs address K-Means' limitations by introducing probability. The idea is that the data is generated from a **mixture** of `K` Gaussian (Normal) distributions.

**The Generative Story (How we imagine the data was created):**
1.  First, pick a cluster `k` with probability `φₖ`. (e.g., roll a weighted die with `K` sides).
    `P(zₙ = k) = φₖ` where `Σ φₖ = 1` and `φₖ ≥ 0`. (`zₙ` is the latent cluster label for point `xₙ`).
2.  Then, generate a data point `xₙ` from the Gaussian distribution of that cluster.
    `xₙ | zₙ = k ~ N(μₖ, Σₖ)`

**The Resulting Probability:** The total probability of any data point `xₙ` is a weighted sum of all Gaussians:
`p(xₙ) = Σₖ φₖ * N(xₙ | μₖ, Σₖ)`
This is a **soft assignment**. A point has a probability of belonging to *every* cluster.

#### 2.2 The Maximum Likelihood Estimation (MLE) Problem
We want to find the parameters `Θ = {φ₁..φₖ, μ₁..μₖ, Σ₁..Σₖ}` that **maximize the probability (likelihood) of observing our data** `X = {x₁,..,x_N}`.

The likelihood is: `p(X | Θ) = Πₙ p(xₙ) = Πₙ [ Σₖ φₖ * N(xₙ | μₖ, Σₖ) ]`

**The Problem (The Math That Breaks):** Take the log to get the log-likelihood:
`L(Θ) = ln p(X | Θ) = Σₙ ln [ Σₖ φₖ * N(xₙ | μₖ, Σₖ) ]`
See the **log-of-a-sum**? This makes the function analytically intractable. You cannot just take the derivative, set it to zero, and solve for the parameters. The parameters are entangled inside the sum.

This is the fundamental problem that the **Expectation-Maximization (EM)** algorithm solves.

---

### Part 3: The Expectation-Maximization (EM) Algorithm for GMMs

EM is an iterative algorithm that finds a (locally) optimal solution to the MLE problem for latent variable models.

#### 3.1 The E-Step (Expectation)
**Goal:** Given the current parameters `Θᵒˡᵈ`, compute the **responsibility** `γ(zₙₖ)` that cluster `k` takes for explaining data point `xₙ`. This is the posterior probability `P(zₙ=k | xₙ)`.

**The Formula (Derived from Bayes' Theorem):**
`γ(zₙₖ) = P(zₙ=k | xₙ) = [ P(zₙ=k) * P(xₙ | zₙ=k) ] / [ Σ_j P(zₙ=j) * P(xₙ | zₙ=j) ] = [ φₖ * N(xₙ | μₖ, Σₖ) ] / [ Σ_j φ_j * N(xₙ | μ_j, Σ_j) ]`

**What it means:** For every single point `xₙ`, you get a probability distribution over the `K` clusters. This is the **"soft assignment"**. It's a set of numbers, not just a 0 or 1.

#### 3.2 The M-Step (Maximization)
**Goal:** Given the soft assignments `γ(zₙₖ)` from the E-Step, update the parameters `Θ` to maximize the expected complete-data log-likelihood. This step has beautiful, intuitive closed-form solutions that look very much like the standard MLE for Gaussians, but now **weighted by the responsibilities**.

**The Formulas:**
1.  **New Mixing Coefficient (φₖ):** The total responsibility assigned to cluster `k`, averaged over all points.
    `φₖⁿᵉʷ = (1/N) * Σₙ γ(zₙₖ)`

2.  **New Mean (μₖ):** The weighted average of all points, where the weight is the responsibility of cluster `k` for that point.
    `μₖⁿᵉʷ = ( Σₙ γ(zₙₖ) * xₙ ) / ( Σₙ γ(zₙₖ) )`

3.  **New Covariance (Σₖ):** The weighted covariance of all points, where the weight is the responsibility of cluster `k` for that point.
    `Σₖⁿᵉʷ = ( Σₙ γ(zₙₖ) * (xₙ - μₖⁿᵉʷ)(xₙ - μₖⁿᵉʷ)ᵀ ) / ( Σₙ γ(zₙₖ) )`

#### 3.3 Hand-Written Example: EM for GMM
Let's use the same 1D data: `X = [1, 2, 3, 10, 11, 12]`, `K=2`. Assume simple initial parameters.

```plaintext
Step 0: Initialize.
Let's set: μ₁=1, μ₂=12, σ₁²=1, σ₂²=1, φ₁=0.5, φ₂=0.5.

Iteration 1: E-Step
Calculate responsibility γ for each point. Let's do one point, x=2.
N(2 | μ₁=1, σ₁²=1) = (1/√(2π)) * exp(-(2-1)²/2) ≈ 0.24
N(2 | μ₂=12, σ₂²=1) ≈ 0
γ(z₂₁) = (0.5 * 0.24) / ( (0.5*0.24) + (0.5*0) ) = 1.0
γ(z₂₂) = 0
We'd do this for all points. Points 1,2,3 will have γ≈1 for cluster 1, γ≈0 for cluster 2. Points 10,11,12 will be the opposite.

Iteration 1: M-Step
Now update parameters using these soft counts.
N₁ (effective count of cluster 1) = γ(z₁₁) + γ(z₂₁) + γ(z₃₁) + ... ≈ 1+1+1+0+0+0 = 3
N₂ ≈ 3
φ₁ⁿᵉʷ = 3/6 = 0.5
φ₂ⁿᵉʷ = 3/6 = 0.5

μ₁ⁿᵉʷ = (γ(z₁₁)*1 + γ(z₂₁)*2 + γ(z₃₁)*3 + ...) / N₁ ≈ (1*1 + 1*2 + 1*3 + 0*10 + 0*11 + 0*12)/3 = 6/3 = 2
μ₂ⁿᵉʷ ≈ (0*1 + 0*2 + 0*3 + 1*10 + 1*11 + 1*12)/3 = 33/3 = 11

σ₁²ⁿᵉʷ = ( γ(z₁₁)*(1-2)² + γ(z₂₁)*(2-2)² + γ(z₃₁)*(3-2)² ) / N₁ ≈ (1*1 + 1*0 + 1*1)/3 = 2/3
Similarly for σ₂².

Parameters are already converging to their true values.
```
*(This shows the conceptual flow of the calculation. The first E-step is often crude but quickly improves.)*

#### 3.4 The "Hard-EM" -> K-Means Connection
The slides mention "Hard-EM". This is a variant where instead of a soft probability `γ(zₙₖ)`, you make a **hard assignment** in the E-Step: `γ(zₙₖ) = 1` for the cluster `k` with the highest `N(xₙ | μₖ, Σₖ)`, and `0` for others.

If you then assume all covariance matrices are identical and equal to `σ²I` (spherical clusters), the M-Step for `μₖ` reduces to the standard K-Means centroid update. This is why K-Means is often called a "hard" version of EM for GMMs.

---

### Summary & Key Takeaways

1.  **K-Means:** Simple, fast, intuitive. Good for well-separated, spherical clusters. Uses **hard assignments**.
2.  **GMMs:** More powerful, flexible, and probabilistic. Can handle overlapping clusters and different shapes. Uses **soft assignments**.
3.  **The MLE Problem for GMMs:** The log-likelihood function has a log-of-a-sum, making direct optimization impossible.
4.  **EM Algorithm:** The solution to the MLE problem. It iterates between:
    *   **E-Step:** "Guess" the latent variables (soft assignments `γ(zₙₖ)`).
    *   **M-Step:** "Update" the parameters `(φ, μ, Σ)` based on that guess.
5.  **Connection:** K-Means is a special, limited case of the more general and powerful EM algorithm for GMMs.

The mathematics is dense because it rigorously formalizes the intuitive concept of grouping data. Understanding the derivation of the E and M steps is key to seeing why EM works and is so elegant.
---

### Criticisms and Drawbacks of Each Topic

#### 1. Clustering and K-Means Algorithm

**Drawbacks:**
*   **Sensitivity to Initialization:** As mentioned on slide 14, K-Means is highly sensitive to the initial random placement of cluster centroids. Different initializations can lead to drastically different (and often suboptimal) final clusters.
*   **Choice of K:** The algorithm requires the number of clusters `K` to be specified *a priori*. Choosing the correct `K` is a non-trivial problem and often requires domain knowledge or additional methods like the elbow method or silhouette analysis, which are not covered.
*   **Hard Assignment Limitation:** The binary assignment (`r_nk = 0 or 1`) is a major weakness. It is unsuitable for data points that lie in overlapping regions or are genuine outliers, forcing them into a cluster and distorting the centroid.
*   **Cluster Shape Assumption:** K-Means implicitly assumes clusters are isotropic (circular in 2D), convex, and of similar size. It performs poorly on clusters with complex, elongated, or non-linear shapes.
*   **Sensitivity to Outliers:** The mean (centroid) is highly sensitive to outliers. A single outlier can significantly pull a centroid away from the true cluster center.

#### 2. Gaussian Mixture Models (GMMs) and Expectation-Maximization (EM)

**Drawbacks:**
*   **Computational Complexity:** GMMs are significantly more computationally expensive than K-Means. Calculating the full covariance matrix for each cluster, especially in high-dimensional spaces, is costly. The E-step requires evaluating every point against every Gaussian.
*   **Singularity Issues:** During the M-step, if a cluster is assigned very few points (a common issue in high dimensions or with poor initialization), the covariance matrix can become singular (non-invertible), causing the algorithm to fail. *Thus the algorithm cannot be considered complete*
*   **Non-Convex Optimization:** Like K-Means, the EM algorithm for GMMs optimizes a non-convex log-likelihood function. It is guaranteed to converge, but only to a *local optimum*, not necessarily the global optimum. The final solution is still dependent on initial parameter values.
*   **Model Selection:** Determining the number of components `K` in a GMM is again a challenging model selection problem. Criteria like Bayesian Information Criterion (BIC) or Akaike Information Criterion (AIC) are used but add another layer of complexity.
*   **Assumption of Gaussian Distribution:** The core assumption is that each cluster is Gaussian-distributed. While more flexible than K-Means, GMMs can still fail to model clusters with highly non-Gaussian structures (e.g., curved manifolds).

#### 3. EM in General for Latent Variable Models

**Drawbacks (Generalizing beyond GMMs):**
*   **Slow Convergence:** The EM algorithm can converge very slowly, especially when the "missing information" (the latent variables) is large. It often requires many more iterations than other optimization techniques like gradient descent.
*   **Local Optima Problem:** This is the most significant general drawback. The algorithm provides no mechanism to escape local maxima of the likelihood function. Multiple restarts with different initializations are typically required to find a good solution.
*   **Requires Analytic M-Step:** The standard EM formulation *requires that the M-step has an analytic, closed-form solution* (as it does for GMMs). For many complex models, this is not possible, leading to variants like Generalized EM (GEM) or Monte Carlo EM, which are more complex.

#### 4. EM for Document Clustering (Implied Topic)

While the slides list this topic in the outline, the specific content for document clustering is not detailed. However, applying EM (e.g., with a Mixture of Multinomials model) has its own set of drawbacks:

*   **High-Dimensionality:** Document data is extremely high-dimensional (vocabulary size can be tens of thousands of features). This exacerbates the computational cost and sparsity problems of GMMs.
*   **Choice of Distribution:** The Gaussian distribution is often a poor choice for count data like word frequencies. A Mixture of Multinomials (the likely model for this application) is more appropriate but still has its own challenges with initialization and smoothing to avoid zero probabilities.
*   **Interpretability:** While clusters can be described by their most probable words, the resulting "topics" can sometimes be difficult to interpret meaningfully without human oversight.
