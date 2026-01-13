# Study Sheet: Unsupervised and Self-Taught Learning with Neural Networks

---

## 1. Introduction to Unsupervised Learning

### Supervised vs Unsupervised Learning

**Supervised Learning (Previous Chapters):**

- Network predicts output variables given input variables
- Training data: pairs of inputs and labeled outputs $(\mathbf{x}, \mathbf{y})$
- Goal: learn mapping $\mathbf{x} \rightarrow \mathbf{y}$

**Unsupervised Learning:**

- No output labels provided
- Training data: only input vectors ${\mathbf{x}^{(1)}, \ldots, \mathbf{x}^{(N)}}$
- Goal: discover structure, patterns, or representations in the data
- Application: dimensionality reduction, feature learning

---

## 2. Autoencoders

### 2.1 What is an Autoencoder?

An **autoencoder** is a neural network trained to map input vectors onto themselves by minimizing the **reconstruction error** between inputs and outputs.

**Architecture:**

- **Input layer:** $D$ inputs ($\mathbf{x}$)
- **Hidden layer(s):** $M$ hidden units (bottleneck, where $M < D$)
- **Output layer:** $D$ outputs (reconstruction $\hat{\mathbf{x}}$ or $h_{\boldsymbol{\theta}}(\mathbf{x})$)

**Key Feature:** The network has the **same number of outputs as inputs**, attempting to map each input vector onto itself.

### 2.2 Why Autoencoders Work for Dimensionality Reduction

Since the number of hidden units $M$ is **smaller** than the number of inputs $D$, a perfect reconstruction is generally not possible. The network must learn a **compressed representation** in the hidden layer that captures the most important features of the data.

**Bottleneck Effect:**

- Forces the network to learn efficient encodings
- Hidden layer learns a lower-dimensional representation
- Output layer decodes this representation back to original space

---

### 2.3 Training Objective

We determine network parameters $\boldsymbol{\theta}$ by minimizing the **reconstruction error**:

$$E(\boldsymbol{\theta}) := \frac{1}{2} \sum_{n=1}^{N} ||h_{\boldsymbol{\theta}}(\mathbf{x}_n) - \mathbf{x}_n||_2^2$$

**Interpretation:**

- Minimize sum-of-squares difference between input and reconstruction
- Network learns to preserve most important information through bottleneck
- Similar to supervised learning, but targets are the inputs themselves!

---

## 3. Linear Autoencoders and PCA

### 3.1 Linear Autoencoder Architecture

**Figure 5.5.1** shows an autoassociative multilayer perceptron with two layers of weights:

- $D$ input units: $x_1, \ldots, x_D$
- $M$ hidden units: $z_1, \ldots, z_M$ (where $M < D$)
- $D$ output units: reconstructions of $x_1, \ldots, x_D$

**Key Property:** If hidden units have **linear activation functions**, the autoencoder performs a specific form of dimensionality reduction.

### 3.2 Connection to Principal Component Analysis (PCA)

**Theorem:** If the hidden units have linear activations, then the error function has a **unique global minimum**, and at this minimum the network performs a projection onto the $M$-dimensional subspace spanned by the first $M$ principal components.

**What This Means:**

- The vectors of weights leading into hidden units form a basis set spanning the principal subspace
- This is exactly the subspace obtained by PCA!
- The autoencoder and PCA minimize the same sum-of-squares error function

**Key Difference from PCA:**

- PCA: Vectors must be orthogonal and normalized
- Autoencoder: Vectors need not be orthogonal or normalized
- Both find the same subspace, just with different basis representations

**Conclusion for Linear Case:** The connection between the autoencoder and PCA is unsurprising, since both principal component analysis and the neural network are using linear dimensionality reduction and minimizing the same sum-of-squares error function.

---

## 4. Nonlinear Autoencoders

### 4.1 Overcoming Linear Limitations

**Limitation of Linear Dimensionality Reduction:** Could be overcome by using nonlinear (sigmoidal) activation functions for hidden units.

**But there's a catch!** Even with nonlinear hidden units, the minimum error solution is **still given by projection onto the principal component subspace**.

**Why?** There is no advantage in using two-layer neural networks to perform dimensionality reduction if we only use standard linear techniques (PCA-based linear algebra methods are guaranteed to give correct solution in finite time).

### 4.2 Deep Nonlinear Autoencoders

**Solution:** Add extra hidden layers of nonlinear units!

**Figure 5.5.2** shows a four-layer autoencoder:

- Input layer: $D$ units
- First hidden layer: nonlinear units
- Second hidden layer (bottleneck): $M$ units (can be linear or nonlinear)
- Third hidden layer: nonlinear units
- Output layer: $D$ units (linear)

**The network can be viewed as two successive functional mappings** $\mathbf{F}_1$ and $\mathbf{F}_2$:

**First mapping $\mathbf{F}_1$** (encoder):

- Projects original $D$-dimensional data onto $M$-dimensional subspace $\mathcal{S}$
- Defined by activations of units in second hidden layer
- Because of first hidden layer of nonlinear units, this mapping is **very general** and not restricted to being linear

**Second mapping $\mathbf{F}_2$** (decoder):

- Maps from $M$-dimensional space back into original $D$-dimensional input space
- Arbitrary functional mapping (also general due to nonlinearity)

---

### 4.3 Geometric Interpretation

**Figure 5.5.3** shows the geometric interpretation for $D = 3$ inputs and $M = 2$ hidden units:

**Left panel:** 3D input space ($x_1, x_2, x_3$)

- Data points shown as dots
- $\mathbf{F}_1$ projects them onto 2D manifold $\mathcal{S}$

**Middle panel:** 2D hidden space ($z_1, z_2$)

- The compressed representation
- Shaded region shows the manifold $\mathcal{S}$

**Right panel:** 3D output space

- $\mathbf{F}_2$ maps back to 3D
- Reconstructed points on curved surface

**Key Insight:** Such a network effectively performs a **nonlinear PCA**. It has the advantage of not being limited to linear transformations, although it contains standard PCA as a special case.

**The Challenge:** Training is now **challenging** since the error function is highly nonlinear and non-convex with lots of local optima.

---

## 5. Visualizing Learned Features

### 5.1 Understanding What Hidden Units Learn

**Example:** Train autoencoder on 10×10 images (so $D = 100$ pixels).

Each hidden unit $i$ computes: $$a_i^{(2)} = f\left(\sum_{j=1}^{100} W_{ij}^{(1)} x_j + b_i^{(1)}\right)$$

**Question:** What input image $\mathbf{x}$ would cause $a_i^{(2)}$ to be maximally activated?

### 5.2 Finding Maximally Activating Inputs

We will visualize the function computed by hidden unit $i$, which depends on parameters $W_{ij}^{(1)}$.

**Approach:** Find input that maximally activates the unit, subject to constraint $||\mathbf{x}|| \leq 1$ (norm constraint).

**Solution:** The input that maximally activates hidden unit $i$ is:

$$x_j = \frac{W_{ij}^{(1)}}{\sqrt{\sum_{j=1}^{100} (W_{ij}^{(1)})^2}}$$

for all 100 pixels $j = 1, \ldots, 100$.

**Interpretation:**

- Each $x_j$ is proportional to corresponding weight $W_{ij}^{(1)}$
- Normalized by total magnitude of weight vector
- This gives the "preferred stimulus" for that hidden unit

### 5.3 Visualizing Edge Detectors

By displaying the image formed by these pixel intensity values, we can understand what feature hidden unit $i$ is looking for.

**Figure showing learned features:**

- Grid of small 10×10 images
- Each square shows the maximally activating input for one hidden unit
- Different hidden units learn to detect **edges at different positions and orientations**

**Key Discovery:** Hidden units have learned to detect:

- Vertical edges
- Horizontal edges
- Diagonal edges at various angles
- Edges at different spatial locations

**This is similar to:** Early visual processing in biological systems (V1 neurons in visual cortex are edge detectors)!

---

## 6. Self-Taught Learning

### 6.1 The Motivation

**Quote from the notes:** "Assuming that we have a sufficiently powerful learning algorithm. One of the most reliable ways to get better performance is to give the algorithm more data. This has led to the aphorism in machine learning, 'sometimes it's not who has the best algorithm that wins; it's who has the most data.'"

**The Problem:**

- Getting labeled data can be expensive
- Labeling requires human effort
- Limited labeled datasets restrict model performance

**The Promise:** If we can get algorithms to learn from **unlabeled** data, then we can easily obtain massive amounts of it to significantly reduce efforts in creating large labeled training datasets.

---

### 6.2 What is Self-Taught Learning?

**Definition:** Self-taught learning is an approach to learn from **both** labeled and unlabeled data, a problem scenario called **semi-supervised learning**.

**The Strategy:**

1. Use a large amount of **unlabeled data** to learn good feature representations
2. Use a small amount of **labeled data** to train a classifier on these features

**Key Idea:** Learn feature representations from whatever (perhaps small amount of) labeled data we have for the classification task, and apply supervised learning on that labeled data to solve the classification task.

---

### 6.3 Learning Representations with Autoencoders

**Setup:**

- Unlabeled training set: ${\mathbf{x}_u^{(1)}, \ldots, \mathbf{x}_u^{(N_u)}}$ with $N_u$ unlabeled examples
- Subscript $u$ stands for _unlabeled_

**Step 1: Train Autoencoder on Unlabeled Data**

Train autoencoder on unlabeled data:

- Input: $\mathbf{x}_u$
- Hidden layer: learns features $\mathbf{a}$ (activations)
- Output: reconstruction $\hat{\mathbf{x}}_u$

**Figure 5.5.4** shows a fully connected autoencoder with 3 hidden units.

Having trained the parameters $\mathbf{W}^{(1)}$ of this model, given any new input $\mathbf{x}$, we can compute the corresponding vector of activations $\mathbf{a}$ of the hidden units.

**Key Insight:** These activations $\mathbf{a}$ often give a better representation of the input than the original raw input $\mathbf{x}$.

---

### 6.4 Using Learned Features for Supervised Learning

**Figure 5.5.5** shows the same autoencoder with the final output layer removed:

- Input: $\mathbf{x}$
- Hidden layer: features $\mathbf{a} = (a_1, a_2, a_3)$
- No output layer

This is just the encoder part of the autoencoder.

**Step 2: Extract Features from Labeled Data**

Now suppose we have a labeled training set: $${(\mathbf{x}_\ell^{(1)}, \mathbf{y}^{(1)}), \ldots, (\mathbf{x}_\ell^{(N_\ell)}, \mathbf{y}^{(N_\ell)})}$$ with $N_\ell$ labeled examples (subscript $\ell$ stands for _labeled_).

**Transform the data using learned features:**

For the first training example, rather than representing it as $\mathbf{x}_\ell^{(1)}$, we can feed $\mathbf{x}_\ell^{(1)}$ as input to our autoencoder and obtain the corresponding vector of activations $\mathbf{a}_\ell^{(1)}$.

**Two options for representation:**

1. **Replace** original feature vector with autoencoder features: $\mathbf{a}_\ell^{(1)}$
2. **Concatenate** original and autoencoder features: $(\mathbf{x}_\ell^{(1)}, \mathbf{a}_\ell^{(1)})$

**New training set becomes:**

- Concatenation: ${(\mathbf{x}_\ell^{(1)}, \mathbf{a}_\ell^{(1)}, \mathbf{y}^{(1)}), \ldots, (\mathbf{x}_\ell^{(N_\ell)}, \mathbf{a}_\ell^{(N_\ell)}, \mathbf{y}^{(N_\ell)})}$
- Replacement: ${(\mathbf{a}_\ell^{(1)}, \mathbf{y}^{(1)}), \ldots, (\mathbf{a}_\ell^{(N_\ell)}, \mathbf{y}^{(N_\ell)})}$

---

### 6.5 Training the Final Classifier

**Step 3: Train Supervised Model**

Train a supervised learning algorithm (such as feed-forward neural networks) to obtain a function that makes predictions on the $\mathbf{y}$ values.

**At Test Time:** Given test example $\mathbf{x}^{\text{test}}$:

1. Feed it to autoencoder to get $\mathbf{a}^{\text{test}}$
2. Feed representation (with autoencoder features) into feedforward classifier
3. Get prediction

---

## 7. Key Concepts and Intuitions

### 7.1 Why Autoencoders Learn Useful Features

**Information Bottleneck:**

- Hidden layer has fewer units than input ($M < D$)
- Network must compress information
- Forced to learn most important/salient features
- Removes noise and redundancy

**Reconstruction Pressure:**

- Must reconstruct input from compressed representation
- Only features useful for reconstruction are preserved
- Network learns to capture structure in data

### 7.2 Linear vs Nonlinear Autoencoders

|Aspect|Linear Autoencoder|Nonlinear (Deep) Autoencoder|
|---|---|---|
|**Equivalent to**|PCA|Nonlinear PCA|
|**Subspace**|Linear subspace|Nonlinear manifold|
|**Optimization**|Unique global minimum|Multiple local minima|
|**Expressiveness**|Limited to linear|Can learn complex patterns|
|**Training**|Easy (closed form)|Challenging (gradient descent)|

### 7.3 Benefits of Self-Taught Learning

**Advantages:**

1. **Leverages unlabeled data:** Abundant and cheap
2. **Better features:** Often better than raw inputs
3. **Reduces overfitting:** Pre-trained features are more robust
4. **Requires less labeled data:** Can work with small labeled datasets
5. **Transfer learning:** Features from one domain can help another

**When to Use:**

- Limited labeled data available
- Abundant unlabeled data available
- High-dimensional input (images, audio, text)
- Complex patterns that benefit from learned representations

---

## 8. Practical Implementation

### 8.1 Training an Autoencoder

**Step-by-Step:**

1. **Architecture Design:**
    
    - Input layer: $D$ units (dimensionality of data)
    - Hidden layer(s): $M$ units where $M < D$ (bottleneck)
    - Output layer: $D$ units (same as input)
2. **Activation Functions:**
    
    - Linear autoencoder: linear activations
    - Nonlinear autoencoder: sigmoid/tanh/ReLU in hidden layers
    - Output layer: typically linear (for real-valued inputs)
3. **Loss Function:**
    
    - Reconstruction error: $\frac{1}{2}\sum ||h_{\boldsymbol{\theta}}(\mathbf{x}_n) - \mathbf{x}_n||^2$
4. **Training:**
    
    - Use backpropagation and gradient descent
    - Train on unlabeled data
    - Can add regularization (weight decay, early stopping)
5. **Feature Extraction:**
    
    - After training, remove output layer
    - Use hidden layer activations as features

---

### 8.2 Complete Self-Taught Learning Pipeline

**Full Workflow:**

```
1. Collect unlabeled data: {x_u^(1), ..., x_u^(N_u)}

2. Train autoencoder:
   - Design architecture with bottleneck
   - Train to minimize reconstruction error
   - Extract encoder part (input → hidden layer)

3. Transform all data through encoder:
   - Unlabeled: x_u → a_u
   - Labeled: x_ℓ → a_ℓ

4. Create new training set:
   - Option A: {(a_ℓ^(n), y^(n))}
   - Option B: {(x_ℓ^(n), a_ℓ^(n), y^(n))}

5. Train supervised classifier:
   - Use transformed labeled data
   - Train feedforward NN or other classifier

6. Test:
   - Transform test input: x_test → a_test
   - Feed to classifier for prediction
```

---

## 9. Advanced Considerations

### 9.1 Deep Autoencoders

**Architecture:**

- Multiple encoding layers: $D \rightarrow D_1 \rightarrow D_2 \rightarrow M$
- Multiple decoding layers: $M \rightarrow D_2 \rightarrow D_1 \rightarrow D$
- Symmetric structure

**Advantages:**

- Can learn hierarchical features
- More expressive than shallow autoencoders
- Better for complex data

**Training Challenge:**

- Highly non-convex optimization
- Many local minima
- Requires careful initialization (next section addresses this)

### 9.2 Variations of Autoencoders

**Denoising Autoencoders:**

- Corrupt input with noise: $\tilde{\mathbf{x}} = \mathbf{x} + \epsilon$
- Train to reconstruct clean input: $\hat{\mathbf{x}} \approx \mathbf{x}$
- Forces learning of robust features

**Sparse Autoencoders:**

- Add sparsity penalty on hidden activations
- Encourages only a few hidden units to be active
- Learns more interpretable features

**Variational Autoencoders (VAEs):**

- Probabilistic framework
- Hidden layer represents distribution (not just point)
- Can generate new samples

---

## 10. Mathematical Prerequisites Refresher

### Understanding Reconstruction Error

**Sum-of-Squares Error:** $$E(\boldsymbol{\theta}) = \frac{1}{2} \sum_{n=1}^{N} ||h_{\boldsymbol{\theta}}(\mathbf{x}_n) - \mathbf{x}_n||_2^2$$

**Component-wise:** $$= \frac{1}{2} \sum_{n=1}^{N} \sum_{d=1}^{D} (h_{\boldsymbol{\theta}}(\mathbf{x}_n)_d - x_{n,d})^2$$

**Why squared error?**

- Corresponds to Gaussian noise assumption
- Differentiable everywhere
- Penalizes large errors heavily

### Normalization of Weights

For visualizing features, we normalize: $$x_j = \frac{W_{ij}^{(1)}}{\sqrt{\sum_{j=1}^{D} (W_{ij}^{(1)})^2}}$$

This ensures $||\mathbf{x}|| = 1$ (unit norm constraint).

**Why normalize?**

- Makes visualization scale-invariant
- Focuses on direction, not magnitude
- Comparable across different hidden units

---

## Summary: Key Takeaways

1. **Autoencoders** learn compressed representations by reconstructing inputs through a bottleneck:
    
    - Same number of inputs and outputs
    - Fewer hidden units than inputs ($M < D$)
    - Minimize reconstruction error
2. **Linear autoencoders** are equivalent to PCA:
    
    - Find same subspace as PCA
    - Vectors need not be orthogonal
    - Unique global minimum
3. **Nonlinear (deep) autoencoders** perform nonlinear PCA:
    
    - More expressive than linear
    - Can learn complex manifolds
    - Training is challenging (local minima)
4. **Learned features** are interpretable:
    
    - Visualize by finding maximally activating inputs
    - Often learn edge detectors for images
    - Similar to biological visual processing
5. **Self-taught learning** leverages unlabeled data:
    
    - Train autoencoder on unlabeled data
    - Extract features from labeled data
    - Train classifier on learned features
    - Works with limited labeled data
6. **Feature transformation** can be done two ways:
    
    - Replace: use only autoencoder features
    - Concatenate: combine original and autoencoder features
7. **Semi-supervised learning** combines supervised and unsupervised:
    
    - Unlabeled data: learn representations
    - Labeled data: train classifier
    - Best of both worlds

**The Big Idea:** Autoencoders discover useful representations from unlabeled data, which can then be used to improve supervised learning with limited labeled data! 🎯