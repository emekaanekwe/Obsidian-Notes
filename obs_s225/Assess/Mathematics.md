
---

### **Linear Algebra**

This is the language of deep learning. Data, weights, and transformations are all represented as vectors and matrices.

1.  **Vectors & Matrices:** Definitions, addition, scalar multiplication.
2.  **Matrix Multiplication:** The core operation of neural networks. Understand the rules for when it's defined and the resulting dimensions.
    *   **Key Rule:** For matrices A (m x n) and B (p x q), multiplication AB is only defined if n = p. The result is a matrix of shape (m x q)
3.  **Dot Product:** A special case of matrix multiplication for vectors. Geometrically, it measures similarity and projection.
4.  **Transpose of a Matrix:** Flipping a matrix over its diagonal. Crucial for understanding weight dimensions and the attention score calculation $QK^T$.
5.  **Matrix Inverse:** Conceptually, "undoing" a linear transformation. Important for understanding some derivations, though not used directly in most forward/backward passes.
6.  **Eigenvalues & Eigenvectors:** Represent the "principal components" of a transformation. Fundamental to understanding the stability of RNNs and the theory behind many linear transformations.

---

### **Calculus**

Needed to understand how models learn by optimizing a loss function.

7.  **Derivatives:** The instantaneous rate of change. Represents slope.
8.  **Partial Derivatives:** The derivative of a multi-variable function with respect to one variable, holding the others constant. This is how we find the gradient.
9.  **The Chain Rule:** The fundamental rule for calculating derivatives of composite functions. This is the engine of **Backpropagation**.
    *   If $y = f(g(x))$, then $\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$.
10. **Gradient:** A vector of all partial derivatives of a multi-variable function. It points in the direction of the steepest ascent. The negative gradient points in the direction of steepest descent, which is the basis for **Gradient Descent**.
11. **Jacobian Matrix:** A matrix of all first-order partial derivatives of a vector-valued function. Generalizes the gradient for multiple outputs.
12. **Hessian Matrix:** A square matrix of second-order partial derivatives. It describes the local curvature of the loss function, important for advanced optimization.

---

### **Probability & Statistics**

Essential for understanding loss functions, generative models, and uncertainty.

13. **Random Variables:** Variables whose values are outcomes of a random phenomenon.
14. **Probability Distributions:** Describes the likelihood of different outcomes.
    *   **Gaussian/Normal Distribution:** The "bell curve." Central to noise models, initialization, and Diffusion Models.
    *   **Bernoulli Distribution:** Distribution over two outcomes (0/1). Used in binary classification (like the GAN discriminator).
    *   **Categorical/Multinoulli Distribution:** Distribution over multiple discrete categories. Used in multi-class classification (e.g., predicting the next word).
15. **Probability Density Function (PDF) & Probability Mass Function (PMF):** Functions that describe a continuous and discrete probability distribution, respectively.
16. **Expectation ($\mathbb{E}$):** The average value of a random variable, weighted by its probability. Used in the formulation of loss functions for GANs and Diffusion Models.
17. **Bayes' Theorem:** Describes the probability of an event based on prior knowledge. It's the foundation for the derivation of the optimal GAN discriminator: $P(\text{real} | x) = \frac{p_{data}(x)}{p_{data}(x) + p_g(x)}$.
18. **Conditional Probability ($P(A|B)$):** The probability of event A given that event B has occurred. Fundamental to sequence modeling (e.g., $P(y_j | y_{1:j-1}, c)$ in seq2seq).
19. **Chain Rule of Probability:** Allows breaking down the joint probability of multiple events into a product of conditional probabilities. Used to derive the log-likelihood of a sequence.
    *   $P(A, B, C) = P(A) \cdot P(B|A) \cdot P(C|A, B)$.
20. **Independence & Conditional Independence:** Key assumptions that simplify models (e.g., the Naive Bayes assumption, i.i.d. data assumption).
21. **Maximum Likelihood Estimation (MLE):** A method for estimating the parameters of a statistical model by maximizing the likelihood function. The core principle behind training most generative models.
22. **Kullback-Leibler (KL) Divergence:** A measure of how one probability distribution differs from a second, reference distribution. Used in model training and variational inference.
23. **Jensen-Shannon Divergence:** A symmetric and smoothed version of the KL Divergence. It is the loss function that GANs implicitly minimize.

---

### **Information Theory**

Closely related to probability and crucial for understanding model behavior and training objectives.

24. **Entropy:** A measure of the uncertainty or randomness in a system.
25. **Cross-Entropy:** A measure of the difference between two probability distributions. It is the most common loss function for classification tasks.
26. **Log-Likelihood:** The logarithm of the likelihood function. Used because it turns products into sums, is numerically more stable, and doesn't change the location of the optimum.

---

### **Optimization**

The field dedicated to finding the inputs that minimize or maximize a function.

27. **Convex vs. Non-Convex Functions:** Convex functions have a single global minimum, making them easy to optimize. Neural network loss landscapes are highly non-convex, which is why training is challenging.
28. **Gradient Descent / Stochastic Gradient Descent (SGD):** The core iterative algorithm for minimizing the loss function.
29. **Momentum:** A technique to accelerate SGD by navigating along the relevant directions and softening oscillations in irrelevant ones.
30. **Optimizers (Adam, RMSProp):** Adaptive learning rate algorithms that are the default choice for training most deep learning models.
31. **Constrained Optimization:** Optimizing a function subject to constraints (e.g., $\sum \varphi_k = 1$ in a GMM). Often handled with **Lagrange Multipliers**.
32. **Minimax Optimization:** A scenario in game theory where one player seeks to minimize a function and another seeks to maximize it. This is the formal framework for **training GANs**: $\min_G \max_D V(D, G)$.
33. **Nash Equilibrium:** A concept from game theory where no player can benefit by unilaterally changing their strategy. The theoretical optimum for a GAN is a Nash Equilibrium where $p_g = p_{data}$ and $D(x) = 0.5$.

---

This list forms the essential mathematical toolkit for a deep learning practitioner. Mastering these concepts will allow you to understand not just *how* to implement models, but *why* they are designed the way they are and how they learn.