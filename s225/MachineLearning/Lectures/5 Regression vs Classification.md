
---

### **Study Sheet: Introduction to Classification & The Perceptron Algorithm**

#### **1. Core Concepts: Regression vs. Classification**

*   **Regression:** Models the relationship between input features and a **continuous** target output (e.g., house price, exam score).
*   **Classification:** Models the relationship between input features and a **discrete** target output, known as a **class label** or **category** (e.g., spam/ham, digit 0-9, fish species).
*   **Classifier:** An algorithm that maps an input data point `x` to one of `K` discrete classes `{C₁, C₂, ..., Cₖ}`.

#### **2. Feature Representation & Geometric Intuition**

*   A data point is represented by a **feature vector** `x = [x₁, x₂, ..., x_d]ᵀ`, where each component `x_i` is a measurement of a specific feature (e.g., length, width, pixel intensity).
*   This representation allows us to view each data point as a point in a `d`-dimensional space, called the **feature space** or **input space**. This geometric interpretation is crucial.

#### **3. Linear Classification Models**

The core idea is to use a **linear function** to define a classifier.

*   **Discriminant Function:** A function `f(x)` whose sign determines the class label.
    *   **General Form:** `f(x) = w₀ + w₁x₁ + w₂x₂ + ... + w_dx_d`
    *   **Vector Form (Compact Notation):** `f(x) = w₀ + 𝐰ᵀ𝐱`
        *   `𝐰 = [w₁, w₂, ..., w_d]ᵀ` is the **weight vector**. It controls the **orientation** (slope) of the decision boundary.
        *   `w₀` is the **bias term** or **threshold**. It controls the **position** (shift from origin) of the decision boundary.
*   **Classification Rule:**
    *   `If f(𝐱) >= 0, then predict Class +1 (e.g., Tuna)`
    *   `If f(𝐱) < 0, then predict Class -1 (e.g., Bass)`
*   **Decision Boundary:** The set of points `{𝐱 : f(𝐱) = 0}`. This is a **line** (in 2D), a **plane** (in 3D), or a **hyperplane** (in higher dimensions) that separates the classes.
*   **Decision Regions:** The regions in feature space where the classifier predicts a specific class (e.g., all points where `f(𝐱) > 0`).

#### **4. Important Mathematical Notation**

*   **Input Vector:** `𝐱 = [x₁, x₂, ..., x_d]ᵀ`
*   **Augmented Input Vector (for compactness):** `𝐱 = [1, x₁, x₂, ..., x_d]ᵀ` (This incorporates the bias `w₀` into the weight vector, so `f(𝐱) = 𝐰ᵀ𝐱` where `𝐰 = [w₀, w₁, ..., w_d]ᵀ`).
*   **Weight Vector:** `𝐰 = [w₀, w₁, w₂, ..., w_d]ᵀ`
*   **Target Label:** `tₙ ∈ {-1, +1}` (the true label for data point `n`)
*   **Predicted Label:** `y(𝐱)`
*   **Sign Function:** `y(𝐱) = sign(𝐰ᵀ𝐱) = { +1 if 𝐰ᵀ𝐱 >= 0; -1 if 𝐰ᵀ𝐱 < 0 }`

#### **5. Linear Separability**

*   A dataset is **linearly separable** if a **single hyperplane** can perfectly separate all data points of different classes. If no such hyperplane exists, the problem is **non-linearly separable**.
*   For linearly separable problems, there are **infinitely many** perfect classifiers (hyperplanes) on the training data. The goal is to find the one that **generalizes** best to new data.

#### **6. From Linear Model to Classifier: Activation Functions**

A classifier is built by applying an **activation function** (or **link function**) `F` to the linear function `f(𝐱)`.
`y(𝐱) = F( 𝐰ᵀ𝐱 )`

*   **Perceptron (Discriminative Model):** Uses a **step/sign function**.
    *   `F(z) = sign(z)`
    *   **Output:** A definite class label (`+1` or `-1`). **No notion of probability or uncertainty.**
*   **Logistic Classifier (Probabilistic Discriminative Model):** Uses a **logistic function**.
    *   `F(z) = σ(z) = 1 / (1 + e^{-z})`
    *   **Output:** A number between `0` and `1`, interpreted as the **probability** of class `+1`. Allows for expressing uncertainty.

#### **7. Multi-Class Classification (K > 2)**

Three main strategies were discussed:
1.  **One-vs-Rest (OvR):** Train `K` separate classifiers. Classifier `i` is trained to distinguish class `C_i` (label `+1`) from all other classes (label `-1`). To classify a new point, run all `K` classifiers and choose the class whose classifier outputs the highest value. Can lead to ambiguous regions.
2.  **One-vs-One (OvO):** Train a classifier for every pair of classes (`K(K-1)/2` classifiers). To classify a new point, run all classifiers and choose the class that "wins" the most pairwise comparisons (majority vote). Can also lead to ambiguities (ties).
3.  **Single Machine (K Scoring Functions):** The preferred method. Learn `K` weight vectors `{𝐰₁, 𝐰₂, ..., 𝐰_𝐾}`.
    *   **Scoring Function for class k:** `f_k(𝐱) = 𝐰ₖᵀ𝐱`
    *   **Decision Rule:** `Predicted class = argmax_{k} (f_k(𝐱))`
    *   The decision boundary between any two classes `i` and `j` is the hyperplane where `f_i(𝐱) = f_j(𝐱)`.

#### **8. The Perceptron Learning Algorithm**

A foundational algorithm for learning the parameters `𝐰` of a **linear binary classifier** for a linearly separable problem.

*   **Goal:** Find a weight vector `𝐰` such that the number of misclassified training examples is minimized.
*   **Error Function:** The total error is the sum over misclassified points. However, the **sign function is not differentiable**, so we use a trick:
    *   **Misclassification Condition:** `tₙ (𝐰ᵀ𝐱ₙ) <= 0`
        *   If `tₙ = +1` and `𝐰ᵀ𝐱ₙ` is negative, the product is negative. **(Error)**
        *   If `tₙ = -1` and `𝐰ᵀ𝐱ₙ` is positive, the product is negative. **(Error)**
        *   If the product is positive, the prediction is correct.
*   **The Algorithm:**
    1.  Initialize the weight vector `𝐰` (often randomly or to zeros).
    2.  **Repeat** for a number of epochs or until no mistakes are made:
        *   **For each** training example `(𝐱ₙ, tₙ)`:
            *   Compute the prediction: `y = sign(𝐰ᵀ𝐱ₙ)`
            *   **If** `y == tₙ`: Do nothing. (Correct classification)
            *   **Else** (Misclassification): Update the weights
                `𝐰^{new} = 𝐰^{old} + η * tₙ * 𝐱ₙ`
                *   `η` is the **learning rate** (a small positive constant, often set to 1).
*   **Geometric Interpretation:** The update rule **adds the feature vector to the weights** if the point was incorrectly classified as negative (`tₙ=+1`), and **subtracts the feature vector** if the point was incorrectly classified as positive (`tₙ=-1`). This "tugs" the decision boundary towards the misclassified point.
*   **Properties:**
    *   **Convergence Theorem:** If the training data is **linearly separable**, the Perceptron algorithm is guaranteed to find an exact solution (a separating hyperplane) in a finite number of steps.
    *   It may **not converge** if the data is not linearly separable (it will oscillate indefinitely).
    *   The solution is **not unique**; it depends on the initial weights and the order in which data points are processed.

#### **9. The Perceptron as a Neural Network**

*   The Perceptron is a **single artificial neuron**.
*   **Architecture:**
    *   **Inputs:** The features `x₁, x₂, ..., x_d` (and a constant input `1` for the bias).
    *   **Weights:** `w₀, w₁, ..., w_d` (synaptic strengths).
    *   **Summing Junction:** Computes the linear combination `z = 𝐰ᵀ𝐱`.
    *   **Activation Function:** The step function `F(z) = sign(z)`.
    *   **Output:** The class label `y(𝐱)`.

---