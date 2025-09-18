## DSeek Study Sheet

Of course. Here is a detailed and comprehensive study sheet based on your lecture slides for FIT5201 - Module 3, Part B: Linear Models for Classification.

---

## 1. Core Concept: Two Probabilistic Approaches

Classification can be approached by modeling probabilities:
1.  **Discriminative Models:** Model the **decision boundary** or the **posterior probability `p(y|x)`** directly.
    *   **Example:** Logistic Regression.
2.  **Generative Models:** Model the **underlying distribution of the data `p(x|y)`** and the **class priors `p(y)`**. Use Bayes' Theorem to invert these and find `p(y|x)`.
    *   **Example:** Gaussian Discriminant Analysis (Gaussian Bayes Classifier).

---

# 2. Logistic Regression (Discriminative Model)

## Concept
*   **Goal:** Directly model the probability that a given input `x` belongs to class 1: `P(y=1 | x)`.
*   **Method:** Apply a **sigmoid (logistic) function** to a linear function of the data. This squashes the output to the range [0, 1], interpreting it as a probability.

## Model and Formulas
*   **Sigmoid Function:** `σ(z) = 1 / (1 + exp(-z))`
*   **Logistic Regression Model:**
    `P(C=1 | x) = σ(wᵀx) = 1 / (1 + exp(-wᵀx))`
    `P(C=0 | x) = 1 - P(C=1 | x) = exp(-wᵀx) / (1 + exp(-wᵀx))`
*   **Decision Boundary:** This is where `P(C=1|x) = P(C=0|x) = 0.5`. This happens when `wᵀx = 0`. **The decision boundary is linear.**
![[Pasted image 20250901182344.png]]
## Parameter Learning: Maximum Likelihood Estimation (MLE)
We find parameters `w` that maximize the likelihood of observing the training data.
![[Pasted image 20250901180556.png]]
*   **Likelihood Function (Bernoulli):**
    `L(w) = ∏ [P(C=1|x_n)]^{t_n} * [1 - P(C=1|x_n)]^{1 - t_n}`
    where `t_n` is 1 if the true class is 1, and 0 otherwise.
*   **Log-Likelihood (Easier to maximize):**
    `LL(w) = log L(w) = ∑ [ t_n * log(σ(wᵀx_n)) + (1 - t_n) * log(1 - σ(wᵀx_n)) ]`
    ![[Pasted image 20250901180642.png]]
*   **Optimization:** There is **no analytical solution** for `w`. We use iterative methods like **Gradient Descent**.
*   **Gradient Update Rule (Stochastic Gradient Descent):**
    `w^{(r+1)} := w^{(r)} - η * (y_n - t_n) * x_n`
    where `y_n = σ(wᵀx_n)` is the current predicted probability.

### Example Calculation: Prediction
![[Pasted image 20250901180230.png]]
**Model:** `P(Pass | Hours) = σ(-4.078 + 1.5 * Hours)`
**Task:** Find the probability of passing for 3 hours of study.
1.  Calculate linear function: `z = -4.078 + (1.5 * 3) = -4.078 + 4.5 = 0.422`
2.  Apply sigmoid: `σ(0.422) = 1 / (1 + exp(-0.422)) ≈ 1 / (1 + 0.655) ≈ 1 / 1.655 ≈ 0.604`
*Answer: The probability is approximately 61%, matching the slide's example.*

---

# 3. Probabilistic Generative Models (Gaussian Discriminant Analysis)

## Concept
![[Pasted image 20250901180342.png]]
*   **Goal:** Model `p(x|y)` and `p(y)`, then use **Bayes' Theorem** to find `p(y|x)` for classification.
*   **Bayes' Theorem:**
    `P(C_k | x) = [ p(x | C_k) * P(C_k) ] / p(x)`
*   **For Prediction:** We don't need the exact probability, just which class is most likely.
    `argmax_{k} P(C_k | x) = argmax_{k} [ p(x | C_k) * P(C_k) ]`
![[Pasted image 20250901180408.png]]
## Model Assumptions
1.  **Class Prior `P(C_k)`:** Modeled by a **Bernoulli distribution** for two classes.
    *   `P(C_1) = φ`
    *   `P(C_0) = 1 - φ`
    *   **MLE Estimate:** `φ = (Number of class 1 samples) / (Total samples) = N₁ / N`
![[Pasted image 20250901180439.png]]
2.  **Class-Conditional Density `p(x|C_k)`:** Assumed to follow a **Gaussian (Normal) Distribution**.
3. ![[Pasted image 20250901180740.png]]
    *   **1D Case:** `p(x | C_k) = (1 / √(2πσₖ²)) * exp( - (x - μₖ)² / (2σₖ²) )`
    *   **Multivariate Case:** `p(x | C_k) = (1 / ( (2π)^{D/2} |Σ|^{1/2} )) * exp( -½ (x - μₖ)ᵀ Σ⁻¹ (x - μₖ) )`
    *   **Note:** A common simplification is to assume all classes share the **same covariance matrix `Σ`**.

## Parameter Learning via MLE (1D Example)
**Given:** Data points for class `C_1`: `[1, 4, 3, 5]`
**Task:** Find MLE parameters `μ₁` and `σ₁²` for `p(x|C_1)`.
1.  **Calculate Mean `μ`:**
    `μ = (1 + 4 + 3 + 5) / 4 = 13 / 4 = 3.25`
2.  **Calculate Variance `σ²`:**
    `σ² = [ (1-3.25)² + (4-3.25)² + (3-3.25)² + (5-3.25)² ] / 4`
    `= [ (-2.25)² + (0.75)² + (-0.25)² + (1.75)² ] / 4`
    `= [ 5.0625 + 0.5625 + 0.0625 + 3.0625 ] / 4`
    `= [ 8.75 ] / 4 = 2.1875`
**Answer: `p(x|C_1)` is a Gaussian with `μ = 3.25` and `σ² = 2.1875`.**
![[Pasted image 20250901180839.png]]
![[Pasted image 20250901180901.png]]
![[Pasted image 20250901180941.png]]
## Parameter Learning via MLE (Multivariate, Shared Covariance)
For classes `C_1` and `C_2`:
*   **Priors:** `φ = N₁ / N`, `1 - φ = N₂ / N`
*   **Means:**
    `μ₁ = (1 / N₁) * ∑_{n: t_n=1} x_n`
    `μ₂ = (1 / N₂) * ∑_{n: t_n=0} x_n`
*   **Shared Covariance Matrix `Σ`:**
    `Σ = (1/N) * [ N₁*S₁ + N₂*S₂ ]`
    where `S₁` is the covariance matrix of class 1 data, and `S₂` is the covariance matrix of class 2 data.
![[Pasted image 20250901181035.png]]
## The Decision Boundary
*   With **shared covariance** `Σ`, the log-odds function `a = ln[ P(C₁|x) / P(C₂|x) ]` simplifies to a **linear function** of `x`.
    `a = wᵀx + w₀`
    where:
    `w = Σ⁻¹(μ₁ - μ₂)`
    `w₀ = -½ μ₁ᵀΣ⁻¹μ₁ + ½ μ₂ᵀΣ⁻¹μ₂ + ln(φ / (1-φ))`
*   **Decision Rule:** Predict class 1 if `a > 0` (i.e., `wᵀx + w₀ > 0`), else predict class 2.
*   If classes have **different covariance matrices (`Σ₁`, `Σ₂`)**, the decision boundary becomes **quadratic**.
![[Pasted image 20250901181133.png]]
![[Pasted image 20250901181153.png]]
![[Pasted image 20250901181218.png]]
![[Pasted image 20250901181258.png]]

---

# 4. Comparison & Summary

| Feature                  | **Logistic Regression (Discriminative)** | **Gaussian Discriminant Analysis (Generative)**                 |      |                |
| :----------------------- | :--------------------------------------- | :-------------------------------------------------------------- | ---- | -------------- |
| **What it models**       | `P(y                                     | x)` directly                                                    | `P(x | y)` and `P(y)` |
| **Decision Boundary**    | Linear                                   | Linear (if shared Σ) or Quadratic (if class-specific Σ)         |      |                |
| **Parameter Estimation** | Iterative (Gradient Descent)             | Analytical, closed-form solution (MLE)                          |      |                |
| **Assumptions**          | None on data distribution                | Assumes data per class is Gaussian                              |      |                |
| **Pros**                 | Often more accurate with enough data     | Faster training, can handle missing data, can generate new data |      |                |
| **Cons**                 | Requires optimization, needs more data   | Performance suffers if Gaussian assumption is violated          |      |                |

---

# 5. Key Takeaways for the Exam

1.  ***Know the Difference:*** Understand the fundamental philosophical difference between *discriminative* (`p(y|x)`) and *generative* (`p(x|y)`) models.
2.  ***Logistic Regression:***
    *   Know the *model*: `P(y=1|x) = σ(wᵀx)`.
    *   Understand why we use the *sigmoid function (to output a probability)*.
    *   Be able to *derive the decision boundary* `wᵀx = 0`.
    *   Understand the *gradient descent update rule* `w := w - η*(y_n - t_n)*x_n`.
3.  ***Generative Models:***
    *   Know *Bayes' Theorem* for classification and the *prediction rule argmax* p(x|C_k)P(C_k)`.
    *   Be able to *calculate the MLE parameters* (μ, σ²) for a Gaussian distribution from a small dataset (this is a very likely exam question).
    *   Understand that *shared covariance* leads to a linear decision boundary.
4.  ***Advantages/Disadvantages:*** Be prepared to discuss when you might *choose one model over the other*.

---
Of course. This is an excellent question that gets to the heart of how to interpret a logistic regression model.

The values **1.5** and **-4.078** are the learned parameters (weights or coefficients) of the model, `w₁` and `w₀` respectively, in the equation:

`z = w₀ + w₁ * x`  
`P(Pass | Hours) = σ(z) = 1 / (1 + e^{-z})`

Where:
*   `x` is the input feature (Hours of study).
*   `w₀` is the **intercept** or **bias** term.
*   `w₁` is the **coefficient** for the feature `x`.

Here’s what they mean individually and together:

---

### 1. The Coefficient (`w₁ = 1.5`): The "Direction and Strength"

The coefficient tells you the relationship between the input feature (Hours studied) and the **log-odds** of the outcome (Passing the exam).

*   **Sign (Positive):** A positive value (`+1.5`) means there is a **positive relationship**. As the number of study hours **increases**, the probability of passing the exam **increases**. This makes intuitive sense.
*   **Magnitude (1.5):** The size of the coefficient indicates **how strong** that relationship is. A larger absolute value means a stronger effect. For every **one additional hour** studied, the **log-odds** of passing the exam increases by **1.5**.

**What are "log-odds"?** It's the natural logarithm of the odds. Odds are defined as `P(pass) / P(fail)`. While this isn't intuitively easy to think about, it's the value that the linear part of the model (`w₀ + w₁x`) is actually predicting.

---

### 2. The Intercept (`w₀ = -4.078`): The "Baseline Odds"

The intercept represents the predicted log-odds of passing the exam **when the input feature is zero** (i.e., when `x = 0` hours studied).

*   **Sign (Negative):** A very negative value like `-4.078` means that the log-odds of passing are highly negative when a student hasn't studied at all. Translating this to probability:
    `Log-odds = -4.078`
    `Odds = e^{-4.078} ≈ 0.017`
    `Probability = Odds / (1 + Odds) ≈ 0.017 / (1.017) ≈ 0.0167 or 1.67%`

This also makes sense: a student who doesn't study at all has a very low (接近零) probability of passing.

---

### 3. Putting It All Together: How the Model Makes a Prediction

Let's calculate the probability for a student who studies for **3 hours**, as in the slides.

1.  **Calculate the linear combination (log-odds):**
    `z = w₀ + w₁*x = -4.078 + (1.5 * 3) = -4.078 + 4.5 = 0.422`

2.  **Squash the log-odds into a probability using the sigmoid function:**
    `P(Pass | Hours=3) = σ(0.422) = 1 / (1 + e^{-0.422})`

3.  **Calculate the exponent:**
    `e^{-0.422} ≈ 0.6558` (You can calculate this with a scientific calculator)

4.  **Final calculation:**
    `P(Pass | Hours=3) = 1 / (1 + 0.6558) = 1 / 1.6558 ≈ 0.604`

**Result:** The model predicts a **60.4% chance** of passing for a student who studies 3 hours.

### Summary Table

| Parameter | Value | Interpretation |
| :--- | :--- | :--- |
| **Intercept (`w₀`)** | -4.078 | The log-odds of passing when `Hours = 0`. Indicates a very low baseline probability. |
| **Coefficient (`w₁`)** | 1.5 | For every **one additional hour** studied, the **log-odds** of passing increases by **1.5**. |

**In simple terms:**
*   The negative **intercept** means "Starting from a very low chance..."
*   The positive **coefficient** means "...each hour of study significantly increases your odds of passing."

These values are found during the **training process** by maximizing the likelihood of the observed data (i.e., finding the parameters that make the actual passing outcomes seen in the training data most probable). The algorithm landed on `-4.078` and `1.5` as the best values to explain the relationship in the provided dataset.

---

Of course. Let's break down generative models in a clear, intuitive way.

### The Core Idea: The Storyteller vs. The Boundary Drawer

Imagine you need to tell the difference between **cats** and **dogs**.

1.  **Discriminative Model (The Boundary Drawer):**
    *   This person is a **judge**. They look at many pictures of cats and dogs and only care about one thing: **"What is the rule that separates these two groups?"**
    *   They learn to draw a line (a boundary). They don't need to know what a cat *is* or what a dog *is*. They just need to know how to tell them apart.
    *   **It models the decision boundary.** (e.g., Logistic Regression)

2.  **Generative Model (The Storyteller):**
    *   This person is a **biologist**. They study all the pictures of cats and, separately, all the pictures of dogs. They want to build a complete understanding of each animal.
    *   They learn: "Cats tend to be smaller, have pointy ears, and whiskers. Dogs tend to be bigger, have floppy ears, and a longer snout."
    *   Now, when they see a new animal, they ask: **"Which of my internal descriptions does this new animal match better? Does it look more like the 'cat' I've learned or the 'dog' I've learned?"**
    *   **It models what each class "looks like."**

---

### The Formal Definition, Made Simple

A generative model learns the **joint probability distribution, `P(X, Y)`**. This means it learns two things:

1.  **`P(Y)` (The Class Prior):** How likely is each category on its own?
    *   Example: In your email, what's the probability that any given email is `spam` vs. `not spam`? Maybe `P(spam) = 0.2` and `P(not spam) = 0.8`.

2.  **`P(X|Y)` (The Class-Conditional Distribution):** What do the features look like **for each specific class**?
    *   Example: What does spam email *look like*? It probably has words like "WIN", "FREE", "$$$". What does non-spam email look like? It has words like "meeting", "project", "lunch".
    *   This is the "generative" part. The model is learning the probability distribution of features **given a class**.

Once it knows these two things, it can use **Bayes' Theorem** to flip them around and calculate what we actually want for classification: `P(Y|X)` (the probability of a class *given* the features).

`P(Spam | Email)` = `[ P(Email | Spam) * P(Spam) ] / P(Email)`

**"The probability this email is spam is proportional to (how much it looks like spam) multiplied by (how common spam is)."**

### A Concrete Example: Fish Classification

Let's use the example from your slides.

**Task:** Classify a fish as a **Bass** or a **Tuna** based on its length (`x`).

**Generative Approach:**

1.  **Learn `P(Y)` (The Priors):**
    *   We go to the ocean and catch 100 fish.
    *   40 are Bass, 60 are Tuna.
    *   So, `P(Bass) = 0.4`, `P(Tuna) = 0.6`.

2.  **Learn `P(X|Y)` (The Class-Conditionals):**
    *   We measure the length of all 40 Bass. We see they are generally shorter. We fit a Gaussian (bell curve) to their lengths. This gives us `p(Length | Bass)`.
    *   We measure the length of all 60 Tuna. We see they are generally longer. We fit a different Gaussian to their lengths. This gives us `p(Length | Tuna)`.

    *Bass and Tuna have different length distributions (different means and variances).*

3.  **Classify a New Fish:**
    *   We catch a new fish that is 20 cm long.
    *   We ask our model two questions:
        1.  **How Tuna-like is it?** `p(Length=20 | Tuna) * P(Tuna)`
        2.  **How Bass-like is it?** `p(Length=20 | Bass) * P(Bass)`
    *   We predict the class that gives the **higher** value. The fish is assigned to the class it most likely **came from** or was **generated by**.

### Why is it Called "Generative"?

Because you can **generate** new, synthetic data points from it.

*   Once you have `P(X|Y)` for "Tuna", you can **create a fake, but plausible, Tuna length** by sampling from the "Tuna" Gaussian distribution.
*   This is how AI art generators like DALL-E work. They are incredibly complex generative models that have learned `P(Pixels | "a picture of a teddy bear researching AI")`. When you give it the text, it samples from that distribution to generate a new image.

### Summary: Key Characteristics of Generative Models

| Feature | Explanation | Simple Analogy |
| :--- | :--- | :--- |
| **Models `P(X, Y)`** | Learns the full probability distribution of the data and labels. | Learns the complete "description" of each class. |
| **Can Generate Data** | Can create new, realistic data points for any class. | An artist who can paint a new picture of a cat from memory. |
| **Robust to Missing Data**| If some features of a new data point are missing, it can often "fill in the blanks" using its understanding of the distribution. | If you only see an animal's silhouette, you can still guess what it is based on its overall shape. |
| **Can Be Less Accurate** | If its assumption about the data distribution (e.g., that it's Gaussian) is wrong, its performance can suffer. | If the biologist assumes all fish are shaped like bells, they'll be wrong about flatfish. |
| **Computational Cost** | Often has more parameters to learn than a discriminative model. | The biologist has to do more work (study each animal in detail) than the judge (who just learns the border). |

In essence, a generative model is a **world-building** algorithm. It doesn't just learn to separate categories; it learns to understand and describe the essence of each category itself. This makes it powerful not just for classification, but for creation and imagination.

---

Of course. Let's break down multivariate Gaussian (or Normal) distributions. This is a key concept for understanding advanced generative models.

### The Core Idea: From One Dimension to Many

1.  **Univariate Gaussian (The Familiar One):**
    *   This is the classic "bell curve." It describes the distribution of a **single variable** (like height, test scores, or fish length).
    *   It's defined by two parameters:
        *   **Mean (μ):** The center/average value. Where the peak of the bell is.
        *   **Variance (σ²):** The spread or width of the bell. A high variance means the data is spread out; a low variance means it's clustered near the mean.

2.  **Multivariate Gaussian (The Generalization):**
    *   This is the bell curve for **multiple variables at once**. It describes how several related variables are distributed together (like height *and* weight, or study hours *and* exam score).
    *   It's defined by two new parameters that generalize the concepts of mean and variance:
        *   **Mean Vector (μ):** A vector that contains the mean value for *each* variable.
        *   **Covariance Matrix (Σ):** A matrix that describes **how each variable relates to the others**.

---

### The Covariance Matrix: The Heart of the Matter

The covariance matrix (`Σ`) is what makes the multivariate Gaussian so powerful. It doesn't just describe the spread of each individual variable (like variance did), but also the **relationships between them**.

Let's say we have two variables: Height (`X₁`) and Weight (`X₂`).

*   The **diagonal** elements of `Σ` (e.g., `Σ[1,1]` and `Σ[2,2]`) are simply the **variances** of `X₁` and `X₂` themselves. They control the spread along each axis.

*   The **off-diagonal** elements (e.g., `Σ[1,2]` and `Σ[2,1]`) are the **covariances** between `X₁` and `X₂`.
    *   **Positive Covariance:** When one variable is above its mean, the other tends to be above its mean too. (Taller people tend to be heavier). This creates an **oval-shaped** cloud that slopes **upwards**.
    *   **Negative Covariance:** When one variable is above its mean, the other tends to be below its mean. This creates an oval-shaped cloud that slopes **downwards**.
    *   **Zero Covariance:** There is no linear relationship between the variables. The cloud would be a circle (if variances are equal) or an axis-aligned oval.

**Visual Guide to Covariance:**

```python
# Imagine these are the parameters for a distribution of people's Height and Weight.
mean_vector = [175, 75]  # μ = [mean_height = 175cm, mean_weight = 75kg]

# Covariance Matrix 1: Height and Weight are positively correlated.
cov_matrix_1 = [[50, 40],  # Variance of Height is 50, Covariance with Weight is 40
                [40, 60]]  # Covariance with Height is 40, Variance of Weight is 60

# Covariance Matrix 2: Height and Weight are not correlated.
cov_matrix_2 = [[50, 0],  # Variance of Height is 50, Covariance is 0
                [0, 60]]  # Covariance is 0, Variance of Weight is 60
```
*A plot of samples from these distributions would show an upward-sloping oval for `cov_matrix_1` and a vertical/horizontal oval for `cov_matrix_2`.*

---

### The Formula (For Understanding, Not Memorization)

The probability density function looks complex, but its parts make sense:

$p(\mathbf{x}) = \frac{1}{(2\pi)^{D/2} |\Sigma|^{1/2}} \exp \left( -\frac{1}{2} (\mathbf{x} - \mathbf{\mu})^\top \Sigma^{-1} (\mathbf{x} - \mathbf{\mu}) \right)$

Let's dissect it:

1.  **`(x - μ)`:** This is just the distance of a data point from the mean center.
2.  **`(x - μ)ᵀ Σ⁻¹ (x - μ)`:** This is the **Mahalanobis distance**. It's a critical concept. It's like a "super-powered" Euclidean distance that accounts for correlation.
    *   It measures how many "standard deviations" away a point is from the mean, but it does so in a way that understands the shape of the distribution. A point that is far out along the "long axis" of the oval is not considered unusual, but a point that is far out along the "short axis" is.
3.  **`(2π)^{D/2} |Σ|^{1/2}`:** This is the normalization constant. It ensures the total area under the multi-dimensional "bell" equals 1, making it a valid probability distribution. `|Σ|` is the determinant of the covariance matrix, which roughly corresponds to the overall "volume" of the distribution.

---

### Why is this Useful in Machine Learning?

In your lecture on **Gaussian Discriminant Analysis (GDA)**, this is exactly what's used.

*   **For each class** (e.g., Bass and Tuna), you don't just model one feature (e.g., length). You model *multiple features together* (e.g., length *and* lightness of skin color).
*   You fit a **separate multivariate Gaussian distribution** to the data points of each class.
    *   **Class "Bass":** `N(μ_bass, Σ_bass)`
    *   **Class "Tuna":** `N(μ_tuna, Σ_tuna)`
*   To classify a new fish, you calculate its Mahalanobis distance to the center of each class's distribution (adjusted by the class prior `P(class)`). You assign it to the class for which this probability is highest.
*   **The "Shared Covariance" trick:** The model often assumes `Σ_bass = Σ_tuna = Σ`. This simplifies the math and means the decision boundary between classes becomes a straight line (or hyperplane). If they are different, the boundary becomes curved (quadratic).

### Summary Table

| Concept | Univariate Gaussian | Multivariate Gaussian |
| :--- | :--- | :--- |
| **What it describes** | Distribution of **one variable** | Joint distribution of **multiple related variables** |
| **"Center"** | Mean (μ) - a number | Mean Vector (μ) - a list of numbers (e.g., [mean_x, mean_y]) |
| **"Spread & Shape"** | Variance (σ²) - a number. Controls width. | Covariance Matrix (Σ) - a matrix. Controls width, and **orientation** (correlation). |
| **The "Bell Curve"** | A bell curve on a line. | A multi-dimensional bell-shaped "hill" (e.g., a 2D mountain). |
| **Key Calculation** | Standard Deviation `(x - μ)/σ` | Mahalanobis Distance `(x - μ)ᵀΣ⁻¹(x - μ)` |

In short, a multivariate Gaussian distribution is the natural extension of the familiar bell curve into higher dimensions. It's powerful because it doesn't just model where the data is centered and how spread out it is, but also **how the different features of the data move together**.