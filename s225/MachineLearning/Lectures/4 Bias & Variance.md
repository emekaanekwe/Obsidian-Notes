
---

### **Study Sheet: Bias-Variance Decomposition & Diagnosis**

#### **1. Core Concepts & Intuition**

*   **Generalization Error:** The true error of a model on the entire distribution of data, not just the training set. It's what we ultimately want to minimize.
*   **Bias:** An indicator of a model's **accuracy**.
    *   **Definition:** The difference between the *average prediction* of our model (over many datasets) and the *true target value*.
    *   **High Bias:** The model is systematically wrong (inaccurate). It often leads to **underfitting** (model is too simple to capture patterns).
*   **Variance:** An indicator of a model's **consistency** or stability.
    *   **Definition:** The variability of the model's predictions for a given data point across different training datasets.
    *   **High Variance:** The model is overly sensitive to the noise in the training data. It often leads to **overfitting** (model is too complex and learns the noise).

*   **The Dartboard Analogy:**
    *   **Bullseye (Center):** The true target value.
    *   **Dart Throws (Blue Dots):** Predictions from models trained on different datasets.
    *   **Average of Throws:** The model's bias.
    *   **Spread of Throws:** The model's variance.
    *   **Four Quadrants:**
        1.  **Low Bias, Low Variance (Ideal):** Accurate and consistent.
        2.  **Low Bias, High Variance:** On average correct, but unreliable and inconsistent (overfitting).
        3.  **High Bias, Low Variance:** Consistently but systematically wrong (underfitting).
        4.  **High Bias, High Variance (Worst):** Inaccurate and inconsistent.

#### **2. Mathematical Derivation (Key Exam Takeaway)**

The **generalization error for a regression problem** can be decomposed as follows:

**Generalization Error = Variance + Bias²**

*   **Formula:** `E[(y(x; D) - h(x))²] = Var[y(x; D)] + (E[y(x; D)] - h(x))²`
    *   `E[]`: Expectation (average) over all possible training datasets `D`.
    *   `y(x; D)`: Prediction of a model (trained on dataset `D`) for input `x`.
    *   `h(x)`: The true, unknown target function.
    *   `E[y(x; D)]`: The average prediction over all models (bootstrap process).
*   **How to achieve this decomposition:**
    1.  Start with the error: `(y(x; D) - h(x))²`
    2.  Add and subtract the average prediction: `( (y(x; D) - E[y(x; D)]) + (E[y(x; D)] - h(x)) )²`
    3.  Expand the square. The cross-term cancels out to zero, leaving:
        *   `(y(x; D) - E[y(x; D)])²` → **Variance**
        *   `(E[y(x; D)] - h(x))²` → **Bias²**

**Why this matters:** It proves mathematically that to minimize total error, you must minimize **both** bias and variance. Improving one often worsens the other (**Bias-Variance Tradeoff**).

#### **3. Empirical Demonstration & The Tradeoff**

*   **Tool:** **Bootstrap Sampling** is used to simulate having multiple datasets. We build many models on different bootstrap samples to estimate the average prediction `E[y(x; D)]` and the variance.
*   **Model Complexity:** Bias and variance change with model complexity.
    *   **Simple Models (e.g., High λ in Ridge Regression, Low-Degree Polynomials):**
        *   **High Bias:** Too rigid to capture true patterns.
        *   **Low Variance:** Not sensitive to small changes in data.
    *   **Complex Models (e.g., Low λ, High-Degree Polynomials):**
        *   **Low Bias:** Flexible enough to approximate the truth.
        *   **High Variance:** Highly sensitive to noise in the training data.
*   **The "Sweet Spot":** The optimal model complexity is where the **sum of (Bias² + Variance)** is minimized. This is the best trade-off.

#### **4. Practical Diagnosis & Remedies**

Use the behavior of your model on validation/test data to diagnose issues and apply fixes.

| If your model has... | **High Variance (Overfitting)**                                  | **High Bias (Underfitting)**                                   |
| :------------------- | :--------------------------------------------------------------- | :------------------------------------------------------------- |
| **Symptoms**         | Low training error, high test error, inconsistent predictions    | High training error, high test error, consistently poor        |
| **Possible Fixes**   | 1. Get **more training data**.<br>2. **Reduce features**.<br>3. **Increase regularization** (λ). | 1. **Add more / better features**.<br>2. **Use a more complex model**.<br>3. **Decrease regularization** (λ). |
| **Will it help?**    | ✅ Adding data & Reducing complexity tackles variance.            | ✅ Adding features & Increasing complexity tackles bias.        |
| **Will it help?**    | ❌ These actions will make high bias worse.                      | ❌ These actions will make high variance worse.                |

#### **5. Key Definitions & Formulas for Recall**

*   **Average Prediction:** `ȳ(x) = (1/L) * Σ [y^l(x)]` (average over `L` bootstrap models)
*   **Bias²:** `(1/N) * Σ [ (ȳ(x_n) - h(x_n))² ]` (average over `N` data points)
*   **Variance:** `(1/N) * Σ [ (1/L) * Σ (y^l(x_n) - ȳ(x_n))² ] ]`
*   **Test Error (Empirical):** `(1/N) * (1/L) * Σ Σ (y^l(x_n) - h(x_n))²`


---

