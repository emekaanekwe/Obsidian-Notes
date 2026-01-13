# Machine Learning Study Sheet: Bias-Variance Decomposition (From Lecture Transcription)

## **1. What Is the Goal of Studying Bias and Variance?**
- To **diagnose** model performance and identify **overfitting** or **underfitting**.
- To understand how to **fix** models using techniques like **regularization**.
- Provides a **mathematical framework** for decomposing generalization error into interpretable components.

## **2. What Are Bias and Variance Intuitively?**
- **Bias**: Measures **accuracy** of a model.
  - **High bias** → low accuracy.
  - Difference between **average prediction** and **true target**.
- **Variance**: Measures **consistency** or **stability**.
  - **High variance** → low consistency.
  - Measures **spread** of predictions across different training datasets.

## **3. How Are Bias and Variance Illustrated with a Dartboard?**
- **Center** = true target value.
- **Darts** = predictions from models trained on different datasets.
- **Four scenarios**:
  - **Low bias, low variance**: Darts clustered near center (**ideal**).
  - **Low bias, high variance**: Darts scattered but centered near target (inconsistent).
  - **High bias, low variance**: Darts clustered away from center (consistent but inaccurate).
  - **High bias, high variance**: Darts scattered far from center (**worst case**).

## **4. How Do We Empirically Estimate Bias and Variance?**
- Use **bootstrap sampling**:
  1. Generate multiple datasets via resampling.
  2. Train a model on each dataset.
  3. For a given input \(x\), collect predictions from all models.
  4. Compute **average prediction** → estimate bias.
  5. Compute **variance** of predictions → estimate variance.

## **5. What Is the Mathematical Definition of Generalization Error?**
- For regression, generalization error is:
  \[
  \text{Error} = \mathbb{E}_{x \sim p(x)}\left[(y(x) - h(x))^2\right]
  \]
  where:
  - \(h(x)\) = true target function
  - \(y(x)\) = model prediction
  - \(p(x)\) = input distribution

## **6. How Is Generalization Error Decomposed Mathematically?**
- **Derivation**:
  \[
  \mathbb{E}\left[(y(x) - h(x))^2\right] = \underbrace{\left(\mathbb{E}[y(x)] - h(x)\right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\left[(y(x) - \mathbb{E}[y(x)])^2\right]}_{\text{Variance}}
  \]
  - **Key step**: Adding and subtracting \(\mathbb{E}[y(x)]\) and expanding the square.
  - Cross-term vanishes because \(\mathbb{E}[y(x) - \mathbb{E}[y(x)]] = 0\).

## **7. What Are the Practical Implications of Bias and Variance?**
- **High bias (underfitting)**:
  - Model too simple.
  - **Fix**: Add features, increase model complexity, reduce regularization.
- **High variance (overfitting)**:
  - Model too complex.
  - **Fix**: Reduce features, add regularization, use more training data.

## **8. How Does Regularization Affect Bias and Variance?**
- **Ridge regression example**:
  - **High λ (strong regularization)** → simpler model → **low variance, high bias**.
  - **Low λ (weak regularization)** → complex model → **high variance, low bias**.
  - **Optimal λ** balances bias and variance → minimizes generalization error.

## **9. What Is the Bias-Variance Trade-off?**
- **Flexible models** (e.g., high-degree polynomials, neural networks):
  - **Low bias, high variance**.
- **Rigid models** (e.g., linear regression):
  - **High bias, low variance**.
- **Trade-off**: Reducing bias tends to increase variance, and vice versa.

## **10. How Do We Choose Model Complexity?**
- **Polynomial regression example**:
  - Degree 0: High bias, low variance.
  - Degree 3: Low bias, low variance (**optimal**).
  - Degree 15: Low bias, high variance (overfitting).
- **Test error** = bias² + variance → minimized at optimal complexity.

## **11. How Do We Diagnose and Fix Models in Practice?**
- **High test error** → diagnose bias/variance:
  - **High variance?** → Get more data, reduce features, increase regularization.
  - **High bias?** → Add features, increase model complexity, reduce regularization.
- **Key questions**:
  - **More data?** Helps if high variance, not if high bias.
  - **Fewer features?** Helps if high variance, hurts if high bias.
  - **New features?** Helps if high bias, may hurt if high variance.

## **12. What Is the Takeaway from the Experimental Examples?**
- **Bias² + variance** is a **good proxy** for generalization error.
- **Empirical plots** (bias, variance, error vs. λ) confirm theoretical decomposition.
- **Optimal model** balances bias and variance → neither too simple nor too complex.

---

## **Key Formulas to Remember**
- **Bias**: \(\text{Bias}(x) = \mathbb{E}[y(x)] - h(x)\)
- **Variance**: \(\text{Var}(x) = \mathbb{E}\left[(y(x) - \mathbb{E}[y(x)])^2\right]\)
- **Generalization error**: \(\text{Error}(x) = \text{Bias}^2(x) + \text{Var}(x)\)
- **Average over dataset**: \(\text{Error} = \frac{1}{N}\sum_{i=1}^N \left[\text{Bias}^2(x_i) + \text{Var}(x_i)\right]\)

---

Let me know if you’d like a **flashcard set** or **practice problems** based on this lecture.