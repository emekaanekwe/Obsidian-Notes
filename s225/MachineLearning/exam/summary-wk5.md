# Machine Learning Study Sheet: Classification & Linear Models

## **1. What Is Classification?**
- **Goal**: Associate input with a **discrete target** (class label).
- **Examples**:
  - Spam vs. non‑spam emails.
  - Handwritten digit recognition (0–9).
  - Credit risk: good vs. bad credit.
- **Classifier**: Algorithm that maps input → one of \(K\) classes.

---

## **2. How Is a Data Point Represented in Classification?**
- Each instance is a **feature vector** \( \mathbf{x} = [x_1, x_2, …, x_D] \).
- Features are measurable properties (e.g., length, width, color).
- Geometrically: each data point is a **point** in a \(D\)-dimensional space.

---

## **3. What Is a Linear Classifier?**
- A classifier defined by a **linear function**:
  \[
  f(\mathbf{x}) = w_0 + \mathbf{w}^T \mathbf{x}
  \]
  where:
  - \(w_0\) = **bias** (threshold)
  - \(\mathbf{w}\) = **weight vector**
- **Decision rule**:
  - If \(f(\mathbf{x}) \ge 0\) → class \(+1\)
  - If \(f(\mathbf{x}) < 0\) → class \(-1\)
- **Decision boundary**: Hyperplane where \(f(\mathbf{x}) = 0\).

---

## **4. How Is the Decision Boundary Visualized?**
- In 2D: a **line**.
- In \(D\)D: a **hyperplane**.
- **Weight vector \(\mathbf{w}\)** is **orthogonal** to the decision boundary.
- **Bias \(w_0\)** shifts the boundary parallel to itself.

---

## **5. What Is Linear Separability?**
- A dataset is **linearly separable** if a **single hyperplane** can perfectly separate the classes.
- **Not all problems** are linearly separable (e.g., “moon‑shaped” data).
- For separable problems, **infinitely many** perfect classifiers exist on training data.

---

## **6. What Is the Connection Between Regression and Classification?**
- A classifier can be built by applying an **activation/link function** to a regression output:
  \[
  y(\mathbf{x}) = f(w_0 + \mathbf{w}^T\mathbf{x})
  \]
- **Two common activations**:
  1. **Sign/step function** → **deterministic** class label (e.g., perceptron).
  2. **Logistic function** → **probabilistic** output (e.g., logistic regression).

---

## **7. What Are the Three Main Approaches to Classification?**
1. **Discriminative models** (e.g., perceptron):
   - Model \(P(y \mid \mathbf{x})\) directly.
   - No modeling of input distribution.
2. **Probabilistic discriminative models** (e.g., logistic regression):
   - Model \(P(y \mid \mathbf{x})\) with probabilities.
   - Outputs class probabilities.
3. **Generative models** (e.g., Naive Bayes):
   - Model \(P(\mathbf{x} \mid y)\) and \(P(y)\).
   - Can generate synthetic data.

---

## **8. How Do We Extend Binary Classification to Multi‑Class (\(K > 2\))?**
- **Three strategies**:
  1. **One‑vs‑Rest (OvR)**: Train \(K\) classifiers, each separating one class from all others.
     - **Problem**: Ambiguous regions where multiple classifiers claim “+1”.
  2. **One‑vs‑One (OvO)**: Train \(K(K-1)/2\) classifiers for each pair.
     - **Problem**: Voting ties and uncovered regions.
  3. **K discriminant functions**: Learn \(K\) scoring functions \(f_k(\mathbf{x}) = \mathbf{w}_k^T\mathbf{x} + w_{k0}\).
     - **Decision rule**: Choose class with **highest score**.
     - **Geometrically**: Each point assigned to **closest** hyperplane.

---

## **9. What Is the Perceptron?**
- **Structure**:
  - Single neuron with weights \(\mathbf{w}\) and bias \(w_0\).
  - Activation = **sign/step function**.
- **Learning goal**: Minimize **number of misclassifications**.
- **Error function** (for a misclassified point):
  \[
  E(\mathbf{w}) = -\sum_{n \in \mathcal{M}} t_n \mathbf{w}^T\mathbf{x}_n
  \]
  where \(\mathcal{M}\) = set of misclassified examples.
- **Why this form?**: Differentiable → enables gradient descent.

---

## **10. How Does the Perceptron Learning Algorithm Work?**
1. Initialize \(\mathbf{w}\) randomly.
2. For each data point \((\mathbf{x}_n, t_n)\):
   - Compute prediction \(y_n = \text{sign}(\mathbf{w}^T\mathbf{x}_n)\).
   - If \(y_n = t_n\) → do nothing.
   - If \(y_n \neq t_n\) → update:
     \[
     \mathbf{w} \leftarrow \mathbf{w} + \eta \, t_n \mathbf{x}_n
     \]
     where \(\eta\) = learning rate.
3. Repeat until **no errors** (or max iterations).

---

## **11. What Are the Key Properties of the Perceptron?**
- **Convergence**: Guaranteed if data is **linearly separable** (but may take many iterations).
- **Non‑separable data**: May never converge (cycles indefinitely).
- **Sensitivity**: Depends on **initialization** and **order** of data presentation.
- **Multiple solutions**: Many weight vectors yield zero training error.

---

## **12. How Is Multi‑Class Perceptron Learned?**
- Maintain \(K\) weight vectors \(\mathbf{w}_1, \dots, \mathbf{w}_K\).
- For each data point \((\mathbf{x}_n, t_n)\):
  - Predict \(y_n = \arg\max_k \mathbf{w}_k^T\mathbf{x}_n\).
  - If \(y_n = t_n\) → no change.
  - If \(y_n \neq t_n\):
    - **Increase** score of correct class: $\mathbf{w}_{t_n} \leftarrow \mathbf{w}_{t_n} + \eta \mathbf{x}_n$
    - **Decrease** score of wrong class: $\mathbf{w}_{y_n} \leftarrow \mathbf{w}_{y_n} - \eta \mathbf{x}_n$

---

## **13. What Are the Geometric Insights?**
- **Feature space**: Data points live in \(\mathbb{R}^D\).
- **Decision boundary**: Hyperplane orthogonal to \(\mathbf{w}\).
- **Distance to boundary**: $\frac{|w_0 + \mathbf{w}^T\mathbf{x}|}{\|\mathbf{w}\|}$.
- **Multi‑class decision regions**: Convex polygons formed by intersecting half‑spaces.

---

## **Key Formulas to Remember**
- Linear classifier: $f(\mathbf{x}) = w_0 + \mathbf{w}^T\mathbf{x}$
- Decision rule: $\hat{y} = \text{sign}(f(\mathbf{x}))$
- Perceptron update: $\mathbf{w} \leftarrow \mathbf{w} + \eta \, t_n \mathbf{x}_n$
- Multi‑class prediction: $\hat{y} = \arg\max_k (\mathbf{w}_k^T\mathbf{x} + w_{k0})$