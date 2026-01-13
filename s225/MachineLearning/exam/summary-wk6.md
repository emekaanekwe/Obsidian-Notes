# Machine Learning Study Sheet: Probabilistic Classification & Generative Models  

## **1. What Are Probabilistic Discriminative Models?**  
- **Goal**: Directly model $P(y \mid \mathbf{x})$ → predict **class probabilities**.  
- **Example**: **Logistic regression**.  
- **Advantage**: Outputs **uncertainty** (e.g., “80% cat, 20% dog”).  
- **Comparison**:  
  - **Non‑probabilistic** (e.g., perceptron): outputs class label only.  
  - **Probabilistic** (e.g., logistic regression): outputs class probability.  

---

## **2. What Is Logistic Regression?**  
- A **generalized linear model** with a **logistic (sigmoid) link function**:  
  $$  
  P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x}) = \frac{1}{1+e^{-\mathbf{w}^T\mathbf{x}}}  
  $$  
- **Sigmoid function** $\sigma(z)$:  
  - Squashes real values $z \in (-\infty, \infty)$ → $(0,1)$.  
  - $\sigma(0)=0.5$ → decision boundary at $\mathbf{w}^T\mathbf{x}=0$.  
- **Decision rule**:  
  - If $P(y=1 \mid \mathbf{x}) \ge 0.5$ → predict class 1.  
  - Otherwise → predict class 2.  

---

## **3. How Is Logistic Regression Trained?**  
- **Maximum likelihood estimation (MLE)**.  
- **Likelihood function** for binary labels $t_n \in \{0,1\}$:  
  $$  
  L(\mathbf{w}) = \prod_{n} P(y=1 \mid \mathbf{x}_n)^{t_n} \; P(y=0 \mid \mathbf{x}_n)^{1-t_n}  
  $$  
- **Log‑likelihood** (simplified):  
  $$  
  \ell(\mathbf{w}) = \sum_n \big[ t_n \ln \sigma(\mathbf{w}^T\mathbf{x}_n) + (1-t_n) \ln(1-\sigma(\mathbf{w}^T\mathbf{x}_n)) \big]  
  $$  
- **Optimization**: No closed‑form solution → use **gradient descent** (or SGD).  
- **Gradient** for one data point:  
  $$  
  \nabla_{\mathbf{w}} \ell(\mathbf{w}) = (t_n - \sigma(\mathbf{w}^T\mathbf{x}_n)) \; \mathbf{x}_n  
  $$  
- **Update rule** (SGD):  
  $$  
  \mathbf{w} \leftarrow \mathbf{w} + \eta \, (t_n - \sigma(\mathbf{w}^T\mathbf{x}_n)) \; \mathbf{x}_n  
  $$  

---

## **4. What Are Generative Models for Classification?**  
- **Goal**: Model $P(\mathbf{x} \mid y)$ and $P(y)$ → generate data given class.  
- **Use Bayes’ rule** for prediction:  
  $$  
  P(y \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid y) \, P(y)}{P(\mathbf{x})}  
  $$  
- **Two components**:  
  1. **Class prior** $P(y)$: probability of each class.  
  2. **Class‑conditional distribution** $P(\mathbf{x} \mid y)$: distribution of features given class.  

---

## **5. How Are Class Priors Modeled?**  
- For **K classes**, use a **categorical distribution**.  
- **Binary case** (two classes):  
  $$  
  P(y=1) = \phi, \quad P(y=0) = 1-\phi  
  $$  
- **MLE estimate**:  
  $$  
  \hat{\phi} = \frac{\text{\# class‑1 examples}}{\text{total examples}}  
  $$  

---

## **6. How Are Class‑Conditional Distributions Modeled?**  
- **Common choice**: **Gaussian distribution**.  
- **Univariate Gaussian** (single feature):  
  $$  
  P(x \mid y=k) = \mathcal{N}(x \mid \mu_k, \sigma_k^2)  
  $$  
- **Multivariate Gaussian** (D features):  
  $$  
  P(\mathbf{x} \mid y=k) = \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)  
  $$  
  where:  
  - $\boldsymbol{\mu}_k$ = mean vector (D‑dimensional).  
  - $\boldsymbol{\Sigma}_k$ = covariance matrix (D×D).  

---

## **7. How Are Gaussian Parameters Learned?**  
- **MLE estimates**:  
  - **Mean** for class $k$:  
    $$  
    \hat{\boldsymbol{\mu}}_k = \frac{1}{N_k} \sum_{n: y_n=k} \mathbf{x}_n  
    $$  
  - **Covariance** (shared across classes):  
    $$  
    \hat{\boldsymbol{\Sigma}} = \frac{1}{N} \sum_k N_k \, \mathbf{S}_k  
    $$  
    where $\mathbf{S}_k$ is empirical covariance of class‑$k$ data.  
  - **Covariance** (class‑specific):  
    $$  
    \hat{\boldsymbol{\Sigma}}_k = \frac{1}{N_k} \sum_{n: y_n=k} (\mathbf{x}_n - \hat{\boldsymbol{\mu}}_k)(\mathbf{x}_n - \hat{\boldsymbol{\mu}}_k)^T  
    $$  

---

## **8. What Is the Decision Boundary for Gaussian Generative Models?**  
- **Prediction rule**: Choose class with highest $P(y \mid \mathbf{x})$.  
- **Log‑odds ratio** (binary case):  
  $$  
  \ln \frac{P(y=1 \mid \mathbf{x})}{P(y=0 \mid \mathbf{x})} = \mathbf{w}^T \mathbf{x} + w_0  
  $$  
- **If covariance matrices are shared** ($\boldsymbol{\Sigma}_1 = \boldsymbol{\Sigma}_2$):  
  - Decision boundary is **linear**.  
- **If covariances are different**:  
  - Decision boundary is **quadratic**.  

---

## **9. How Do Generative Models Differ from Discriminative Models?**  
| **Aspect**               | **Generative Models**                          | **Discriminative Models**               |  
|--------------------------|-----------------------------------------------|-----------------------------------------|  
| **Models**               | $P(\mathbf{x} \mid y)$ and $P(y)$             | $P(y \mid \mathbf{x})$                  |  
| **Data generation**      | Can **synthesize** new data                   | Cannot generate data                    |  
| **Data efficiency**      | Usually requires **more data**                | Often needs less data                   |  
| **Example algorithms**   | Naive Bayes, Gaussian discriminant analysis   | Logistic regression, perceptron          |  
| **Decision boundary**    | Can be linear or quadratic                    | Linear (for linear models)              |  

---

## **10. What Are the Key Steps for Gaussian Discriminant Analysis (GDA)?**  
1. **Estimate class priors** $\hat{\phi}_k$ from class frequencies.  
2. **Estimate class means** $\hat{\boldsymbol{\mu}}_k$ as class‑wise averages.  
3. **Estimate covariance** $\hat{\boldsymbol{\Sigma}}$ (shared or class‑specific).  
4. **For a new $\mathbf{x}$**, compute:  
   $$  
   P(y=k \mid \mathbf{x}) \propto \mathcal{N}(\mathbf{x} \mid \hat{\boldsymbol{\mu}}_k, \hat{\boldsymbol{\Sigma}}_k) \; \hat{\phi}_k  
   $$  
5. **Predict** class with highest posterior probability.  

---

## **Key Formulas to Remember**  
- **Logistic regression**: $P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x})$  
- **Sigmoid derivative**: $\sigma'(z) = \sigma(z)(1-\sigma(z))$  
- **Gaussian PDF**:  
  $$  
  \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}, \boldsymbol{\Sigma}) = \frac{1}{(2\pi)^{D/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\!\big(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\big)  
  $$  
- **Log‑odds decision**: $\mathbf{w}^T\mathbf{x} + w_0 = 0$ (linear boundary if shared covariance).  

---

Let me know if you’d like a **comparison table** or **worked examples** for logistic regression vs. Gaussian generative models.