

Training set:  
$$  
\mathcal{D}={(x_n,t_n)}_{n=1}^N  
$$

Deterministic model:  
$$  
y(x,w)  
$$

Gaussian noise model (this is the key assumption):  
$$  
t_n = y(x_n,w) + \epsilon_n,\qquad \epsilon_n\sim \mathcal{N}(0,\sigma^2)\ \text{i.i.d.}  
$$
## Prerequisite math rules you need

### Gaussian density

$$  
\mathcal{N}(t\mid \mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(t-\mu)^2}{2\sigma^2}\right)  
$$

### i.i.d. likelihood is a product

If samples are conditionally independent given $w$:  
$$  
p(\mathbf{t}\mid \mathbf{x},w)=\prod_{n=1}^N p(t_n\mid x_n,w)  
$$

### Log turns products into sums

$$  
\ln\left(\prod_{n=1}^N a_n\right)=\sum_{n=1}^N \ln a_n  
$$

---

## Step 1: Write the likelihood under Gaussian noise

Because $t_n\mid x_n,w \sim \mathcal{N}(y(x_n,w),\sigma^2)$:  
$$  
p(t_n\mid x_n,w,\sigma^2)=\mathcal{N}(t_n\mid y(x_n,w),\sigma^2)  
$$

So the likelihood of the whole dataset is:  
$$  
p(\mathbf{t}\mid \mathbf{x},w,\sigma^2)=\prod_{n=1}^N \mathcal{N}(t_n\mid y(x_n,w),\sigma^2)  
$$

---

## Step 2: Take log-likelihood

Plug in the Gaussian formula and take logs:

 $$  
\ln p(\mathbf{t}\mid \mathbf{x},w,\sigma^2)

\sum_{n=1}^N \left[  
-\frac{1}{2}\ln(2\pi\sigma^2)  
-\frac{(t_n-y(x_n,w))^2}{2\sigma^2}  
\right]  
$$

# Rearrange:  
$$  
\ln p(\mathbf{t}\mid \mathbf{x},w,\sigma^2)

-\frac{N}{2}\ln(2\pi\sigma^2)  
-\frac{1}{2\sigma^2}\sum_{n=1}^N (t_n-y(x_n,w))^2  
$$

---

## Step 3: Convert to a loss (negative log-likelihood)

Define the loss as negative log-likelihood:  
$$  
\mathcal{L}(w)= -\ln p(\mathbf{t}\mid \mathbf{x},w,\sigma^2)  
$$

So:  
$$  
\mathcal{L}(w)=  
\frac{N}{2}\ln(2\pi\sigma^2)  
+\frac{1}{2\sigma^2}\sum_{n=1}^N (t_n-y(x_n,w))^2  
$$

If $\sigma^2$ is fixed, the first term is a constant w.r.t. $w$, so **maximizing likelihood** is equivalent to **minimizing sum of squared errors**:  
$$  
w_{\text{ML}}=\arg\min_w \sum_{n=1}^N (t_n-y(x_n,w))^2  
$$

**Common confusion:** MSE vs SSE: MSE is $\frac{1}{N}\sum(\cdot)^2$; SSE is $\sum(\cdot)^2$. Same minimizer.

---

# Hand-worked numeric example (full MLE)

Let’s choose a very simple deterministic model:  
$$  
y(x,w)=wx  
$$

Data ($N=3$):  
$$  
(x_1,t_1)=(1,1),\quad (x_2,t_2)=(2,2),\quad (x_3,t_3)=(3,2)  
$$

### 1) Derive the MLE for $w$ (closed form)

We minimize:  
$$  
S(w)=\sum_{n=1}^3 (t_n-wx_n)^2  
$$

# Differentiate and set to zero:  
$$  
\frac{dS}{dw}

# \sum_{n=1}^3 2(t_n-wx_n)(-x_n)

-2\sum_{n=1}^3 x_n t_n + 2w\sum_{n=1}^3 x_n^2  
$$

Set $\frac{dS}{dw}=0$:  
$$  
-2\sum x_n t_n + 2w\sum x_n^2=0  
;\Rightarrow;  
w_{\text{ML}}=\frac{\sum_{n=1}^N x_n t_n}{\sum_{n=1}^N x_n^2}  
$$

Now compute the sums:  
$$  
\sum x_n t_n = 1\cdot 1 + 2\cdot 2 + 3\cdot 2 = 1+4+6=11  
$$  
$$  
\sum x_n^2 = 1^2+2^2+3^2 = 1+4+9=14  
$$

So:  
$$  
w_{\text{ML}}=\frac{11}{14}\approx 0.786  
$$

### 2) Compute predictions and squared errors

Predictions $\hat t_n = wx_n$:

- $x_1=1$: $\hat t_1=\frac{11}{14}$
    
- $x_2=2$: $\hat t_2=\frac{22}{14}=\frac{11}{7}$
    
- $x_3=3$: $\hat t_3=\frac{33}{14}$
    

Residuals $r_n=t_n-\hat t_n$:

$$  
r_1 = 1-\frac{11}{14}=\frac{3}{14}  
\quad\Rightarrow\quad  
r_1^2=\frac{9}{196}  
$$  
$$  
r_2 = 2-\frac{11}{7}=\frac{3}{7}  
\quad\Rightarrow\quad  
r_2^2=\frac{9}{49}  
$$  
$$  
r_3 = 2-\frac{33}{14}=-\frac{5}{14}  
\quad\Rightarrow\quad  
r_3^2=\frac{25}{196}  
$$

# Sum of squared errors:  
$$  
\text{SSE}=\frac{9}{196}+\frac{9}{49}+\frac{25}{196}

# \frac{9}{196}+\frac{36}{196}+\frac{25}{196}

# \frac{70}{196}

\frac{5}{14}  
\approx 0.357  
$$

### 3) (Optional) MLE for $\sigma^2$

If $\sigma^2$ is also unknown, the Gaussian MLE is:  
$$  
\sigma^2_{\text{ML}}=\frac{1}{N}\sum_{n=1}^N (t_n-w_{\text{ML}}x_n)^2  
=\frac{1}{N}\text{SSE}  
$$

So here:  
$$  
\sigma^2_{\text{ML}}=\frac{1}{3}\cdot \frac{5}{14}=\frac{5}{42}\approx 0.119  
$$

**Common confusion:** In statistics you often see $\frac{1}{N-1}$ for an _unbiased_ variance estimate; MLE uses $\frac{1}{N}$.

---

## In what context is this used?

This is the standard justification for the squared-error loss in **regression**:

- If you assume **Gaussian noise** around a deterministic prediction $y(x,w)$, then **maximum likelihood training** is equivalent to **least squares**.
    
- This underpins **linear regression**, polynomial regression, and many “regression head” neural nets trained with MSE.
    

If you tell me whether your course uses the vector/matrix form $y(\mathbf{x},\mathbf{w})=\mathbf{w}^\top\mathbf{x}$ (and whether bias is included), I can do the same derivation ending in:  
$$  
\mathbf{w}_{\text{ML}}=(X^\top X)^{-1}X^\top \mathbf{t}  
$$  
with a tiny 2D numeric worked example.

---

	Here’s the same pipeline, written tightly in PRML terms:

1. **Data**  
    $$  
    \mathcal{D}={(\mathbf{x}_n,t_n)}_{n=1}^N  
    $$
    
2. **Deterministic model (mean function)**  
    $$  
    y(\mathbf{x},\mathbf{w})=\mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x})  
    $$
    
3. **Noise model (where probability enters)**  
    Assume:  
    $$  
    t_n = y(\mathbf{x}_n,\mathbf{w})+\epsilon_n,\quad \epsilon_n\sim\mathcal{N}(0,\sigma^2)  
    $$  
    So:  
    $$  
    p(t_n\mid \mathbf{x}_n,\mathbf{w})=\mathcal{N}!\left(t_n\mid y(\mathbf{x}_n,\mathbf{w}),\sigma^2\right)  
    $$
    
4. **Likelihood (product) + log rules (sum)**  
    $$  
    p(\mathbf{t}\mid\mathbf{w})=\prod_{n=1}^N p(t_n\mid \mathbf{x}_n,\mathbf{w})  
    \quad\Rightarrow\quad  
    \ln p(\mathbf{t}\mid\mathbf{w})=\sum_{n=1}^N \ln p(t_n\mid \mathbf{x}_n,\mathbf{w})  
    $$
    
5. **Negative log-likelihood (this becomes the loss)**  
    For Gaussian:  
    $$  
    -\ln p(\mathbf{t}\mid\mathbf{w})  
    =  
    \text{const}+\frac{1}{2\sigma^2}\sum_{n=1}^N\left(t_n-y(\mathbf{x}_n,\mathbf{w})\right)^2  
    $$
    

So (dropping constants/scaling):  
$$  
\mathcal{L}(\mathbf{w}) \propto \sum_{n=1}^N\left(t_n-\mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x}_n)\right)^2  
$$

6. **Optimize**  
    Either:
    

- set gradient to zero (closed form for linear-in-parameters), or
    
- do gradient descent:  
    $$  
    \mathbf{w}\leftarrow \mathbf{w}-\eta\nabla \mathcal{L}(\mathbf{w})  
    $$
    

**Common confusion (one line):** In this pipeline, the “loss” in step 3 is _ultimately_ the **negative log-likelihood** from step 5; squared error appears because Gaussian noise makes the NLL proportional to SSE.

If you want, I can write the exact PRML “error function” $E(\mathbf{w})=\tfrac12\sum(\cdot)^2$ and show the precise constant relationship:  
$$  
-\ln p(\mathbf{t}\mid\mathbf{w})=\frac{N}{2}\ln(2\pi\sigma^2)+\frac{1}{\sigma^2}E(\mathbf{w})  
$$  
so you can see why minimizing $E(\mathbf{w})$ is equivalent to MLE.
