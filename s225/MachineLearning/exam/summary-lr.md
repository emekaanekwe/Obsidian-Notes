# complete summary of steps for LR

### Designing the structure

##### have training Data

*You have $N$ training pairs:*  
$$  
\mathcal{D}={(\mathbf{x}_n,t_n)}_{n=1}^N  
$$

##### make linear in parameters using basis functions

*Pick $M$ basis functions $\phi_1,\dots,\phi_M$ and define the feature vector:*  
$$  
\boldsymbol{\phi}(\mathbf{x})=  
\begin{bmatrix}  
\phi_1(\mathbf{x})\  
\vdots\  
\phi_M(\mathbf{x})  
\end{bmatrix}\in\mathbb{R}^{M\times 1}  
$$

*Weights:*  
$$  
\mathbf{w}\in\mathbb{R}^{M\times 1}  
$$

*Prediction for one point:*  
$$  
y(\mathbf{x},\mathbf{w})=\mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x})  
$$

##### Design matrix

*Stack all feature row vectors into a matrix:*  
$$  
\Phi=  
\begin{bmatrix}  
\boldsymbol{\phi}(\mathbf{x}_1)^\top\  
\vdots\  
\boldsymbol{\phi}(\mathbf{x}_N)^\top  
\end{bmatrix}  
\in\mathbb{R}^{N\times M}  
$$

*Targets:*  
$$  
\mathbf{t}=  
\begin{bmatrix}  
t_1\ \vdots\ t_N  
\end{bmatrix}\in\mathbb{R}^{N\times 1}  
$$

***FINALLY**, get predictions for all points at once:*  
$$  
\mathbf{y}=  
\begin{bmatrix}  
y(\mathbf{x}_1,\mathbf{w})\ \vdots\ y(\mathbf{x}_N,\mathbf{w})  
\end{bmatrix}  
=\Phi\mathbf{w}  
\qquad (N\times 1)  
$$

### Analyzing the Prediction (squared error, normal eq, closed form sol)

#####  Squared error (sum-of-squares):  
$$  
E(\mathbf{w})=\frac{1}{2}\sum_{n=1}^N\left(t_n-\mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x}_n)\right)^2  
$$

##### Squared Error in Matrix Form  
$$  
E(\mathbf{w})=\frac{1}{2}\lVert \mathbf{t}-\Phi\mathbf{w}\rVert^2  
=\frac{1}{2}(\mathbf{t}-\Phi\mathbf{w})^\top(\mathbf{t}-\Phi\mathbf{w})  
$$

##### Key derivative identity (the one you need to remember)

*If*  
$$  
E(\mathbf{w})=\frac{1}{2}\lVert \mathbf{t}-\Phi\mathbf{w}\rVert^2,  
$$  
*then*  
$$  
\nabla E(\mathbf{w})=\Phi^\top(\Phi\mathbf{w}-\mathbf{t})  
= -\Phi^\top(\mathbf{t}-\Phi\mathbf{w})  
$$

*Set gradient to zero for the optimum:*  
$$  
\nabla E(\mathbf{w})=\mathbf{0}  
\quad\Rightarrow\quad  
\Phi^\top(\Phi\mathbf{w}-\mathbf{t})=\mathbf{0}  
$$

*Rearrange:*  
$$  
\Phi^\top\Phi,\mathbf{w}=\Phi^\top\mathbf{t}  
$$

##### Establish the normal equations.

*If $\Phi^\top\Phi$ is invertible:*  
$$  
\mathbf{w}^*=(\Phi^\top\Phi)^{-1}\Phi^\top\mathbf{t}  
$$

**Common confusion:** If $\Phi^\top\Phi$ is not invertible (collinearity / too many features), you use a pseudoinverse or add ridge regularization (next step).

---

***Tiny hand-worked example (so this feels real)***

Let’s model $y=mx+b$ using basis  
$$  
\boldsymbol{\phi}(x)=\begin{bmatrix}x\ 1\end{bmatrix},\quad  
\mathbf{w}=\begin{bmatrix}m\ b\end{bmatrix}  
$$

Data:  
$$  
(x_1,t_1)=(1,2),\quad (x_2,t_2)=(2,2)  
$$

Build $\Phi$:  
$$  
\Phi=  
\begin{bmatrix}  
1 & 1\\  
2 & 1  
\end{bmatrix},\quad  
\mathbf{t}=  
\begin{bmatrix}  
2\\  
2  
\end{bmatrix}  
$$

##### Compute:  
$$  
\Phi^\top\Phi=  
\begin{bmatrix}  
1 & 2\\  
1 & 1  
\end{bmatrix}  
\begin{bmatrix}  
1 & 1\\  
2 & 1  
\end{bmatrix}
=
\begin{bmatrix}  
5 & 3\\  
3 & 2  
\end{bmatrix}  
$$  
$$  
\Phi^\top\mathbf{t}=  
\begin{bmatrix}  
1 & 2\\  
1 & 1  
\end{bmatrix}  
\begin{bmatrix}  
2\\  
2  
\end{bmatrix}
=
\begin{bmatrix}  
6\\  
4  
\end{bmatrix}  
$$

 Solve $(\Phi^\top\Phi)\mathbf{w}=\Phi^\top\mathbf{t}$:  
$$  
\begin{bmatrix}  
5 & 3\\  
3 & 2  
\end{bmatrix}  
\begin{bmatrix}  
m\\ b  
\end{bmatrix}
=
\begin{bmatrix}  
6\\4  
\end{bmatrix}  
$$

This gives:  
$$  
m=0,\quad b=2  
$$

So the best-fitting line is $y=2$ (flat), which matches the data.

---

### Gradient descent version (iterative optimization)

##### Same objective:  
$$  
E(\mathbf{w})=\frac{1}{2}\lVert \mathbf{t}-\Phi\mathbf{w}\rVert^2  
$$

*Gradient:*  
$$  
\nabla E(\mathbf{w})=\Phi^\top(\Phi\mathbf{w}-\mathbf{t})  
$$

*Gradient descent update:*  
$$  
\mathbf{w}_{k+1}=\mathbf{w}_k-\eta,\Phi^\top(\Phi\mathbf{w}_k-\mathbf{t})  
$$

Equivalent (often more intuitive):  
$$  
\mathbf{w}_{k+1}=\mathbf{w}_k+\eta,\Phi^\top(\mathbf{t}-\Phi\mathbf{w}_k)  
$$

**Common confusion:** Normal equations give the _exact minimizer_ (when solvable). Gradient descent is what you use when you prefer iterative updates (large-scale, streaming, neural nets, etc.).

---

## Step A4: Regularization (ridge + lasso) and the Bayesian connection

### Ridge regression (L2)

#### Add an L2 penalty:  
$$  
E_{\text{ridge}}(\mathbf{w})

\frac{1}{2}\lVert \mathbf{t}-\Phi\mathbf{w}\rVert^2  
+\frac{\lambda}{2}\lVert \mathbf{w}\rVert_2^2  
$$

Gradient:  
$$  
\nabla E_{\text{ridge}}(\mathbf{w})=\Phi^\top(\Phi\mathbf{w}-\mathbf{t})+\lambda \mathbf{w}  
$$

Closed-form solution:  
$$  
(\Phi^\top\Phi+\lambda I)\mathbf{w}=\Phi^\top\mathbf{t}  
\quad\Rightarrow\quad  
\mathbf{w}=(\Phi^\top\Phi+\lambda I)^{-1}\Phi^\top\mathbf{t}  
$$

**Purpose:** stabilizes inversion, shrinks weights, improves generalization.

#### Bayesian/MAP meaning (this is where Bayes “fits”)

- Likelihood (Gaussian noise): $\beta = 1/\sigma^2$
    
- Prior on weights (Gaussian): $p(\mathbf{w}\mid \alpha)=\mathcal{N}(\mathbf{0},\alpha^{-1}I)$
    

Then MAP equals minimizing:  
$$  
\frac{\beta}{2}\lVert \mathbf{t}-\Phi\mathbf{w}\rVert^2+\frac{\alpha}{2}\lVert \mathbf{w}\rVert^2  
$$

Divide by $\beta$:  
$$  
\frac{1}{2}\lVert \mathbf{t}-\Phi\mathbf{w}\rVert^2+\frac{\alpha}{2\beta}\lVert \mathbf{w}\rVert^2  
$$

So:  
$$  
\lambda=\frac{\alpha}{\beta}  
$$

**Common confusion:** MLE = no prior (just fit data). MAP = prior + likelihood → looks like “loss + regularization”.

---



---

**What you should do next (still Step A, but the “finish line”)**

If you want Step A “complete,” the next micro-goals are:

1. Be able to build $\Phi$ from basis functions and write $\mathbf{y}=\Phi\mathbf{w}$ without hesitation.
    
2. Memorize/derive:  
    $$  
    \nabla\left(\frac{1}{2}\lVert \mathbf{t}-\Phi\mathbf{w}\rVert^2\right)=\Phi^\top(\Phi\mathbf{w}-\mathbf{t})  
    $$
    
3. Know the normal equations and ridge-modified normal equations.
    
4. Understand: **Gaussian likelihood ⇒ squared error**, and **Gaussian prior ⇒ L2 regularization**.


Cool — Step A is basically: **(i) write linear regression in matrix form, (ii) get the closed-form solution (normal equations), (iii) show the gradient + gradient descent, (iv) add regularization (ridge + lasso) and connect ridge to the Bayesian/MAP view.**

I’ll keep it PRML-style and be explicit about shapes so you don’t get lost.

---

# Step A1: Design-matrix view (do all data points at once)

## Data

You have $N$ training pairs:  
$$  
\mathcal{D}={(\mathbf{x}_n,t_n)}_{n=1}^N  
$$

## Basis-function model (linear in parameters)

Pick $M$ basis functions $\phi_1,\dots,\phi_M$ and define the feature vector:  
$$  
\boldsymbol{\phi}(\mathbf{x})=  
\begin{bmatrix}  
\phi_1(\mathbf{x})\  
\vdots\  
\phi_M(\mathbf{x})  
\end{bmatrix}\in\mathbb{R}^{M\times 1}  
$$

Weights:  
$$  
\mathbf{w}\in\mathbb{R}^{M\times 1}  
$$

Prediction for one point:  
$$  
y(\mathbf{x},\mathbf{w})=\mathbf{w}^\top \boldsymbol{\phi}(\mathbf{x})  
$$

### Design matrix

Stack all feature _row vectors_ into a matrix:  
$$  
\Phi=  
\begin{bmatrix}  
\boldsymbol{\phi}(\mathbf{x}_1)^\top\  
\vdots\  
\boldsymbol{\phi}(\mathbf{x}_N)^\top  
\end{bmatrix}  
\in\mathbb{R}^{N\times M}  
$$

Targets:  
$$  
\mathbf{t}=  
\begin{bmatrix}  
t_1\ \vdots\ t_N  
\end{bmatrix}\in\mathbb{R}^{N\times 1}  
$$

Predictions for all points at once:  
$$  
\mathbf{y}=  
\begin{bmatrix}  
y(\mathbf{x}_1,\mathbf{w})\ \vdots\ y(\mathbf{x}_N,\mathbf{w})  
\end{bmatrix}  
=\Phi\mathbf{w}  
\qquad (N\times 1)  
$$

**Common confusion:** $\boldsymbol{\phi}(\mathbf{x})$ is a _feature/basis vector_; in linear regression it’s not a neural-net “activation after $wx+b$”.

---

# Step A2: Error function and normal equations (closed-form solution)

PRML’s squared error (sum-of-squares):  
$$  
E(\mathbf{w})=\frac{1}{2}\sum_{n=1}^N\left(t_n-\mathbf{w}^\top\boldsymbol{\phi}(\mathbf{x}_n)\right)^2  
$$

In matrix form this becomes:  
$$  
E(\mathbf{w})=\frac{1}{2}\lVert \mathbf{t}-\Phi\mathbf{w}\rVert^2  
=\frac{1}{2}(\mathbf{t}-\Phi\mathbf{w})^\top(\mathbf{t}-\Phi\mathbf{w})  
$$

## Key derivative identity (the one you need to remember)

If  
$$  
E(\mathbf{w})=\frac{1}{2}\lVert \mathbf{t}-\Phi\mathbf{w}\rVert^2,  
$$  
then  
$$  
\nabla E(\mathbf{w})=\Phi^\top(\Phi\mathbf{w}-\mathbf{t})  
= -\Phi^\top(\mathbf{t}-\Phi\mathbf{w})  
$$

Set gradient to zero for the optimum:  
$$  
\nabla E(\mathbf{w})=\mathbf{0}  
\quad\Rightarrow\quad  
\Phi^\top(\Phi\mathbf{w}-\mathbf{t})=\mathbf{0}  
$$

Rearrange:  
$$  
\Phi^\top\Phi,\mathbf{w}=\Phi^\top\mathbf{t}  
$$

These are the **normal equations**.

If $\Phi^\top\Phi$ is invertible:  
$$  
\mathbf{w}^*=(\Phi^\top\Phi)^{-1}\Phi^\top\mathbf{t}  
$$

**Common confusion:** If $\Phi^\top\Phi$ is not invertible (collinearity / too many features), you use a pseudoinverse or add ridge regularization (next step).

---

## Tiny hand-worked example (so this feels real)

Let’s model $y=mx+b$ using basis  
$$  
\boldsymbol{\phi}(x)=\begin{bmatrix}x\1\end{bmatrix},\quad  
\mathbf{w}=\begin{bmatrix}m\b\end{bmatrix}  
$$

Data:  
$$  
(x_1,t_1)=(1,2),\quad (x_2,t_2)=(2,2)  
$$

Build $\Phi$:  
$$  
\Phi=  
\begin{bmatrix}  
1 & 1\  
2 & 1  
\end{bmatrix},\quad  
\mathbf{t}=  
\begin{bmatrix}  
2\  
2  
\end{bmatrix}  
$$

# Compute:  
$$  
\Phi^\top\Phi=  
\begin{bmatrix}  
1 & 2\  
1 & 1  
\end{bmatrix}  
\begin{bmatrix}  
1 & 1\  
2 & 1  
\end{bmatrix}

# \begin{bmatrix}  
5 & 3\  
3 & 2  
\end{bmatrix}  
$$  
$$  
\Phi^\top\mathbf{t}=  
\begin{bmatrix}  
1 & 2\  
1 & 1  
\end{bmatrix}  
\begin{bmatrix}  
2\  
2  
\end{bmatrix}

\begin{bmatrix}  
6\  
4  
\end{bmatrix}  
$$

# Solve $(\Phi^\top\Phi)\mathbf{w}=\Phi^\top\mathbf{t}$:  
$$  
\begin{bmatrix}  
5 & 3\  
3 & 2  
\end{bmatrix}  
\begin{bmatrix}  
m\ b  
\end{bmatrix}

\begin{bmatrix}  
6\4  
\end{bmatrix}  
$$

This gives:  
$$  
m=0,\quad b=2  
$$

So the best-fitting line is $y=2$ (flat), which matches the data.

---

# Step A3: Gradient descent version (iterative optimization)

Same objective:  
$$  
E(\mathbf{w})=\frac{1}{2}\lVert \mathbf{t}-\Phi\mathbf{w}\rVert^2  
$$

Gradient:  
$$  
\nabla E(\mathbf{w})=\Phi^\top(\Phi\mathbf{w}-\mathbf{t})  
$$

Gradient descent update:  
$$  
\mathbf{w}_{k+1}=\mathbf{w}_k-\eta,\Phi^\top(\Phi\mathbf{w}_k-\mathbf{t})  
$$

Equivalent (often more intuitive):  
$$  
\mathbf{w}_{k+1}=\mathbf{w}_k+\eta,\Phi^\top(\mathbf{t}-\Phi\mathbf{w}_k)  
$$

**Common confusion:** Normal equations give the _exact minimizer_ (when solvable). Gradient descent is what you use when you prefer iterative updates (large-scale, streaming, neural nets, etc.).

---

# Step A4: Regularization (ridge + lasso) and the Bayesian connection

## Ridge regression (L2)

# Add an L2 penalty:  
$$  
E_{\text{ridge}}(\mathbf{w})

\frac{1}{2}\lVert \mathbf{t}-\Phi\mathbf{w}\rVert^2  
+\frac{\lambda}{2}\lVert \mathbf{w}\rVert_2^2  
$$

Gradient:  
$$  
\nabla E_{\text{ridge}}(\mathbf{w})=\Phi^\top(\Phi\mathbf{w}-\mathbf{t})+\lambda \mathbf{w}  
$$

Closed-form solution:  
$$  
(\Phi^\top\Phi+\lambda I)\mathbf{w}=\Phi^\top\mathbf{t}  
\quad\Rightarrow\quad  
\mathbf{w}=(\Phi^\top\Phi+\lambda I)^{-1}\Phi^\top\mathbf{t}  
$$

**Purpose:** stabilizes inversion, shrinks weights, improves generalization.

**Another option is LASSO regression (L1) to set a lot of weights to 0**
*not differentiable at 0, so no closed form solution*
$$  
E_{\text{lasso}}(\mathbf{w})

\frac{1}{2}\lVert \mathbf{t}-\Phi\mathbf{w}\rVert^2  
+\lambda\lVert \mathbf{w}\rVert_1,  
\qquad  
\lVert \mathbf{w}\rVert_1=\sum_i |w_i|  
$$
	add the L1 penalty, 


**Common confusion:** Ridge shrinks weights smoothly; LASSO can set them to exactly zero.


### Bayesian/MAP meaning (this is where Bayes “fits”)

- Likelihood (Gaussian noise): $\beta = 1/\sigma^2$
    
- Prior on weights (Gaussian): $p(\mathbf{w}\mid \alpha)=\mathcal{N}(\mathbf{0},\alpha^{-1}I)$
    

Then MAP equals minimizing:  
$$  
\frac{\beta}{2}\lVert \mathbf{t}-\Phi\mathbf{w}\rVert^2+\frac{\alpha}{2}\lVert \mathbf{w}\rVert^2  
$$

Divide by $\beta$:  
$$  
\frac{1}{2}\lVert \mathbf{t}-\Phi\mathbf{w}\rVert^2+\frac{\alpha}{2\beta}\lVert \mathbf{w}\rVert^2  
$$

So:  
$$  
\lambda=\frac{\alpha}{\beta}  
$$

**Common confusion:** MLE = no prior (just fit data). MAP = prior + likelihood → looks like “loss + regularization”.

---

## LASSO regression (L1)

# Add an L1 penalty:  
$$  
E_{\text{lasso}}(\mathbf{w})

\frac{1}{2}\lVert \mathbf{t}-\Phi\mathbf{w}\rVert^2  
+\lambda\lVert \mathbf{w}\rVert_1,  
\qquad  
\lVert \mathbf{w}\rVert_1=\sum_i |w_i|  
$$

**Key difference:** no simple closed-form solution, because $|w_i|$ is not differentiable at $0$.

**Purpose:** encourages sparsity (many weights exactly 0).

**Common confusion:** Ridge shrinks weights smoothly; LASSO can set them to exactly zero.

---

# What you should do next (still Step A, but the “finish line”)

If you want Step A “complete,” the next micro-goals are:

1. Be able to build $\Phi$ from basis functions and write $\mathbf{y}=\Phi\mathbf{w}$ without hesitation.
    
2. Memorize/derive:  
    $$  
    \nabla\left(\frac{1}{2}\lVert \mathbf{t}-\Phi\mathbf{w}\rVert^2\right)=\Phi^\top(\Phi\mathbf{w}-\mathbf{t})  
    $$
    
3. Know the normal equations and ridge-modified normal equations.
    
4. Understand: **Gaussian likelihood ⇒ squared error**, and **Gaussian prior ⇒ L2 regularization**.
    