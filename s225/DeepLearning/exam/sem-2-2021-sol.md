
---

# PART A — Multiple Choice (12 questions)

## Q1 (Total parameters)

Layer 1: `Dense(20, input_shape=(10,))`

- Weights: $10 \times 20 = 200$
    
- Biases: $20$
    
- Total layer 1: $200 + 20 = 220$
    

Layer 2: `Dense(40)` with input size 20

- Weights: $20 \times 40 = 800$
    
- Biases: $40$
    
- Total layer 2: $800 + 40 = 840$
    

Total parameters:  
$$220 + 840 = 1060$$

✅ **Answer: a. 1060**

---

## Q2 (One gradient descent step)

Given $f(w) = 3w^2 - 4w + 10$.

Derivative:  
$$f'(w) = 6w - 4$$

At $w_t = 2$:  
$$\nabla f(w_t) = f'(2) = 6(2) - 4 = 8$$

Update with $\eta = 0.05$:  
$$w_{t+1} = w_t - \eta \nabla f(w_t) = 2 - 0.05(8) = 2 - 0.4 = 1.6$$

✅ **Answer: a. 1.6**

---

## Q3 (SGD update rule)

SGD uses a minibatch of size $b$ with indices $i_1,\dots,i_b$:

$$\theta_{t+1}=\theta_t-\frac{\eta}{b}\sum^{b}_{k=1}\nabla_{\theta}, l(x_{i_k},y_{i_k};\theta_t)$$

✅ **Answer: a**

---

## Q4 (Conv2D output shape, padding = same)

Input: $[64,64,3]$  
Filters: $15$  
Kernel: $[5,5,3]$  
Stride: $[3,3]$  
Padding: `same`

For `same` padding (spatial):  
$$H_{\text{out}} = \left\lceil \frac{H_{\text{in}}}{S} \right\rceil,\quad W_{\text{out}} = \left\lceil \frac{W_{\text{in}}}{S} \right\rceil$$

So:  
$$H_{\text{out}} = W_{\text{out}} = \left\lceil \frac{64}{3} \right\rceil = \lceil 21.33 \rceil = 22$$

Channels out = number of filters = $15$.

✅ **Answer: c. $[22,22,15]$**

---

## Q5 (Cross-entropy loss)

Classes: ${\text{cat}=1,\text{dog}=2,\text{lion}=3,\text{flower}=4,\text{cow}=5}$  
Prediction: $f(x) = [0.4,0.2,0.1,0.2,0.1]$  
True label: $\text{flower}$ so $p_y = 0.2$

Cross-entropy:  
$$\ell = -\log p_y = -\log(0.2)$$

✅ **Answer: b. $-\log 0.2$**

---

## Q6 (Softmax probability for lion)

Scores: $h_1=-3,;h_2=10,;h_3=5,;h_4=-1$  
Probability for lion (class 3):  
$$p(y=\text{lion}\mid x)=\frac{e^{h_3}}{e^{h_1}+e^{h_2}+e^{h_3}+e^{h_4}}  
=\frac{e^5}{e^{-3}+e^{10}+e^5+e^{-1}}$$

✅ **Answer: b**

---

## Q7 (CNN output tensor shape)

Input: $[64,32,32,10]$  
Conv2D: 20 filters, stride $[3,3]$, padding `same`

Spatial:  
$$H_{\text{out}} = W_{\text{out}} = \left\lceil \frac{32}{3} \right\rceil = \lceil 10.67 \rceil = 11$$

Channels out = $20$.

✅ **Answer: c. $[64,11,11,20]$**

---

## Q8 (Adversarial example interpretation)

Given:  
$$x_{\text{adv}}=\arg\max_{x'\in B_\epsilon(x)} \ \ell(f(x';\theta), y)$$

This **maximizes the loss wrt the true label** $y$:

- It **maximally decreases** the chance of predicting label $y$ correctly.
    
- It **maximally increases** the chance of predicting some label $y'\neq y$.
    
- No specific target label is chosen $\Rightarrow$ **untargeted**.
    

✅ Correct statements: **b, c, e**

---

## Q9 (Skip-gram shapes and correct statements)

Vocab size: $500$  
Embedding size: $150$  
Target index: $2$, context index: $8$

Weight matrices:

- Input $\to$ hidden embedding: $U\in\mathbb{R}^{500\times 150}$
    
- Hidden embedding $\to$ output vocab: $V\in\mathbb{R}^{150\times 500}$
    

Input is one-hot of target word: one-hot$(2)$.

Hidden vector $h$ equals row 2 of $U$ (because multiplying one-hot by $U$ picks that row).

✅ Correct: **b, d**

---

## Q10 (Denoising autoencoder objective)

Denoising AE: corrupt $x$ to $x'$ (noisy), encode then decode to reconstruct original $x$:

$$\underset{\theta, \phi}{\min}\ \mathbb{E}_{x\sim P}\Big[\mathbb{E}_{x' \sim \mathcal{N}(x,\eta I)}\big[d(x,\ g_{\phi}(f_{\theta}(x')))\big]\Big]$$

✅ **Answer: a**

---

## Q11 (GAN training objective)

Standard GAN:  
$$\min_G\max_D\ J(G,D)=\mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z\sim p(z)}[\log(1-D(G(z)))]$$

✅ **Answer: b**

---

## Q12 (Global attention context)

Weights: $a=[0.3,0.1,0.4,0.2]$ on encoder states $\bar h_1,\bar h_2,\bar h_3,\bar h_4$

Context:  
$$c_t = 0.3\bar h_1 + 0.1\bar h_2 + 0.4\bar h_3 + 0.2\bar h_4$$

Largest weight is $0.4$ (third word) so third word is most important.

✅ Correct: **c, e**

---

# PART B — Short Workout (8 questions)

## Q13 (Forward pass + softmax + CE loss)

Given input:  
$$x=\begin{bmatrix}-1\1\end{bmatrix},\quad y=2$$

Layer 1:  
$$W^{(1)}=\begin{bmatrix}1&0\0&1\-1&0\end{bmatrix},\quad  
b^{(1)}=\begin{bmatrix}-2\0\1\end{bmatrix}$$

### 13a) Compute $\tilde h^{(1)}$ and $h^{(1)}$

# Pre-activation:  
$$\tilde h^{(1)} = W^{(1)}x + b^{(1)}

# \begin{bmatrix}1&0\0&1\-1&0\end{bmatrix}  
\begin{bmatrix}-1\1\end{bmatrix}  
+  
\begin{bmatrix}-2\0\1\end{bmatrix}

# \begin{bmatrix}-1\1\1\end{bmatrix}  
+  
\begin{bmatrix}-2\0\1\end{bmatrix}

\begin{bmatrix}-3\1\2\end{bmatrix}$$

# ReLU:  
$$h^{(1)}=\text{ReLU}(\tilde h^{(1)})=  
\begin{bmatrix}\max(0,-3)\\max(0,1)\\max(0,2)\end{bmatrix}

\begin{bmatrix}0\1\2\end{bmatrix}$$

---

### 13b) Compute logits $h^{(2)}$ and probabilities $p$

Using:  
$$W^{(2)}=  
\begin{bmatrix}  
1&0&0\  
-1&1&1  
\end{bmatrix},\quad  
b^{(2)}=  
\begin{bmatrix}0\1\end{bmatrix}$$

# Logits:  
$$h^{(2)} = W^{(2)}h^{(1)} + b^{(2)}

# \begin{bmatrix}  
1&0&0\  
-1&1&1  
\end{bmatrix}  
\begin{bmatrix}0\1\2\end{bmatrix}  
+  
\begin{bmatrix}0\1\end{bmatrix}

# \begin{bmatrix}  
0\  
0+1+2  
\end{bmatrix}  
+  
\begin{bmatrix}0\1\end{bmatrix}

\begin{bmatrix}0\4\end{bmatrix}$$

Softmax:  
$$p=\text{softmax}(h^{(2)})=  
\left[  
\frac{e^0}{e^0+e^4},  
\frac{e^4}{e^0+e^4}  
\right]  
\approx [0.0180,\ 0.9820]$$

---

### 13c) Predicted label and cross-entropy

Predicted label:  
$$\hat y = \arg\max_k p_k = 2$$

Cross-entropy for $y=2$:  
$$\ell = -\log(p_2) = -\log(0.9820)\approx 0.0182$$

Prediction is correct because $\hat y = y$.

---

## Q14 (Conv → Pool → Flatten neurons)

Start: $[32,64,64,10]$

### Step 1: Conv2D (valid, kernel $5$, stride $3$, filters $10$)

$$H'=\left\lfloor\frac{H-K}{S}\right\rfloor+1=  
\left\lfloor\frac{64-5}{3}\right\rfloor+1=\lfloor 19.67\rfloor+1=20$$

So:  
$$[32,64,64,10]\to [32,20,20,10]$$

### Step 2: MaxPool2D (same, kernel $2$, stride $3$)

Same pooling spatial:  
$$H''=\left\lceil\frac{20}{3}\right\rceil=7$$

So:  
$$[32,20,20,10]\to [32,7,7,10]$$

### Step 3: Flatten neurons

Neurons per example:  
$$7\times 7\times 10=490$$

✅ **Answer: $490$ neurons**

---

## Q15 (Embedding numeric tensor)

Given embedding rows:

- $U_1=[-1,2]$
    
- $U_2=[1,-1]$
    
- $U_3=[-1,-2]$
    
- $U_4=[-1,3.5]$
    
- $U_5=[1,-2.5]$
    
- $U_6=[-1.5,-0.5]$
    
- $U_7=[-4,2]$
    
- $U_8=[1,-3]$
    

Batch sentences (padded length $T=5$):

- Seq1: $\text{This, movie, is, fantastic, pad} \Rightarrow [8,7,8,4,8]$
    
- Seq2: $\text{I, really, love, this, movie} \Rightarrow [7,8,2,7,7]$
    

### 15a) Numeric inputs per timestep

At each timestep $t$:

$t=1$: $[U_8 \mid U_7] = [[1,-3]\mid [-4,2]]$  
$t=2$: $[U_7 \mid U_8] = [[-4,2]\mid [1,-3]]$  
$t=3$: $[U_8 \mid U_2] = [[1,-3]\mid [1,-1]]$  
$t=4$: $[U_4 \mid U_7] = [[-1,3.5]\mid [-4,2]]$  
$t=5$: $[U_8 \mid U_7] = [[1,-3]\mid [-4,2]]$

### 15b) 3D embedding tensor $[B,T,E]=[2,5,2]$

$$  
X=  
\begin{bmatrix}  
[[1,-3],[-4,2],[1,-3],[-1,3.5],[1,-3]]\  
[[-4,2],[1,-3],[1,-1],[-4,2],[-4,2]]  
\end{bmatrix}  
$$

---

## Q16 (RNN tensor shapes)

Given:

```python
x = Input(shape=[5])
h1 = Embedding(vocab_size, 64)(x)
h2 = GRU(8, return_sequences=True)(h1)
h3 = GRU(8, return_sequences=True)(h2)
h4 = GRU(16, return_sequences=True)(h3)
h5 = Flatten()(h4)
h6 = Dense(100, softmax)(h5)
```

Shapes (batch is `None`):

- $x$: $[None,5]$
    
- $h_1$: $[None,5,64]$
    
- $h_2$: $[None,5,8]$
    
- $h_3$: $[None,5,8]$
    
- $h_4$: $[None,5,16]$
    
- $h_5$: $[None,5\cdot 16]=[None,80]$
    
- $h_6$: $[None,100]$
    

---

## Q17 (CNN shapes)

Input:  
$$x:[None,32,32,3]$$

Conv SAME, $k=3$, $s=1$, filters $10$:  
$$h_1:[None,32,32,10]$$

MaxPool VALID, $k=2$, $s=2$:  
$$h_2:[None,16,16,10]$$

Conv VALID, $k=3$, $s=1$, filters $20$:  
$$H=16-3+1=14\Rightarrow h_3:[None,14,14,20]$$

MaxPool SAME, $k=2$, $s=2$:  
$$H=\left\lceil\frac{14}{2}\right\rceil=7\Rightarrow h_4:[None,7,7,20]$$

Flatten:  
$$h_5:[None,7\cdot 7\cdot 20]=[None,980]$$

Dense(10):  
$$p:[None,10]$$

---

## Q18 (Pooling outputs)

Input is $6\times 6$, kernel $2\times 2$, stride $2$, valid $\Rightarrow 3\times 3$ output.

### 18a) Max pooling output

$$  
\text{MaxPool}=  
\begin{bmatrix}  
1 & 4 & 3\  
3 & 6 & 1\  
2 & 1 & 2  
\end{bmatrix}  
$$

### 18b) Average pooling output

$$  
\text{AvgPool}=  
\begin{bmatrix}  
-0.75 & 0.25 & 0.75\  
0 & -0.25 & -0.5\  
-0.25 & 0.25 & 0.25  
\end{bmatrix}  
$$

---

## Q19 (Dot-product attention formulas)

Alignment scores:  
$$e_{t,s}=h_t^\top \bar h_s,\quad s\in{1,2,3}$$

Alignment weights:  
$$a_{t,s}=\frac{\exp(e_{t,s})}{\sum_{j=1}^3 \exp(e_{t,j})},\quad s\in{1,2,3}$$

---

## Q20 (Sign-score attention numeric)

Given:  
$$\bar h_1=1,\quad \bar h_2=-1,\quad \bar h_3=2,\quad h_t=1$$

Score:  
$$e_{t,s}=\text{sign}(h_t\cdot \bar h_s)$$

So:  
$$e_{t,1}=\text{sign}(1\cdot 1)=1,\quad e_{t,2}=\text{sign}(1\cdot -1)=-1,\quad e_{t,3}=\text{sign}(1\cdot 2)=1$$

Thus $e=[1,-1,1]$. Softmax:  
$$a_{t,1}=\frac{e^1}{e^1+e^{-1}+e^1},\quad a_{t,2}=\frac{e^{-1}}e^1+e^{-1}+e^1,\quad a_{t,3}=\frac{e^1}{e^1+e^{-1}+e^1}$$

Context:  
$$c_t=a_{t,1}\bar h_1+a_{t,2}\bar h_2+a_{t,3}\bar h_3$$

---

# PART C — Written Answer

## Q21 Word2Vec

### 21a Purpose

Word2Vec learns dense vectors $v_w\in\mathbb{R}^d$ such that semantically similar words have embeddings close under a metric like cosine similarity.

### 21b Skip-gram task + drawback

Skip-gram predicts context words given a target word:  
$$\max_\theta \sum_t \sum_{j=-k,\ j\neq 0}^{k} \log P(w_{t+j}\mid w_t)$$

Drawbacks: large softmax cost over vocab, ignores order within window, and basic models struggle with polysemy.

### 21c Negative sampling

Objective for one positive pair $(w_t,w_c)$ with $K$ negatives $w_{n_1},\dots,w_{n_K}$:  
$$\max\ \log\sigma(u_{w_c}^\top v_{w_t})+\sum_{i=1}^{K}\log\sigma(-u_{w_{n_i}}^\top v_{w_t})$$  
with $\sigma(z)=\frac{1}{1+e^{-z}}$.

---

## Q22 Overfitting + early stopping

Overfitting occurs when training loss keeps decreasing while validation loss stops decreasing and begins increasing.

From the plot: validation loss reaches minimum around epoch $\approx 12.5$ then increases.

Early stopping: stop around epoch $\approx 12$–$13$.

---

## Q23 Roles of $D$ and $G$

Generator: $x_{\text{fake}}=G(z)$ with $z\sim p(z)$.  
Discriminator: outputs probability of being real, $D(x)\in[0,1]$.

---

## Q24 Optimization problems

GAN objective:  
$$\min_G\max_D\ \mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z\sim p(z)}[\log(1-D(G(z)))]$$

Common generator alternative:  
$$\max_G\ \mathbb{E}_{z\sim p(z)}[\log D(G(z))]$$

---

## Q25 Why $D^*(x)=0.5$

Optimal discriminator:  
$$D^*(x)=\frac{p_{\text{data}}(x)}{p_{\text{data}}(x)+p_g(x)}$$

If $p_g(x)=p_{\text{data}}(x)$ then:  
$$D^*(x)=\frac{p_{\text{data}}(x)}{2p_{\text{data}}(x)}=\frac{1}{2}=0.5$$

---

## Q26 CNN spatial relationship issue

Pooling/stride increases invariance and reduces precise spatial arrangement. CNN can detect parts (eyes/nose/mouth) but not enforce correct relative geometry, so it can misclassify “scrambled” parts as a face.

---

## Q27 Underfitting

Underfitting: high bias, poor fit even on training data.  
Example: too few hidden units, too strong regularization, or too few epochs.

---

## Q28 Overfitting

Overfitting: training error low but validation/test error high; model memorizes noise.

---

## Q29 Example of overfitting

Large CNN trained on small dataset without augmentation; training accuracy $\to 1$ but validation accuracy stagnates or decreases.

---

## Q30 Two solutions

1. Weight decay (L2 regularization): adds penalty:  
    $$\min_\theta\ \mathcal{L}(\theta)+\lambda|\theta|_2^2$$
    
2. Dropout: randomly disables units during training, reducing co-adaptation and variance.
    

---

If you want, I can now reformat this into **flashcards** where each card has: **Prompt → Step-by-step solution → Final answer** (still with full MathJax).