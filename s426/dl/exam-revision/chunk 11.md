

## tags: [FIT5215, deep-learning, ViT, fine-tuning, LoRA, adapters, GAN, generative-models] pages: 201-220

# FIT5215 — Deep Learning: Pages 201–220

## 1. ViT Fine-Tuning Strategies

### 1.1 Pretrained ViT Overview (p. 202)

Pretrained **ViT/16** models are typically trained on very large datasets:

- **ImageNet-21K** (~21,000 classes)
- **JFT-300M** (~300 million images, Google's internal dataset)

These large-scale pretrained models are then adapted to smaller, downstream/target datasets via **fine-tuning**.

### 1.2 Full Fine-Tuning (p. 203)

**Approach:** replace the class token and MLP head with **new** versions suited to the new dataset, then fine-tune (update) the relevant parameters using the new data.

- **New class token** — a fresh learnable classification token
- **New MLP Head** — a fresh classification head sized for the new dataset's number of classes
- The rest of the pretrained Transformer Encoder can be fine-tuned (fully or partially) on the new dataset

> This is the most straightforward fine-tuning strategy: swap in task-specific heads and continue training all (or most) parameters — but it is also the most compute- and memory-intensive, since it updates the full model.

### 1.3 General Principle: Fine-Tuning with Additional Components (p. 204)

**Core idea:** rather than updating **all** parameters of a large pretrained ViT, insert some **additional, small, learnable components** into the frozen pretrained model that "favour" (adapt) its computations, and fine-tune **only these new components** on the new dataset.

**Three common parameter-efficient fine-tuning (PEFT) strategies covered:**

1. **Prompts** (prompt tuning)
2. **Adapter** modules
3. **LoRA** (Low-Rank Adaptation)

> **Motivation:** large pretrained models (like ViT) are expensive to fully fine-tune (in compute, memory, and storage — a full fine-tuned copy per downstream task). PEFT methods freeze the bulk of the pretrained weights and add a much smaller set of trainable parameters, drastically reducing the cost of adapting to new tasks while retaining most of the pretrained knowledge.

### 1.4 Fine-Tuning with Prompts (p. 205)

**Approach:** insert a set of **learnable prompt tokens** alongside the patch tokens (and class token) as additional input to the Transformer Encoder.

- The **backbone Transformer Encoder remains frozen** (pretrained weights unchanged)
- Only the **learnable prompts** (and typically the new class token / MLP head) are updated during fine-tuning
- These prompt tokens participate in the self-attention computation alongside the real patch tokens, allowing them to influence (and be influenced by) the image representation without changing any of the pretrained weights

```python
import torch
import torch.nn as nn

class PromptTunedViT(nn.Module):
    def __init__(self, pretrained_vit, num_prompts, d_model, num_classes):
        super().__init__()
        self.vit = pretrained_vit
        for p in self.vit.parameters():
            p.requires_grad = False              # freeze pretrained backbone

        self.learnable_prompts = nn.Parameter(torch.randn(1, num_prompts, d_model))  # trainable
        self.new_mlp_head = nn.Linear(d_model, num_classes)  # trainable

    def forward(self, patch_embeddings):          # patch_embeddings: (batch, num_patches+1, d_model), incl. class token
        batch_size = patch_embeddings.size(0)
        prompts = self.learnable_prompts.expand(batch_size, -1, -1)
        x = torch.cat([patch_embeddings, prompts], dim=1)   # append learnable prompts to the sequence
        encoded = self.vit.encoder(x)                        # frozen encoder processes everything together
        cls_output = encoded[:, 0]                            # class token's output
        return self.new_mlp_head(cls_output)
```

### 1.5 Fine-Tuning with Adapters (p. 206)

**Approach:** insert small **bottleneck adapter modules** in parallel with (or after) the existing (frozen) point-wise FFN inside each encoder block.

**Fixed point-wise FFN** (pretrained, frozen):

$$g(X) = \sigma(XW_1)W_2$$

**Learnable adapter** (small bottleneck: down-project then up-project):

$$h(X) = \sigma(XW_{down})W_{up}$$

**Combined output:**

$$f(X) = g(X) + \sigma(XW_{down})W_{up}$$

- $W_{down}$ projects down to a small bottleneck dimension, $W_{up}$ projects back up — keeping the adapter's parameter count small
- Only $W_{down}, W_{up}$ (plus typically the new class token / MLP head) are trained; $W_1, W_2$ (the original FFN) remain **fixed**

```python
import torch
import torch.nn as nn

class AdapterFFN(nn.Module):
    def __init__(self, fixed_ffn, d_model, bottleneck_dim):
        super().__init__()
        self.fixed_ffn = fixed_ffn                 # g(X) = sigma(X W1) W2, frozen
        for p in self.fixed_ffn.parameters():
            p.requires_grad = False

        self.w_down = nn.Linear(d_model, bottleneck_dim)   # trainable
        self.w_up = nn.Linear(bottleneck_dim, d_model)     # trainable
        self.activation = nn.ReLU()

    def forward(self, X):
        g_X = self.fixed_ffn(X)                                  # frozen FFN path
        h_X = self.w_up(self.activation(self.w_down(X)))          # learnable adapter path
        return g_X + h_X                                          # f(X) = g(X) + adapter(X)
```

### 1.6 Fine-Tuning with LoRA — Low-Rank Adaptation (p. 207)

**Approach:** instead of fine-tuning the full query/key/value projection matrices $W_Q, W_K, W_V$, add a **low-rank update** to each:

$$W_Q + B_Q \times A_Q, \qquad W_K + B_K \times A_K, \qquad W_V + B_V \times A_V$$

where:

- $W_Q, W_K, W_V \in \mathbb{R}^{[m,n]}$ — the original (frozen) pretrained projection matrices
- $B \in \mathbb{R}^{[m,r]}$, $A \in \mathbb{R}^{[r,n]}$ — the new, **learnable, low-rank** matrices
- $r \ll m, n$ — the rank $r$ is chosen to be much smaller than $m, n$, so $B\times A$ has far fewer parameters than a full $[m,n]$ update matrix would

> **Why this works:** the hypothesis behind LoRA is that the _change_ needed in the weight matrices during fine-tuning has a low "intrinsic rank" — i.e., it can be well-approximated by a low-rank matrix $B\times A$, even though the original weight matrix itself is full-rank. This lets LoRA achieve fine-tuning quality comparable to full fine-tuning while training orders of magnitude fewer parameters (only $B$ and $A$, not the full $W$).

```python
import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, r, pretrained_weight):
        super().__init__()
        self.W = nn.Parameter(pretrained_weight, requires_grad=False)  # frozen pretrained W: [m, n]
        self.A = nn.Parameter(torch.randn(r, out_features) * 0.01)      # trainable: [r, n]
        self.B = nn.Parameter(torch.zeros(in_features, r))               # trainable: [m, r], init to zero

    def forward(self, X):
        return X @ (self.W + self.B @ self.A)    # W + B x A, with B x A low-rank (rank r << m, n)

# Applied to Q, K, V projections within each attention block:
# WQ_lora = LoRALinear(d_model, d_model, r=8, pretrained_weight=original_WQ)
# WK_lora = LoRALinear(d_model, d_model, r=8, pretrained_weight=original_WK)
# WV_lora = LoRALinear(d_model, d_model, r=8, pretrained_weight=original_WV)
```

> **PEFT comparison — Prompts vs. Adapters vs. LoRA:**
> 
> - **Prompts**: adds trainable _tokens_, backbone weights fully frozen and unmodified
> - **Adapters**: adds trainable _bottleneck layers_ inside each block, in parallel with existing frozen layers
> - **LoRA**: adds trainable _low-rank weight updates_ directly to existing weight matrices (particularly attention's Q/K/V), without adding new layers or tokens All three share the same underlying principle from Section 1.3: freeze the large pretrained backbone, add a small number of new trainable parameters, and fine-tune only those.

---

## 2. Generative Adversarial Networks (GAN)

### 2.1 General Formulation for Deep Generative Models (DGM) (p. 209)

**Setup:** given a training set $D = {x_1, x_2, \dots, x_N}$ where each $x_i \sim p_d(x)$ — the **true data distribution** $p_d(x)$ **exists but is unknown**.

**Goal:** learn a **generator** $G$ mapping from a 'noise'/**latent space** $\mathcal{Z}$ to the data space:

$$z \sim p(z) \quad \to \quad \tilde{x} = G(z) \sim p_d(x)$$

- $\tilde{x} = G(z)$ should look **'similar'** to samples in $D$
- $G$ is parameterized by a **deep neural network**
- Once trained, use $G$ to **generate novel and new data samples** by sampling fresh $z$ values

### 2.2 Overview of Current DGM Methods (p. 210)

|Method|Core mechanism|
|---|---|
|**GAN**|Adversarial training — a discriminator $D(x)$ competes against a generator $G(z)$|
|**VAE**|Maximize a **variational lower bound** — encoder $q_\phi(z\mid x)$, decoder $p_\theta(x\mid z)$|
|**Flow-based models**|**Invertible transform** of distributions — flow $f(x) \to z$, inverse $f^{-1}(z) \to x'$|
|**Diffusion models**|Gradually **add Gaussian noise** ($x_0 \to x_1 \to x_2 \to \dots \to z$) and then **reverse** the process to generate|

_(Source: [lilianweng.github.io/posts/2021-07-11-diffusion-models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/))_

---

## 3. GAN: The Minimax Game Analogy (p. 212)

**Analogy: counterfeiter vs. police officer**

|Player|Task|Goal|
|---|---|---|
|**Generator** (counterfeiter)|Produce counterfeit money|Try to **fool/confuse** the police officer|
|**Discriminator** (police officer)|Distinguish real from counterfeit money|Try to **sabotage** the counterfeit maker|

- The **final output from GANs** is the trained **Generator** — once training is complete, the discriminator has served its purpose (as a training signal) and is typically discarded
- The relationship is **adversarial** during training but can be seen as jointly **"cooperative"** in the sense that both networks improve together, pushing each other toward better performance
- Training **terminates** at a **Nash equilibrium point**: neither player can improve any further given the other's fixed strategy

---

## 4. GAN Formulation (p. 213)

**Formal setup:**

1. Introduce a **noise variable** $z \sim p_z$ — the **prior distribution** (analogous to the "raw resources" used to make counterfeit money)
    
2. A **generator** $G$ that maps $z$ to $\tilde{x}$ in data space:
    

$$\tilde{x} = G(z)$$

- Parameterized by a deep neural network with parameters $\theta_g$

3. A **discriminator** $D(x)$ maps $x$ to a probability in $[0,1]$ representing the probability that $x$ comes from the **real data** rather than from $G$:
    - Parameterized by another deep neural network with parameters $\theta_d$
    - $D(x)$ close to **1** → judged **real**
    - $D(x)$ close to **0** → judged **fake**

**The GAN minimax objective:**

$$\min_G \max_D J(G,D) = \mathbb{E}_{x\sim p_d(x)}\big[\log D(x)\big] + \mathbb{E}_{z\sim p_z}\big[\log(1-D(G(z)))\big]$$

_(Figure credit: Goodfellow, NeurIPS workshop 2016)_

---

## 5. Training GAN

### 5.1 Training the Generator $G$ (p. 214)

**Goal:** minimize or **fool** the discriminator $D$ — i.e., minimize $(1 - D(G(z)))$ toward 0 (making the discriminator think generated samples are real).

**Gradient descent update for $G_{\theta_g}$:**

$$\min_{\theta_g} \ \mathbb{E}_{z\sim p_z}\big[\log(1-D(G(z)))\big]$$

**Mini-batch descent update** (sample $z^{(i)} \sim p_z$ for $i=1,\dots,M$):

$$-\nabla_{\theta_g} \frac{1}{M}\sum_{i=1}^{M}\Big[\log\big(1-D(G(z^{(i)}))\big)\Big]$$

### 5.2 Training the Discriminator $D$ (p. 215)

**Goal:** maximize the probability of **detecting correct labels** (correctly classify real vs. fake).

**Gradient ascent update for $D_{\theta_d}$:**

$$\max_{\theta_d} \ \mathbb{E}_{x\sim p_{data}(x)}\big[\log D_{\theta_d}(x)\big] + \mathbb{E}_{z\sim p_z}\big[\log(1-D_{\theta_d}(G(z)))\big]$$

**Mini-batch ascent update** (sample $z^{(i)}\sim p_z$, $x^{(i)}\sim p_{data}$):

$$\nabla_{\theta_d} \frac{1}{M}\sum_{i=1}^{M}\Big[\log D(x^{(i)}) + \log\big(1-D(G(z^{(i)}))\big)\Big]$$

### 5.3 Combined Training Summary (p. 216)

Both updates together implement the minimax objective:

$$\min_G \max_D J(G,D) = \mathbb{E}_{x\sim p_{data}(x)}\big[\log D(x)\big] + \mathbb{E}_{z\sim p_z}\big[\log(1-D(G(z)))\big]$$

**Full training loop (standard GAN alternating updates):**

```python
import torch
import torch.nn as nn
import torch.optim as optim

def train_gan_step(G, D, optimizer_G, optimizer_D, real_batch, latent_dim, M):
    # --- 1. Ascent update for D: maximize log D(x) + log(1 - D(G(z))) ---
    optimizer_D.zero_grad()
    z = torch.randn(M, latent_dim)
    fake_batch = G(z).detach()                       # detach so gradient doesn't flow into G here
    loss_D = -(torch.log(D(real_batch) + 1e-10).mean()
               + torch.log(1 - D(fake_batch) + 1e-10).mean())
    loss_D.backward()
    optimizer_D.step()

    # --- 2. Descent update for G: minimize log(1 - D(G(z))) ---
    optimizer_G.zero_grad()
    z = torch.randn(M, latent_dim)
    fake_batch = G(z)
    loss_G = torch.log(1 - D(fake_batch) + 1e-10).mean()
    loss_G.backward()
    optimizer_G.step()

    return loss_D.item(), loss_G.item()
```

> **Practical note:** in practice, the generator loss $\log(1-D(G(z)))$ is often replaced with $-\log(D(G(z)))$ (the "non-saturating" generator loss) because the original formulation provides very weak gradients early in training when $D$ easily rejects poor generated samples — but the slides here present the original/theoretical minimax formulation.

---

## 6. Basic Theory of GAN (p. 217–218)

### 6.1 Reframing as a Classification Problem

**Goal:** train a NN-based function $G_\theta(\cdot)$ such that $p_g(x) = p_d(x)$ (generator's distribution matches the true data distribution) — but we **don't have $p_d(x)$ directly available**.

**Solution:** introduce a strong discriminator $D(x)$ to **implicitly quantify** how far $p_d(x)$ and $p_g(x)$ are from each other.

**Reframing as binary classification:**

- $D(x) = P(y=1\mid x)$: probability $x$ is true data
- $1-D(x) = P(y=-1\mid x)$: probability $x$ is fake data

**Log-likelihood function to maximize:**

$$J(D,G) = \mathbb{E}_{x\sim P_d}\big[\log D(x)\big] + \mathbb{E}_{x\sim P_g}\big[\log(1-D(x))\big]$$

$$= \mathbb{E}_{x\sim P_d}\big[\log D(x)\big] + \mathbb{E}_{z\sim P_z}\big[\log(1-D(G(z)))\big]$$

where $z \sim p_z = N(0,I)$ or $\text{Uni}([0,1]^d)$, and $x_{fake} = G_\theta(z) \sim p_g(x)$.

### 6.2 Geometric Intuition: Separating Distributions (p. 218)

The discriminator $D^*$ acts like a **classifier boundary** separating the real distribution $p_d(x)$ (labeled $y=1$) from the generated distribution $p_g(x)$ (labeled $y=-1$):

|Scenario|Objective value|
|---|---|
|$p_d$ and $p_g$ are **well-separated** (easy to distinguish)|$J(D^*,G)$ is **large**|
|$p_d$ and $p_g$ **overlap partially**|$J(D^*,G)$ **decreases**|
|$p_d$ and $p_g$ **fully overlap** (indistinguishable)|$J(D^*,G)$ is **smallest**|

> This geometric picture directly motivates the minimax structure: $D$ wants to **maximize** $J$ (push the distributions apart / classify well), while $G$ wants to **minimize** $J$ (make $p_g$ overlap with $p_d$ so $D$ can no longer tell them apart) — hence:

$$\min_G \max_D J(D,G)$$

---

## 7. Optimal Discriminator Solution (p. 219)

**Recall the GAN objective**, rewritten as an integral over $x$:

$$J(G,D) = \int_x p_d(x)\log D(x),dx + \int_x p_g(x)\log(1-D(x)),dx$$

**Maximize $D$ pointwise**, at each position of $x$:

$$\max_D \Big{ p_d(x)\log D(x) + p_g(x)\log(1-D(x)) \Big}$$

**Taking the derivative and setting it to zero:**

$$\nabla_D = \frac{p_d(x)}{D} - \frac{p_g(x)}{1-D} = 0$$

**Solving for the optimal discriminator:**

$$D^*(x) = \frac{p_d(x)}{p_d(x)+p_g(x)}$$

> **Special case:** when $G$ generates **perfect samples**, $p_g = p_d$ everywhere, giving $D^*(x) = 0.5$ for all $x$ — the discriminator **can no longer distinguish** real from fake at all, exactly matching the intuition from the counterfeiter analogy (p.212): "when training is perfect, discriminator $D$ can NO longer tell the fake money from the real money, i.e., 0.5 probability."

---

## 8. Optimal Generator Solution (p. 220)

**Fix $D^*$** in the minimax objective:

$$\min_G \max_D J(D,G) = \min_G J(D^*, G)$$

**Recall the Jensen-Shannon (JS) divergence:**

$$D_{JS}(d,g) = \frac{1}{2}KL(d|m) + \frac{1}{2}KL(g|m), \qquad m = \frac{1}{2}(d+g)$$

**Substituting $D^_(x) = \frac{p_d(x)}{p_d(x)+p_g(x)}$ into $J(D^_,G)$:**

$$J(D^_,G) = \mathbb{E}_{x\sim p_d}\big[\log D^_(x)\big] + \mathbb{E}_{x\sim p_g}\big[\log(1-D^*(x))\big]$$

$$= \mathbb{E}_{x\sim p_d}\left[\log\frac{p_d(x)}{p_d(x)+p_g(x)}\right] + \mathbb{E}_{x\sim p_g}\left[\log\left(1-\frac{p_d(x)}{p_d(x)+p_g(x)}\right)\right]$$

$$= \mathbb{E}_{x\sim p_d}\left[\log\frac{p_d(x)}{p_d(x)+p_g(x)}\right] + \mathbb{E}_{x\sim p_g}\left[\log\frac{p_g(x)}{p_d(x)+p_g(x)}\right]$$

**This simplifies exactly to:**

$$J(D^*,G) = D_{JS}\big(p_d(x),|,p_g(x)\big) - 2\log 2$$

**Therefore, the minimization over $G$ becomes:**

$$\min_G D_{JS}\big(p_d(x),|,p_g(x)\big) \quad \Rightarrow \quad p_{g^*} = p_d$$

> **Key theoretical result:** minimizing the GAN objective over $G$ (with the discriminator held at its optimum $D^_$) is **exactly equivalent** to minimizing the **Jensen-Shannon divergence** between the generator's distribution $p_g$ and the true data distribution $p_d$. Since JS divergence is minimized (equal to 0) if and only if the two distributions are identical, the **globally optimal generator** satisfies $p_{g^_} = p_d$ — i.e., the generator's distribution exactly matches the true data distribution, and at this point $D^*(x) = 0.5$ everywhere (as derived in Section 7).

---

## Cross-Topic Connections

|Concept|Connects to|Relationship|
|---|---|---|
|PEFT methods (prompts/adapters/LoRA, p.205–207)|ViT architecture (previous chunk, p.193–194)|All three insert small trainable components into the _same_ frozen ViT architecture — the class token, patch embeddings, and Q/K/V projections identified earlier are exactly where these modifications attach|
|LoRA's low-rank decomposition (p.207)|Embedding matrices, weight matrices throughout (pages 1–200)|Demonstrates a general technique (low-rank approximation) that could in principle apply to any large weight matrix in a deep network, not just attention projections|
|GAN's minimax objective (p.213)|Adversarial training in CNN robustness (p.114)|Both use a $\min\max$ structure, but for different purposes: adversarial training minimizes worst-case loss over input perturbations to make a classifier robust; GAN's minimax is between two separate networks (G and D) with fundamentally opposed objectives over the _data itself_|
|GAN's discriminator as binary classifier (p.217)|Basic binary classification with sigmoid/logistic regression (pages 1–100)|The discriminator is literally trained as a standard binary classifier (real vs. fake), reusing the same $\log D(x)$ cross-entropy-style loss structure seen throughout the course|
|Optimal $D^*(x) = \frac{p_d}{p_d+p_g}$ (p.219)|Softmax/probability interpretation (pages 1–140)|Directly parallels how softmax normalizes competing "scores" into probabilities — here $D^*$ normalizes the relative "evidence" for real vs. fake at each point $x$|
|JS divergence result (p.220)|KL divergence (implicit throughout probabilistic ML)|Shows that the GAN's adversarial game has a precise information-theoretic interpretation: it's implicitly minimizing a symmetrized, bounded version of KL divergence between real and generated distributions|

---

## Quick-Reference Formula Sheet

$$\text{Adapter: } f(X) = g(X) + \sigma(XW_{down})W_{up}, \quad g(X)=\sigma(XW_1)W_2$$

$$\text{LoRA: } W_Q + B_QA_Q,\ \ W_K+B_KA_K,\ \ W_V+B_VA_V, \quad r \ll m,n$$

$$\text{GAN objective: } \min_G\max_D J(G,D) = \mathbb{E}_{x\sim p_d}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]$$

$$\text{Optimal discriminator: } D^*(x) = \frac{p_d(x)}{p_d(x)+p_g(x)}$$

$$J(D^_,G) = D_{JS}(p_d|p_g) - 2\log 2 \quad\Rightarrow\quad \min_G J(D^_,G) \Rightarrow p_{g^*}=p_d$$