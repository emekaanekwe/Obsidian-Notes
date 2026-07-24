
## tags: [FIT5215, deep-learning, GAN, generative-models, mode-collapse, inception-score] pages: 221-232

# FIT5215 — Deep Learning: Pages 221–232

## 1. The Nash Equilibrium Point (p. 221)

**The solution of the minimax problem** $\min_G\max_D J(D,G)$ is the **Nash equilibrium point** $(D^_, G^_)$, which satisfies:

$$p_{g^*} = p_d$$

$$D^_(x) = \frac{p_d(x)}{p_d(x)+p_{g^_}(x)} = 0.5, \quad \text{for all } x$$

_(This confirms and restates the theoretical results derived in the previous chunk's Sections 7–8: the optimal discriminator formula and the JS-divergence-minimizing optimal generator.)_

**Visualizing convergence (1D GAN simulation, Goodfellow et al., NeurIPS 2014):**

The classic 4-panel figure (a)–(d) shows how $p_g$ (green, generated distribution) and $D$ (blue dashed, discriminator) evolve over training against a fixed $p_d$ (black dotted, real data distribution):

- **(a)** Early training: $p_g$ and $p_d$ are quite different; $D$ is noisy/wiggly but roughly separates them
- **(b)–(c)** As training progresses, $p_g$ gradually shifts to **overlap more** with $p_d$, and $D$'s decision boundary **flattens** (becomes less able to discriminate)
- **(d)** At convergence: $p_{g^_} = p_d$ (the two distributions coincide exactly), and $D^_(x) = 0.5$ for all $x$ — a **flat line** at 0.5, confirming the discriminator can no longer tell real from fake anywhere

**Course's own simulation** (right panel, "Histogram of GAN1D"): shows the decision boundary (red, flat at 0.5), real data density (blue), and generated data density (purple) after training — illustrating the same theoretical convergence point in a concrete numerical example.

---

## 2. GAN: Example Results

### 2.1 Classic Results — MNIST and CIFAR-10 (p. 222)

_(From Goodfellow et al., NeurIPS 2014's original paper)_

- **MNIST**: generated handwritten digit samples compared against their **nearest-neighbour samples from the training set** — a common way to check the generator isn't simply memorizing/copying training examples, but producing genuinely novel (if similar-looking) samples
- **CIFAR-10**: generated natural image samples, similarly compared to nearest training neighbours

### 2.2 Harder Cases — ImageNet (p. 223)

For a **complex dataset such as ImageNet**:

- GANs **can generate sharp images**, but they **look unrealistic**
- It's often **hard to tell what objects** the generated images are even supposed to depict

> **Key practical finding: Conditional GANs with labels work much better in practice!** — providing class label information to both generator and discriminator (rather than expecting the GAN to learn the full unconditional, highly multi-modal distribution of a complex dataset like ImageNet from scratch) substantially improves sample quality and coherence.

---

## 3. Issues with GAN

GANs suffer from **two serious technical problems**.

### 3.1 Problem #1: Mode Collapsing (p. 224–225)

**Description:** mode collapse **impeaches (undermines) the generator's ability to generate diverse, realistic data/images**.

- Generated (fake) examples $G(z)$ where $z\sim p(z)$ **can only cover a few modes** in the data distribution $p_d(x)$
- The generator **misses many other modes** in the data distribution — i.e., it learns to produce only a narrow subset of the true variety present in real data, even if those samples individually look plausible

**Concrete examples from the slides:**

- **MNIST mode collapse example**: the generator produces **only some digits** (e.g., ${0, 1, 3, 5, 7}$) while learning from a dataset that contains all ten digit classes — entire digit classes are effectively "missing" from what the generator can produce
- **CelebA mode collapse example**: the generator produces **only some types of faces**, failing to capture the full diversity of faces present in the training data

> **Why this happens (intuition):** if the generator finds a small set of outputs that reliably fool the current discriminator, gradient descent on the minimax objective has no explicit pressure forcing it to also cover other, harder-to-generate modes — it can get "stuck" repeatedly producing variations on the few modes it has already learned fool $D$, rather than exploring the full data distribution.

### 3.2 Problem #2: Convergence Is Hard (p. 226–227)

**Description:** convergence is difficult due to the fundamentally **minimax** nature of the formulation — **training GAN is very challenging!**

**Recall the minimax problem:**

$$\min_G\max_D J(G,D) = \mathbb{E}_{x\sim p_d(x)}\big[\log D(x)\big] + \mathbb{E}_{z\sim p_z}\big[\log(1-D(G(z)))\big]$$

**Key issue: no unique loss function.** Unlike standard supervised learning (where a single scalar loss is minimized), GAN training involves **two separate, competing loss functions** (discriminator loss and generator loss) being optimized in opposite directions simultaneously — there's no single objective value that monotonically indicates "training is going well."

**Empirical illustration (loss curves over training iterations):**

- **Discriminator loss**: rises sharply early, then fluctuates in a roughly stable but noisy band (~0.60–0.61) rather than smoothly converging to a fixed value
- **Generator loss**: drops sharply early, then **gradually increases** over the rest of training with continued fluctuation, rather than monotonically decreasing

> **Interpretation:** unlike a normal training loss curve which should trend toward a minimum, GAN loss curves for both networks tend to oscillate rather than settle — because **each network's optimal response depends on the other network's current state**, which keeps changing during training. This lack of a clean convergence signal makes it hard to know when to stop training or diagnose training problems, and is a core reason GAN training is notoriously unstable in practice (motivating later variants such as WGAN, spectral normalization, and other stabilization techniques not covered directly in these slides but often referenced as follow-ups to this exact issue).

---

## 4. Evaluating GANs: The Inception Score

### 4.1 The Evaluation Problem (p. 228–229)

**Core challenges:**

1. Except for trivial cases, **we don't know the true data distribution $p_d$** — so how can we quantify how good the generated distribution $p_g$ is?
2. How do we know that a generative model can generate **diverse examples** that cover **many modes** in the data?
3. How do we **compare two generative models** ($G_1$ vs. $G_2$) trained on the same training set?

**Illustrative example:** two generators $G_1$ and $G_2$ both map from the same latent space $z\sim p(z)$ to a data space with 4 modes (labeled $y=1,2,3,4$).

- $G_1$'s generated samples (red dots) **miss data mode $y=4$** — its samples cluster only around modes 1, 2, and 3
- $G_2$'s generated samples cover **all four modes** more evenly

**Conclusion:** $G_2$ is **better** because it generates **more diverse** examples, while $G_1$ **misses the data mode $y=4$** — this is a direct visual illustration of the mode collapse problem (Section 3.1) and motivates the need for a quantitative diversity metric.

### 4.2 Inception Score Formulation (p. 230)

**Setup:** assume training examples have labels $y \in {1,2,3,4}$.

**Step 1 — Train a good classifier $C$** on the labeled training set:

$$C(x) = p(y\mid x) = \big[p(y=k\mid x)\big]_{k=1}^{4}$$

is the vector of prediction probabilities, where $p(y=k\mid x)$ is the probability of classifying $x$ into class $k$.

**Step 2 — Two desired properties, given generated samples** $\tilde{x}_i = G(z_i)$ for $i=1,\dots,T$:

**(a) Confidence (low entropy per sample):** $C(\tilde{x}_i) = p(y\mid \tilde{x}_i = G(z_i))$ should be **close to a one-hot vector** — i.e., $C(\tilde{x}_i)$ is confident about which class each individual generated sample belongs to, and has **small entropy**.

**(b) Diversity (high entropy on average):** the average prediction across all generated samples,

$$\frac{1}{T}\sum_{i=1}^{T} C(\tilde{x}_i) = \frac{1}{T}\sum_{i=1}^{T} p(y\mid \tilde{x}_i = G(z_i))$$

should be **close to the uniform distribution** $[0.25, 0.25, 0.25, 0.25]$ — i.e., across many generated samples, all classes should be represented roughly equally (the uniform distribution has the **largest entropy** among distributions over 4 classes).

**Inception Score formula:**

$$\text{IS} \approx \frac{1}{T}\sum_{i=1}^{T} KL\left(C(\tilde{x}_i),\ \frac{1}{T}\sum_{j=1}^{T} C(\tilde{x}_j)\right) \approx \mathbb{E}_z\Big[KL\big(p(y\mid \tilde{x}=G(z)),\ p(y)\big)\Big]$$

- This is the **KL divergence** between each individual sample's (confident, low-entropy) prediction distribution and the (diverse, high-entropy) average prediction distribution across all samples
- **Higher is better** → indicates **more diversity** of generated images (while each individual sample remains confidently classifiable)

**Paper:** Salimans et al., _Improved Techniques for Training GANs_, 2016.

> **Intuition for why KL divergence captures both properties simultaneously:** the KL divergence $KL(C(\tilde{x}_i) | \bar{C})$ is large when $C(\tilde{x}_i)$ (confident/peaked per-sample) differs a lot from $\bar{C}$ (the diverse/spread average). This happens exactly when: (1) each sample is confidently classified into _some_ specific class (peaked $C(\tilde{x}_i)$), AND (2) different samples are confidently classified into _different_ classes (so the average $\bar{C}$ ends up spread out/uniform rather than also peaked at the same class). If the generator suffered from mode collapse (all samples confidently predicted as the _same_ class), $C(\tilde{x}_i)$ and $\bar{C}$ would look similar (both peaked at that one class), giving a **low** KL divergence and thus a **low** (bad) Inception Score — correctly penalizing lack of diversity.

```python
import torch
import torch.nn.functional as F

def inception_score(generated_samples, classifier, num_samples):
    """
    generated_samples: tensor of generated images, shape (T, C, H, W)
    classifier: pretrained classifier C(x) -> class probabilities
    """
    with torch.no_grad():
        preds = F.softmax(classifier(generated_samples), dim=-1)   # C(x_tilde_i) for each i, shape (T, num_classes)

    marginal = preds.mean(dim=0, keepdim=True)                      # (1/T) sum_j C(x_tilde_j), shape (1, num_classes)

    # KL(C(x_i) || marginal) for each sample, then average
    kl_divs = (preds * (torch.log(preds + 1e-10) - torch.log(marginal + 1e-10))).sum(dim=-1)
    inception_score_value = torch.exp(kl_divs.mean())   # commonly exponentiated in practice for reporting
    return inception_score_value.item()
```

---

## 5. GAN: Summary (p. 231)

**Adversarial training** is fundamentally **a game between two players** [Goodfellow et al., 2014]:

- **Generator $G$**: generates fake samples that are **indistinguishable** from real samples
- **Discriminator $D$**: discriminates between real and fake samples

**The min-max problem:**

$$\min_G\max_D J(G,D) = \mathbb{E}_{x\sim p_d(x)}\big[\log D(x)\big] + \mathbb{E}_{z\sim p_z}\big[\log(1-D(G(z)))\big]$$

**Issues and challenges** (both covered in this chunk):

- **Mode collapse** — limited diversity in generated samples
- **Min-max optimization problem** — inherent training instability, no unique/monotonic loss signal

---

## Cross-Topic Connections

|Concept|Connects to|Relationship|
|---|---|---|
|Nash equilibrium's $D^*=0.5$ everywhere (p.221)|Optimal discriminator derivation (previous chunk, p.219)|Direct restatement/visualization of the closed-form result derived analytically — the 1D simulation plots make the abstract $D^*(x)=\frac{p_d}{p_d+p_g}$ formula concrete|
|Mode collapse (p.224)|GAN's minimax objective structure (previous chunk, p.213)|Mode collapse is a direct practical consequence of the adversarial objective having no explicit diversity-promoting term — $G$ only needs to fool $D$, not cover all modes|
|Conditional GANs improving on ImageNet (p.223)|Label smoothing, cross-entropy with labels (pages 81–140)|Conditioning the generator/discriminator on class labels is analogous to how supervised classification uses labels to structure the learning signal — it breaks down a hard, highly multi-modal unconditional generation problem into easier conditional sub-problems|
|Inception Score's KL divergence (p.230)|Jensen-Shannon divergence in optimal generator proof (previous chunk, p.220)|Both use divergence measures (KL vs. JS) to quantify distributional (mis)match — one as a training-time proof tool, the other as a post-hoc evaluation metric|
|GAN training instability (p.226)|Optimization/gradient descent challenges (pages 1–80)|Extends the course's earlier optimization content (SGD, momentum, saddle points) to the harder setting of _simultaneous_ two-player optimization, where standard single-objective convergence guarantees don't directly apply|

---

## Quick-Reference Formula Sheet

$$\text{Nash equilibrium: } p_{g^_}=p_d, \quad D^_(x)=\frac{p_d(x)}{p_d(x)+p_{g^*}(x)}=0.5\ \forall x$$

$$\text{GAN minimax: } \min_G\max_D J(G,D) = \mathbb{E}_{x\sim p_d}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1-D(G(z)))]$$

$$\text{Inception Score: } \frac{1}{T}\sum_{i=1}^{T}KL\left(C(\tilde{x}_i),\ \frac{1}{T}\sum_{j=1}^{T}C(\tilde{x}_j)\right) \approx \mathbb{E}_z\big[KL(p(y\mid\tilde{x}=G(z)),\ p(y))\big]$$

---

## End-of-Deck Note

This concludes the FIT5215 Deep Learning slide deck (232 pages total). Across the full set of notes (pages 1–232), the material progressed through:

1. **Foundations** (p.1–80): FFNs, activations, backprop, optimization, CNNs
2. **Regularization & robustness** (p.81–120): dropout, batch norm, data augmentation, adversarial examples, RNN basics
3. **Sequence modelling** (p.121–170): LSTM/GRU, Word2Vec, seq2seq, attention mechanisms
4. **Transformers** (p.170–200): self-attention, positional encoding, Vision Transformers
5. **Fine-tuning & generative models** (p.201–232): PEFT methods (prompts/adapters/LoRA), GANs

All 12 chunk-notes together form a complete, cross-referenced study set for exam revision — the "Cross-Topic Connections" tables in each note were designed to help trace recurring themes (gradient flow, distribution matching, parameter sharing, attention mechanisms) across the entire course.