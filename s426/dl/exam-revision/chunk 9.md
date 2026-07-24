

## tags: [FIT5215, deep-learning, attention, seq2seq, transformers, self-attention] pages: 161-180

# FIT5215 — Deep Learning: Pages 161–180

## 1. Beam Search — Worked Example (p. 161)

Illustration of **beam search with beam width $k=3$**:

- **Step 1:** given input $x$, generate 3 candidate first words with their probabilities: "Hi" ($p_1$), "Yes" ($p_2$), "Please" ($p_3$)
- **Step 2:** for **each** of these 3 candidates, generate 3 candidate second words (e.g. for "Hi": ",": $p_{11}$, "How": $p_{12}$, "What": $p_{13}$; for "Yes": "sure": $p_{21}$, "certainly": $p_{22}$, "I": $p_{23}$; for "Please": "I": $p_{31}$, "see": $p_{32}$, "come": $p_{33}$) — giving $3\times 3 = 9$ two-word candidate sequences
- **Joint probabilities** computed for each 2-word sequence: $p_1p_{11}, p_1p_{12}, p_1p_{13}, p_2p_{11}, \dots$
- **Prune:** keep only the **top-3** sequences by joint probability, discard the rest
- Repeat this expand-then-prune cycle at each subsequent step

> This confirms the general beam search algorithm from p.160: at every step, the search **branches** ($k$ candidates per surviving beam) then **prunes back down** to the top-$k$ by joint probability — preventing the candidate pool from growing exponentially while still exploring more than the single greedy path.

---

## 2. Drawback of Fixed Context (p. 162)

**Problem:** the fixed context vector $c$ (from basic seq2seq, p.155–157) is **easily overwhelmed by long inputs or long outputs** — a single fixed-size vector must encode arbitrarily long source information.

**Key insight:** at a specific decoding timestep $j$, some words/items in the **input** sequence might contribute **more** to generating the next output word than others — but a fixed $c$ cannot reflect this varying relevance.

**Illustrative example (English→French):**

- "I want to see you every day" → "Je veux te voir chaque jour" — translates cleanly
- But if we only have "I want to see **you** every day" → "Je veux te **?** (voir) …" — the model must correctly align "see" with "voir" at exactly the right decoding step, which is hard with one static $c$

**Solution — timestep-varying context vector:**

$$c_j = \alpha(h_1, \dots, h_{T_x}, q_{j-1})$$

computed using an **attention mechanism** — i.e., $c_j$ is now a function of **all** encoder hidden states **and** the current decoder state $q_{j-1}$, allowing it to dynamically emphasize different source positions at each decoding step.

**Paper:** Bahdanau, Cho, Bengio — _Neural Machine Translation by Jointly Learning to Align and Translate_, ICLR 2015.

---

## 3. Attention Mechanism: Global vs. Local (p. 163)

**Attention allows the decoding network to "refer back" to the input** rather than relying solely on one fixed summary vector.

|Type|Description|
|---|---|
|**Global attention**|Uses **all** input hidden states of the encoder when deriving context $c_t$|
|**Local attention**|Uses a **selective window** of input hidden states when deriving context $c_t$|

**Key papers:**

- Bahdanau, Cho, Bengio — _Neural Machine Translation by Jointly Learning to Align and Translate_, ICLR 2015 (global attention)
- Luong, Pham, Manning — _Effective Approaches to Attention-based Neural Machine Translation_, EMNLP 2015 (both global and local attention)

---

## 4. Global Attention

### 4.1 Main Idea and Alignment Weights (p. 164)

**Context vector** — weighted sum over **all** source hidden states (example shown with 3 source positions):

$$c_t = \sum_{s=1}^{3} a_t(s), h_s$$

**Alignment weights** — a softmax over alignment scores, giving how much attention to pay to each source position $s$ at decode step $t$:

$$a_t(s) = \text{align}(q_t, h_s) = \frac{\exp\big(\text{score}(q_t,h_s)\big)}{\sum_{s'}\exp\big(\text{score}(q_t,h_{s'})\big)}$$

where $q_t$ is the current target (decoder) state, and $h_s$ is each source (encoder) state.

**Alignment score function** — three common choices:

$$ \text{score}(q_t, h_s) = \begin{cases} q_t^{\mathsf{T}} h_s & \text{dot product} \ q_t^{\mathsf{T}} W_a h_s & \text{general metric} \ v_a^{\mathsf{T}}\tanh\big(W_a[q_t;h_s]\big) & \text{concat} \end{cases} $$

> **Terminology bridge:** $h_s$ plays the role of **keys** (and, implicitly, values), while $q_t$ is the **query** — this is the same query/key/value vocabulary that will reappear explicitly in Transformer self-attention (Section 7).

### 4.2 Full Global Attention Pipeline (p. 165)

**Context vector** (general $S$ source positions):

$$c_t = \sum_{s=1}^{3} a_t(s), h_s$$

**Attentional hidden state** — combine the context with the current decoder state:

$$\tilde{q}_t = \tanh\big(W_c[c_t; q_t]\big)$$

**Predictive distribution** over the output vocabulary:

$$p(y_t \mid y_{<t}, x) = \text{softmax}(W_s\tilde{q}_t)$$

### 4.3 Global Attention — Worked Procedure (p. 166)

1. Compute alignment scores and **convert into alignment weights** via softmax: $$a_t(s) = \frac{\exp(\text{score}(q_t,h_s))}{\sum_{s'}\exp(\text{score}(q_t,h_{s'}))}$$
2. **Build the context vector** as a weighted average of source hidden states: $$c_t = \sum_s a_t(s), h_s$$
3. **Compute the next (attentional) hidden state:** $$\tilde{q}_t = \tanh(W_c[c_t; q_t])$$
4. **Predictive distribution:** $$p(y_t\mid y_{<t}, x) = \text{softmax}(W_s\tilde{q}_t)$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GlobalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)   # "general" score
        self.Wc = nn.Linear(2*hidden_size, hidden_size)
        self.Ws = nn.Linear(hidden_size, hidden_size)  # project to vocab size in practice

    def forward(self, q_t, h_all):
        # q_t: (batch, hidden_size); h_all: (batch, seq_len, hidden_size)
        scores = torch.bmm(self.Wa(q_t).unsqueeze(1), h_all.transpose(1, 2)).squeeze(1)  # (batch, seq_len)
        a_t = F.softmax(scores, dim=-1)                                                   # alignment weights
        c_t = torch.bmm(a_t.unsqueeze(1), h_all).squeeze(1)                               # weighted sum -> context
        q_tilde = torch.tanh(self.Wc(torch.cat([c_t, q_t], dim=-1)))                      # attentional hidden state
        return q_tilde, a_t
```

### 4.4 Global Attention — Drawback (p. 167)

- **Employs all items on the source side** to derive each target item
- **Expensive computation** — scales with source sequence length at every decoding step
- **Impractical for translating longer sequences**

→ Motivates **Local attention**.

---

## 5. Local Attention (p. 168–169)

### 5.1 Main Idea

- Selectively focuses on a **small window** of source context, and remains **differentiable** (unlike hard/discrete attention)
- $c_t$ is derived as a **weighted average** over source hidden states within a window $[p_t - D,\ p_t + D]$, where $D$ is chosen empirically

### 5.2 Predicted Alignment Position

For the current target word, first **predict an aligned source position** $p_t$:

$$p_t = S \cdot \text{sigmoid}\big(v_p^{\mathsf{T}}\tanh(W_p q_t)\big)$$

where $W_p, v_p$ are learnable model parameters, and $S$ is the source sentence length — since sigmoid $\in (0,1)$, we get $0 \le p_t \le S$.

### 5.3 Windowed Alignment Weights

$$a_t(s) = \text{align}(q_t, h_s)\cdot \exp\left(-\frac{(s-p_t)^2}{2\sigma^2}\right), \qquad \sigma = D/2$$

where the **temporary alignment weights** are computed exactly as in global attention:

$$\text{align}(q_t,h_s) = \frac{\exp(\text{score}(q_t,h_s))}{\sum_{s'}\exp(\text{score}(q_t,h_{s'}))}$$

> **Interpretation:** the Gaussian factor $\exp(-\frac{(s-p_t)^2}{2\sigma^2})$ **down-weights** source positions far from the predicted center $p_t$, effectively restricting attention to a window around $p_t$ while keeping everything smooth and differentiable (rather than hard-cutting off positions outside the window).

**Context** is then computed exactly as before (weighted sum), but restricted to (effectively) the window of size $D$ centered at $p_t$.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LocalAttention(nn.Module):
    def __init__(self, hidden_size, D=5):
        super().__init__()
        self.D = D
        self.Wp = nn.Linear(hidden_size, hidden_size)
        self.vp = nn.Linear(hidden_size, 1, bias=False)
        self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, q_t, h_all, S):
        # Predict aligned position p_t
        p_t = S * torch.sigmoid(self.vp(torch.tanh(self.Wp(q_t))))  # (batch, 1)

        scores = torch.bmm(self.Wa(q_t).unsqueeze(1), h_all.transpose(1, 2)).squeeze(1)  # (batch, seq_len)
        align = F.softmax(scores, dim=-1)

        positions = torch.arange(h_all.size(1), device=h_all.device).float()  # (seq_len,)
        sigma = self.D / 2
        gaussian = torch.exp(-((positions - p_t) ** 2) / (2 * sigma ** 2))    # (batch, seq_len)

        a_t = align * gaussian
        a_t = a_t / (a_t.sum(dim=-1, keepdim=True) + 1e-10)  # re-normalize
        c_t = torch.bmm(a_t.unsqueeze(1), h_all).squeeze(1)
        return c_t, a_t, p_t
```

---

## 6. Transformers — Motivation

### 6.1 Recap: RNN-Based Seq2Seq (p. 171)

Standard encoder-decoder recap: encoder RNN processes "The cat jumps the wall" → hidden states $h_0,\dots,h_4$ → context vector $c$ → decoder RNN generates "Il gatto salta il muro" token by token, starting from $\langle\text{BOS}\rangle$ and ending at $\langle\text{EOS}\rangle$.

### 6.2 Two Core Problems with RNNs (p. 172)

1. **We forget tokens too far in the past** in the context vector $c$ — the long-term dependency problem revisited (cf. p.125, p.157's fixed-$c$ drawback)
2. **We need to wait for the previous token** to compute the next hidden state — the **sequential** nature of RNNs prevents parallelization across time steps

### 6.3 Solving Problem 1: Attention (p. 173)

**Solution:** add an attention mechanism, exactly as covered in Sections 2–5 — instead of a single fixed $c$, compute a **timestep-specific** context $c_t$ as a weighted combination ($\alpha_0, \alpha_1, \alpha_2, \alpha_3, \alpha_4$) of **all** encoder hidden states, re-derived fresh at each decoding step.

### 6.4 Solving Problem 2: Remove Recurrence Entirely (p. 174)

**The 2017 paper "Attention Is All You Need"** proposes: **throw away the recurrent connections entirely.**

- If attention alone can access **all** past information at once (as shown in solving Problem 1), why keep the sequential recurrence at all?
- Removing recurrence means all positions can be processed **in parallel**, directly addressing Problem 2 (no need to "wait for the previous token").

This is the core idea behind the **Transformer** architecture.

### 6.5 Transformer Motivations Summary (p. 175)

**Problems using RNNs for Seq2Seq:**

- **Slow** due to sequential nature (cannot parallelize across time steps within a sequence)
- **Poor long-range dependency modelling**

**Transformer solutions:**

- Enables **parallel computation**, fully exploiting GPU power — achieved by **removing the dependency between words** (no recurrence)
- But since removing recurrence loses all notion of word order/position, this is resolved via **positional encoding**
- Extremely good at capturing **long-range dependencies** (via self-attention, which connects any two positions directly regardless of distance)

---

## 7. Transformer — General Architecture (p. 176)

**Some key building blocks:**

- **Positional encoding**
- **Self-attention** and **Multi-Head Self-attention**
- **Masked Multi-Head Attention** (used in the decoder, to prevent attending to future tokens)
- **Residual add and layer norm**

**Overall structure** (per the original paper and the "illustrated transformer" diagram):

- **Encoder stack:** Input Embedding + Positional Encoding → [Multi-Head Attention → Add & Norm → Feed Forward → Add & Norm] $\times N$
- **Decoder stack:** Output Embedding + Positional Encoding → [Masked Multi-Head Attention → Add & Norm → Multi-Head Attention (over encoder output) → Add & Norm → Feed Forward → Add & Norm] $\times N$ → Linear → Softmax → Output Probabilities

Multiple stacked encoders feed into multiple stacked decoders — each decoder layer's second attention sub-block attends over the **final encoder output** (all arrows from top encoder to every decoder layer).

---

## 8. Transformer Encoder — Overview (p. 177)

**Full shape pipeline for the encoder** (example: batch of tokenized sentences like "I love deep learning" / "Hello the world <pad>"):

|Stage|Shape|
|---|---|
|Raw token indices|$[batch_size,\ seq_len]$|
|After Embedding Layer ($E$: $[vocab_size, embed_size]$)|$[batch_size,\ seq_len,\ d_model]$ _(note: embed_size = d_model)_|
|After Self-Attention (SA) / Multi-head SA|$[batch_size,\ seq_len,\ d_model]$|
|After **Add & Normalize** (Layer-Norm + Skip Connection)|$[batch_size,\ seq_len,\ d_model]$|
|After Point-wise Feed-forward NN|$[batch_size,\ seq_len,\ d_model]$|
|After **Add & Normalize** (Layer-Norm + Skip Connection)|$[batch_size,\ seq_len,\ d_model]$|
|... repeated for $\times n$ **Encoders** ...|$[batch_size,\ seq_len,\ d_model]$|
|**Encoder output**|$[batch_size,\ seq_len,\ d_model]$|

> **Key structural property:** the output shape of each encoder block is **identical** to its input shape ($[batch_size, seq_len, d_model]$) — this is what allows $n$ identical encoder blocks to be **stacked** directly on top of one another.

---

## 9. Transformer — Self-Attention

### 9.1 Core Concept (p. 178)

**Self-attention** operates **among a sequence of items/tokens** — each token attends to (potentially) every other token in the same sequence, including itself.

- **Keys and queries** are used to compute self-attention weights
- $W^Q, W^K, W^V$ are **learnable matrices**
- Multiplying each token embedding $x_1, x_2, \dots$ by $W^Q, W^K, W^V$ produces its **query**, **key**, and **value** vectors

**Example computation** for output $z_1$ (illustrative "Thinking Machines" example):

$$z_1 = 0.88 \times v_1 + 0.12 \times v_2$$

i.e., $z_1$ is a weighted combination of the **value** vectors, where the weights (0.88, 0.12) come from the attention distribution.

### 9.2 Full Self-Attention Formulas (p. 179)

**Per-token queries, keys, values:**

$$q_i = x_i W^Q, \qquad k_i = x_i W^K, \qquad v_i = x_i W^V$$

**Matrix/stacked version** (with $X = \begin{bmatrix}x_1\ \vdots \ x_L\end{bmatrix}$, $L = seq_len$):

$$Q = XW^Q, \qquad K = XW^K, \qquad V = XW^V$$

**Scaled dot-product attention** — attention probabilities between query and key:

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

**Weighted sum of values:**

$$Z = AV$$

> **Why divide by $\sqrt{d_k}$?** As $d_k$ (the key/query dimension) grows, raw dot products $QK^T$ can grow large in magnitude, pushing the softmax into regions with extremely small gradients. Scaling by $\sqrt{d_k}$ keeps the dot products in a numerically well-behaved range, stabilizing training — directly analogous to the initialization-variance concerns from He/Xavier init (p.81) and batch normalization (p.93–95).

### 9.3 Worked Numeric Example — "Thinking Machines" (p. 178–179)

For two tokens $x_1$="Thinking", $x_2$="Machines":

|Step|Value|
|---|---|
|Embeddings $x_1, x_2$|4-dim vectors (example)|
|Queries $q_1, q_2$|computed via $W^Q$|
|Keys $k_1, k_2$|computed via $W^K$|
|Values $v_1, v_2$|computed via $W^V$|
|Score: $q_1\cdot k_1$|112|
|Score: $q_1\cdot k_2$|96|
|Divide by $\sqrt{d_k}=8$|14, 12|
|Softmax|0.88, 0.12|
|$z_1 = 0.88\times v_1 + 0.12\times v_2$|weighted-sum output for token 1|

> This shows explicitly how $z_1$ (the self-attention output for "Thinking") is dominated by its **own** value $v_1$ (weight 0.88) but still incorporates a smaller contribution (weight 0.12) from "Machines" — this is the mechanism that lets each token's representation be informed by the full context.

**Source:** [jalammar.github.io/illustrated-transformer](https://jalammar.github.io/illustrated-transformer/)

### 9.4 Self-Attention — Full Shape Trace (p. 180)

Starting from token indices $[batch_size, seq_len]$ → embedding → $X \in [batch_size, seq_len, d_model]$:

$$Q = XW^Q \ \in [batch_size, seq_len, d_Q{=}d_K]$$

$$K = XW^K \ \in [batch_size, seq_len, d_Q{=}d_K]$$

$$V = XW^V \ \in [batch_size, seq_len, d_V]$$

**Attention scores:**

$$B = \frac{QK^T}{\sqrt{d_K}} \ \in [batch_size, seq_len, seq_len]$$

where individual entries represent similarity: $B_{ij}^1 = Q^1[i,:], K^1[j,:]^T \to \text{sim}(x_i^1, x_j^1)$

**Attention probabilities** (softmax over the **last** dimension, i.e. `dim=2`):

$$A = \text{softmax}(B,\ \text{dim}=2) \ \in [batch_size, seq_len, seq_len]$$

**Weighted sum of values:**

$$Z = AV \ \in [batch_size, seq_len, d_V]$$

**Final output projection:**

$$Z = ZW^O \ \in [batch_size, seq_len, d_model]$$

> Note: the output shape after $W^O$ matches the **original input shape** $[batch_size, seq_len, d_model]$ — consistent with the requirement noted in Section 8 that encoder blocks preserve shape end-to-end, enabling stacking.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v):
        super().__init__()
        self.WQ = nn.Linear(d_model, d_k, bias=False)
        self.WK = nn.Linear(d_model, d_k, bias=False)
        self.WV = nn.Linear(d_model, d_v, bias=False)
        self.WO = nn.Linear(d_v, d_model, bias=False)
        self.d_k = d_k

    def forward(self, X):                          # X: (batch, seq_len, d_model)
        Q = self.WQ(X)                              # (batch, seq_len, d_k)
        K = self.WK(X)                              # (batch, seq_len, d_k)
        V = self.WV(X)                              # (batch, seq_len, d_v)

        B = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(self.d_k)  # (batch, seq_len, seq_len)
        A = F.softmax(B, dim=2)                      # attention probabilities
        Z = torch.bmm(A, V)                          # (batch, seq_len, d_v)
        Z = self.WO(Z)                               # (batch, seq_len, d_model)
        return Z, A
```

---

## Cross-Topic Connections

|Concept|Connects to|Relationship|
|---|---|---|
|Global attention's query/key terminology (p.164)|Self-attention Q/K/V (p.178–180)|Direct conceptual predecessor — RNN-based attention already used "query" ($q_t$) and "key" ($h_s$) roles; the Transformer generalizes this into learnable $W^Q, W^K, W^V$ projections applied **within** a single sequence|
|Local attention's Gaussian windowing (p.169)|Masked attention in Transformer decoders (p.176)|Both are ways of **restricting** which positions can be attended to — local attention softly restricts by distance, masked attention hard-restricts to only past positions|
|Fixed-context drawback → attention (p.162)|RNN long-term dependency problem (p.125), LSTM gating (p.127–129)|All three are responses to the same core issue: fixed-size summaries (or purely sequential compression) lose information over long sequences; attention is the most general fix, since it avoids compression entirely|
|Removing recurrence for parallelism (p.174)|RNN's sequential computation (pages 115–140)|Directly resolves the "must wait for previous token" bottleneck inherent to every RNN/LSTM/GRU architecture covered earlier|
|Scaled dot-product attention's $\sqrt{d_k}$ scaling (p.179)|Xavier/He initialization (p.81), Batch Normalization (p.93–96)|All three techniques manage the **variance** of internal signals to keep softmax/activations in a well-behaved numerical range and preserve healthy gradients|
|Encoder shape-preservation property (p.177, p.180)|Stacked/deeper RNNs (p.122)|Same architectural principle — a building block whose output shape matches its input shape can be stacked arbitrarily deep|

---

## Quick-Reference Formula Sheet

$$c_j = \alpha(h_1,\dots,h_{T_x}, q_{j-1}) \quad \text{(attention-based context)}$$

$$\text{Global attention: } a_t(s)=\frac{\exp(\text{score}(q_t,h_s))}{\sum_{s'}\exp(\text{score}(q_t,h_{s'}))}, \quad c_t=\sum_s a_t(s)h_s$$

$$\text{score}(q_t,h_s) \in {q_t^Th_s,\ q_t^TW_ah_s,\ v_a^T\tanh(W_a[q_t;h_s])}$$

$$\text{Local attention: } p_t = S\cdot\text{sigmoid}(v_p^T\tanh(W_pq_t)), \quad a_t(s)=\text{align}(q_t,h_s)\exp\left(-\frac{(s-p_t)^2}{2\sigma^2}\right)$$

$$\text{Self-attention: } Q=XW^Q,\ K=XW^K,\ V=XW^V,\quad A=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right),\quad Z=AV$$