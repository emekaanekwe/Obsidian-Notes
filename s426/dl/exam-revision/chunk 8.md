## tags: [FIT5215, deep-learning, word2vec, skip-gram, CBOW, seq2seq, encoder-decoder] pages: 141-160

# FIT5215 — Deep Learning: Pages 141–160

## 1. Word2Vec: Pretext Task Overview (p. 141)

Word2Vec learns word embeddings by casting representation learning as a **supervised pretext task** over a sliding window across a sentence, e.g.:

> _"The quick brown fox jumps over the lazy dogs."_

Two complementary formulations:

|Model|Pretext task|
|---|---|
|**Skip-gram**|Target word → **predict** context words|
|**Continuous Bag of Words (CBOW)**|Context words → **predict** target word|

---

## 2. Skip-gram

### 2.1 Pretext Task (p. 142)

For each **target word** (center word), Skip-gram generates training pairs with each **context word** within a window around it. Example pairs generated from the sentence (window size effectively $\pm 2$):

```
(brown, the), (brown, quick), (brown, for), (brown, jumps)
(fox, quick), (fox, brown), (fox, jumps), (fox, overs)
(jumps, brown), (jumps, fox), (jumps, over), (jumps, the)
(over, fox), (over, jumps), (over, the), (over, lazy)
(the, jumps), (the, over), (the, lazy), (the, dogs)
```

Each pair is (target word, context word) — the model is trained to predict the **second** element given the **first**.

### 2.2 Modelling (p. 143)

**Setup:** current window _"The quick brown fox jumps"_, target word $tw = \text{brown}$.

**Joint probability of context words given target** (treating context word predictions as conditionally independent given $tw$):

$$P(\text{the}, \text{quick}, \text{fox}, \text{jumps} \mid \text{brown}) = P(\text{the}\mid\text{brown}) \times P(\text{quick}\mid\text{brown}) \times P(\text{fox}\mid\text{brown}) \times P(\text{jumps}\mid\text{brown})$$

**Log-likelihood form:**

$$\log P(\text{the},\text{quick},\text{fox},\text{jumps}\mid\text{brown}) = \log P(\text{the}\mid\text{brown}) + \dots + \log P(\text{jumps}\mid\text{brown})$$

**Architecture — two embedding matrices:**

$$U \in \mathbb{R}^{N\times d} \quad (N = \text{vocabulary size},\ d = \text{embedding size})$$

$$V \in \mathbb{R}^{d\times N}$$

Let $t$ be the index of target word $tw$, and $c$ the index of context word $cw$ (both in $1,\dots,N$).

**Forward propagation:**

$$h = \mathbf{1}_t U = U_t^r \in \mathbb{R}^{1\times d} \qquad \text{(select target word's row from } U\text{)}$$

$$o = hV \in \mathbb{R}^{1\times N}$$

$$p = \text{softmax}(o) \in \mathbb{R}^{1\times N}$$

**Probability of a specific context word:**

$$P(cw=\text{the}\mid tw=\text{brown}) = p_c$$

**Log-probability, expanded via the softmax definition:**

$$\log P(cw=\text{the}\mid tw=\text{brown}) = \log p_c = U_t^r V_c^c - \log\left(\sum_{k=1}^{N}\exp(U_t^r V_k^c)\right)$$

Train by **maximizing log-likelihood** over all (target, context) pairs in the corpus.

> **Key idea:** $U$ provides the **input/target-word embeddings** (rows of $U$), while $V$ provides the **output/context-word embeddings** (columns of $V$). After training, $U$ (or sometimes $U$ and $V$ averaged/concatenated) is used as the final word embedding matrix.

### 2.3 Toy Example Setup (p. 144)

**Corpus:** _"the quick brown fox jumps over the lazy dog"_

**Vocabulary (tokens):** ${$'brown': 1, 'lazy': 2, 'over': 3, 'fox': 4, 'dog': 5, 'quick': 6, 'the': 7, 'jumps': 8$}$

- Number of tokens: $N = 8$
- Context (window) size: $C = 3$
- Embedding dimension: $d = 3$
- $U, V$: collections of input & output vectors

Example one-hot encoding of "quick" (index 6): $[0,0,0,0,0,1,0,0]$

### 2.4 Skip-gram Forward Propagation — Worked Example (p. 145)

**Setup:** center word $w_t = \mathbf{1}_6$ ("quick"), predicting context words $w_{t-1}=$"the" and $w_{t+1}=$"brown".

**Step 1 — Hidden layer (embedding lookup):**

$$h = \mathbf{1}_6 U = [0.7, \ 0.7,\ 0.6] \in \mathbb{R}^{1\times d}$$

(This picks out row 6 of the $8\times 3$ matrix $U$.)

**Step 2 — Output scores:**

$$o = hV = [0.45,\ 0.53,\ 0.89,\ 0.48,\ 0.67,\ 1.32,\ 0.77,\ 0.81]^T \in \mathbb{R}^{1\times N}$$

**Step 3 — Softmax → probability distribution over vocabulary:**

$$p(cw\mid tw) = [.09,\ .10,\ .14,\ .09,\ .11,\ .21,\ .12,\ .13]^T$$

**Step 4 — Cross-entropy loss** against each true context word's one-hot vector — e.g. for context word "brown" (index 1), target one-hot is $[1,0,0,0,0,0,0,0]^T$; for context word "the" (index 7), target one-hot is $[0,0,0,0,0,0,1,0]^T$. Loss is computed and backpropagated for **each** context word in the window separately, using the **same** hidden vector $h$.

```python
import torch
import torch.nn as nn

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.U = nn.Embedding(vocab_size, embed_dim)   # input/target embeddings
        self.V = nn.Linear(embed_dim, vocab_size, bias=False)  # output/context embeddings (V^T as weight)

    def forward(self, target_word_idx):
        h = self.U(target_word_idx)      # (batch, embed_dim)
        o = self.V(h)                    # (batch, vocab_size)
        return o                         # pass through CrossEntropyLoss (applies softmax internally)

# Training: for each (target, context) pair, treat context word index as the class label
criterion = nn.CrossEntropyLoss()
# loss = criterion(model(target_idx), context_idx)
```

### 2.5 Skip-gram Drawback (p. 146)

- Computing $p = \text{softmax}(o)$ is **computationally expensive** — requires summing over the entire vocabulary $\sum_{k=1}^{N}\exp(U_t^r V_k^c)$ for every training pair
- $p \in \mathbb{R}^{1\times N}$ is a distribution over a (usually very large) vocabulary $N$ → individual probabilities $p_i$ are **very tiny** → **hard to train** (vanishing gradient-like issue in the softmax normalization)

**Two standard fixes:**

1. **Hierarchical Softmax**
2. **Negative sampling** (more popular and efficient)

### 2.6 Negative Sampling (p. 147)

**Idea:** transform the expensive $N$-class softmax prediction into a cheap **binary prediction** problem using sampled negative examples.

**Setup:** consider a **positive (true) pair** $(tw=\text{brown}, cw=\text{the})$, labeled $y=1$:

$$[(\text{brown}, \text{the}), 1]$$

**Sample** a small number (e.g. two) random "negative" context words that did **not** actually appear near "brown":

$$[(\text{brown}, ng_1=\text{hello}), 0] \quad \text{and} \quad [(\text{brown}, ng_2=\text{awesome}), 0]$$

Let $n_1, n_2$ denote the vocabulary indices of $ng_1, ng_2$.

**Forward propagation** (using **sigmoid** instead of softmax — this is the key efficiency gain, since sigmoid only needs a single dot product per pair, not a full-vocabulary normalization):

$$h = \mathbf{1}_t U = U_t^r \in \mathbb{R}^{1\times d}, \qquad o = hV \in \mathbb{R}^{1\times N}, \qquad p = \text{sigmoid}(o) \in \mathbb{R}^{1\times N}$$

$$P(y=1 \mid tw=\text{brown}, cw=\text{the}) = p_c$$

$$P(y=1 \mid tw=\text{brown}, ng_1=\text{hello}) = p_{n_1}$$

$$P(y=1 \mid tw=\text{brown}, ng_2=\text{awesome}) = p_{n_2}$$

**Optimization problem:**

$$\max\ \Big[\log p_c - \alpha\log p_{n_1} - \alpha\log p_{n_2}\Big], \qquad \alpha > 0 \text{ is a trade-off parameter}$$

> **Intuition:** we want $p_c$ (probability assigned to the _true_ context word) to be **high**, while $p_{n_1}, p_{n_2}$ (probabilities assigned to _randomly sampled, likely-irrelevant_ words) should be **low**. This only requires evaluating the sigmoid at a handful of specific word pairs per training step — not a full softmax over the entire vocabulary — giving a large speedup.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkipGramNegSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.U = nn.Embedding(vocab_size, embed_dim)
        self.V = nn.Embedding(vocab_size, embed_dim)

    def forward(self, target_idx, context_idx, negative_idxs):
        h = self.U(target_idx)                        # (batch, d)
        v_pos = self.V(context_idx)                    # (batch, d)
        v_neg = self.V(negative_idxs)                  # (batch, k, d)

        pos_score = torch.sigmoid((h * v_pos).sum(-1))                 # p_c
        neg_score = torch.sigmoid(-(h.unsqueeze(1) * v_neg).sum(-1))   # 1 - p_{n_i}, k negatives

        loss = -(torch.log(pos_score + 1e-10).mean()
                 + torch.log(neg_score + 1e-10).sum(-1).mean())
        return loss
```

---

## 3. Continuous Bag of Words (CBOW)

### 3.1 Pretext Task (p. 148)

CBOW **reverses** Skip-gram's direction: **context words predict the target word**.

Example generated training instances from the sentence (format: context words | target word):

```
(the | quick | for | jumps, brown)
(quick | brown | jumps | over, fox)
(brown | fox | over | the, jumps)
(for | jumps | the | lazy, over)
(jumps | over | lazy | dog, the)
```

### 3.2 Modelling (p. 149)

**Setup:** current window _"The quick brown fox jumps"_, need to formulate $P(\text{brown}\mid\text{the},\text{quick},\text{fox},\text{jumps})$.

Let $tw = \text{brown}$ and $cw_1=\text{the}, cw_2=\text{quick}, cw_3=\text{for}, cw_4=\text{jumps}$.

**Same two matrices as Skip-gram:**

$$U \in \mathbb{R}^{N\times d}, \qquad V \in \mathbb{R}^{d\times N}$$

**Forward propagation — the key difference from Skip-gram is the averaging step:**

$$h = \frac{\mathbf{1}_{c_1}+\dots+\mathbf{1}_{c_4}}{4} U = \frac{1}{4}\big(U_{c_1}^r + \dots + U_{c_4}^r\big) = \overline{U^r} \in \mathbb{R}^{1\times d}$$

$$o = hV \in \mathbb{R}^{1\times N}, \qquad p = \text{softmax}(o) \in \mathbb{R}^{1\times N}$$

**Probability of the target word given all context words:**

$$P(\text{brown}\mid\text{the},\text{quick},\text{fox},\text{jumps}) = p_t$$

$$\log P(\text{brown}\mid\text{the},\text{quick},\text{fox},\text{jumps}) = \log p_t = \overline{U^r}V_t^c - \log\left(\sum_{k=1}^{N}\exp(\overline{U^r}V_k^c)\right)$$

Train by **maximizing log-likelihood**, same as Skip-gram.

> **Skip-gram vs. CBOW — key structural difference:** Skip-gram takes **one** target word's embedding ($h = U_t^r$) and predicts **multiple** context words (one softmax per context word in the window). CBOW **averages multiple** context word embeddings into a single $h = \overline{U^r}$ and predicts **one** target word (single softmax per window).

### 3.3 CBOW Forward Propagation — Worked Example (p. 150)

**Setup:** context words "the" (index 7, one-hot $\mathbf{1}_7$) and "quick" (index 6, one-hot $\mathbf{1}_6$) — averaging over 2 context words in this simplified example — predicting center word "brown".

**Step 1 — Averaged hidden layer:**

$$h = \frac{\mathbf{1}_1 + \mathbf{1}_7}{2}U = [0.75,\ 0.15,\ 0.4]$$

_(Note: the slide's specific indices used in the average may reflect the two nearest context words in that particular worked window.)_

**Step 2 — Output scores:**

$$o = Vh = [0.55,\ 0.44,\ 0.93,\ 0.40,\ 0.46,\ 0.94,\ 0.60,\ 0.97]^T$$

**Step 3 — Softmax:**

$$p(o\mid i) = [.11,\ .10,\ .16,\ .09,\ .10,\ .16,\ .11,\ .17]^T$$

**Step 4 — CE loss** against the one-hot target vector for "brown" (index 1).

```python
import torch
import torch.nn as nn

class CBOW(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.U = nn.Embedding(vocab_size, embed_dim)
        self.V = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, context_idxs):          # context_idxs: (batch, num_context_words)
        h = self.U(context_idxs).mean(dim=1)  # average context embeddings -> (batch, embed_dim)
        o = self.V(h)                          # (batch, vocab_size)
        return o

criterion = nn.CrossEntropyLoss()
# loss = criterion(model(context_idxs), target_idx)
```

---

## 4. Advanced Sequential Models: Encoder-Decoder (Seq2Seq)

### 4.1 Encoder-Decoder Overview (p. 152)

**General framework:** transform **source structured data** into a fixed **context/encoding vector** $c$, then **decode** $c$ into **target structured data**.

$$c = h_{last}$$

**Applications:**

- **Image captioning:** CNN encodes image → feature vector $c$ → LSTM decoder generates caption words sequentially
- **Machine translation:** RNN/LSTM encoder reads source sentence → $c$ → RNN/LSTM decoder generates target sentence
- Many more: C++ → Java code translation, text → image generation, question answering (questions → answers), etc.

### 4.2 Seq2Seq Problem Statement (p. 153)

- **Source sequence:** $x = (x_1, x_2, \dots, x_{T_x})$ where $x_i \in V_x$ (source vocabulary)
- **Target sequence:** $y = (y_1, y_2, \dots, y_{T_y})$ where $y_i \in V_y$ (target vocabulary)
- **Training set:**

$$\mathfrak{D} = \left{\big(x^{(1)}, y^{(1)}\big), \big(x^{(2)}, y^{(2)}\big), \dots, \big(x^{(N)}, y^{(N)}\big)\right} = \left{\big(x^{(i)}, y^{(i)}\big)\right}_{i=1}^{N}$$

- $S_x$: set of possible source sequences ($x^{(i)} \subseteq S_x$)
- $S_y$: set of possible target sequences ($y^{(i)} \subseteq S_y$)
- **Task:** learn a function $f: S_x \to S_y$

**Applications:** Machine Translation, Image Captioning.

### 4.3 Machine Translation Example (p. 154)

**Task:** translate English → French.

- Input: "He loved to eat." (English)
- Output: "Il aimait manger." (French)

**Unrolled structure:**

- **Encoder** (reads source): $h_1 (\text{He}) \to h_2 (\text{loved}) \to h_3 (\text{to}) \to h_4 (\text{eat})$, producing $c = h_4$
- **Decoder** (generates target, conditioned on $c$): starts at $q_0$ with $\langle\text{BOS}\rangle$, produces $q_1 \to \text{Il}$, $q_2 \to \text{aimait}$, $q_3 \to \text{manger}$, then $\langle\text{EOS}\rangle$

At each decoder step, the output is a distribution over "vocabulary size + 1" classes (the $+1$ accounts for the special $\langle\text{EOS}\rangle$ token).

### 4.4 Encoder-Decoder Architecture (p. 155)

**Encoder:**

- Produces the **context vector** $c = h_{T_x}$ of the input sequence
- $c$ summarizes the entire input sequence $[x_1,\dots,x_{T_x}]$

**Decoder:**

- Decodes $c$ into the output sequence, one token at a time

**Special symbols:**

- $\langle\text{EOS}\rangle$ signifies the **end** of a sequence
- $\langle\text{BOS}\rangle$ signifies the **beginning** of a sequence

**Structure:**

$$\underbrace{h_1 \to h_2 \to \dots \to h_{T_x-1} \to h_{T_x}}_{\text{Encoder}} = c \quad\longrightarrow\quad \underbrace{q_0(\langle\text{BOS}\rangle) \to q_1(y_1) \to \dots \to q_{T_y}(y_{T_y}) \to \langle\text{EOS}\rangle}_{\text{Decoder}}$$

Each decoder step $q_j$ takes the **previous generated token** $y_{j-1}$ (or $\langle\text{BOS}\rangle$ initially) as input and produces the next token $y_j$.

---

## 5. Training of Seq2Seq (p. 156–157)

### 5.1 Objective: Maximize Log-Likelihood

$$\max_{\theta} \ J(\theta) = \sum_{(x,y)\in\mathfrak{D}} \log P(y\mid x,\theta)$$

where $\theta = [\theta_e, \theta_d]$ — $\theta_e$ (encoder parameters) and $\theta_d$ (decoder parameters).

### 5.2 Product Rule Decomposition

$$P(y\mid x,\theta) = P(y_{1:T_y}\mid x_{1:T_x},\theta) = P(y_{1:T_y}\mid c,\theta)$$

$$= P(y_1\mid c,\theta), P(y_2\mid y_1,c,\theta), \cdots, P(y_j\mid y_{1:j-1},c,\theta), \cdots, P(y_{T_y}\mid y_{1:T_y-1},c,\theta)$$

$$= \prod_{j=1}^{T_y} P(y_j \mid y_{1:j-1}, c, \theta)$$

**Taking logs:**

$$\log P(y\mid x,\theta) = \log P(y\mid c,\theta) = \sum_{j=1}^{T_y}\log P(y_j\mid y_{1:j-1},c,\theta) = \sum_{j=1}^{T_y}\log P(y_j\mid q_{j-1},c,\theta)$$

- $q_{j-1}$ is the decoder's hidden state, which implicitly **contains the information of $y_{1:j-1}$** (via the recurrence)
- We compute $P(y_j\mid q_{j-1},c) = g(y_j, q_{j-1}, c)$ where $g$ is a **nonlinear, potentially multi-layered NN** (e.g. an LSTM/GRU cell followed by softmax) outputting the probability of $y_j$

### 5.3 How $c$ Is Incorporated: Two Variants

The slide highlights **two ways** the context vector $c$ can be used in the decoder recurrence — pay attention to how $c$ enters at every step:

**Variant (1)** [Sutskever et al. 2014]:

$$P(y_j\mid q_{j-1},c,\theta) = P(y_j\mid q_{j-1},\theta), \qquad q_{j-1} = f(q_{j-2}, y_{j-1})$$

- Here $c$ is used **only once**, to **initialize** the decoder's first hidden state ($c = h_{T_x}$ feeds directly into $q_0$); it is not explicitly re-injected at every subsequent step.

**Variant (2):**

$$P(y_j\mid q_{j-1},c,\theta) = P(y_j\mid q_{j-1},\theta), \qquad q_{j-1} = f(q_{j-2}, \text{concat}(c, y_{j-1}))$$

- Here $c$ is **concatenated with the previous output token** $y_{j-1}$ at **every** decoder time step, so the decoder is explicitly reminded of the full source context throughout generation (not just at initialization).

In both variants, $f$ is a **memory cell** (e.g., LSTM or GRU).

### 5.4 Big Drawback of Basic Seq2Seq

> **The context vector $c$ is fixed across all decoding timesteps** — a single fixed-length vector must summarize the _entire_ (potentially very long) source sequence, creating an information bottleneck. This motivates the **attention mechanism** (to be covered in a subsequent chunk), which allows the decoder to dynamically attend to different parts of the source sequence at each decoding step rather than relying on one static $c$.

```python
import torch
import torch.nn as nn

class Seq2SeqEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

    def forward(self, x):                       # x: (batch, T_x)
        x = self.embedding(x)                    # (batch, T_x, embed_size)
        _, (h_n, c_n) = self.lstm(x)             # h_n: (1, batch, hidden_size) -- this is c = h_{T_x}
        return h_n, c_n

class Seq2SeqDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, y_prev_token, hidden_state):   # single-step decoding
        y_emb = self.embedding(y_prev_token)          # (batch, 1, embed_size)
        output, hidden_state = self.lstm(y_emb, hidden_state)
        logits = self.fc(output.squeeze(1))           # (batch, vocab_size)
        return logits, hidden_state

# Training loop sketch (teacher forcing):
# h_n, c_n = encoder(x)                 # context vector c = h_n
# hidden = (h_n, c_n)                   # decoder initialized with encoder's final state (Variant 1 style)
# for j in range(T_y):
#     logits, hidden = decoder(y[:, j:j+1], hidden)
#     loss += criterion(logits, y[:, j+1])   # predict next token
```

---

## 6. Inference for Seq2Seq (p. 158–160)

### 6.1 Inference Overview (p. 158)

**Given:** a trained model and an input sequence $x$. **Need:** infer the corresponding output sequence $y$.

**Two common strategies:** **Greedy Decoding** and **Beam Search Decoding**.

At each decoder step $j$, we know the probability distribution $P(y_j = \circ \mid c)$ (or more precisely, conditioned on $c$ and all previously generated tokens) — the question is how to pick a full sequence $y_1, \dots, y_{T_y}$ from these per-step distributions.

### 6.2 Greedy Decoding (p. 159)

**Algorithm:**

1. Given $x$, find word $y_1$ with **highest probability** $P(y_1=\circ\mid c)$
2. Given $y_1$ and $x$, find word $y_2$ with **highest probability**
3. Continue…
4. **Stop** when $\langle\text{EOS}\rangle$ is generated

> **Limitation:** greedy decoding commits to the single best token at each step without considering how that choice affects future steps — this can lead to a **locally optimal but globally suboptimal** overall sequence (a classic "greedy algorithm" pitfall).

```python
def greedy_decode(decoder, encoder_hidden, bos_token, eos_token, max_len=50):
    hidden = encoder_hidden
    y_prev = bos_token
    output_sequence = []
    for _ in range(max_len):
        logits, hidden = decoder(y_prev, hidden)
        y_next = logits.argmax(dim=-1)          # pick highest-probability token
        if y_next.item() == eos_token:
            break
        output_sequence.append(y_next.item())
        y_prev = y_next.unsqueeze(1)
    return output_sequence
```

### 6.3 Beam Search Decoding (p. 160)

**Algorithm with beam width $k$:**

1. Given $x$, find **$k$ candidates** for $y_1$ with highest probability
2. For **each** candidate $y_1$, find $k$ candidates for word $y_2$ with highest probability (giving up to $k\times k$ combined candidates)
3. Pick the **top-$k$ sequences** $y_1y_2$ with highest **joint probability**
4. For each surviving $y_1y_2$, find $k$ candidates for word $y_3$
5. Pick top-$k$ sequences $y_1y_2y_3$ with highest joint probability
6. … repeat …
7. **Stop** when $\langle\text{EOS}\rangle$ is seen on each beam
8. Finally, **pick the single sequence** with highest overall probability from the surviving top-$k$ sequences

> **Beam search vs. greedy:** beam search is a middle ground between greedy decoding ($k=1$) and exhaustive search (all possible sequences) — it keeps $k$ partial hypotheses alive at each step, which usually produces higher-quality (higher joint-probability) sequences than greedy decoding, at the cost of $k\times$ more computation per step.

```python
import torch
import torch.nn.functional as F

def beam_search_decode(decoder, encoder_hidden, bos_token, eos_token, beam_width=3, max_len=50):
    # Each beam entry: (sequence_so_far, cumulative_log_prob, hidden_state)
    beams = [([bos_token], 0.0, encoder_hidden)]
    completed = []

    for _ in range(max_len):
        candidates = []
        for seq, log_prob, hidden in beams:
            if seq[-1] == eos_token:
                completed.append((seq, log_prob))
                continue
            logits, new_hidden = decoder(torch.tensor([[seq[-1]]]), hidden)
            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
            topk_log_probs, topk_tokens = log_probs.topk(beam_width)
            for lp, tok in zip(topk_log_probs, topk_tokens):
                candidates.append((seq + [tok.item()], log_prob + lp.item(), new_hidden))

        # Keep only the top-k candidates by cumulative joint log-probability
        candidates.sort(key=lambda x: x[1], reverse=True)
        beams = candidates[:beam_width]

        if all(seq[-1] == eos_token for seq, _, _ in beams):
            break

    completed.extend([(seq, lp) for seq, lp, _ in beams])
    best_seq, _ = max(completed, key=lambda x: x[1])
    return best_seq
```

---

## Cross-Topic Connections

|Concept|Connects to|Relationship|
|---|---|---|
|Skip-gram/CBOW two-matrix architecture (p.143, 149)|Sentiment analysis embedding lookup (pages 121–140)|Both use a lookup matrix $U$ selecting rows via one-hot multiplication — but Word2Vec's $U$ is the _end product_ being learned, while the sentiment model's embedding was a _means_ to a downstream classification task|
|Skip-gram/CBOW softmax + CE loss (p.143–150)|Softmax + cross-entropy (pages 1–80), label smoothing (p.100)|Same fundamental $CE(\mathbf{1}_y, p)$ framework, applied here to predicting words instead of image classes|
|Negative sampling's sigmoid trick (p.147)|Binary classification via logistic regression (foundational ML)|Converts an $N$-way softmax into $k+1$ independent binary sigmoid decisions — a classic way to sidestep expensive normalization|
|Seq2seq context vector $c=h_{T_x}$ (p.152–155)|RNN/LSTM hidden states (pages 115–140)|Directly reuses the "final hidden state summarizes the whole sequence" idea from many-to-one RNN architectures (e.g. sentiment analysis, p.138)|
|Seq2seq's fixed-$c$ drawback (p.157)|RNN's long-term dependency problem (p.125)|Same underlying theme: compressing arbitrarily long information into one fixed-size vector loses information — this motivates both LSTM gating (local fix) and attention (global fix, upcoming)|
|Greedy vs. beam search decoding (p.159–160)|—|Analogous to the classic tradeoff between greedy algorithms and search-based algorithms in general CS — sacrificing some computation for a better (though not guaranteed globally optimal) solution|

---

## Quick-Reference Formula Sheet

$$\text{Skip-gram: } h = \mathbf{1}_tU = U_t^r,\quad o=hV,\quad p=\text{softmax}(o),\quad P(cw\mid tw)=p_c$$

$$\text{CBOW: } h = \frac{1}{4}\sum_{i=1}^{4}U_{c_i}^r = \overline{U^r},\quad o=hV,\quad p=\text{softmax}(o),\quad P(tw\mid cw_{1:4})=p_t$$

$$\text{Negative sampling: } \max\big[\log p_c - \alpha\log p_{n_1}-\alpha\log p_{n_2}\big]$$

$$\text{Seq2seq objective: } \max_\theta \sum_{(x,y)\in\mathfrak{D}} \log P(y\mid x,\theta) = \sum_{j=1}^{T_y}\log P(y_j\mid q_{j-1},c,\theta)$$

$$\text{Context vector: } c = h_{T_x}$$