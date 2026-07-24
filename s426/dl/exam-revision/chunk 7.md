

## tags: [FIT5215, deep-learning, RNN, LSTM, GRU, sentiment-analysis, word2vec] pages: 121-140

# FIT5215 — Deep Learning: Pages 121–140

## 1. Shape Transformation Across a Full Sequence (p. 121)

Generalizing the 2-timestep example (pp. 116–120) to a full sequence of length $L$:

**Per-timestep chain:** $x_t \xrightarrow{U} h_t \xrightarrow{V} \hat{y}_t$, with $h_{t-1} \xrightarrow{W} h_t$ connecting consecutive hidden states.

**Shape pipeline (batched):**

|Stage|Shape|
|---|---|
|Raw input sequence|$[seq_len, batch_size, input_size]$ or $[batch_size, seq_len, input_size]$|
|After $U$ (input→hidden) at each $t$|$[batch_size, hidden_size]$ per step → stacked: $[batch_size, seq_len, hidden_size]$|
|After $V$ (hidden→output) at each $t$|$[batch_size, num_classes]$ per step|

**Layout conversion:** `torch.transpose(X, 0, 1)` converts between $[seq_len, batch_size, input_size]$ ("sequence-first") and $[batch_size, seq_len, input_size]$ ("batch-first") conventions — matching PyTorch's `batch_first=True/False` argument in `nn.RNN`/`nn.LSTM`/`nn.GRU`.

> **Key takeaway:** regardless of sequence length, the _same_ parameters $U, W, V$ (and biases $b, c$) are reused at every time step — this is the parameter-sharing principle that makes RNNs scale to arbitrary-length sequences.

---

## 2. Deeper (Stacked) RNNs (p. 122)

**Idea:** stack multiple RNN layers on top of each other — the hidden states of layer 1 become the _inputs_ to layer 2, and so on.

- Layer 1 hidden states: $h_t^{1}$ (computed from $x_t$ and $h_{t-1}^{1}$)
- Layer 2 hidden states: $h_t^{2}$ (computed from $h_t^{1}$ and $h_{t-1}^{2}$)
- Layer 3 hidden states: $h_t^{3}$ (computed from $h_t^{2}$ and $h_{t-1}^{3}$)
- Output $\hat{y}_t$ computed from the **top layer's** hidden state $h_t^{3}$

> **Intuition:** just like stacking layers in a feedforward/CNN increases representational capacity, stacking RNN layers lets the network learn hierarchical temporal features — lower layers might capture local/short-range patterns, higher layers more abstract/long-range patterns.

```python
import torch.nn as nn

# 3-layer stacked RNN, equivalent to the diagram on p.122
stacked_rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=3,
                      nonlinearity='tanh', batch_first=True)
# Or for LSTM/GRU:
stacked_lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=3, batch_first=True)
```

---

## 3. RNN Architecture Zoo (p. 123–124)

RNNs are flexible enough to handle inputs/outputs of varying length combinations:

|Architecture|Structure|Example tasks|
|---|---|---|
|**Many-to-one**|Multiple inputs $x_1,\dots,x_T$ → single output $y$|Sentiment analysis, image classification|
|**One-to-many**|Single input $x$ → multiple outputs $y_1,\dots,y_T$|Image captioning|
|**Many-to-many (1)**|Inputs and outputs both sequences, often different lengths, output produced only after reading full input|Machine translation|
|**Many-to-many (2)**|Inputs and outputs are sequences of the **same length**, output produced at every input step|Video classification (frame-by-frame), part-of-speech tagging|

> **Note on many-to-many (1) vs (2):** in translation (1), the model typically needs to see the entire source sentence before generating any target words (encoder-decoder structure) — input and output lengths can differ. In video classification (2), a prediction is made at every time step in lock-step with the input (input and output lengths match).

---

## 4. Problems with Basic RNNs (p. 125)

### 4.1 Long-Term Dependency Limitation

- A basic RNN's hidden state $h_t$ is computed from **only the immediately preceding state** $h_{t-1}$ and current input $x_t$
- This means information from many steps back must survive being repeatedly compressed and transformed — in practice, basic RNNs **struggle with long-term dependencies**

**Illustrative examples from the slide:**

- _"The **clouds** are in the **sky**"_ — predicting "sky" only requires looking back a few words ("clouds") — a **short-term dependency**, well within basic RNN capability.
- _"I was **born** in **Vietnam**... I speak fluent **Vietnamese**"_ — predicting "Vietnamese" requires remembering "Vietnam" from much earlier in the sequence — a **long-term dependency**, which basic RNNs handle poorly.

### 4.2 Modelling Drawbacks

- **Vanishing gradient problem**: a technical training issue for long sequences
- Many layers of nonlinear transformation (one per time step, when unrolled) prevent gradients (and data signals) from flowing easily backward through the network — analogous to vanishing gradients in very deep feedforward networks (pages 1–80), but here "depth" comes from **sequence length**, not layer count

### 4.3 Solution: Gating Mechanisms

- Add a **linear component** carried forward from the previous state (bypassing the repeated nonlinear squashing)
- This upgrades: **basic RNN → LSTM / GRU**

---

## 5. Memory Cells (p. 126)

**Basic RNN cell recap:**

- Input to cell: $h_{t-1}$ (previous hidden state), $x_t$ (current input token)
- Output: $h_t = \tanh(x_t U + h_{t-1}W + b)$
- $h_t$ can only capture **short-term dependency** → acts as **short-term memory**

**Question:** how to capture long-term memory more efficiently? → **LSTM cell** and **GRU cell**

**LSTM's key structural idea:** maintain **two separate state vectors**:

- $c_t$ — **long-term memory** (carried via a mostly-linear path, allowing gradients to flow across many steps largely unimpeded)
- $h_t$ — **short-term memory** (similar role to the basic RNN's hidden state)

---

## 6. Long Short-Term Memory (LSTM) (p. 127–129)

### 6.1 Structure Overview

At each time step, the LSTM cell receives $x_t$, $h_{t-1}$ (short-term state), and $c_{t-1}$ (long-term state), and outputs updated $h_t$, $c_t$, and prediction $\hat{y}_t$.

### 6.2 Full Gate Equations (p. 128)

**Candidate cell content** (like the basic RNN's hidden state computation):

$$g_t = \tanh(h_{t-1}W + x_t U + b)$$

**Forget gate** — decides how much of the previous long-term memory $c_{t-1}$ to keep:

$$f_t = \sigma\big(x_t U^f + h_{t-1}W^f + b^f\big)$$

**Input gate** — decides how much of the new candidate $g_t$ to write into memory:

$$i_t = \sigma\big(x_t U^i + h_{t-1}W^i + b^i\big)$$

**LSTM long-term state update:**

$$c_t = f_t \odot c_{t-1} + g_t \odot i_t$$

**Output gate** — decides how much of the (squashed) long-term memory to expose as the short-term state:

$$o_t = \sigma\big(x_t U^o + h_{t-1}W^o + b^o\big)$$

**LSTM short-term state:**

$$h_t = o_t \odot \tanh(c_t)$$

**LSTM output:**

$$ \hat{y}_t = \begin{cases} h_t V + c & \text{(regression)} \ \text{softmax}(h_t V + c) & \text{(classification)} \end{cases} $$

> **Note on notation clash:** the output bias is also called $c$ in the regression/classification formula — this is a **different** $c$ from the cell state $c_t$; the slide reuses the symbol from the basic RNN's output-layer bias convention (see p.119–120).

### 6.3 Gate Summary Table

|Gate/quantity|Formula|Role|
|---|---|---|
|Forget gate $f_t$|$\sigma(x_tU^f + h_{t-1}W^f+b^f)$|Controls how much old long-term memory ($c_{t-1}$) to retain|
|Input gate $i_t$|$\sigma(x_tU^i+h_{t-1}W^i+b^i)$|Controls how much new candidate info ($g_t$) to add|
|Candidate $g_t$|$\tanh(h_{t-1}W+x_tU+b)$|New candidate content to potentially store|
|Output gate $o_t$|$\sigma(x_tU^o+h_{t-1}W^o+b^o)$|Controls how much of $c_t$ to expose as $h_t$|
|Long-term state $c_t$|$f_t\odot c_{t-1}+g_t\odot i_t$|The persistent "conveyor belt" memory|
|Short-term state $h_t$|$o_t\odot\tanh(c_t)$|The exposed working memory / output basis|

### 6.4 LSTM Summary (p. 129)

- LSTM belongs to a class of **gated RNN models**
- LSTM introduces **self-loops** creating paths where the gradient can flow for long durations (the $c_t = f_t\odot c_{t-1} + \dots$ update is largely **additive/linear** in $c_{t-1}$, unlike the fully nonlinear basic RNN recurrence — this is what mitigates vanishing gradients)
- Improvements over basic RNN cell:
    - Can capture **long-term dependency**
    - **Faster and more robust to train**, often with quicker convergence
- LSTM manages **two state vectors** kept separate by default: $h_t$ (short-term), $c_t$ (long-term)
- **Gates** (forget, input, output) can remove or add information to the cell state

```python
import torch.nn as nn

lstm_cell = nn.LSTMCell(input_size=10, hidden_size=20)
# Manual step: h_t, c_t = lstm_cell(x_t, (h_prev, c_prev))

# Or for a full sequence:
lstm = nn.LSTM(input_size=10, hidden_size=20, batch_first=True)
# output, (h_n, c_n) = lstm(x_seq)   # x_seq: [batch, seq_len, input_size]
```

---

## 7. Gated Recurrent Unit (GRU) (p. 130–131)

**Proposed by:** Cho et al., 2014 — originally for Encoder-Decoder networks (machine translation).

**Paper:** Cho et al., _Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation_, EMNLP 2014.

### 7.1 GRU as a Simplified LSTM

- Long-term state $c_t$ and short-term state $h_t$ are **merged into a single state** $h_t$
- A **single update gate** $z_t$ controls **both** the forget and input roles simultaneously:
    - $z_t = 1$: open input gate, close forget gate (fully replace memory with new content)
    - $z_t = 0$: close input gate, open forget gate (fully retain old memory)
    - i.e., whenever a memory location is written to, it must first be "erased" — forget and input are complementary, not independent
- The **output gate is removed** entirely
- An additional **reset gate** $r_t$ controls how much of the previous state is carried forward into the _candidate_ computation

### 7.2 GRU Equations (p. 131)

**Update gate** — decides how much the unit updates its state:

$$z_t = \sigma\big(x_t U^z + h_{t-1}W^z\big)$$

**Reset gate** — controls which parts of the previous state are used to compute the candidate:

$$r_t = \sigma\big(x_t U^r + h_{t-1}W^r\big)$$

**Candidate state** (pre-computed, using the reset gate to modulate $h_{t-1}$'s contribution):

$$g_t = \tanh\big(x_t U^g + (r_t \odot h_{t-1})W^g\big)$$

**Memory state update** — linear interpolation between old state and candidate:

$$h_t = (1-z_t)\odot h_{t-1} + z_t \odot g_t$$

**Reduction to basic RNN:** when $z_t$ and $r_t$ are both close to 1, the GRU update reduces to something resembling the basic RNN recurrence (candidate fully replaces state, full previous state used).

```python
import torch.nn as nn

gru = nn.GRU(input_size=10, hidden_size=20, batch_first=True)
# output, h_n = gru(x_seq)   # x_seq: [batch, seq_len, input_size]
```

> **LSTM vs GRU — quick comparison:**
> 
> - LSTM: 3 gates (forget, input, output) + 2 states ($h_t$, $c_t$) → more expressive, more parameters
> - GRU: 2 gates (update, reset) + 1 state ($h_t$) → simpler, fewer parameters, often comparable performance and faster to train

---

## 8. Sentiment Analysis — Worked Example

### 8.1 Problem Setup (p. 132)

**Movie review dataset** (toy example):

1. "I like this movie" (pos: 1)
2. "This is a bad movie to watch" (neg: 0)
3. "I love this movie" (pos: 1)
4. "I do not recommend you to watch this movie" (neg: 0)
5. "This movie is fantastic" (pos: 1)

**Architecture:** feed one word at a time into a chain of RNN cells; final hidden state feeds a sigmoid output layer.

$$\hat{y} \ge 0.5 \Rightarrow \text{positive}, \qquad \hat{y} < 0.5 \Rightarrow \text{negative}$$

**Key question:** how to feed _symbolic_ words into a numeric RNN? → need a **vocabulary + embedding** step.

### 8.2 Vocabulary Construction (p. 133–134)

Build a vocabulary from frequent/informative words, with an **out-of-vocabulary (OOV) bucket** for everything else:

|Word|Index|
|---|---|
|like|1|
|love|2|
|bad|3|
|fantastic|4|
|not|5|
|recommend|6|
|(OOV bucket 1: I, movie, to, pad)|7|
|(OOV bucket 2: This, is, watch)|8|

Example sentence _"This movie is fantastic <pad>"_ → indices $[8, 7, 8, 4, 7]$, represented as one-hot vectors, e.g. index 8 → $[0,0,0,0,0,0,0,1]$.

**Embedding matrix** $E \in \mathbb{R}^{8\times 6}$ (vocab_size $=8$, embed_size $=6$) — each row $E_i$ is the learned embedding vector for vocabulary index $i$.

### 8.3 Embedding Lookup Mechanics (p. 135)

For a one-hot vector $\mathbf{1}_i$ (1 at position $i$, 0 elsewhere), the embedding lookup is a matrix multiplication:

$$e = \mathbf{1}_i E \in \mathbb{R}^{1\times 6}$$

Since $\mathbf{1}_i$ is one-hot, this simply **selects row $E_i$** of the embedding matrix:

$$e_1 = \mathbf{1}_8 E = E_8 \in \mathbb{R}^{1\times 6}, \qquad [1\times 8]\times[8\times 6] = [1\times 6]$$

**For the full sequence** "This movie is fantastic <pad>" (indices 8, 7, 8, 4, 7):

$$e = \begin{bmatrix} e_1 \ e_2 \ e_3 \ e_4 \ e_5\end{bmatrix} \in \mathbb{R}^{5\times 6} = \mathbb{R}^{seq_len \times embed_size}$$

- This is equivalent to an **embedding lookup operation**: pick rows with indices $8, 7, 8, 4, 7$ directly from $E$ (no explicit one-hot multiplication needed in practice — this is exactly what `nn.Embedding` does internally, efficiently).

### 8.4 Batched Sequence Embedding (p. 136)

- Embedding for **one** sequence: $\mathbb{R}^{seq_len \times embed_size}$
- Embedding for an **entire batch** of `batch_size` sequences: $\mathbb{R}^{batch_size \times seq_len \times embed_size}$

Example: with `batch_size=2`, input indices shape is $[batch_size, seq_len]$, e.g.:

```
[[8, 7, 8, 4, 7],
 [3, 2, 8, 3, 4]]
```

which maps through the embedding layer to shape $[batch_size, seq_len, embed_size] = [2, 5, 6]$.

### 8.5 Full Shape Transformation with RNN/LSTM (p. 137)

**Pipeline shapes:**

$$[batch_size, seq_len] \xrightarrow{\text{Embedding Layer } (E: [vocab_size, embed_size])} [batch_size, seq_len, embed_size]$$

$$\xrightarrow{\text{RNN/LSTM}} [batch_size, seq_len, hidden_size\ 1]$$

Each embedded token $e_t$ feeds through $U$ into the recurrence $h_t = f(h_{t-1}W, e_t U, b)$, producing hidden states $h_1, \dots, h_5$, one per position in the sequence.

### 8.6 Full PyTorch Implementation (p. 138)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init

embed_size = 128

class Model(nn.Module):
    def __init__(self, vocab_size):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.lstm1 = nn.LSTM(embed_size, 128, batch_first=True)
        self.lstm2 = nn.LSTM(128, 128, batch_first=True)

        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):                      # x: (batch_size, seq_len)
        x = self.embedding(x)                   # (batch_size, seq_len, embed_size)
        x, (_, _) = self.lstm1(x)                # (batch_size, seq_len, 128)
        _, (x, _) = self.lstm2(x)                # (batch_size, 128)  -- final hidden state
        x = self.fc(x.view(-1, 128))             # (batch_size, 1)
        return x.squeeze(1)
```

**Shape trace, matching comments in the code:**

|Step|Shape|
|---|---|
|Raw token indices|$(batch_size, seq_len)$|
|After embedding|$(batch_size, seq_len, embed_size)$|
|After `lstm1` (full sequence output)|$(batch_size, seq_len, 128)$|
|After `lstm2` (final hidden state only, via `(x, _)`)|$(batch_size, 128)$|
|After `fc`|$(batch_size, 1)$|

> Note: `self.lstm2` returns `_, (x, _) = self.lstm2(x)` — this takes the **final hidden state** `h_n` (not the full output sequence), collapsing the sequence dimension before the final linear + sigmoid layer, appropriate for a many-to-one (sentiment classification) architecture.

---

## 9. Introduction to Word2Vec (p. 139–140)

### 9.1 Motivation

**Goal:** learn dense vector representations of words that **preserve semantic and linguistic relationships** present in a large text corpus (e.g., Wikipedia).

**Classic analogy example:**

$$\text{Canberra} : \text{Australia} = \text{Paris} : ; ??? $$

Formally, find the vector $v$ that best completes the analogy by solving:

$$\arg\min_{v} \left| \big(v_{Canberra} - v_{Australia} + v_{Paris}\big) - v \right| $$

The expected answer is $v \approx v_{France}$ — i.e., the same "capital-of" relationship vector ($v_{Canberra}-v_{Australia}$) should transfer to other country/capital pairs when embeddings capture meaningful linguistic structure.

### 9.2 The Learning Setup

- **We have:** many texts (e.g., Wikipedia corpus)
- **We want:** vector representations for words that preserve semantic/linguistic relationships carried in the text corpus
- **We need:** to devise a **pretext task** that casts learning word representations as a **supervised learning problem** (since raw text has no explicit labels for "meaning")

> This sets up the core Word2Vec idea (to be covered in detail in the next chunk): using surrounding context words to predict a target word (CBOW) or vice versa (Skip-gram), turning an unsupervised representation-learning goal into a supervised prediction task.

---

## Cross-Topic Connections

|Concept|Connects to|Relationship|
|---|---|---|
|Deeper/stacked RNNs (p.122)|Stacking layers in FFNs/CNNs (pages 1–80)|Same general principle: more layers → more representational capacity, applied along the "depth" axis rather than or in addition to the "time" axis|
|Vanishing gradients in RNNs (p.125)|Vanishing/exploding gradients, Xavier/He init (pages 1–80)|Same fundamental issue (repeated nonlinear transformations shrink/blow up gradients), but here caused by sequence length rather than network depth|
|LSTM/GRU gating (p.127–131)|Batch normalization, residual/skip connections (pages 81–100, ResNet)|All are mechanisms that create more direct paths for gradient flow — LSTM's additive $c_t$ update is conceptually similar to a ResNet skip connection, applied across time instead of across layers|
|Embedding lookup (p.133–136)|Softmax/one-hot vectors (pages 1–80, p.100 label smoothing)|Reuses the one-hot vector notation $\mathbf{1}_y$ from earlier, but here as an _input_ representation (via lookup) rather than a target label|
|Sentiment analysis LSTM (p.138)|RNN architecture zoo — many-to-one (p.123)|Concrete implementation of the many-to-one pattern: full sequence in, single scalar sentiment score out|
|Word2Vec motivation (p.140)|Embedding matrix used in sentiment analysis (p.133–136)|The sentiment analysis example used a **randomly-initialized, task-trained** embedding matrix; Word2Vec (next chunk) shows how to **pre-train** semantically meaningful embeddings independent of any downstream task|

---

## Quick-Reference Formula Sheet

$$\text{Basic RNN: } h_t = \tanh(x_tU + h_{t-1}W+b)$$

$$\text{LSTM:} \quad \begin{aligned} f_t &= \sigma(x_tU^f+h_{t-1}W^f+b^f) \ i_t &= \sigma(x_tU^i+h_{t-1}W^i+b^i) \ g_t &= \tanh(h_{t-1}W+x_tU+b) \ c_t &= f_t\odot c_{t-1}+g_t\odot i_t \ o_t &= \sigma(x_tU^o+h_{t-1}W^o+b^o) \ h_t &= o_t\odot\tanh(c_t) \end{aligned}$$

$$\text{GRU:} \quad \begin{aligned} z_t &= \sigma(x_tU^z+h_{t-1}W^z) \ r_t &= \sigma(x_tU^r+h_{t-1}W^r) \ g_t &= \tanh(x_tU^g+(r_t\odot h_{t-1})W^g) \ h_t &= (1-z_t)\odot h_{t-1}+z_t\odot g_t \end{aligned}$$

$$\text{Embedding lookup: } e = \mathbf{1}_i E = E_i \in \mathbb{R}^{1\times embed_size}$$

$$\text{Word2Vec analogy: } \arg\min_v |(v_{Canberra}-v_{Australia}+v_{Paris}) - v|$$