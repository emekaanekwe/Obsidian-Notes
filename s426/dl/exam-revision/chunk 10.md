
## tags: [FIT5215, deep-learning, transformers, multi-head-attention, positional-encoding, vision-transformers, ViT] pages: 181-200

# FIT5215 — Deep Learning: Pages 181–200

## 1. Multi-Head Self-Attention

### 1.1 Main Idea (p. 181)

- Perform self-attention **multiple times in parallel** and **combine** the results
- Uses **multiple attention heads**, each with its own set of $Q, K, V$ matrices ($W^Q_h, W^K_h, W^V_h$ per head $h$)
- Each attention head performs attention **independently**
- This allows the model to **attend to different parts of the sequence differently** — e.g., one head might capture syntactic relationships while another captures semantic ones

### 1.2 Matrix Calculations — One-Head vs. Multi-Head (p. 183)

**One-Head computation** (as covered in Section 9 of the previous chunk):

$$Q = XW^Q, \qquad K = XW^K, \qquad V = XW^V$$

$$Z = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Multi-Head computation:**

For each head $i \in {0, 1, \dots, 7}$ (8 heads in the original Transformer):

$$Q_i = XW_i^Q, \qquad K_i = XW_i^K, \qquad V_i = XW_i^V$$

$$Z_i = \text{softmax}\left(\frac{Q_iK_i^T}{\sqrt{d_k}}\right)V_i$$

**Combining heads — 3-step procedure:**

1. **Concatenate** all attention heads' outputs: $Z_0, Z_1, \dots, Z_7$ side by side into one wide matrix
2. **Multiply** by a weight matrix $W^O$ (trained jointly with the model) to project back down to the model dimension
3. The result is the final $Z$ matrix that **captures information from all attention heads**, which is then passed forward to the FFN

$$Z_{\text{combined}} = \text{Concat}(Z_0, Z_1, \dots, Z_7), W^O$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.WQ = nn.Linear(d_model, d_model, bias=False)
        self.WK = nn.Linear(d_model, d_model, bias=False)
        self.WV = nn.Linear(d_model, d_model, bias=False)
        self.WO = nn.Linear(d_model, d_model, bias=False)

    def forward(self, X):                              # X: (batch, seq_len, d_model)
        batch_size, seq_len, d_model = X.shape

        Q = self.WQ(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.WK(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.WV(X).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Q, K, V: (batch, num_heads, seq_len, d_k)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, num_heads, seq_len, seq_len)
        A = F.softmax(scores, dim=-1)
        Z = torch.matmul(A, V)                          # (batch, num_heads, seq_len, d_k)

        Z = Z.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)  # concatenate heads
        Z = self.WO(Z)                                  # final linear projection
        return Z, A
```

---

## 2. Residual Connections and Layer Normalization (p. 184)

### 2.1 Layer Norm vs. Batch Norm

**Layer normalization (LN)** is the same idea as **batch normalization** (p. 93–96), except LN normalizes **across the feature dimension** rather than across the batch dimension.

> **Key practical note:** Batch normalization is usually **empirically less effective than layer normalization in NLP tasks**, whose inputs are often **variable-length sequences** — batch statistics become unstable or ill-defined when sequence lengths vary within/across batches, whereas layer norm's per-example, per-position normalization is unaffected by this.

### 2.2 Full Layer Norm Formula

Given the residual sum $Y = X + Z$ (input $X$ plus self-attention output $Z$), with shape $[batch_size, seq_len, embed_size]$:

$$Y_m = \text{mean}(Y,\ \text{axis}=embed_size,\ \text{keep_dims}=\text{True})$$

$$Y_\sigma = \text{std}(Y,\ \text{axis}=embed_size,\ \text{keep_dims}=\text{True})$$

$$\text{LayerNorm}(Y) = \gamma\frac{Y - Y_m}{Y_\sigma} + \beta$$

where $\gamma, \beta$ are **learnable scale and shift parameters** (directly analogous to batch norm's $\gamma, \beta$ from p.95).

> **Key distinction from Batch Norm:** BN computes $\mu_B, \sigma_B$ across the **batch** dimension (same feature, across examples); LN computes $Y_m, Y_\sigma$ across the **feature/embedding** dimension (same example, across features) — this makes LN's statistics independent of batch composition and sequence length, which is exactly why it suits NLP's variable-length sequences.

### 2.3 Residual (Skip) Connection Placement

Within each encoder block, the residual pattern is applied **twice**:

1. After **Multi-Head Self-Attention**: $\text{LayerNorm}(X + Z)$, where $X$ is the block's input and $Z$ is the self-attention output
2. After the **Point-wise Feed-Forward NN**: $\text{LayerNorm}(\bar{Z}_{\text{prev}} + \text{FFN output})$

```python
import torch
import torch.nn as nn

class AddNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(d_model, eps=eps)  # PyTorch's built-in handles gamma, beta, mean, std internally

    def forward(self, X, sublayer_output):
        return self.ln(X + sublayer_output)   # residual connection + layer norm
```

---

## 3. Point-wise Feed-Forward NN (p. 185)

**Placement:** applied **independently to each position** (each token) in the sequence, after the Add & Normalize step following self-attention, and followed by another Add & Normalize before passing to the **next encoder**.

**Formula:**

$$\bar{Z} = \text{ReLU}(W_1 Z + b_1), W_2 + b_2$$

- Applied identically (same $W_1, b_1, W_2, b_2$) to every position $z_1, z_2, \dots, z_T$ independently — hence "point-wise"
- Typically expands to a larger hidden dimension inside (via $W_1$) then projects back down to $d_model$ (via $W_2$)

**Full block flow:**

$$x_1, x_2, \dots, x_T \xrightarrow{\text{Multi-head Self-attention}} z_1, z_2, \dots, z_T \xrightarrow{\text{FFN (per-position, shared weights)}} \bar{z}_1, \bar{z}_2, \dots, \bar{z}_T \to \text{LayerNorm}(\bar{Z}) \to \text{Next Encoder}$$

```python
import torch.nn as nn

class PointwiseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, Z):                    # Z: (batch, seq_len, d_model)
        return self.fc2(self.relu(self.fc1(Z)))   # applied identically at every seq_len position
```

---

## 4. Positional Encoding (p. 186)

### 4.1 Motivation

- The Transformer has **no recurrence** and **no convolution** — all tokens/embeddings are fed in **simultaneously**
- Without any additional signal, the model has **no information about word order** (e.g., embeddings $E_3, E_2, E_5, E_8$ for "I am a student" carry no positional information on their own)
- **Solution:** inject information about the **relative or absolute position** of each token in the sequence

### 4.2 Sinusoidal Positional Encoding Formulas

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{1000^{2i/embed_size}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{1000^{2i/embed_size}}\right)$$

- $pos$ = the position of the token in the sequence (1, 2, 3, …)
- $i$ = the dimension index within the embedding vector (even dimensions use sine, odd dimensions use cosine)
- $embed_size$ = the total embedding dimension

**Worked example** (embed_size = 3, sentence "I am a student"):

|Word|Vocab index|Position|Embedding|Positional encoding added|
|---|---|---|---|---|
|I|3|pos=1|$E_{31}, E_{32}, E_{33}$|$PE_{(1,1)}, PE_{(1,2)}, PE_{(1,3)}$|
|am|2|pos=2|$E_{21}, E_{22}, E_{23}$|$PE_{(2,1)}, PE_{(2,2)}, PE_{(2,3)}$|
|a|5|pos=3|$E_{51}, E_{52}, E_{53}$|$PE_{(3,1)}, PE_{(3,2)}, PE_{(3,3)}$|
|student|8|pos=4|$E_{81}, E_{82}, E_{83}$|$PE_{(4,1)}, PE_{(4,2)}, PE_{(4,3)}$|

The final input to the Transformer at each position is: **embedding + positional encoding** (element-wise sum), as shown in the architecture diagram's "⊕" symbol right after Input/Output Embedding.

> **Why sinusoidal functions?** Different dimensions oscillate at different frequencies (varying wavelengths across dimensions, as seen in the plotted sine/cosine curves for different columns). This gives each position a **unique** encoding pattern, and critically, allows the model to learn to attend by **relative** position (since $PE_{pos+k}$ can be expressed as a linear function of $PE_{pos}$ for fixed $k$, a property of sinusoids) even though the encoding itself is fixed (not learned).

```python
import torch
import math

def positional_encoding(seq_len, embed_size):
    pe = torch.zeros(seq_len, embed_size)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)          # (seq_len, 1)
    div_term = torch.exp(torch.arange(0, embed_size, 2).float() *
                          (-math.log(1000.0) / embed_size))                       # per the 1000^(2i/embed_size) term
    pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions
    pe[:, 1::2] = torch.cos(position * div_term)   # odd dimensions
    return pe   # (seq_len, embed_size), to be added elementwise to token embeddings
```

---

## 5. Transformer's Cross-Attention Mechanism (Encoder-Decoder Attention)

### 5.1 Overview (p. 187)

In the **decoder**, besides self-attention over the target sequence, there is a second attention sub-layer that attends **from the target sequence to the source sequence** — this is the direct architectural descendant of the RNN-based attention mechanisms (Global/Local attention, p.163–169).

**Structure:**

- **Queries** come from the **target sequence** ("Il", "gatto", "salta", …) — i.e., "target tokens from the point of view of the source sequence"
- **Keys & Values** come from the **source sequence** ("The", "cat", "jumps", "the", …), each processed through its own FFN
- Attention computed via: **dot product** → **Norm & Softmax** → weighted combination of values

> This is exactly the query/key/value framework of self-attention (Section 9, previous chunk), but here **queries** originate from one sequence (decoder/target) while **keys and values** originate from a **different** sequence (encoder/source) — hence "cross-attention" as opposed to "self-attention" (where Q, K, V all come from the same sequence).

### 5.2 Alternative Perspective: Lookup Table / Soft-Matching (p. 188)

A more intuitive framing: think of the source sequence as populating a **lookup table** of (Key, Value) pairs:

|Key|Value|
|---|---|
|"A4"|(vector)|
|"N9"|(vector)|
|"O7"|(empty/low relevance)|
|"A4"|(vector)|
|"N2"|(vector)|

For a target query (e.g., generating "gatto", query = "N5"), the mechanism performs **soft-matching** against all keys in the table — rather than a hard/discrete lookup (which would require an exact key match), it computes a similarity-weighted blend.

**Result:** the "gatto" token's representation is built by **aggregating (weighted-averaging) the value vectors** in the source dictionary, weighted by how well the query "N5" soft-matches each key.

> This reframing makes explicit why attention is often called a **"soft" dictionary/lookup mechanism**: instead of retrieving exactly one value for an exact key match (as in a hash map), it retrieves a **weighted blend of all values**, where the weights come from a learned similarity function between the query and each key.

---

## 6. Vision Transformers (ViT)

### 6.1 Motivation: Can Self-Attention Work on Images? (p. 190–192)

**The core challenge:** the Transformer works with a **set of tokens** — but what counts as a token/"visual word" in an image?

**Naive approach — treat every pixel as a token:** for a $224\times 224$ image, this means $224^2 \approx 50,176$ tokens. Since self-attention computes pairwise interactions between **all** tokens, this requires roughly $224^4$ calculations — **computationally impossible** at this scale.

**Intermediate approach — object detector features as tokens:** use an object detector (e.g., with **ROI Pooling**) to extract a manageable number of region-based feature vectors from the image, and treat each region's feature vector as one token.

### 6.2 ViT's Solution: Patches ("An Image is Worth 16x16 Words") (p. 193)

**Paper:** Dosovitskiy et al., 2020 — _"An Image is Worth 16x16 Words"_

**Procedure:**

1. Take a $224\times 224$ input image
2. Divide it into a grid of **16×16 pixel patches** — this yields a $14\times 14$ grid (i.e., $14\times 14 = 196$ patches total)
3. **Flatten** each patch and apply a **linear projection** to turn it into a token vector

$$\text{Image} \to \text{Patches (16px} \times \text{16px each)} \to \text{Flatten} + \text{Linear Projection} \to \text{Tokens}$$

> This reduces the effective "sequence length" from ~50,000 pixels down to 196 patch-tokens — making full self-attention computationally tractable, since cost now scales with $196^2$ rather than $(224^2)^2$.

### 6.3 Full ViT Architecture (p. 193–194)

**Pipeline:**

1. **$N$ input patches**, each of shape $3\times 16\times 16$ (RGB channels × patch height × patch width)
2. **Linear projection** to a $D$-dimensional vector per patch
3. **Add positional embedding** — a **learned** $D$-dim vector per position (unlike NLP Transformers' fixed sinusoidal encoding, ViT's original design uses learned positional embeddings)
4. **Prepend a special classification token** ("class token" / `[CLS]`) — also a learned $D$-dim vector, marked with `*` in the diagram — this extra token's final representation is used for the classification decision
5. Feed the full sequence (class token + patch tokens + positional embeddings) into a **standard Transformer** — "**Exact same as NLP Transformer!**"
6. **Output vectors** — one per input position, plus one for the class token
7. The **class token's output vector** is linearly projected to a $C$-dimensional vector of predicted class scores (where $C$ = number of classes)

**Example scale (from slide):** 224×224 image → 14×14 grid of 16×16 patches (or equivalently 16×16 grid of 14×14 patches, depending on patch size choice); with 48 layers, 16 heads per layer, all attention matrices together take ~112MB (or 192MB) — illustrating the substantial memory footprint of large ViT models.

```python
import torch
import torch.nn as nn

class ViTPatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # Equivalent to: flatten each patch + linear projection, done via a strided conv
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))  # learned positional embedding

    def forward(self, x):                        # x: (batch, 3, 224, 224)
        x = self.proj(x)                          # (batch, embed_dim, 14, 14)
        x = x.flatten(2).transpose(1, 2)           # (batch, num_patches=196, embed_dim)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)      # (batch, num_patches+1, embed_dim)
        x = x + self.pos_embed                     # add learned positional embedding
        return x                                    # feed into a standard Transformer encoder next
```

---

## 7. CNN vs. Vision Transformer — Comparison (p. 196–200)

### 7.1 Training Dynamics (p. 197)

||Convolutional Neural Network|Vision Transformer|
|---|---|---|
|**Learned knowledge over training**|**Not improving anymore** (plateaus)|**Still improving** (given enough data)|

### 7.2 Why Are They Different? (p. 198)

**CNN characteristics:**

- **Locality sensitive** — convolution kernels only look at local neighborhoods
- **Translation invariant** — a learned filter detects the same pattern regardless of where it appears in the image
- **Learns inductive biases** — the convolutional structure itself encodes assumptions (locality, translation invariance) into the architecture, which helps with limited data
- **Lack of global understanding** — harder to directly relate very distant parts of an image without many stacked layers

**Vision Transformer characteristics:**

- **Able to find long-term dependencies** — self-attention connects any two patches directly, regardless of distance
- **Needs a very large dataset for training** — without the built-in inductive biases of convolution, ViT must learn spatial relationships from data alone, requiring much more training data (or large-scale pretraining, e.g., on datasets like JFT) to reach strong performance

> **Core tradeoff:** CNNs bake in useful assumptions (locality, translation invariance) that work well with less data but constrain what the model can represent; ViTs make fewer assumptions (more flexible, better at long-range dependencies) but need much more data to learn those useful patterns from scratch.

### 7.3 A Different Point of View: ViTs Are Both Local and Global (p. 199–200)

**With low amounts of data (p. 199):**

- ViT attention heads mostly learn **global** information
- "Heads focus on farther patches" — attention weight examples like 0.7, 0.0, 1 show heads attending broadly across the image regardless of spatial proximity

**With more data (p. 200):**

- ViT **also** learns **local** information, not just global
- **Higher layers'** heads still focus on farther patches (similar pattern to the low-data case: e.g., weights 0.0, 0.01, 0.7)
- **Lower layers'** heads focus on **both** farther **and** closer patches (e.g., weights showing 0.4, 0.3, 0.01 in one head and 0.06, 0.6 in another — mixing near and far attention)

**Supporting evidence — "Mean Distance" plot (ViT-L/16, JFT→ImageNet):**

- Plots mean attention distance (y-axis) against sorted attention heads (x-axis), across different encoder blocks (encoder_block0, 1, 22, 23)
- **Early encoder blocks** (e.g., block0): attention heads show a wide range of mean distances — some heads attend very locally (low mean distance), others more globally — demonstrating a **mix of local and global attention** at lower layers
- **Later encoder blocks** (e.g., block22, block23): nearly all heads show consistently **high** mean distance — i.e., by the later layers, attention has become predominantly **global**

> **Key takeaway:** the popular narrative that "CNNs are local, Transformers are global" is an oversimplification. With sufficient training data, ViTs develop **both** local attention patterns (especially in early layers, functionally similar to what a convolution would capture) **and** global attention patterns (especially in later layers) — they are not restricted to one or the other, unlike a CNN's architecturally fixed receptive field growth.

---

## Cross-Topic Connections

|Concept|Connects to|Relationship|
|---|---|---|
|Multi-head attention (p.181–183)|Self-attention basics (previous chunk, p.178–180)|Multi-head is simply running several independent self-attention computations in parallel and combining — same $QK^TV$ machinery, just replicated across heads with separate learned projections|
|Layer Norm (p.184)|Batch Normalization (p.93–96)|Same normalize-scale-shift structure ($\gamma, \beta$), but normalizes across a different axis (features vs. batch) — chosen specifically because NLP sequences have variable length, unlike fixed-size image batches|
|Point-wise FFN's ReLU (p.185)|Activation functions, gradient flow (pages 1–80)|Standard feedforward block reused as a per-position "processing" step after attention has mixed information across positions|
|Positional encoding (p.186)|Embedding lookup (pages 121–160)|Both are added/looked-up vectors combined with token representations — but positional encoding is fixed/deterministic (or learned as in ViT) and encodes _position_, not _identity_|
|Cross-attention's Q from target, K/V from source (p.187–188)|Global/Local attention in RNN seq2seq (p.163–169)|Directly the same query=decoder-state, key/value=encoder-states structure — the Transformer's cross-attention is the RNN attention mechanism generalized and formalized with explicit learned $W^Q, W^K, W^V$ projections|
|ViT's patch tokenization (p.193)|CNN's convolution + pooling (pages 1–80)|Both are ways of reducing the effective spatial resolution to a manageable number of "units" before further processing — CNNs do this via learned local filters and pooling, ViT via non-overlapping patch extraction and a single linear projection|
|ViT needing large datasets (p.198)|CNN inductive biases (locality/translation invariance)|Directly explains the data-efficiency gap: CNN's built-in assumptions substitute for data, while ViT must learn analogous patterns purely from data volume|
|ViT's local+global attention emerging with data (p.199–200)|Attention mechanism dot-product foundations (p.178–180)|Confirms empirically that self-attention is flexible enough to _learn_ CNN-like locality when useful, without it being architecturally hard-coded|

---

## Quick-Reference Formula Sheet

$$\text{Multi-head: } Z_{\text{combined}} = \text{Concat}(Z_0,\dots,Z_7),W^O, \quad Z_i = \text{softmax}\left(\frac{Q_iK_i^T}{\sqrt{d_k}}\right)V_i$$

$$\text{LayerNorm}(Y) = \gamma\frac{Y-Y_m}{Y_\sigma}+\beta$$

$$\text{FFN: } \bar{Z} = \text{ReLU}(W_1Z+b_1)W_2+b_2$$

$$PE_{(pos,2i)}=\sin\left(\frac{pos}{1000^{2i/embed_size}}\right), \quad PE_{(pos,2i+1)}=\cos\left(\frac{pos}{1000^{2i/embed_size}}\right)$$

$$\text{ViT patches: } 224\times224 \to 14\times14 \text{ grid of } 16\times16 \text{ patches} \to \text{flatten} + \text{linear projection} \to \text{tokens}$$