
---


# Part A — Multiple Choice (with quick justifications)

**Q1 (p.4). Self-Attention facts**  
**Correct:** a, d

- a) Output is a weighted sum of **values** (V) using attention weights.
    
- d) In self-attention, each token can attend to **all tokens** in the same sequence.  
    **Incorrect:** b (works beyond strictly sequential data), c (scores use **queries vs keys**, not queries vs values), e (compare **queries and keys**).
    

---

**Q2 (p.4). Skip-gram shapes & one-hot**  
**Correct:** a, b, f

- (U\in\mathbb{R}^{400\times 200}), (V\in\mathbb{R}^{200\times 400}).
    
- Input is one-hot of the **target** word (index 15).
    
- Hidden embedding (h) is the **row 15 of (U)** (multiplying a one-hot row by (U) “selects” that row).  
    (If an option said “column 15 of (U)”: that’s **wrong** here.)
    

---

**Q3 (p.5). Multi-head Self-Attention**  
**Correct:** b, c, e

- b) Heads are computed **independently**.
    
- c) Head outputs are **concatenated**, then projected by (W_O).
    
- e) Each head has its own (W_Q, W_K, W_V).  
    **Incorrect:** a (not shared), d (you still apply (W_O)), f (heads are not conditionally dependent during the per-head computation).
    

---

**Q4 (p.5). Conv2D output shape**  
Input ([30,64,64]), filters: 20 of shape ([30,3,3]), stride ([2,2]), padding (=1).  
Spatial: (\big\lfloor\frac{64-3+2}{2}\big\rfloor + 1=\lfloor 63/2\rfloor+1=31+1=32).  
**Answer:** ([20,32,32]) → option **b**.

---

**Q5 (p.5). One SGD step**  
Per-sample loss ( \tfrac12(w-i)^2 \Rightarrow \nabla = (w-i)).  
Mini-batch ({1,2,3,4}), use average gradient:  
(\bar{g}=\frac{1}{4}\sum (16-i)=\frac{15+14+13+12}{4}=13.5).  
Update (w_{t+1}=16-0.1\cdot 13.5=14.65).

> **Note:** 14.65 isn’t in the options; if the examiner used a different convention (e.g., extra factor 2 or a non-averaged batch), the nearest listed choice would differ. Show this derivation and state the convention you used.

---

**Q6 (p.6). ResNet tensor shapes**  
Without the figure’s exact blocks you pick the option consistent with standard downsampling (×2 per stage) and final flatten to logits. **Answer likely:** **b**. (Explain: input ([16,3,64,64]) → stem → ([16,64,16,16]) then stages ([16,128,8,8]), ([16,256,4,4]), global pool ([16,256,1,1]), FC ([16,10])).

---

**Q7 (p.6). Cross-entropy**  
True label **car** (index 6), predicted prob (=0.2).  
Loss (= -\log 0.2). → option **a**.

---

**Q8 (p.7). Softmax prob for “lion”**  
Scores (h=[-3,6,-2,-1,3]).  
(p(y=\text{lion}\mid x) = \dfrac{e^{-2}}{e^{-3}+e^{6}+e^{-2}+e^{-1}+e^{3}}).  
Pick the option that matches this form. (Usually labeled **c**.)

---

**Q9 (p.7). LSTM layer-2 output shape**  
Batch (=2), seq len (=5), hidden size (layer 2) (=35).  
Output (all time steps): ([2,5,35]). → option **a**.

---

**Q10 (p.8). Last CNN tensor shape**  
Before last: ([128,10,32,32]); 16 filters ([10,5,5]); stride 2; pad 1.  
Spatial: (\lfloor (32-5+2)/2\rfloor+1=\lfloor 29/2\rfloor+1=14+1=15).  
**Answer:** ([128,16,15,15]) → option **c**.

---

**Q11 (p.8). Adversarial example**  
(x_{\text{adv}}=\arg\max_{x'\in B_\epsilon(x)} \ell(f(x';\theta),y)).  
Maximizing loss **decreases** (p(y\mid x')) and **increases** some other class.  
**Correct:** a, c, d. (Untargeted; not b/e.)

---

**Q12 (p.8). Optimal GAN discriminator**  
(D^_(x)=\dfrac{p_d(x)}{p_d(x)+p_g(x)}).  
At the Nash equilibrium (p_g=p_d\Rightarrow D^_(x)=\tfrac12). Pick the option(s) stating this.

---

# Part B — Short Workouts (with key math)

**Q13 (p.9). 2-layer MLP for spam**  
Let (x=[-1,0]). With weights (W^{(1)}, b^{(1)}, W^{(2)}, b^{(2)}):

- **(a)** (\bar h_1=W^{(1)}x+b^{(1)}); (h_1=\sigma(\bar h_1)) (e.g., ReLU or sigmoid).
    
- **(b)** (h_2=W^{(2)}h_1+b^{(2)}) (logits). For 2-class softmax:  
    (p=\mathrm{softmax}(h_2)_1=\dfrac{e^{h_{2,1}}}{e^{h_{2,1}}+e^{h_{2,2}}}).
    
- **(c)** (\hat y=\arg\max_k h_{2,k}); CE loss (\ell=-\log p) (if (y=1)).
    
- **(d)** (\frac{\partial \ell}{\partial h_2}=\mathrm{softmax}(h_2)-\mathrm{onehot}(y)).
    
- **(e)** (\frac{\partial \ell}{\partial W^{(2)}}=\frac{\partial \ell}{\partial h_2}, h_1^\top).
    
- **(f)** SGD update: (W^{(2)}\leftarrow W^{(2)}-\eta,\frac{\partial \ell}{\partial W^{(2)}}).  
    (Plug the actual numbers from the figure’s weights to compute values.)
    

---

**Q14 (p.10). CNN shapes (CIFAR-100, [128,3,64,64])**  
Use the conv formula (H_{\text{out}}=\big\lfloor \frac{H_{\text{in}}-K+2P}{S}\big\rfloor+1) and similarly for (W).

- **(a)** Provide ([A_1,B_1,C_1,D_1]) by applying the formula layer-by-layer per the figure.
    
- **(b)** Same for ([A_2,B_2,C_2,D_2]).
    
- **(c)** **Flatten path**: tensor ([C,D]=[128, C_\text{last}!\times H!\times W]) stores per-example features; (W\in\mathbb{R}^{(C!HW)\times 100}).
    
- **(d)** **Global max-pool path**: ([C,D]=[128, C_\text{last}]) stores channel-wise maxima; (W\in\mathbb{R}^{C_\text{last}\times 100}).
    
- **(e)** ([E,F]=[128, 100]) stores logits for 100 classes.
    
- **(f)** Batch loss: mean CE over batch, (\tfrac{1}{128}\sum_{i=1}^{128}-\log \mathrm{softmax}(z_i)_{y_i}).  
    (Compute exact numbers from the figure’s kernels/strides.)
    

---

**Q15 (p.11). ViT with 3×3 patches on [3,30,30]**

- **(a)** One patch has shape ([3,10,10]).
    
- **(b)** Flatten each patch to (3\cdot10\cdot10=300), project to (C=768). With class token, seq len (= 1+9=10).  
    **Answer:** ([A,B,C]=[1,10,768]).
    
- **(c)** Transformer encoder preserves ([1,10,768]) (same (B), (L), (C)).
    
- **(d)** The **class token** attends to all patch tokens; after multiple layers, its embedding aggregates **global** information via attention (it collects a weighted mixture of all patch features at each layer).
    

---

**Q16 (p.12). CNN code shapes**  
From the figure’s code:

```
conv1: in 3 → 32, k=3,s=2,p=1      ⇒ h1: [128, 32, 32, 32]
maxpool k=2,s=2                    ⇒ h2: [128, 32, 16, 16]
conv2: 32 → 64, k=3,s=1,p=1        ⇒ h3: [128, 64, 16, 16]
dropout (no shape change)          ⇒ h4: [128, 64, 16, 16]
conv3: 64 → 128, k=3,s=1,p=1       ⇒ h5: [128, 128, 16, 16]
AdaptiveAvgPool2d((1,1))           ⇒ h6: [128, 128, 1, 1]
Flatten                            ⇒ h7: [128, 128]
Linear(128→20)                     ⇒ h8: [128, 20]
```

(Use (\lfloor (H-K+2P)/S\rfloor+1) per conv.)

---

**Q17 (p.13). Tiny RNN, two time-steps “I love”**  
Let embeddings (x_0, x_1), weights (W_{xh}, W_{hh}), no bias, ReLU:

- **(a)** (h_0=\mathrm{ReLU}(W_{xh}x_0)).
    
- **(b)** (h_1=\mathrm{ReLU}(W_{xh}x_1+W_{hh}h_0)).
    
- **(c)** Logits (z=Vh_1), probs (p=\mathrm{softmax}(z)), (\hat y=\arg\max p).  
    (Plug the numeric matrices from the figure to compute values.)
    

---

# Part C — Conceptual / Short-Derivation

**Q18 (p.14). Why pick best on validation, not test?**

- To avoid **test leakage** and optimistic bias. The test set must estimate **generalization** only once, **after** model selection. Validation accuracy drives **hyper-parameter/model** choice; the test set remains untouched until the end.
    

---

**Q19 (p.14). Conv vs Max-pool**

- **Difference 1:** Conv has **learned weights**; max-pool has **no learnable params**.
    
- **Difference 2:** Conv produces **linear combinations** of local neighborhoods (plus nonlinearity); max-pool computes a **fixed reduction** (max).
    
- **Similarity:** Both are **translation-equivariant** local operators that change spatial resolution and operate channel-wise (pool over spatial dims per channel).
    

---

**Q20 (p.15). DCGAN on MNIST [1,28,28]**

- **(a) Generator:** (z\in\mathbb{R}^{30}) → **Unflatten** to a small 3-D seed (e.g., ([N,c_0,h_0,w_0])) → series of **ConvTranspose2d** (stride (>1)) to upsample until ([N,1,28,28]). Purpose: map noise to **realistic images**.
    
- **(b) Discriminator:** Image ([N,1,28,28]) → **Conv2d** (downsample) → **Flatten** → **Linear** → scalar logit. Purpose: distinguish **real vs fake**.
    
- **(c) Minimax objective:**  
    [  
    \min_G\max_D ;; \mathbb{E}_{x\sim p_d}![\log D(x)] + \mathbb{E}_{z\sim p_z}![\log(1-D(G(z)))].  
    ]  
    Update (D) to **increase** both terms; update (G) to **increase** (\log D(G(z))) (or minimize (-\log D(G(z)))).
    
- **(d) Optimal (D^*):** (D^_(x)=\frac{p_d(x)}{p_d(x)+p_g(x)}). At Nash equilibrium (p_g=p_d\Rightarrow D^_(x)=\tfrac12) for all (x).
    

---

**Q21 (p.16). Seq2Seq (Encoder–Decoder)**

- **(a) BOS/EOS:** **BOS** starts decoding; **EOS** marks sequence end.
    
- **(b)** Last encoder hidden state (h_T) summarizes the **source** via recurrent updates, so it’s used as the fixed **context** (c).
    
- **(c) MLE objective:** For dataset (\mathcal{D}={(x,y)}) with (\theta=[\theta_e,\theta_d]),  
    [  
    \max_\theta \sum_{(x,y)\in\mathcal{D}} \log P(y\mid x;\theta).  
    ]
    
- **(d)** Autoregressive factorization with product rule:  
    [  
    \log P(y\mid x;\theta)=\sum_{t=1}^{T_y}\log P(y_t\mid y_{<t},,c,,\theta).  
    ]
    
- **(e)** Drawback of fixed (c): information **bottleneck**—long sequences lose detail; (c) is time-invariant, so decoder can’t focus on different source parts.
    
- **(f) Global attention:** For decoder state (q_t), compute scores (e_{t,i}=a(q_t,h_i)); weights (\alpha_{t,i}=\mathrm{softmax}_i(e_{t,i})); context  
    [  
    c_t=\sum_i \alpha_{t,i}h_i,  
    ]  
    then use (c_t) (time-varying) with (q_t) for predicting (y_t).
    

---

## Handy formulas (put these on your one-pager)

- **Conv2D size:** (H'=\big\lfloor\frac{H-K+2P}{S}\big\rfloor+1,\quad W'=\big\lfloor\frac{W-K+2P}{S}\big\rfloor+1.)
    
- **Cross-entropy:** (\ell=-\log p_{y}).
    
- **Softmax:** (p_k=\dfrac{e^{h_k}}{\sum_j e^{h_j}}).
    
- **Skip-gram shapes:** (U\in\mathbb{R}^{|V|\times d},;V\in\mathbb{R}^{d\times |V|}); one-hot selects a **row** of (U).
    
- **Self-attention:** (\mathrm{Attn}(Q,K,V)=\mathrm{softmax}!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V).
    
- **GAN (D^*):** (\dfrac{p_d}{p_d+p_g}).
    
- **Seq2Seq MLE:** (\sum_t\log P(y_t\mid y_{<t},c,\theta)).
## Handy formulas (put these on your one-pager)

- **Conv2D size:** H′=⌊H−K+2PS⌋+1,W′=⌊W−K+2PS⌋+1.H'=\big\lfloor\frac{H-K+2P}{S}\big\rfloor+1,\quad W'=\big\lfloor\frac{W-K+2P}{S}\big\rfloor+1.H′=⌊SH−K+2P​⌋+1,W′=⌊SW−K+2P​⌋+1.
    
- **Cross-entropy:** ℓ=−log⁡py\ell=-\log p_{y}ℓ=−logpy​.
    
- **Softmax:** pk=ehk∑jehjp_k=\dfrac{e^{h_k}}{\sum_j e^{h_j}}pk​=∑j​ehj​ehk​​.
    
- **Skip-gram shapes:** U∈R∣V∣×d,  V∈Rd×∣V∣U\in\mathbb{R}^{|V|\times d},\;V\in\mathbb{R}^{d\times |V|}U∈R∣V∣×d,V∈Rd×∣V∣; one-hot selects a **row** of UUU.
    
- **Self-attention:** Attn(Q,K,V)=softmax ⁣(QK⊤dk)V\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)VAttn(Q,K,V)=softmax(dk​​QK⊤​)V.
    
GAN D∗D* D∗:pdpd+pg\dfrac{p_d}{p_d+p_g}pd​+pg​pd​​.