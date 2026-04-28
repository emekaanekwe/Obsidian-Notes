# FIT5217 — Week 3: POS Tagging & Hidden Markov Models

**Unit:** FIT5217 Natural Language Processing · Monash University  
**Topics:** Sequence Labeling · Markov Chains · HMM · Forward Algorithm · Viterbi · MLE Learning

---

## Table of Contents

1. [[#1. Word Categories & POS Tags]]
2. [[#2. POS Tagging Task & Ambiguity]]
3. [[#3. Why Not Simple Classification?]]
4. [[#4. Sequence Labeling]]
5. [[#5. Markov Chains]]
6. [[#6. Hidden Markov Models (HMM)]]
7. [[#7. Task 1 — Observation Likelihood]]
8. [[#8. The Forward Algorithm]]
9. [[#9. Task 2 — Viterbi Decoding]]
10. [[#10. Task 3 — Learning the HMM (MLE)]]
11. [[#11. Exam Preparation]]
12. [[#12. Connections to Other Topics]]

---
## 1. Word Categories & POS Tags

### The Idea

Classifying words into grammatical categories is ancient — Sanskrit grammarians (~100 BC) already proposed eight categories. The goal: understand how words **function** in a sentence and combine into meaningful phrases.

### Two Broad Super-Categories

|Class|Membership|Examples|Why?|
|---|---|---|---|
|**Closed Class**|Fixed, rarely grows|Prepositions, determiners, pronouns|No new prepositions appear in English|
|**Open Class**|Constantly expanding|Nouns, verbs, adjectives|"iPhone", "Google", "tweet" — new words arise constantly|

### Penn Treebank Tag Set (~48 tags — the standard)

| Tag   | Meaning              | Example           |
| ----- | -------------------- | ----------------- |
| `NN`  | Singular common noun | _dog, city_       |
| `NNP` | Proper noun          | _John, Sydney_    |
| `VB`  | Verb, base form      | _run, eat_        |
| `VBD` | Verb, past tense     | _ran, ate_        |
| `JJ`  | Adjective            | _blue, fast_      |
| `RB`  | Adverb               | _quickly, back_   |
| `DT`  | Determiner           | _the, a, an_      |
| `IN`  | Preposition          | _in, of, with_    |
| `MD`  | Modal auxiliary      | _can, will, must_ |

> **Exam note:** You are not expected to memorise all tags — they will be provided in exam problems.

---

## 2. POS Tagging Task & Ambiguity

### Task Definition

**POS Tagging:** Given a sentence (sequence of words), assign a grammatical tag to each word.

**Example:**

```
John/NNP  saw/VBD  the/DT  saw/NN
```

POS tagging is the _lowest level of syntactic analysis_. It feeds into parsing, NER, sentiment analysis, TTS, and language modeling.

### The Ambiguity Problem

Many words can be multiple POS depending on context. The word **"back"**:

| Sentence        | POS       | Reason                    |
| --------------- | --------- | ------------------------- |
| _the back door_ | Adjective | Modifies a noun           |
| _win the back_  | Noun      | Refers to a physical part |
| _he came back_  | Adverb    | Modifies a verb           |
| _back the car_  | Verb      | Action                    |

**From the Brown Corpus:**

- ~**11% of word types** are ambiguous
- ~**40% of word tokens** are ambiguous

### Key Statistics — Know These

| Method                                      | Accuracy |
| ------------------------------------------- | -------- |
| Baseline (most-frequent tag + unknown → NN) | ~90%     |
| State of the art (neural/transformer)       | ~97–98%  |

> **Why study HMMs if baseline is 90%?** Because the _algorithms_ used to solve POS tagging (Viterbi, forward-backward) generalise to many other NLP tasks — machine translation, NER, speech recognition, and more.

---

## 3. Why Not Simple Classification?

> **Analogy:** Imagine labelling each puzzle piece without ever looking at its neighbours. You'd get the shape wrong most of the time. Words are the same — context is everything.

### The Independence Assumption Fails

Standard classifiers (logistic regression, SVM, etc.) assume **i.i.d. (independent and identically distributed)** inputs. Each prediction is made independently.

But in language:

- The tag of word $w_t$ depends on the tags of surrounding words $w_{t-1}, w_{t+1}$
- Labelling "saw" in isolation tells you nothing — is it a verb (action) or a noun (tool)?

> **We need models that capture dependencies between labels in a sequence, i.e., models that violate the i.i.d. assumption**

---

## 4. Sequence Labeling

### Formal Definition

Given an input sequence $w_1, w_2, \ldots, w_T$, assign a label $t_1, t_2, \ldots, t_T$ where each label **can depend on other labels** in the sequence.

This intentionally **violates the i.i.d. assumption**.

### Other Sequence Labeling Tasks in NLP

| Task                           | Input | Output Labels               | Why Sequence?                                  |
| ------------------------------ | ----- | --------------------------- | ---------------------------------------------- |
| POS Tagging                    | Words | NN, VB, DT…                 | Tag depends on neighbours                      |
| Named Entity Recognition (NER) | Words | PERSON, ORG, DATE…          | "Dell" = person or company? Depends on context |
| Semantic Role Labeling         | Words | Agent, Patient, Instrument… | Role depends on position relative to verb      |

---

## 5. Markov Chains

### The Markov Assumption

> The probability of the next state depends **only** on the current state, not the full history.

$$P(q_t \mid q_1, q_2, \ldots, q_{t-1}) = P(q_t \mid q_{t-1})$$

This is a **first-order Markov assumption**, equivalent to a **bigram model** — but over POS tags rather than words.

> **Analogy:** A Markov chain is like navigating a city where the only thing determining your next turn is your current street — you forget everywhere you've been.

### Probability of a Tag Sequence

Given a finite-state machine with transition probabilities, the joint probability of a POS sequence is:

$$P(q_1, q_2, \ldots, q_T) = P(q_1 \mid \text{START}) \cdot \prod_{t=2}^{T} P(q_t \mid q_{t-1}) \cdot P(\text{END} \mid q_T)$$

This is just the **chain rule + Markov assumption** applied to a sequence of tags.

### ✏️ Worked Example — POS Sequence Probability

**Setup:** Transition probabilities for a tiny 3-state model:

| From → To | DT   | NNP | VB  |
| --------- | ---- | --- | --- |
| START     | 0.5  | 0.4 | 0.1 |
| NNP       | 0.0  | 0.0 | 0.8 |
| VB        | 0.25 | 0.0 | 0.0 |

**Task:** Compute $P(\text{NNP} \to \text{VB} \to \text{DT})$, i.e. a sentence like _"John ate the"_

**Step 1:** Apply the chain rule with the Markov assumption:

$$P = P(\text{NNP} \mid \text{START}) \times P(\text{VB} \mid \text{NNP}) \times P(\text{DT} \mid \text{VB})$$

**Step 2:** Plug in values:

$$P = 0.4 \times 0.8 \times 0.25 = \boxed{0.08}$$

The probability of this POS sequence is **8%**. We could compare other sequences to find the most likely one.

---

## 6. Hidden Markov Models (HMM)

### The Key Insight — "Hidden" States

In a plain Markov chain, we **observe** the states directly (we know the POS sequence). But in POS tagging:

- We **observe** the words: _"John bit the apple"_
- The POS tags are **hidden** — we must infer them

An HMM models _how words are generated from hidden POS tags_.

> **Analogy:** A writer first plans a sentence structure in their head (hidden plan: NNP → VB → DT → NN), then picks specific words to fill each slot (observable result: _"John bit the apple"_). The HMM models both layers simultaneously.

### Formal Definition

An HMM $\lambda$ is defined by:

| Symbol                   | Name              | Description                                   |
| ------------------------ | ----------------- | --------------------------------------------- |
| $Q = {q_1, \ldots, q_N}$ | States            | The $N$ POS tags (+ START, END)               |
| $V = {v_1, \ldots, v_M}$ | Observations      | The vocabulary (all possible words)           |
| $A = a_{ij}$             | Transition Matrix | $P(\text{state } s_j \mid \text{state } s_i)$ |
| $B = b_j(v_k)$           | Emission Matrix   | $P(\text{word } v_k \mid \text{state } s_j)$  |

**HMM parameters:** $\lambda = (A, B)$

### Two Key Assumptions

**Assumption 1 — Markov Property (First-Order):**

$$P(q_t \mid q_1, \ldots, q_{t-1}) = P(q_t \mid q_{t-1})$$

Current state depends only on the immediately previous state.

**Assumption 2 — Output Independence:**

$$P(o_t \mid q_1, \ldots, q_T,\ o_1, \ldots, o_T) = P(o_t \mid q_t)$$

Each word depends only on its current POS tag — not on any other words or tags.

### Three Core HMM Tasks

| #   | Task                       | Given           | Find                    | Algorithm         |
| --- | -------------------------- | --------------- | ----------------------- | ----------------- |
| 1   | **Observation Likelihood** | $\lambda,\ O$   | $P(O \mid \lambda)$     | Forward Algorithm |
| 2   | **Decoding**               | $\lambda,\ O$   | Best tag sequence $Q^*$ | Viterbi Algorithm |
| 3   | **Learning**               | Labelled corpus | $\lambda = (A, B)$      | MLE counting      |

---

## 7. Task 1 — Observation Likelihood

### The Problem

Given $O = o_1, o_2, \ldots, o_T$ and model $\lambda$, compute $P(O \mid \lambda)$.

**Why is this useful?**

- **Language ID:** Which language model best explains this sentence?
- **Speech recognition:** Which word sequence most likely produced this audio?
- **Sequence classification:** Which HMM (one per class) best explains the observation?

### Naïve Approach — Exponential Cost

Sum over all possible hidden state sequences:

$$P(O \mid \lambda) = \sum_{\text{all } Q} P(O, Q \mid \lambda)$$

For $T$ words and $N$ POS tags, there are $N^T$ possible sequences.  
With $N = 45$, $T = 10$: $45^{10} \approx 3.4 \times 10^{16}$ — completely infeasible.

> ⚠️ **Complexity of naïve approach:** $O(N^T \cdot T)$ — exponential in sequence length.  
> The Forward Algorithm reduces this to $O(N^2 T)$ — polynomial — by exploiting the Markov assumption with dynamic programming.

---

## 8. The Forward Algorithm

### Core Idea — Dynamic Programming

Instead of summing all global paths, build up probabilities **incrementally**. At each time step, we only need the values from the previous step.

> **Analogy:** Instead of counting every possible route from city A to city Z, keep a running tally of probabilities at each intermediate city as you move forward — never backtracking, never re-enumerating.

### Forward Probability Definition

$$\alpha_t(j) = P(o_1, o_2, \ldots, o_t,\ q_t = s_j \mid \lambda)$$

The probability of having observed the first $t$ words **AND** being in state $s_j$ at time $t$.

### Algorithm

**Step 1 — Initialisation:**

$$\alpha_1(j) = a_{0j} \cdot b_j(o_1) \quad \text{for each state } s_j$$

Transition from START $\times$ probability of emitting the first word from state $s_j$.

**Step 2 — Recursion** (for $t = 2, \ldots, T$):

$$\alpha_t(j) = \left[\sum_{i=1}^{N} \alpha_{t-1}(i) \cdot a_{ij}\right] \cdot b_j(o_t)$$

Sum over all ways to arrive at state $s_j$ from any previous state, multiplied by the emission probability of word $o_t$.

**Step 3 — Termination:**

$$P(O \mid \lambda) = \sum_{i=1}^{N} \alpha_T(i) \cdot a_{i,\text{END}}$$

Sum all forward probabilities at the final time step, weighted by their transition into END.

**Complexity:** $O(N^2 T)$ ✓

### ✏️ Worked Example — Forward Algorithm

**Setup (tiny HMM):**

- States: $s_1 =$ NNP (proper noun), $s_2 =$ VB (verb)
- Sentence: $O = (\text{John},\ \text{bit})$, so $T = 2$

**Transition Matrix $A$:**

| NNP   | VB  | END |     |
| ----- | --- | --- | --- |
| START | 0.7 | 0.3 | —   |
| NNP   | 0.0 | 0.8 | 0.2 |
| VB    | 0.0 | 0.0 | 1.0 |

**Emission Matrix $B$:**

| John | bit | apple |     |
| ---- | --- | ----- | --- |
| NNP  | 0.9 | 0.0   | 0.1 |
| VB   | 0.0 | 0.8   | 0.2 |

---

**Step 1 — Initialisation** ($t=1$, word = _John_):

$$\alpha_1(\text{NNP}) = a_{\text{START,NNP}} \cdot b_{\text{NNP}}(\text{John}) = 0.7 \times 0.9 = 0.63$$

$$\alpha_1(\text{VB}) = a_{\text{START,VB}} \cdot b_{\text{VB}}(\text{John}) = 0.3 \times 0.0 = 0.0$$

---

**Step 2 — Recursion** ($t=2$, word = _bit_):

For state NNP:

$$\alpha_2(\text{NNP}) = \left[\alpha_1(\text{NNP}) \cdot a_{\text{NNP,NNP}} + \alpha_1(\text{VB}) \cdot a_{\text{VB,NNP}}\right] \cdot b_{\text{NNP}}(\text{bit})$$

$$= \left[0.63 \times 0.0 + 0.0 \times 0.0\right] \times 0.0 = 0.0$$

For state VB:

$$\alpha_2(\text{VB}) = \left[\alpha_1(\text{NNP}) \cdot a_{\text{NNP,VB}} + \alpha_1(\text{VB}) \cdot a_{\text{VB,VB}}\right] \cdot b_{\text{VB}}(\text{bit})$$

$$= \left[0.63 \times 0.8 + 0.0 \times 0.0\right] \times 0.8 = 0.504 \times 0.8 = 0.4032$$

---

**Step 3 — Termination:**

$$P(O \mid \lambda) = \alpha_2(\text{NNP}) \cdot a_{\text{NNP,END}} + \alpha_2(\text{VB}) \cdot a_{\text{VB,END}}$$

$$= 0.0 \times 0.2 + 0.4032 \times 1.0 = \boxed{0.4032}$$

The sentence _"John bit"_ has probability $\approx 0.40$ under this model.

---

## 9. Task 2 — Viterbi Decoding

### The Goal

Find the **single most probable POS tag sequence** $Q^*$ for a given sentence $O$:

$$Q^* = \underset{Q}{\arg\max}\ P(Q \mid O, \lambda)$$

This is the actual POS tagging task — labelling each word in a sentence.

> **Analogy:** The Forward Algorithm asks _"what is the total traffic flow across all routes?"_ The Viterbi Algorithm asks _"what is the single fastest route?"_

### Viterbi Score Definition

$$v_t(j) = \max_{q_1, \ldots, q_{t-1}} P(q_1, \ldots, q_{t-1},\ q_t = s_j,\ o_1, \ldots, o_t \mid \lambda)$$

The probability of the **most likely path** ending in state $s_j$ at time $t$.

**Key difference from Forward:** Replace $\sum$ with $\max$.

### Algorithm

**Step 1 — Initialisation:**

$$v_1(j) = a_{0j} \cdot b_j(o_1)$$

Identical to the Forward initialisation step.

**Step 2 — Recursion** (for $t = 2, \ldots, T$):

$$v_t(j) = \max_{i}\ \left[v_{t-1}(i) \cdot a_{ij}\right] \cdot b_j(o_t)$$

$$\text{bt}_t(j) = \underset{i}{\arg\max}\ \left[v_{t-1}(i) \cdot a_{ij}\right]$$

$\text{bt}_t(j)$ is the **backpointer** — it records _which previous state_ led to the maximum. This is how we reconstruct the best path at the end.

**Step 3 — Termination + Backtracking:**

$$q_T^* = \underset{i}{\arg\max}\ \left[v_T(i) \cdot a_{i,\text{END}}\right]$$

Then trace back through backpointers from $q_T^*$ to $q_1^*$ to recover the full tag sequence.

**Complexity:** $O(N^2 T)$ ✓

### ✏️ Worked Example — Viterbi

**Same HMM as the Forward example.** States: NNP, VB. Sentence: $O = (\text{John},\ \text{bit})$.

---

**Step 1 — Initialisation** ($t=1$, word = _John_):

$$v_1(\text{NNP}) = 0.7 \times 0.9 = 0.63 \quad \text{bt}_1(\text{NNP}) = \text{START}$$

$$v_1(\text{VB}) = 0.3 \times 0.0 = 0.0 \quad \text{bt}_1(\text{VB}) = \text{START}$$

---

**Step 2 — Recursion** ($t=2$, word = _bit_):

For state NNP — find the best incoming path:

$$\max_i\left[v_1(i) \cdot a_{i,\text{NNP}}\right] = \max(0.63 \times 0.0,\ 0.0 \times 0.0) = 0.0$$

$$v_2(\text{NNP}) = 0.0 \times b_{\text{NNP}}(\text{bit}) = 0.0 \times 0.0 = 0.0$$

For state VB — find the best incoming path:

$$\max_i\left[v_1(i) \cdot a_{i,\text{VB}}\right] = \max(0.63 \times 0.8,\ 0.0 \times 0.0) = 0.504$$

$$v_2(\text{VB}) = 0.504 \times b_{\text{VB}}(\text{bit}) = 0.504 \times 0.8 = 0.4032$$

$$\text{bt}_2(\text{VB}) = \text{NNP} \quad \text{(NNP produced the max)}$$

---

**Step 3 — Termination:**

$$q_2^* = \underset{i}{\arg\max}\ \left[v_2(i) \cdot a_{i,\text{END}}\right] = \arg\max(0.0 \times 0.2,\ 0.4032 \times 1.0) = \text{VB}$$

**Backtrack:** $q_2^* = \text{VB}$, $\text{bt}_2(\text{VB}) = \text{NNP}$, so $q_1^* = \text{NNP}$

$$\boxed{\text{John/NNP} \quad \text{bit/VB}}$$

### Forward vs. Viterbi — Side-by-Side

|Property|Forward Algorithm|Viterbi Algorithm|
|---|---|---|
|Computes|Total probability $P(O \mid \lambda)$|Best sequence $Q^*$|
|Recursion operator|$\sum$ (sum all paths)|$\max$ (keep best path)|
|Stores backpointers?|No|Yes (for backtracking)|
|Use case|Language scoring, classification|POS tagging, decoding|
|Complexity|$O(N^2 T)$|$O(N^2 T)$|

---

## 10. Task 3 — Learning the HMM (MLE)

### Goal

Given a **labelled training corpus** (words with gold-standard POS tags), estimate the parameters $\lambda = (A, B)$ using **Maximum Likelihood Estimation (MLE)**.

### Learning the Transition Probabilities $A$

This is simply a **bigram model over POS tags**:

$$\hat{a}_{ij} = P(s_j \mid s_i) = \frac{C(s_i \to s_j)}{C(s_i)}$$

Count how often tag $s_i$ is followed by tag $s_j$, divide by total occurrences of $s_i$.

### Learning the Emission Probabilities $B$

$$\hat{b}_j(v_k) = P(v_k \mid s_j) = \frac{C(s_j,\ v_k)}{C(s_j)}$$

Count how often word $v_k$ appears labelled with tag $s_j$, divide by total occurrences of $s_j$.

### ✏️ Worked Example — MLE Learning

**Training corpus** (2 sentences):

|Word|Tag|
|---|---|
|John|NNP|
|bit|VB|
|Mary|NNP|
|ran|VB|

---

**Step 1 — Count tag bigrams (for transitions):**

|Bigram|Count|
|---|---|
|START → NNP|2|
|NNP → VB|2|
|VB → END|2|

**Step 2 — Compute transition probabilities:**

$$\hat{a}_{\text{START,NNP}} = \frac{2}{2} = 1.0$$

$$\hat{a}_{\text{NNP,VB}} = \frac{2}{2} = 1.0$$

---

**Step 3 — Count (word, tag) pairs (for emissions):**

|Pair|Count|
|---|---|
|(John, NNP)|1|
|(Mary, NNP)|1|
|(bit, VB)|1|
|(ran, VB)|1|

Total NNP = 2, Total VB = 2

**Step 4 — Compute emission probabilities:**

$$\hat{b}_{\text{NNP}}(\text{John}) = \frac{1}{2} = 0.5 \qquad \hat{b}_{\text{NNP}}(\text{Mary}) = \frac{1}{2} = 0.5$$

$$\hat{b}_{\text{VB}}(\text{bit}) = \frac{1}{2} = 0.5 \qquad \hat{b}_{\text{VB}}(\text{ran}) = \frac{1}{2} = 0.5$$

---

> ⚠️ **Data Sparsity — Emission Probabilities:**  
> If a word like "Google" never appeared in training, then $\hat{b}_{\text{NNP}}(\text{Google}) = 0$, causing zero probabilities to propagate through Viterbi and Forward.  
> **Fix:** Apply smoothing (e.g. Add-1/Laplace) to emission probabilities.  
> Transition probabilities rarely suffer from this since the POS tag set is fixed and small.

---

## 11. Exam Preparation

### Likely Question Type 1 — Conceptual

> _"Why can't we use a standard classifier for POS tagging?"_

**Model answer:** Standard classifiers assume i.i.d. inputs — each prediction is independent. In POS tagging, a word's tag depends on its neighbours. "Back" can be a noun, verb, adjective, or adverb — the correct tag can only be determined from context. Sequence models like HMMs explicitly capture label-to-label dependencies.

---

### Likely Question Type 2 — Algorithm Tracing

> _"Trace the Viterbi algorithm for the sentence 'X Y Z' given matrices A and B."_

**Strategy:**

1. Draw a trellis: states as rows, time steps $t=1,\ldots,T$ as columns
2. Initialise column 1 using $v_1(j) = a_{0j} \cdot b_j(o_1)$
3. Fill each subsequent column left-to-right using $v_t(j) = \max_i[v_{t-1}(i) \cdot a_{ij}] \cdot b_j(o_t)$
4. Record the backpointer $\text{bt}_t(j)$ at each cell
5. At the final column, pick $\arg\max_i [v_T(i) \cdot a_{i,\text{END}}]$
6. Follow backpointers right-to-left to read off the tag sequence

---

### Likely Question Type 3 — Comparison

> _"Compare the Forward Algorithm and the Viterbi Algorithm."_

Both are dynamic programming algorithms with complexity $O(N^2 T)$. Forward uses $\sum$ to compute the total probability $P(O \mid \lambda)$ over all paths. Viterbi uses $\max$ and stores backpointers to find and reconstruct the single best path $Q^*$.

---

### Likely Question Type 4 — MLE Calculation

> _"Given a small labelled corpus, compute transition and emission probabilities."_

Count tag bigrams → normalise by tag frequency (for $A$). Count (word, tag) pairs → normalise by tag frequency (for $B$). This is the same MLE counting from Week 2 n-gram models, applied to POS tags instead of words.

---

## 12. Connections to Other Topics

|This Week|Connection|Linked Topic|
|---|---|---|
|Markov Assumption|Same assumption|n-gram Language Models (Wk 2)|
|MLE for HMM|Same counting principle|n-gram LM estimation (Wk 2)|
|Smoothing on emissions|Same sparsity problem|Laplace / Kneser-Ney (Wk 2)|
|POS tags as features|Tags improve downstream parsing|Syntax / Parsing (Wk 4)|
|Viterbi decoding|General DP decoding pattern|Beam Search, Machine Translation|
|HMM as sequence model|Foundation before neural models|RNNs, seq2seq, Transformers|

---

## Summary

**Why POS Tagging?** Words are ambiguous; their grammatical role depends on context. Getting tags right improves parsing, NER, sentiment, LMs, and TTS.

**Why Not Classification?** Labels in a sequence are interdependent — the i.i.d. assumption fails.

**Markov Chain:** Model POS sequences by assuming each tag depends only on the previous tag. Joint probability = product of bigram transition probabilities.

**HMM:** Extends Markov chains to the case where states (POS tags) are _hidden_ and we only observe words. Parameters: transition matrix $A$ and emission matrix $B$.

**Three HMM Tasks:**

$$\text{Likelihood} \rightarrow \text{Forward (}\sum\text{, DP, } O(N^2 T))$$

$$\text{Decoding} \rightarrow \text{Viterbi (}\max\text{ + backpointers, } O(N^2 T))$$

$$\text{Learning} \rightarrow \text{MLE counting on labelled corpus}$$

**Data Sparsity:** MLE produces zero emission probabilities for unseen words → apply smoothing.

---