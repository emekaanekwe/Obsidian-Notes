 # Self Assess
- Given a set of PCFG rules and probabilities, and a given sentence, write down in details steps for CKY parsing, recover all possible parse trees for the sentence, and calculate the final probability of the sentence.
- Why conversion to CNF is needed?
- In a given parse tree, lexicalize all its non-terminal nodes
- Given a balanced bracket format, draw the corresponding parse tree.
- Given a pair of predicted and gold parse trees, calculate Precision, Recall, F1 (need to know the formulas).
- How would you learn the parameters of a PCFG given a treebank?

---
# FIT5217 — Week 4: Syntactic Parsing, CFG, CKY & PCFG

**Topics:** Syntax · Context-Free Grammar · CKY Parsing · Probabilistic CFG · Treebanks · Evaluation

---

## Table of Contents

1. [[#1. The Big Picture — Parallel with Week 3]]
2. [[#2. Sentence Structure & Syntax]]
3. [[#3. Phrase Chunking vs. Full Parsing]]
4. [[#4. Context-Free Grammar (CFG)]]
5. [[#5. Top-Down vs. Bottom-Up Parsing]]
6. [[#6. CKY Parsing Algorithm]]
7. [[#7. Chomsky Normal Form (CNF)]]
8. [[#8. Probabilistic CFG (PCFG)]]
9. [[#9. Training a PCFG]]
10. [[#10. Lexicalised PCFG]]
11. [[#11. Evaluation — Precision, Recall, F1]]
12. [[#12. Exam Preparation]]
13. [[#13. Connections to Other Topics]]

---

## 1. The Big Picture — Parallel with Week 3

A central theme of this lecture is that **Week 4 is structurally identical to Week 3**, just one level higher in language.

| Dimension           | Week 3 (POS Tagging)       | Week 4 (Syntactic Parsing) |
| ------------------- | -------------------------- | -------------------------- |
| Unit of analysis    | Word                       | Sentence                   |
| Structure sought    | Flat tag sequence          | Hierarchical parse tree    |
| Grammar formalism   | Finite-state machine       | Context-Free Grammar (CFG) |
| Probabilistic model | HMM                        | Probabilistic CFG (PCFG)   |
| Learning method     | MLE on POS-tagged corpus   | MLE on treebank            |
| Efficient search    | Viterbi (DP on tags)       | CKY (DP on spans)          |
| Ambiguity handling  | Most-probable tag sequence | Most-probable parse tree   |

---

## 2. Sentence Structure & Syntax

### What is Syntax?

Syntax is the study of which sentences are **well-formed** in a given grammar. Two important clarifications:

- **Well-formed ≠ meaningful.** Chomsky's famous example: _"Colourless green ideas sleep furiously."_ — grammatically perfect, semantically nonsense.
- **Ill-formed ≠ incomprehensible.** Broken English can still be understood.

### Hierarchical Sentence Structure

Sentences are not flat sequences — words group into **phrases (constituents)**, which group into larger phrases, forming a tree.

**Example:** _"I ate the spaghetti with meatballs"_

```
                  S
                 / \
                NP   VP
                |   /  \
                I  VB    NP
                   |    /   \
                  ate  NP    PP
                       |    /  \
                      the  P    NP
                           |    |
                          with  meatballs
                               and spaghetti
```

- **Constituent (phrase):** A group of words that function as a unit. Also called a _chunk_.
- **NP** = Noun Phrase, **VP** = Verb Phrase, **PP** = Prepositional Phrase

### What is a Parser?

A **parser** is an algorithm that analyses a sentence given a grammar. It can:

1. Return a **binary decision**: can this grammar generate this sentence?
2. Return the **parse tree(s)**: the hierarchical structure of the sentence

In practice, we almost always want (2).

---

## 3. Phrase Chunking vs. Full Parsing

### Phrase Chunking (Shallow Parsing)

Phrase chunking (from Week 3's sequence labeling) extracts **flat, non-recursive** noun phrases and verb phrases. It uses **BIO tagging**:

|Tag|Meaning|
|---|---|
|`B`|Beginning of a phrase|
|`I`|Inside a phrase (not first word)|
|`O`|Outside any phrase|

**Example:** _"The current account deficit strikes again"_

```
The/B-NP  current/I-NP  account/I-NP  deficit/I-NP  strikes/B-VP  again/O
```

### Why Chunking Is Not Enough

Chunking identifies _what_ the phrases are, but **not how they relate to each other**. It cannot capture:

- The hierarchical dependency between phrases
- Which PP modifies which NP
- Complex nested structures in scientific or legal text

This is why we need **full syntactic parsing**.

---

## 4. Context-Free Grammar (CFG)

### Formal Definition

A CFG is defined by four components:

|Symbol|Name|Description|
|---|---|---|
|$N$|Non-terminals|Variables: NP, VP, PP, S, NNP, VB… Never appear in final sentence|
|$\Sigma$|Terminals|The actual words in the vocabulary. Cannot generate further|
|$R$|Production rules|Rules of the form $A \to \alpha$ where $A \in N$ and $\alpha \in (N \cup \Sigma)^*$|
|$S$|Start symbol|The root of every parse tree (often written as `S` for Sentence)|

> **Key constraint:** $N \cap \Sigma = \emptyset$ — non-terminals and terminals never overlap.

### CFG Rules — Two Types

**1. Grammar rules** (non-terminal → string of non-terminals):

$$\text{S} \to \text{NP VP} \qquad \text{NP} \to \text{DT NN} \qquad \text{VP} \to \text{VB NP}$$

**2. Lexicon rules** (non-terminal → single terminal/word):

$$\text{DT} \to \text{the} \qquad \text{VB} \to \text{book} \qquad \text{NNP} \to \text{Houston}$$

### Reading a Parse Tree

- **Leaf nodes** = terminal symbols (words)
- **Internal nodes** = non-terminal symbols
- **Root** = start symbol $S$
- If the root of a tree is $S$, the sentence is grammatical in that CFG

### ✏️ Worked Example — Sentence Derivation from CFG

**Grammar (toy):**

```
S   → NP VP          NP → DT NN         VP → VB NP
DT → the             NN → flight         VB → book
NN → book            NNP → Houston      PP → with Japsley
```

**Deriving:** _"book the flight"_

$$S \Rightarrow \text{NP VP} \Rightarrow \text{NP VB NP} \Rightarrow \text{NP VB DT NN} \Rightarrow \ldots \Rightarrow \text{book the flight}$$

Each application of a rule is called a **rewrite step**. The full sequence is called a **derivation**.

---

## 5. Top-Down vs. Bottom-Up Parsing

### Top-Down Parsing

Start from $S$, recursively expand rules until the generated string matches the input.

- ✅ Always produces valid parse trees (rooted at $S$)
- ❌ Wastes effort exploring branches that can never match the input sentence
- ❌ Lots of backtracking

### Bottom-Up Parsing

Start from the words (terminals), find rules that match them, and work upward toward $S$.

- ✅ Always grounded in the actual input
- ❌ Can explore branches that never reach $S$
- ❌ Lots of backtracking

### Why Both Are Inefficient

Both approaches require roughly 20–25 steps (with heavy backtracking) for even a short 4-word sentence. The core problems are:

1. **Repeated work** — same sub-phrases are re-computed many times
2. **Wasted work** — branches are explored that can never lead to a valid parse

> **Solution:** Use **dynamic programming** to cache intermediate results → the **CKY Algorithm**.

---

## 6. CKY Parsing Algorithm

### Overview

CKY (Cocke-Kasami-Younger) is a **bottom-up, chart-based** parsing algorithm using dynamic programming.

- **Complexity:** $O(n^3)$ where $n$ is sentence length
- **Requirement:** Grammar must be in **Chomsky Normal Form (CNF)** — see Section 7
- **Key idea:** Fill a triangular table (chart) with constituents for every substring, from length 1 up to the full sentence

### The CKY Table

For a sentence of length $n$, the table has rows $i$ (start position) and columns $j$ (end position), with $0 \le i < j \le n$.

- **Cell $(i, j)$** holds all non-terminals that can generate the substring from position $i$ to $j$
- **First diagonal** (length-1 spans): filled using lexicon rules
- **Higher diagonals**: filled by trying all split points $k$, $i < k < j$

If the start symbol $S$ appears in cell $(0, n)$, the sentence is grammatical.

### Algorithm (Pseudocode Explanation)

**Phase 1 — Initialise (length-1 spans):**

For each word $w_j$ at position $j$, find all non-terminals $A$ such that $A \to w_j$ is a lexicon rule:

$$\text{table}[j-1][j] = {A \mid A \to w_j \in \text{grammar}}$$

**Phase 2 — Fill (length 2 to n spans):**

For each span length $\ell = 2, 3, \ldots, n$:  
For each start position $i$:  
Let $j = i + \ell$. Try all split points $k$ where $i < k < j$:

$$\text{If } A \to B\ C \in \text{grammar},\quad B \in \text{table}[i][k],\quad C \in \text{table}[k][j]$$

$$\text{Then add } A \text{ to } \text{table}[i][j]$$

Store with each entry: (rule used, split point $k$) as **backpointers** for tree reconstruction.

**Phase 3 — Check:**

If $S \in \text{table}[0][n]$ → sentence is grammatical. Follow backpointers to reconstruct the parse tree.

### ✏️ Worked Example — CKY Table

**Sentence:** _"book the flight"_ (positions 0, 1, 2, 3)

**Grammar (CNF):**

```
S    → NP VP    |  VP NP    |  VB NP
VP   → VB NP
NP   → DT NN
VB   → book
DT   → the
NN   → flight
NN   → book
```

**Step 1 — Length-1 spans (lexicon):**

|Cell|Span|Constituents|
|---|---|---|
|table[0][1]|_book_|VB, NN|
|table[1][2]|_the_|DT|
|table[2][3]|_flight_|NN|

**Step 2 — Length-2 spans:**

Cell table[1][3] covers _"the flight"_. Split at $k=2$:

- table[1][2] = {DT}, table[2][3] = {NN}
- Rule $\text{NP} \to \text{DT NN}$ matches → **NP** added to table[1][3]

Cell table[0][2] covers _"book the"_. Split at $k=1$:

- table[0][1] = {VB, NN}, table[1][2] = {DT}
- No rule $A \to \text{VB DT}$ or $A \to \text{NN DT}$ → **empty**

**Step 3 — Length-3 spans:**

Cell table[0][3] covers _"book the flight"_. Try split points:

Split at $k=1$: table[0][1] = {VB, NN}, table[1][3] = {NP}

- Rule $\text{S} \to \text{VB NP}$ matches → **S** added ✓
- Rule $\text{VP} \to \text{VB NP}$ matches → **VP** added

Split at $k=2$: table[0][2] = {}, table[2][3] = {NN} → nothing

**Result:** $S \in \text{table}[0][3]$ → sentence is grammatical ✓

**Parse tree recovered via backpointers:**

```
        S
       / \
      VB   NP
      |   / \
     book DT  NN
          |   |
         the flight
```

---

## 7. Chomsky Normal Form (CNF)

### What is CNF?

CKY requires that every production rule in the grammar has **exactly one of these two forms**:

$$A \to B\ C \quad \text{(two non-terminals)}$$

$$A \to w \quad \text{(exactly one terminal/word)}$$

Rules with 3+ symbols on the right, or mixed terminal/non-terminal rules, are **not** CNF.

### Why CNF?

CNF ensures that every span can be split into exactly **two** sub-spans. This is what makes the $O(n^3)$ DP possible — we only need to try binary splits.

### Converting Any CFG to CNF

**Problem 1 — Rule with 3+ non-terminals:**

$$\text{VP} \to \text{VB NP PP}$$

**Fix:** Introduce a dummy non-terminal:

$$\text{VP} \to \text{VB X}_1 \qquad \text{X}_1 \to \text{NP PP}$$

> The dummy symbol $X_1$ captures the last two elements. Recurse if needed for 4+ symbols.

**Problem 2 — Unit rule** (one non-terminal on right-hand side):

$$\text{VP} \to \text{VB}$$

**Fix:** Replace by collapsing the chain. If $\text{VB} \to \text{book}$, add:

$$\text{VP} \to \text{book}$$

This eliminates the unit rule while preserving the language.

**Problem 3 — Mixed terminal/non-terminal:**

$$\text{VP} \to \text{VB NP PP}$$

where VB here is used as a terminal — wrap terminals in dummy rules:

$$\text{VP} \to \text{X}_{\text{VB}}\ \text{NP PP} \qquad \text{X}_{\text{VB}} \to \text{book}$$

### ✏️ Worked Example — CFG to CNF Conversion

**Original rule (3 symbols on right):**

$$\text{VP} \to \text{ADV VB NP}$$

**Step 1 — Introduce dummy non-terminal:**

$$\text{VP} \to \text{ADV X}_1 \qquad \text{X}_1 \to \text{VB NP}$$

Both rules now satisfy CNF. $X_1$ appears in only one rule, so its probability is 1.0.

> **Key:** The original rule's probability stays on $\text{VP} \to \text{ADV } X_1$. The new $X_1 \to \text{VB NP}$ has probability 1.0 since it is the only rule for $X_1$.

---

## 8. Probabilistic CFG (PCFG)

### Motivation

A plain CFG can generate multiple parse trees for the same sentence (structural ambiguity). We need a way to decide which tree is **most likely**.

> **Parallel with Week 3:** CFG → PCFG is the same move as Markov Chain → HMM. We add probabilities to the grammar rules.

### Definition

A PCFG is a CFG where each production rule $A \to \alpha$ is assigned a probability $P(A \to \alpha)$, subject to:

$$\sum_{\alpha} P(A \to \alpha) = 1 \quad \text{for every non-terminal } A$$

All rules with the same left-hand side must sum to 1.

### Probability of a Parse Tree

Applying the **first-order Markov assumption**: each rule application is independent of all others. So the probability of a parse tree $\tau$ is the **product of the probabilities of all rules used**:

$$P(\tau) = \prod_{(A \to \alpha)\ \in\ \tau} P(A \to \alpha)$$

### ✏️ Worked Example — PCFG Tree Probability

**PCFG rules (probabilities in brackets):**

```
S   → NP VP    [0.9]      NP → DT NN     [0.5]
S   → VP       [0.1]      VP → VB NP     [0.7]
DT  → the      [1.0]      VP → VB PP     [0.3]
NN  → flight   [0.5]      VB → book      [0.3]
NN  → book     [0.5]
```

**Parse tree 1** uses rules: $S \to \text{NP VP}$, $\text{NP} \to \text{DT NN}$, $\text{VP} \to \text{VB NP}$, $\text{DT} \to \text{the}$, $\text{NN} \to \text{flight}$, $\text{VB} \to \text{book}$

$$P(\tau_1) = 0.9 \times 0.5 \times 0.7 \times 1.0 \times 0.5 \times 0.3 = \boxed{0.047}$$

**Parse tree 2** uses rules: $S \to \text{NP VP}$, $\text{NP} \to \text{DT NN}$, $\text{VP} \to \text{VB PP}$, $\text{DT} \to \text{the}$, $\text{NN} \to \text{flight}$, $\text{VB} \to \text{book}$

$$P(\tau_2) = 0.9 \times 0.5 \times 0.3 \times 1.0 \times 0.5 \times 0.3 = \boxed{0.020}$$

**Conclusion:** $\tau_1$ is the more probable parse. We select it as our output.

### PCFG Sentence Probability

To compute the probability of a sentence $S$ (as a language model):

$$P(S) = \sum_{\tau : \text{yield}(\tau) = S} P(\tau)$$

This sums over all parse trees that generate the sentence — analogous to the **Forward Algorithm** from Week 3.

To find the **most probable parse tree**:

$$\tau^* = \underset{\tau}{\arg\max}\ P(\tau)$$

This is analogous to the **Viterbi Algorithm** from Week 3.

### PCFG-CKY: Probabilistic Table Filling

In the probabilistic version of CKY, each cell stores not just which constituents are possible, but also their **best probability** and the backpointer that achieved it:

$$\text{table}[i][j][A] = \max_{k,\ B,\ C} \left[ P(A \to BC) \times \text{table}[i][k][B] \times \text{table}[k][j][C] \right]$$

At the end, the tree probability in table[0][n][S] is the most probable parse probability.

---

## 9. Training a PCFG

### From a Treebank

A **treebank** is a corpus of sentences manually annotated with their parse trees. Famous examples:

|Treebank|Language|Notes|
|---|---|---|
|Penn Treebank (Wall Street Journal)|English|Most widely used|
|Penn Chinese Treebank|Chinese||
|Universal Dependencies|100+ languages|Ongoing effort for low-resource languages|

### MLE Estimation

Exactly the same principle as n-gram LMs and HMMs — count and normalise:

$$P(A \to \alpha) = \frac{C(A \to \alpha)}{C(A)}$$

Count how many times rule $A \to \alpha$ appears in the treebank, divide by the total count of $A$ on the left-hand side.

### ✏️ Worked Example — PCFG MLE

**Treebank counts:**

|Rule|Count|
|---|---|
|$S \to \text{NP VP}$|80|
|$S \to \text{VP}$|20|
|Total $S$|100|

$$P(S \to \text{NP VP}) = \frac{80}{100} = 0.80$$

$$P(S \to \text{VP}) = \frac{20}{100} = 0.20$$

$$\text{Check: } 0.80 + 0.20 = 1.0 \checkmark$$

> Same principle as Week 2 n-gram MLE and Week 3 HMM MLE. MLE is the unified learning method across the course so far.

---

## 10. Lexicalised PCFG

### The Problem with Plain PCFG

Plain PCFG cannot distinguish between different uses of the same non-terminal. For example, it cannot tell that a VP headed by _"eat"_ is more likely to take an NP object than a VP headed by _"sleep"_.

### Solution: Lexicalisation

Associate each constituent with its **head word** — the most semantically important word in the span.

- Head of **NP** = main noun
- Head of **VP** = main verb
- Head of **PP** = preposition

**Plain PCFG rule:**

$$\text{VP} \to \text{VB NP} \quad [P = 0.7]$$

**Lexicalised PCFG rule:**

$$\text{VP(ate)} \to \text{VB(ate) NP(spaghetti)} \quad [P = \ldots]$$

### Trade-off

|Approach|Strength|Weakness|
|---|---|---|
|Plain PCFG|Simple, enough data|Cannot distinguish verb types|
|Full lexicalisation|Very specific|Data too sparse — many zero probabilities|
|**Head-word only**|Balance|✓ Standard approach|

> Sparse data → zero MLE probabilities → need smoothing. This is the same problem as Week 2 n-gram models and Week 3 HMM emission probabilities.

---

## 11. Evaluation — Precision, Recall, F1

### Why Not Just Accuracy?

Accuracy (percentage of correctly tagged tokens) is not enough for parse trees because:

- A parse tree is a set of **constituents (spans)**, not just a flat sequence of labels
- There are many ways to split a sentence into spans
- A model could cheat by predicting just one correct constituent and claiming 100% precision

### Constituent-Level Evaluation

We evaluate at the **constituent level**: each predicted span $[i, j, A]$ (start, end, label) is either correct or not.

$$\text{Precision} = \frac{\text{# correctly predicted constituents}}{\text{# predicted constituents}}$$

$$\text{Recall} = \frac{\text{# correctly predicted constituents}}{\text{# gold constituents in reference}}$$

$$F_1 = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

### Why We Need Both Precision and Recall

- **High precision, low recall:** Model is cautious — only predicts constituents it is sure about, but misses many. _E.g._ predict only $S \to \text{whole sentence}$ → precision = 100%, recall ≈ 0%
- **High recall, low precision:** Model predicts every possible span, but many are wrong.
- **F1** balances the two.

---

## 12. Exam Preparation

### Likely Question Type 1 — Conceptual

> _"What is the difference between phrase chunking and syntactic parsing?"_

**Model answer:** Phrase chunking (sequence labeling with BIO tags) identifies flat, non-recursive phrases. It cannot represent hierarchical relationships between phrases. Syntactic parsing produces a full parse tree capturing the nested constituent structure and dependency between phrases. Parsing is more powerful but computationally harder.

---

### Likely Question Type 2 — CFG to CNF Conversion

> _"Convert the rule $\text{VP} \to \text{ADV VB NP}$ to CNF."_

**Model answer:**

Introduce dummy non-terminal $X_1$:

$$\text{VP} \to \text{ADV}\ X_1 \qquad X_1 \to \text{VB NP}$$

Both rules have exactly two non-terminals on the right-hand side. $P(X_1 \to \text{VB NP}) = 1.0$ because $X_1$ appears in only one rule.

---

### Likely Question Type 3 — CKY Table Tracing

> _"Fill in the CKY table for sentence 'X Y Z' given grammar G."_

**Strategy:**

1. Convert grammar to CNF first (if not already)
2. Create a table with rows = start positions, columns = end positions
3. Fill length-1 diagonal using lexicon rules
4. For each longer diagonal, try all binary split points $k$
5. Check all rules $A \to BC$ where $B \in \text{table}[i][k]$ and $C \in \text{table}[k][j]$
6. If $S \in \text{table}[0][n]$, the sentence is grammatical

---

### Likely Question Type 4 — PCFG Probability

> _"Compute the probability of parse tree $\tau$ given PCFG $G$."_

**Strategy:** Multiply the probabilities of every rule used in the tree:

$$P(\tau) = \prod_{r \in \tau} P(r)$$

Make sure all rules for the same left-hand symbol sum to 1 (validity check).

---

### Likely Question Type 5 — Comparison

> _"Compare CKY parsing to top-down and bottom-up parsing."_

**Model answer:** Top-down and bottom-up parsing both suffer from repeated and wasted work, requiring ~20–25 steps (with backtracking) for a 4-word sentence. CKY uses dynamic programming to cache constituent results for all substrings in a table, achieving $O(n^3)$ complexity. It requires CNF grammar but eliminates backtracking entirely by reusing previously computed spans.

---

## 13. Connections to Other Topics

|This Week|Connection|Linked Topic|
|---|---|---|
|CFG structure|Grammar formalisms|Formal Languages (CS theory)|
|PCFG rule probabilities|Same MLE counting|n-gram LM (Wk 2), HMM (Wk 3)|
|Markov assumption in PCFG|Rule applications are independent|HMM independence assumptions (Wk 3)|
|Viterbi-like max over parse trees|Same DP max idea|Viterbi decoding (Wk 3)|
|Forward-like sum over parse trees|Same DP sum idea|Forward Algorithm (Wk 3)|
|Sparsity in lexicalised PCFG|Same zero-probability problem|Smoothing (Wk 2), HMM emissions (Wk 3)|
|POS tags as CFG pre-terminals|Tags used inside grammar rules|POS Tagging (Wk 3)|
|Treebank annotation|Labelled training data|Supervised MLE learning pattern|

---

## Summary

**Syntax** defines well-formed sentences. A parse tree captures the **hierarchical constituent structure** of a sentence — going beyond the flat label sequence of POS tagging.

**CFG** defines grammar through production rules ($A \to \alpha$). Two rule types: grammar rules (non-terminal → non-terminals) and lexicon rules (non-terminal → word).

**Top-down / Bottom-up parsing** are intuitive but inefficient due to repeated and wasted work.

**CKY** solves this with dynamic programming in $O(n^3)$. Requires CNF grammar. Fills a triangular chart from short spans to the full sentence.

**CNF** requires all rules to be either $A \to BC$ or $A \to w$. Any CFG can be converted by introducing dummy non-terminals.

**PCFG** adds probabilities to CFG rules. The probability of a parse tree is the product of all rule probabilities used:

$$P(\tau) = \prod_{r \in \tau} P(r)$$

**Training** uses MLE on a treebank:

$$P(A \to \alpha) = \frac{C(A \to \alpha)}{C(A)}$$

**Lexicalised PCFG** associates constituents with head words to improve accuracy, at the cost of data sparsity.

**Evaluation** uses constituent-level Precision, Recall, and F1 — not accuracy — because parse trees are sets of spans.

---

_FIT5217 · Week 4 Study Sheet · Monash University S1 2026_