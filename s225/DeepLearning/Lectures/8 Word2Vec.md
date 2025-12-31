### Detailed Analysis of Lecture: Learned Representations & Word Embeddings

This lecture's core theme is the evolution from **feature engineering** to **feature learning**. In traditional ML, we designed features (like TF-IDF). In deep learning, the model *learns* the optimal feature representations directly from the data. Word2Vec is a quintessential example of this paradigm.

---

### #1. Automated Feature Extraction vs. Handcrafted Features

**# The Core Concept**
The goal is to find a transformation that maps raw, high-dimensional, and often sparse data (like text) into a lower-dimensional, dense vector space where semantic relationships are preserved.

**## Prerequisite Knowledge & Refresher**
*   **Sparsity:** A one-hot encoded vector for a vocabulary of size 100,000 is 99.999% zeros. This is computationally inefficient and lacks any notion of meaning.
*   **Dense Representations:** A learned embedding of size 300 is a vector of 300 real numbers. This is computationally efficient and can encode meaning in the relative values of its dimensions.

**## The Analogy: Handcrafted vs. Learned Maps**
*   **Handcrafted Features (TF-IDF, SIFT):** Like being given a detailed, pre-drawn map of a city. It's very useful, but it was created by a cartographer (a human expert) with a specific purpose. If the city changes, the map becomes outdated. **TF-IDF is a brilliant, hand-designed feature for text.**
*   **Learned Features (Word2Vec, Autoencoders):** Like learning to navigate a city by walking around and understanding the relationships between neighborhoods. You develop an internal, intuitive "map" based on experience. This map is adaptive and can capture subtle relationships the cartographer might have missed. **Word2Vec learns a "map" of the semantic landscape of a language.**

The lecture correctly states that the "beauty of DNN" is **end-to-end learning**: the feature extraction and classification layers are trained together, allowing the features to be optimally tuned for the final task.

---

### #2. The Importance of Data Representation & Deep Architectures

Good representations are the foundation of effective deep architectures.

*   **Why it Matters:** The lecture uses the example of `"isn't"` vs. `"is not"`. To a computer, these are completely different symbols. A good representation should map them to nearby points in the vector space because they have similar *meanings*.
*   **Deep Architectures:** Stacking layers (e.g., h1, h2, h3) allows the network to build a hierarchy of representations.
    *   **Lower Layers:** Might learn simple features (e.g., edges in an image, or word stems/syntax in text).
    *   **Higher Layers:** Combine these simple features to learn complex, abstract concepts (e.g., object shapes in an image, or sentiment/topic in text).

---

### #3. Text Analytics Pipeline (Pre-Deep Learning)

This is the traditional approach, which is important to understand as a baseline.

1.  **Text Normalization:** Standardizing the text (`"isn't" -> "is not"`, `"running" -> "run"`, remove stopwords). This reduces variability.
2.  **Feature Extraction -> Bag-of-Words (BoW):** Represent a document as a vector of word counts. `Document = [count("word1"), count("word2"), ...]`. This ignores word order.
3.  **Feature Weighting -> TF-IDF:** Improves BoW by weighting terms based on their importance.
    *   **TF (Term Frequency):** How often a term appears in a document. `TF(t, d) = count(t in d) / total words in d`
    *   **IDF (Inverse Document Frequency):** How unique a term is across the corpus. `IDF(t) = log( Total Documents / (Number of documents containing t) )`
    *   **TF-IDF:** `TF-IDF(t, d) = TF(t, d) * IDF(t)`
    *   **Why IDF?** As the lecture explains, a word like `"Cristiano Ronaldo"` that appears in very few documents is a strong signal (high IDF). A word like `"the"` that appears everywhere is not informative (IDF ~ 0).

---

### #4. Word Embeddings with Word2Vec

This is the heart of the lecture. Word2Vec learns dense embeddings by leveraging the **distributional hypothesis**: "You shall know a word by the company it keeps."

**# The Mathematical Model: Skip-gram and CBOW**

Both models use a simple neural network with a single hidden layer to learn the embeddings.

*   **Vocabulary Size:** `V` (e.g., 10,000 words)
*   **Embedding Dimension:** `N` (e.g., 300). This is the size of the learned representation.
*   **Input:** A one-hot encoded vector `x` of size `V`.
*   **Hidden Layer:** The embedding for the input word. It's calculated by a weight matrix `W` (of size `V x N`). The operation `h = x^T * W` essentially **selects the row** of `W` corresponding to the input word. **This weight matrix `W` becomes our lookup table of word embeddings.**
*   **Output Layer:** A weight matrix `W'` (of size `N x V`) projects the hidden layer back to a vector of size `V`, followed by a softmax to get a probability distribution over the vocabulary.

**## Prerequisite Knowledge: Softmax and Computational Cost**
*   **Softmax Function:** Turns a vector of real numbers (logits) into a probability distribution.
    `softmax(z_i) = exp(z_i) / sum(exp(z_j) for j=1 to V)`
*   **The Problem:** Calculating the denominator requires a sum over the *entire vocabulary* `V`. If `V=100,000`, this is computationally expensive for every single training example.

#### Model 1: Skip-gram
*   **Task:** Predict context words given a target word. (Target -> Context)
*   **Architecture:** Input is the one-hot vector of the target word (e.g., `"fox"`). The network tries to predict the one-hot vectors of the surrounding context words (e.g., `"quick"`, `"brown"`, `"jumps"`, `"over"`).
*   **Objective Function:** Maximize the average log probability of the context words given the target word.
    `(1/T) * ∑_{t=1}^T ∑_{-c ≤ j ≤ c, j ≠ 0} log p(w_{t+j} | w_t)`
    Where `c` is the context window size.

#### Model 2: Continuous Bag-of-Words (CBOW)
*   **Task:** Predict a target word given its context words. (Context -> Target)
*   **Architecture:** The input is the *average* of the one-hot vectors of the context words. The network tries to predict the one-hot vector of the target word in the middle.
*   **Objective Function:** Maximize the log probability of the target word given the context words.

**## Step-by-Step Intuition**
1.  **Self-Supervised Learning:** The "labels" are generated for free from the text itself. For every sentence, you can create multiple (target, context) pairs.
2.  **Training:** The network is trained to be good at this "fake" prediction task. For example, when it sees `"fox"`, it should assign high probability to words like `"quick"` and `"jumps"`.
3.  **The Byproduct:** The *real* goal is not the output layer, but the hidden layer weights `W`. After training, by performing the task of predicting context, the model is forced to organize the rows of `W` so that words with similar contexts have similar vector representations. **The weights `W` are the word embeddings.**

**### Solving the Softmax Problem: Negative Sampling**
The lecture correctly identifies the computational bottleneck of the full softmax. The solution is **Negative Sampling**, which transforms the problem from a `V`-class classification into a series of binary classification problems.

*   **For each (target, context) pair:** We treat this as a positive example (label=1).
*   **We then sample `k` "negative" words** (e.g., `k=5`) that do *not* appear in the context. These are negative examples (label=0).
*   **The new objective:** The model now only has to decide if a word is a true context word or a negative sample. This is vastly more efficient than predicting over all `V` words.

---

### #5. Properties and Applications of Word Embeddings

*   **Semantic Meaning:** The vectors capture meaning. The famous example: `vec("King") - vec("Man") + vec("Woman") ≈ vec("Queen")`. This shows that relationships like gender can be encoded as vector offsets.
*   **Visualization:** High-dimensional embeddings can be projected to 2D/3D using t-SNE for visualization, where semantically similar words cluster together.
*   **Transfer Learning:** Pre-trained word embeddings (like GloVe or Word2Vec) can be used as a fixed first layer in a network for a specific task (e.g., sentiment analysis), providing a huge head start.
*   **Generalization to Other Domains:** The concept of learning embeddings is universal.
    *   **Doc2Vec:** Learns embeddings for entire documents/paragraphs.
    *   **Node2Vec:** Learns embeddings for nodes in a graph (e.g., in a social network, similar users will have similar embeddings).

### Summary: Key Takeaways for Your Exam

1.  **Feature Learning is Key:** The shift from handcrafted features (TF-IDF) to learned representations (embeddings) is a fundamental principle of deep learning.
2.  **Word2Vec is a Self-Supervised Model:** It creates its own training task from unlabeled text. Understand the difference between Skip-gram (target->context) and CBOW (context->target).
3.  **The Embedding Matrix:** The primary output of Word2Vec is the weight matrix that maps a one-hot input to a dense hidden layer. This matrix is the lookup table for word vectors.
4.  **Negative Sampling is Crucial:** It's an efficient approximation that makes training on large vocabularies feasible by converting a massive softmax into binary logistic regression problems.
5.  **Embeddings Capture Semantics:** The geometry of the embedding space reflects semantic relationships. Analogies can be solved via vector arithmetic.

# Lab: Advanced Sequential Models, Transformers & Seq2seq

