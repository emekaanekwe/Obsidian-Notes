**Study Guide: Week 2 - Language Modelling**

**1. Explain the chain rule for calculating the probability of a sequence.**

The chain rule is a fundamental rule of probability that allows us to calculate the joint probability of a sequence of random variables. In the context of language modeling, this means calculating the probability of a sequence of words, $W = w_1, w_2, \dots, w_n$. The chain rule states that the probability of this sequence can be calculated as the product of the conditional probability of each word given all the preceding words in the sequence.

The formula for the chain rule for a sequence of words $w_1, w_2, \dots, w_n$ is:

$P(w_1, w_2, \dots, w_n) = P(w_1) \times P(w_2 | w_1) \times P(w_3 | w_1, w_2) \times \dots \times P(w_n | w_1, w_2, \dots, w_{n-1})$

This can be written more compactly using product notation:

$P(w_1, \dots, w_n) = \prod_{i=1}^n P(w_i | w_1, \dots, w_{i-1})$

To calculate the probability of a text using a language model, this chain rule is often applied to the entire sequence of words in a test set.

**2. For a dictionary of size 1000, how many parameters need to be estimated for a bigram and trigram LM?**

In n-gram language models, parameters correspond to the conditional probabilities $P(w_i | w_{i-n+1}, \dots, w_{i-1})$. The number of parameters to be estimated is the number of possible n-grams in the language model, which is determined by the size of the vocabulary (dictionary).

Let $V$ be the size of the dictionary (vocabulary). If $V=1000$, the number of parameters to estimate is:

- **Bigram LM (n=2):** A bigram model estimates the probability of a word given the preceding word, $P(w_i | w_{i-1})$. There are $V$ possible words for $w_{i-1}$ and $V$ possible words for $w_i$. Thus, there are $V \times V = V^2$ possible bigrams. For a dictionary size of 1000, this is $1000^2 = \mathbf{1,000,000}$ parameters.
- **Trigram LM (n=3):** A trigram model estimates the probability of a word given the two preceding words, $P(w_i | w_{i-2}, w_{i-1})$. There are $V$ possible words for $w_{i-2}$, $V$ for $w_{i-1}$, and $V$ for $w_i$. Thus, there are $V \times V \times V = V^3$ possible trigrams. For a dictionary size of 1000, this is $1000^3 = \mathbf{1,000,000,000}$ parameters.

It's important to note that this calculation represents the theoretical maximum number of parameters based on all possible combinations of words in the vocabulary. In practice, most of these possible n-grams will never appear in any given training corpus, leading to the issue of **sparsity**.

**3. Explain the Markov assumption. Write down the chain rule and then the corresponding Markov approximation for it.**

- **The Markov Assumption:** The Markov assumption is a simplifying assumption made in n-gram language models to make parameter estimation tractable. It states that the probability of the next word in a sequence depends only on a fixed, limited number of preceding words, rather than the entire history of words. For an n-gram model, this fixed context size is $n-1$ words.
- **Chain Rule:** As explained in question 1, the chain rule for a sequence $w_1, \dots, w_n$ is: $P(w_1, \dots, w_n) = \prod_{i=1}^n P(w_i | w_1, \dots, w_{i-1})$
- **Markov Approximation:** Applying the Markov assumption for an n-gram model, the conditional probability of a word $w_i$ given its entire history $w_1, \dots, w_{i-1}$ is approximated by the probability of $w_i$ given only the $n-1$ preceding words: $P(w_i | w_1, \dots, w_{i-1}) \approx P(w_i | w_{i-n+1}, \dots, w_{i-1})$
- Using this approximation, the chain rule for the probability of the entire sequence is approximated as: $P(w_1, \dots, w_n) \approx \prod_{i=1}^n P(w_i | w_{i-n+1}, \dots, w_{i-1})$
- For a bigram model (n=2), this simplifies to: $P(w_1, \dots, w_n) \approx \prod_{i=1}^n P(w_i | w_{i-1})$
- For a trigram model (n=3), this becomes: $P(w_1, \dots, w_n) \approx \prod_{i=1}^n P(w_i | w_{i-2}, w_{i-1})$

**4. Given a small corpus, list all the n-grams (e.g., n=2) and their MLE probabilities. Then calculate the probability of a given sentence based on this.**

Let's use the "I am Sam" corpus provided in the source: `<s> I am Sam </s>` `<s> Sam I am </s>` `<s> I am Sam </s>` `<s> I do not like green eggs and Sam </s>`

And let's focus on bigrams (n=2). The vocabulary includes `<s>`, `</s>`, I, am, Sam, do, not, like, green, eggs, and.

First, we need the counts of bigrams and unigrams (to serve as denominators) from the corpus:

- `<s> I`: 3
- `I am`: 3
- `am Sam`: 2
- `Sam </s>`: 2
- `am </s>`: 1
- `Sam I`: 1
- `I do`: 1
- `do not`: 1
- `not like`: 1
- `like green`: 1
- `green eggs`: 1
- `eggs and`: 1
- `and Sam`: 1
- Unigram counts (as bigram prefixes): `<s>`: 3, `I`: 4, `am`: 3, `Sam`: 3, `do`: 1, `not`: 1, `like`: 1, `green`: 1, `eggs`: 1, `and`: 1.

Maximum Likelihood Estimation (MLE) probabilities are calculated as the frequency of an n-gram divided by the frequency of its prefix (the history).

For example, some bigram MLE probabilities:

- $P_{MLE}(I | <s>) = \frac{Count(<s> I)}{Count(<s>)} = \frac{3}{3} = 1$
- $P_{MLE}(am | I) = \frac{Count(I am)}{Count(I)} = \frac{3}{4}$
- $P_{MLE}(Sam | am) = \frac{Count(am Sam)}{Count(am)} = \frac{2}{3}$
- $P_{MLE}(</s> | Sam) = \frac{Count(Sam </s>)}{Count(Sam)} = \frac{2}{3}$
- $P_{MLE}(do | I) = \frac{Count(I do)}{Count(I)} = \frac{1}{4}$

Now, let's calculate the probability of the sentence "i want chinese food" using an unsmoothed bigram model, as requested, using the provided bigram counts and "useful probabilities": Corpus from: `i`, `want`, `to`, `eat`, `chinese`, `food`, `lunch`, `spend`. Counts from:

- `i want`: 2
- `want chinese`: 1
- `chinese food`: 1
- `food </s>`: 0 (from this table, though provides one for add-1 calculation)
- Counts of prefixes: `i`: 827+2=829, `want`: 2+608+1+6+6+5+1+2=631, `chinese`: 1+82+1=84, `food`: 15+15+1=31. Let's use the marginals provided in the table where available. `i`: 827, `want`: 608, `chinese`: 82, `food`: 4. (Note: The table counts seem inconsistent with token counts if summing rows/cols, e.g., 'i' row sum is 850, but row marginal is 827. Let's use the marginals as the denominators).
- $P_{MLE}(want | i) = \frac{Count(i want)}{Count(i)} = \frac{2}{827}$
- $P_{MLE}(chinese | want) = \frac{Count(want chinese)}{Count(want)} = \frac{1}{608}$
- $P_{MLE}(food | chinese) = \frac{Count(chinese food)}{Count(chinese)} = \frac{1}{82}$
- $P_{MLE}(</s> | food)$ - not available directly, but we need to include `<s>` and `</s>` context. Let's assume we have $P(i|<s>)$ and $P(</s>|food)$ available or estimated from a larger corpus. If any of the required bigrams have a count of 0 in the training data (and assuming no smoothing is used), the MLE probability for that bigram is 0, and thus the probability of the entire sentence becomes 0. The table shows many zero counts, illustrating the sparsity issue. For instance, `i chinese` has count 0.

**5. Repeat the above calculation with add-1 smoothing.**

Add-one smoothing (also called Laplace smoothing) is a technique to address the sparsity problem by giving a count of one to all unseen n-grams before normalizing. This ensures that no n-gram probability is zero.

The formula for Add-one smoothing for a bigram $P(w_i | w_{i-1})$ is:

$P_{Add-1}(w_i | w_{i-1}) = \frac{Count(w_{i-1}w_i) + 1}{Count(w_{i-1}) + |V|}$

where $|V|$ is the size of the vocabulary (all possible words $w_i$ could be).

Let's calculate the probability of "i want chinese food" using the add-1 smoothed table mentioned in the source. The source mentions assuming $P(i|<s>)= 0.19$ and $P(</s>|food)= 0.40$ for this calculation with the smoothed table (which is not included in the excerpts, but would contain smoothed counts/probabilities like those in Fig 3.7 of the full J&M text). Using these provided smoothed probabilities:

$P(\text{"i want chinese food"})$ $= P(i | <s>) \times P(want | i) \times P(chinese | want) \times P(food | chinese) \times P(</s> | food)$

Using the example add-1 probabilities from the question part 5: $= 0.19 \times P_{add-1}(want | i) \times P_{add-1}(chinese | want) \times P_{add-1}(food | chinese) \times 0.40$

(The specific smoothed probabilities for `want|i`, `chinese|want`, `food|chinese` would come from the smoothed table based on a vocabulary V and counts from or a larger corpus).

Smoothing will result in a non-zero probability for the sentence, even if some bigrams in the sentence were not observed in the original training data. Comparing the unsmoothed and smoothed probabilities (as asked in), the smoothed probability will be higher than the unsmoothed probability if the unsmoothed probability was zero due to unseen n-grams.

**6. Explain the issue of OOV. Would it be more severe for English or Finnish? Why?**

- **Out-of-Vocabulary (OOV) Issue:** The OOV issue arises when a language model encounters a word in the test or runtime data that was not present in its training vocabulary. Standard n-gram models assign a zero probability to any occurrence of an OOV word or any n-gram containing an OOV word. Without smoothing techniques to handle this, the probability of any sequence containing an OOV word would be zero, making the language model unable to process such sequences effectively.
- **English vs Finnish:** The OOV problem is generally **more severe for Finnish than for English**. This is primarily due to Finnish being a morphologically richer language than English.
    - **English** has relatively simple morphology. Word forms change less drastically (e.g., adding '-s' for plurals or third person singular, '-ed' for past tense).
    - **Finnish** is an agglutinative language with complex morphology. Words can have many different endings (suffixes and infixes) indicating grammatical functions like case, possession, and tense, which are often expressed by separate words (like prepositions) in English. This results in a much larger number of unique word forms (types) for the same underlying concepts compared to English. The **type-token ratio** (the ratio of unique words to the total number of words) tends to be higher in Finnish.
    - Consequently, for a given size of training corpus, a model trained on Finnish is much more likely to encounter a word form in unseen test data that it has never seen before (an OOV word) compared to a model trained on English.

**7. Mention a few solutions to handle OOVs in language modelling.**

Several approaches exist to mitigate the OOV problem in language modeling:

- **Replace rare words with a special token:** During training, words that occur very infrequently in the corpus can be replaced with a special out-of-vocabulary token, often denoted as `<UNK>`. The language model then learns probabilities associated with this `<UNK>` token, allowing it to assign non-zero probabilities to sequences containing unseen words during inference.
- **Subword Tokenization:** Instead of treating words as atomic units, the text can be broken down into subword units (like morphemes or character sequences). Algorithms like Byte Pair Encoding (BPE) or WordPiece are used for this. This allows the model to build representations and probabilities for words it hasn't seen before by composing representations of their subword units, as these units are more likely to have appeared in the training data. Models like fastText leverage subword information. This is particularly beneficial for morphologically rich languages like Finnish.

**8. What is perplexity? Write down its formula, and explain if it is bounded or not.**

- **Perplexity:** Perplexity is a common intrinsic evaluation metric for language models. It measures how well a probability model predicts a sample of text. Informally, it can be thought of as the weighted average number of choices the language model has for the next word. A lower perplexity score indicates a better language model because it means the model assigns higher probabilities to the words in the test sequence. The term "perplexity" was first used in the context of language modeling by the IBM speech recognition group in the 1970s and 1980s.
- **Formula:** Given a test sequence of words $W = w_1, w_2, \dots, w_N$, the perplexity of the sequence according to a language model $P$ is defined as the inverse probability of the sequence, normalized by the number of words $N$:
- $PP(W) = P(w_1, \dots, w_N)^{-1/N}$
- Using the chain rule and the Markov approximation for an n-gram model, this is typically computed as:
- $PP(W) = \left( \prod_{i=1}^N P(w_i | w_{i-n+1}, \dots, w_{i-1}) \right)^{-1/N}$
- **Boundedness:** Perplexity is **bounded below by 1**. The minimum possible probability for any sequence is 0, and the maximum is 1.
    - If a language model assigns probability 1 to the entire sequence (which happens if the model is perfect and the sequence is the only possible one), $PP(W) = 1^{-1/N} = 1$. This is the ideal lower bound.
    - The probability of a sequence, $P(W)$, must be $P(W) \le 1$. Therefore, $P(W)^{-1/N} \ge 1^{-1/N} = 1$.
    - Perplexity is **not bounded above in theory**. If a language model assigns a probability of 0 to any word in the test sequence (e.g., due to encountering an unseen n-gram without smoothing), the probability of the entire sequence $P(W)$ becomes 0. Division by zero or $0^{-1/N}$ results in an infinite or undefined perplexity.
    - However, with proper smoothing techniques that guarantee $P(W) > 0$ for any sequence, the perplexity is **bounded above in practice**.

**9. Mention 3 smoothing techniques. Choose 2 and explain them with words and formulas.**

Here are three smoothing techniques discussed in the sources:

1. **Add-one Smoothing (Add-k Smoothing)**
2. **Kneser-Ney Smoothing (and Modified Kneser-Ney)**
3. **Linear Interpolation Smoothing** _(Another mentioned is a non-parametric Bayesian prior__, and Good-Turing__)._

Let's explain Add-one Smoothing and Kneser-Ney Smoothing:

- **Add-one Smoothing (Add-k Smoothing):** This is one of the simplest smoothing techniques. The idea is to add a small constant value (commonly 1, hence "add-one") to all n-gram counts, including those that were zero, before normalizing to calculate probabilities. While easy to implement, adding 1 to every count can significantly distort the original MLE probabilities, especially for low-frequency n-grams, assigning too much probability mass to unseen events. Add-k is a generalization where a value 'k' (e.g., 0.5) is added instead of 1.
- The formula for Add-one smoothing for an n-gram $w_{i-n+1}, \dots, w_i$ is: $P_{Add-1}(w_i | w_{i-n+1}, \dots, w_{i-1}) = \frac{Count(w_{i-n+1}, \dots, w_i) + 1}{Count(w_{i-n+1}, \dots, w_{i-1}) + |V|}$ where $|V|$ is the size of the vocabulary. For bigrams: $P_{Add-1}(w_i | w_{i-1}) = \frac{Count(w_{i-1}w_i) + 1}{Count(w_{i-1}) + |V|}$
- **Kneser-Ney Smoothing (KN) and Modified Kneser-Ney (MKN):** These are more advanced and effective smoothing techniques, considered the de-facto standard for n-gram LMs. They are based on the principle of **absolute discounting**, where a fixed discount value is subtracted from the counts of observed n-grams. The discounted probability mass is then distributed among lower-order n-grams using **interpolation**. A key innovation of Kneser-Ney is that the lower-order distribution is not simply the MLE lower-order probability, but one that considers **how often a word appears as a** **novel continuation** **in different contexts**. This helps assign higher probabilities to words that are likely to appear after various other words, even if they don't appear frequently as unigrams (e.g., "San Francisco" makes "Francisco" a common bigram second word, but "Francisco" might be rare as a unigram. KN handles this).
- Modified Kneser-Ney (MKN) is an improvement that uses different discount parameters for different frequency counts. For instance, one implementation uses three different discount parameters at each n-gram level. The interpolation mixes the discounted higher-order model with the lower-order model.
- A simplified formula illustrating the interpolation idea for bigrams (KN): $P_{KN}(w_i | w_{i-1}) = \frac{\max(Count(w_{i-1}w_i) - D, 0)}{Count(w_{i-1})} + \lambda_{w_{i-1}} P_{KN}(w_i)$
- Here, $D$ is a discount factor, $\frac{\max(Count(w_{i-1}w_i) - D, 0)}{Count(w_{i-1})}$ is the discounted probability mass for the observed bigram, $\lambda_{w_{i-1}}$ is the amount of probability mass "left over" after discounting the counts starting with $w_{i-1}$, and $P_{KN}(w_i)$ is the KN unigram probability (which depends on how many different words precede $w_i$). The calculation of $\lambda$ and the lower-order probabilities are specific to the KN/MKN algorithm. Recent work focuses on optimising these parameters and making computations efficient for large data.

These techniques, particularly MKN, are crucial for building effective n-gram language models by providing better estimates for unseen or rare word sequences and improving performance on tasks like speech recognition by reducing word-error rates.