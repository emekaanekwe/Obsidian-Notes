## Definitions

1. Explain the chain rule for calculating the probability of a sequence
2. For a dictionary of size 1000, how many parameters need to be estimated for a bigram and trigram LM?
3.1 **Explain the Markov assumption.** 
a model that assumes the k-th order where the next state of the model depends on k recent states
3.2 **Write down the chain rule** 

3.3 **then the corresponding Markov approximation for it.**

$$P(W^N_1)=\prod^N_{i=1}P(w_i|w^{i-1}_1) \approx \prod^N_{i=1}P(w_i|w^{i-1}_{i-n+1})$$


3. Given a small corpus, list all the n-grams (e.g., n=2) and their MLE probabilities. Then calculate the probability of a given sentence based on this.
4. Repeat the above calculation with add-1 smoothing.
5. Explain the issue of OOV. Would it be more severe for English or Finnish? Why?
6. Mention a few solutions to handle OOVs in language modelling.
7. What is perplexity, write down its formula, and explain if it is bounded or not.
8. Mention 3 smoothing techniques. Choose 2 and explain them with words and formula.
---
- Explain the difference between open and closed word classes
- Given a sentence, and a set of POS tags, tag each word with the correct tag
- Provide example of POS tagging ambiguity. Could it be resolved by looking at only a single word?
- Explain 3 sequence labelling problems, provide an example for each.
- What is the state transition probability? What is an observation probability?
- Explain the forward algorithm, and compare its complexity with naïve solution for calculating the observation likelihood.
- Write down the mathematical steps for Viterbi algorithm and compare it with the forward algorithm.
- Explain how would you learn the parameters of HMM in supervised, and fully unsupervised (just an sketch of EM algorithm is enough) settings?