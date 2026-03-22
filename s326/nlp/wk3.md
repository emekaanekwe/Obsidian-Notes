### Self-assess checklist

- Explain the difference between open and closed word classes
- Given a sentence, and a set of POS tags, tag each word with the correct tag
- Provide example of POS tagging ambiguity. Could it be resolved by looking at only a single word?
- Explain 3 sequence labelling problems, provide an example for each.
- What is the state transition probability? What is an observation probability?
- Explain the forward algorithm, and compare its complexity with naïve solution for calculating the observation likelihood.
- Write down the mathematical steps for Viterbi algorithm and compare it with the forward algorithm.
- Explain how would you learn the parameters of HMM in supervised, and fully unsupervised (just an sketch of EM algorithm is enough) settings?


### word categories
![[Pasted image 20260319111429.png]]

### parts of speech tagging

*(most popular is the pen treebank)*
**NOUNS**
• Singular (NN): dog, fork
• Plural (NNS): dogs, forks
• Proper (NNP, NNPS): John, Springfields
• Personal pronoun (PRP): I, you, he, she, it
• Wh-pronoun (WP): who, what
**VERBS**
• Base, infinitive (VB): eat
• Past tense (VBD): ate
• Gerund (VBG): is eating
• Past participle (VBN): has eaten
• Non 3rd person singular present tense (VBP): I eat
• 3rd person singular present tense: (VBZ): it eats
• Modal (MD): should, can
• To (TO): to (to eat)
**ADJECTIVES**
• Basic (JJ): red, tall
• Comparative (JJR): redder, taller
• Superlative (JJS): reddest, tallest
**ADVERBS**
• Basic (RB): quickly
• Comparative (RBR): quicker
• Superlative (RBS): quickest
**PREPOSITIONS** 
• (IN): on, in, by, to, with
**DETERMINER**
• Basic (DT) a, an, the
• WH-determiner (WDT): which, that
**COORDINATING CONJUNCTION**
• (CC): and, but, or
**PARTICLE** 
(RP): off (took off), up (put up)

### nlp goal: sequence labeling

**labeled tokens**
*people           org               place*
emeka          monash      clayton

**modeling sequences using probability**
allow integration uncertainty over interdependent classifications

***So, sequence labeling concerns using probabilistic sequence modeling to determine the most likely global assignment labels of tokens that are dependent on the labels of other tokens in a sequence (i.i.d)***

### sequence labeling using the hidden markov model


