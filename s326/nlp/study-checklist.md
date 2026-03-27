# Week 2: Language modelling (n-gram Markov models, sparsity and smoothing)

1. Explain the chain rule for calculating the probability of a sequence
    
2. For a dictionary of size 1000, how many parameters need to be estimated for a bigram and trigram LM?
    
3. Explain the Markov assumption. Write down the chain rule and then the corresponding Markov approximation for it.
    
4. Given a small corpus, list all the n-grams (e.g., n=2) and their MLE probabilities. Then calculate the probability of a given sentence based on this.
    
5. Repeat the above calculation with add-1 smoothing.
    
6. Explain the issue of OOV. Would it be more severe for English or Finnish? Why?
    
7. Mention a few solutions to handle OOVs in language modelling.
    
8. What is perplexity? Write down its formula, and explain if it is bounded or not.
    
9. Mention 3 smoothing techniques. Choose 2 and explain them with words and formulas.
    

# Week 3: Sequence Labelling (Hidden Markov Models, Forward-Backward algorithm)

1. Explain the difference between open and closed word classes
    
2. Given a sentence and a set of POS tags, tag each word with the correct tag
    
3. Provide an example of POS tagging ambiguity. Could it be resolved by looking at only a single word?
    
4. Explain 3 sequence labelling problems, provide an example for each.
    
5. What is the state transition probability? What is an observation probability?
    
6. Explain the forward algorithm, and compare its complexity with naïve solution for calculating the observation likelihood.
    
7. Write down the mathematical steps for the Viterbi algorithm and compare it with the forward algorithm.
    
8. Explain how you would learn the parameters of HMM in supervised and fully unsupervised (just a sketch of the EM algorithm is enough) settings?
    

# Week 4: Syntactic Parsing (Context Free Grammars, Syntax Parsing, Dynamic Programming, Alternative Grammars)

1. Given a set of PCFG rules and probabilities, and a given sentence, write down in detail the steps for CKY parsing, recover all possible parse trees for the sentence, and calculate the final probability of the sentence.
    
2. Why conversion to CNF is needed?
    
3. What is the use of the EM algorithm? How is it used for parsing? (just a rough sketch)
    
4. In a given parse tree, lexicalise all its non-terminal nodes
    
5. Given a balanced bracket format, draw the corresponding parse tree.
    
6. Given a pair of predicted and gold parse trees, calculate Precision, Recall, and F1 (need to know the formulas).
    
7. How would you learn the parameters of a PCFG given a treebank?
    

# Week 5: Linear Text Classification (Naive Bayes and Logistic Regression)

1. What are the similarities/differences between naïve Bayes and logistic regression?
    
2. You are asked to design a system to predict a numerical rating [1-5] a user will give to a movie based on the review she has written. What features will you extract, and what objective function (formula) will you use? What would you change if, instead of the rating, you were to predict the sentiment of the review?
    
3. What is the connection between the Sigmoid and Softmax functions? (More on this in week 6.)
    
4. What is the usage of the learning rate?
    
5. What is the difference between SGD and GD?
    
6. Explain cross-entropy loss with a formula.
    
7. What is the risk of only relying on Precision, Recall, and Accuracy in isolation?
    

# Week 6: Neural Language Models (RNN, RNN LM, Teacher Forcing)

- You are given a sentiment analysis task to predict 2 class labels (+,-) given a tweet. Draw a feedforward neural network, explain the input features you will use, and the formula to convert the output of the network into a probability over class labels + and -. What loss function will you use?
    
- How does the convexity of the loss function affect the optimisation process?
    
- Explain the difference between end-to-end learning and feature engineering.
    
- What are the limitations of n-gram LMs? How are these limitations addressed by RNN LMs?
    
- Explain the Softmax function with a formula. When do we typically use it?
    
- Write down the loss function for training an RNN LM. Explain it.
    
- What is teacher forcing, where is it used, and what issues could it cause?
    
- What does autoregressive mean?
    

# Week 7: Neural Machine Translation (Seq2Seq, Attention)

1. What is a parallel corpus?
    
2. What does alignment mean in MT?
    
3. Explain Seq2Seq models using Machine Translation as an example.
    
4. Name 3 decoding strategies for MT, and explain their differences.
    
5. What is the effect of beam size in beam-search decoding?
    
6. How would you correct for the candidates’ length mismatch in MT?
    
7. What metric to use for evaluating an MT system? Explain without a formula.
    
8. Use an example and explain the Attention mechanism.
    
9. Given vectors for the hidden states of the encoder and decoder, calculate attention scores, attention distribution, and attention output. Write down the formulas used and details of the calculation.
    
10. What is an autoencoder, and why is it typically used?
    
11. Name 2-3 tasks that could be framed as an Encoder-Decoder problem.
    

# Week 8: Static and Contextualised Distributional Semantics (Word2Vec, GLoVe, fastText, ELMo, BERT)

1. What is the difference between Lexical and Distributional semantics?
    
2. What is the problem with count-based representations like tf-idf?
    
3. Given a toy corpus, explain PMI and PMI of word pairs.
    
4. What are the limitations of embedding models like Word2vec and GloVe?
    
5. In full detail, explain different ways of training the Word2Vec model.
    
6. What is the difference between GloVe and the Word2Vec model?
    
7. Different ways of evaluating a representation learning model.
    
8. What is the advantage of fastText over word2vec? Between English and Turkish, which one is more likely to benefit from fastText and why?
    
9. What are the similarities and differences between BERT/ELMo and word2vec/GloVe?
    
10. Why do we need contextualised word embedding models?
    

# Week 9: Transformer Language Models, Impacts and Implications (Transformers, Pre-training, Parameter-Efficient Fine-tuning)

1. Explain self-attention. Is it order sensitive?
    
2. Explain the residual connection and why it is needed in Transformers.
    
3. Explain Layer Norm and why it is needed in Transformers.
    
4. Explain scaling in Dot Product Attention with a formula. Why is it needed?
    
5. What is the difference between self-attention and masked self-attention? Which one is used in the decoder of transformers, and why?
    
6. What is the motivation for having multi-head self-attention?
    
7. Explain one solution to inject positional information into transformers.
    
8. What is the difference between pretraining and fine-tuning?
    
9. What is the difference between LLaMA, T5, and RoBERTa models?
    
10. Explain the 3 token prediction subtasks BERT uses.
    
11. Name and explain 4 ways of fine-tuning BERT for a downstream task. Rank them in order of the number of parameters being updated.
    

# Week 10: Neural Speech Recognition and Translation (Speech Transformers)

1. What’s the key difference between speech signals and text?
    
2. What are the key challenges in dealing with speech in designing a deep learning model, and what are the solutions?
    
3. What are the common speech tasks, and what are the applications of these technologies?
    
4. What are the two approaches to performing speech recognition? What are the pros and cons of these approaches?
    
5. Why is it useful to have a language model added to a speech recognition model?
    
6. What are the two approaches to performing speech translation? What are the pros and cons of these approaches?
    
7. How can one address the data scarcity issue faced with speech translation?
    
8. Why is it more challenging to learn speech representations than text representations?
    
9. How can one make use of wav2vec2 for downstream applications? What are the concerns?
    
10. What is the difference between an ASR system and a text-to-speech synthesiser?
    

# Week 11: Advanced Topics in Large Language Models 1 (In-context Learning, Instruction Tuning, Preference Alignment)

1. If you were asked to learn a reward model for RLHF, would you train it to predict the human rating or pairwise ranking? Why?
    
2. If your language model had racial biases, how would you collect data for RLHF to align your LLM's output to be unbiased? Once you have done this, how would you verify if your approach worked?
    
3. For what sorts of tasks chain-of-thought would be better than standard prompting?
    
4. How about retrieval augmented prompting? Would you use it for numerical reasoning?
    
5. Could we produce both the instructions and feedback from an LLM?
    
6. Explain the difference between System 1 and System 2 modes of thinking and reasoning.
    

# Week 12: Advanced Topics in Large Language Models 2 (Tool/RAG Augmentation, Self-Correction, Language Agents)

1. What limitations of LLMs are addressed by RAG?
    
2. Why the size of LLM contexts is relevant to RAG?
    
3. Consider a mathematical reasoning question (i.e., from GSMK8K). Describe a strategy which uses both an LLM and an external tool to solve the question.
    
4. Describe a multi-agent design which solves a logical reasoning problem. What role does each of the agents serve?
    
5. What elements does a language agent possess that differentiate it from a standard LLM?
    
6. What is the difference between ORM and PRM? How are they used?