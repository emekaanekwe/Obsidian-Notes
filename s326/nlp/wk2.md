**Self-assess checklist**
- Explain the chain rule for calculating the probability of a sequence
- For a dictionary of size 1000, how many parameters need to be estimated for a bigram and trigram LM?
- Explain the Markov assumption. Write down the chain rule and then the corresponding Markov approximation for it.
- Given a small corpus, list all the n-grams (e.g., n=2) and their MLE probabilities. Then calculate the probability of a given sentence based on this.
- Repeat the above calculation with add-1 smoothing.
- Explain the issue of OOV. Would it be more severe for English or Finnish? Why?
- Mention a few solutions to handle OOVs in language modelling.
- What is perplexity, write down its formula, and explain if it is bounded or not.
- Mention 3 smoothing techniques. Choose 2 and explain them with words and formula.

# slides
### context length
kafursky
![[Pasted image 20260318182938.png]]

google
![[Pasted image 20260318183034.png]]

***the maximum amount of tokens that can be processed in one epoch where tokens * length/width is the size of the matrix***


### chain rule of Pr

**assume**
*rules*
$P(A|B)=\frac{P(A,B)}{P(B)}$ and $P(A|B)=P(A|B)P(B)$
*data*
s = "one two three"
a = "one"
b = "two"
c = "three"
*infer*
$P(W)$
p(w) = p(a|b,c)P(b|a)p(c|a,b)p(s)



 ***$P(W)$ is the joint probability over the words in the sentence. In short, $P(W^N_1) = \prod^N_{i=1}P(W_i|W^{i-1}_1)$***

### What the chain rule of Pr measures

***the multinomial distro over the vocab of how likely it is for each word to appear after the sequence***

### how to compute the measurement

$p(s=c|a,b)=\frac{count(s)}{count(a,b)}$

***The Maximum Likelihood Estimation***

#### Problem: for each increase in context, the number of likely predicted words grows exponentially

### solution: markov and n-grams

predict the next word in a sent based on $n-X$ preceding words of context
**1-grams**: predict based on 0 words
**2-grams**: predict based on 1 word
**3-grams**: predict based on 2 words

![[Pasted image 20260318185259.png]]

Consider each sentence placed in a matrix with the first word in the first column. Then count how many occurrences of the words.

---

# n-grams code

```
text = [['a', 'b', 'c'], ['a', 'c', 'd', 'c', 'e', 'f']]
list(bigrams(text[0]))
list(ngrams(text[1], n=3))
--> [('a', 'c', 'd'), ('c', 'd', 'c'), ('d', 'c', 'e'), ('c', 'e', 'f')]
```
***b becomes the first and second member of the bigram***

#### problem: easier to find word that starts with a and ends with c

**Padding used**
```
['<s>', 'a', 'b', 'c', '</s>']
```

