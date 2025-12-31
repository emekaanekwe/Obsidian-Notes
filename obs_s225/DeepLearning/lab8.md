

---

## 1. What is Knowledge Distillation (KD)

At a high level:

- You have a **teacher model**: large, well-trained, high capacity, maybe slow, costly to run.
    
- You want a **student model**: smaller, faster (inference-wise), maybe fewer parameters, less compute.
    

Goal: the student learns _not just from the “hard labels”_ (the ground truth), but also from the “soft outputs” (the probabilities / logits) of the teacher. Those soft outputs carry extra information: how confident the teacher is in non-true classes, similarities among classes, etc. This extra info helps the student generalize better.

Key concepts:

- **Soft targets / soft labels**: teacher’s output probabilities (or logits scaled by some temperature) across all classes, not just the single correct class.
    
- **Temperature (T)**: a scalar > 1 that “softens” the softmax output (makes the probability distribution over classes more spread out). Using higher temperature reveals more about the teacher’s beliefs, not just the top class.
    
- **Loss function combining two things**:
    
    1. Loss between student output and the teacher output distribution (often KL divergence)
        
    2. Loss between student output and the true labels (often cross-entropy)
        

So overall loss might look something like:

L=α⋅CE(student_pred,true_labels)+(1−α)⋅T2⋅KL(soft_teacher,soft_student)L = \alpha \cdot \text{CE}(\text{student\_pred}, \text{true\_labels}) + (1 - \alpha) \cdot T^2 \cdot \text{KL}(\text{soft\_teacher}, \text{soft\_student})

- The factor T2T^2 comes from how the gradients scale when using temperature.
    
- α\alpha is how much weight to put on “hard label loss” vs. “distillation loss”.
    

Why this helps:

- Student gets extra signal from the teacher’s “beliefs” about how non-true classes relate.
    
- Helps regularize student, often yields better generalization than just training from hard labels.
    
- Especially useful when student cannot match teacher capacity, but still wants to do “as well as possible.”
    

Also, there are variants:

- Feature-based distillation: matching intermediate layer activations
    
- Relation-based distillation: matching relationships among examples
    
- Self-distillation, online distillation, etc.
    

---

## 2. What the Kaggle notebook you linked has to do with it

I took a look: the notebook is a working implementation of knowledge distillation on some classification dataset. What it shows / demonstrates:

- They first train a teacher model (often a large network like ResNet or something).
    
- Then they define a smaller student model (fewer layers or smaller width).
    
- They train the student using both the teacher’s outputs (soft labels) and the true labels, using the kind of loss I described above.
    
- They probably experiment with different temperatures, weights α\alpha, maybe compare accuracy of student vs. teacher, maybe see how much smaller the student can be before its performance degrades too much.

---
















# (wk8)
## Word Embedding
❑ Word2Vec Characteristics and Word2Vec for Feature Extraction (Tute 8a) (*****)
❑ Word2Vec for Initializing Embedding Matrix (Tute 8b) (*****)

### Example case

individual words do not have meaning alone. like NN, we are training embedding so that the word can be learned.

example code

```python

word_vectors.get_vectors('king') # embedding is a vector of a certain word

model = KeyedVectors.load("/path/to/bin") 

## Older method (Ngative Sampling)
print_most_similar(word_vectos.most_similar('france, 'berlin'), negative('paris'), 1) # calculate numerical distance between words

```



### RNN revisited


![[Pasted image 20250917142035.png]]
for **sentences**,  has own meaning . carry info from preprocess to postprocess (begin to end)

### Word2Vec
	an already trained word mapping algorithm that is pre-trained (since before RNN)
**note:** RNN are types of transformers

#### Takeaway: Word Embedding can be used in RNNs


### Example Case

1. have a large group of words
2. flatten to 2D using PCA

---

## RNNs Cont.

![[Pasted image 20250917142035.png]]

consider the word "this movie is fun"
	we would have an embedding matrix of row nxm where n contains each word

in the hidden size 1 => thensor is of batch size and sequence length 

| batch size|
| sequence length| 
	| goes to
	embedding layer
		| contains
		batch sizex embed size

resulting in a 3D tensor
[batch size, seq length, embed size]
	| U
	this goes through H_n
resulting in tensor of
[batch size, seq length, hidden layer]

for hidden layers, need to specify the hidden sign

at last step, we can build up connected layers. 

**How do we initialize the embedding matrix?**
	Word2Vec

## !This will be relevant to Q2 of A2

