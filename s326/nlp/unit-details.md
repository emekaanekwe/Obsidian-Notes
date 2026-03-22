***LLM PROMPT***
_Help me create a study guide the Monash University Masters of AI unit FIT5217: Natural Language Processing class_ --- _Use the .md file I atached. It contains summaries of all 12 weeks, and suggested reading material._---  _Create a chart, table, or another appropriate visual to summarize how the most important concepts/rules/definitions in the NLP class relate or differ from one another._ --- _Ignore the "# Assessments" section_ 
# Summary
## FIT5217 2026
    
    Natural language processing (NLP) stands as a cornerstone in the information age, made even more riveting with the rise of Generative AI and the introduction of models like LLM. NLP not only supports artificial intelligence in grasping intricate language nuances but also heralds a range of innovative applications. This unit delves into the fundamental principles of NLP, covering essential techniques for analyzing language syntax and meaning. We will also explore the neural network underpinnings of contemporary language models in the context of important real-world problems such as Machine Translation. Furthermore, we'll delve into the theoretical and practical foundations of recent LLMs.
    
    
### Learning outcomes 2026

    |01|organise core problems and applications in NLP;|
    |02|design systems to tackle NLP problems;|
    |03|Evaluation of NLP models from utility & ethics, and safety perspectives.|
    |04|assess various recent approaches to NLP.|

### Unit Period 

March 2nd - June 26th
### Teaching approach
    
    The course material for this unit will be provided as readings (optional), and live online lectures on Zoom during the timetabled lecture period. The recording of the lectures will be made available within 24 hours of the session.  Applied sessions (also known as tutorials) will then give students the opportunity to develop their knowledge by attempting practical tasks (mostly designed as Jupyter Notebooks) individually, while being supported by teaching staff.  These applied sessions will not be recorded and are in-person & on-campus activities (check Allocate+ for your allocation of time and room). Assessments will be through 2 in-semester assignments, and a final assessment (i.e., exam).

# Schedule


Week 1 - Introduction to Natural Language Processing
### **Overview:**

- Unit overview
- History and overview of NLP
- Common applications of NLP

### **Learning Outcomes**

By the end of this week, you'll be able to:

- Getting familiar with the unit outlines, the teaching and assessment formats
- Understanding fundamental building blocks of a (classical) NLP pipeline



Week 2 - Language modelling (n-gram markov models, sparsity and smoothing)

### **Overview:**

- What is a language model?
- Context Length
- The Chain Rule of probability
- n-gram language models
- Data sparsity issues
- Smoothed n-grams
- Evaluating model performance

### **Learning Outcomes**

By the end of this week, you'll be able to:

- Understand how n-gram language models work
- Evaluate language models
- Explain the need for smoothing in the presence of sparsity

Week 3 - Sequence Labelling (Hidden Markov Models, Forward-Backward algorithm)

### **Overview:**

- Word categories
- Part-of-Speech (POS) Tagging
- Other Sequence Labelling Problems
- Hidden Markov Model (HMM)
- Observation Likelihood
- Most Likely State Sequence
- Supervised Learning of HMM
- Evaluation

### **Learning Outcomes**

By the end of this week, you'll be able to:

- Understand the concept of word categories and their roles in NLP.
- Learn about Part-of-Speech (POS) Tagging, and its use in producing the word categories 
- Explore types of sequence labeling problems in NLP
- Understand the application of HMM to sequence tagging
- Do the calculation of dynamic programming algorithm of viterbi

Week 4 - Syntactic Parsing (Context Free Grammars, Syntax Parsing, Dynamic Programming, Alternative Grammars)

### **Overview:**

- Syntax
- Syntactic Parsing
- CKY Parsing
- Limitations of Context Free Grammars
    - Statistical Parsing
    - Probabilistic CKY Parsing
- PCFG Training
- Limitations of PCFGs
- Treebanks
- Evaluating model performance
- Alternatives

### **Learning Outcomes**

By the end of this week, you'll be able to:

- Understand the principles of Syntax and Parsing
- Apply a common parsing algorithm and understand its limitations
- Evaluate and alternative parsing model

Week 5 - Linear Text Classification (Naive Bayes and Logistic Regression)

### **Overview:**

- Text classification
- Classification methods
- Naïve Bayes Model
- Logistic Regression Model
- Evaluation

### **Learning Outcomes**

By the end of this week, you'll be able to:

- Explain the basics of text classification and its applications.
- Compare the Naïve Bayes and Logistic Regression models for text classification.
- Apply and understand evaluation metrics for text classification models.

Week 6 - Neural Language Models (RNN, RNN LM, Teacher Forcing)

### **Overview:**

- Introduction to (Recurrent) Neural Networks
- The challenge of statistical language modelling
- Neural n-gram language models
- Recurrent language models
- A few key papers (not examinable)

### **Learning Outcomes**

By the end of this week, you'll be able to:

- Understand the basics of (Recurrent) Neural Networks
- Recognize the challenges in statistical language modeling and how neural networks address them.
- Understand the concept and application of recurrent neural in language modelling

Week 7 - Neural Machine Translation (Seq2Seq, Attention)

### **Overview:**

- Machine Translation
- Decoding Algorithms
- Sequence-to-Sequence Models
- Attention Mechanism
- Evaluation of MT systems
- Examples of other Encoder-Decoder tasks

### **Learning Outcomes**

By the end of this week, you'll be able to:

- Understand the fundamentals of Machine Translation (MT) and its significance in NLP.
- Learn about decoding algorithms used in MT to generate target language text.
- Learn about Sequence-to-Sequence models and their application in MT for translating between languages.
- Comprehend the Attention Mechanism and its role in improving the accuracy of MT systems.
- Learn about Evaluating MT systems.
- Examine examples of other tasks utilizing Encoder-Decoder architectures beyond MT.

Week 8 - Static and Contextualized Distributional Semantics (Word2Vec, GLoVe, fastText, ELMo, BERT)

### **Overview:**

- Meaning and Lexical Semantics
- Vector Semantics
- Count-based Distributed Representations 
- Sparse vs. Dense Representation 
- Word Embeddings 
- Contextualized Word Embeddings 

### **Learning Outcomes**

By the end of this week, you'll be able to:

- Grasp the concepts of meaning and lexical semantics
- Understand vector semantics and its use in capturing word meanings.
- Learn about count-based distributed representations
- Differentiate between sparse and dense representations of words
- Explore word embeddings, including Word2Vec, GloVe, and fastText
- Delve into contextualized word embeddings such as ELMo and BERT, understanding their importance in capturing word context.

Week 9 - Transformer Language Models, Impacts and Implications (Transformers, Pre-training, Parameter-Efficient Fine-tuning)

### **Overview:**

- Transformers
- Pretrained Large Language Models
    - Encoders
    - Decoders
    - Encoder-Decoder
- Parameter-efficient Finetuning Methods  
    - Adapters
    - Prefix-Tuning
    - LoRA
- Implications of Pre-trained Language models (environment, privacy, safety, copyright)

### **Learning Outcomes**

By the end of this week, you'll be able to:

- Understand the architecture and principles of Transformers, foundational to modern NLP advancements.
- Explore Pretrained Large Language Models and their impact on a variety of NLP tasks.
- Learn the roles and functionalities of Encoders and Decoders within the Transformer framework.
- Investigate parameter-efficient finetuning methods, including Adapters, Prefix-Tuning, and LoRA, for adapting large models to specific tasks efficiently.
- Examine the limitations, ethical considerations, biases, and environmental impacts associated with the deployment of large-scale NLP models

Week 10 - Neural Speech Recognition and Translation (Speech Transformers) and Introduction to LLMs

### **Overview:**

- Speech Processing Tasks
- Automatic Speech Recognition
    - Model Design
    - Evaluation Metrics
- Speech Translation
    - Model Design
    - Evaluation Metrics
- Pre-trained Speech Transformers
    - Wav2vec 2
    - Whisper Speech Encoder
- SUPERB Evaluation Benchmark

### **Learning Outcomes**

By the end of this week, you'll be able to:

- Understand the various tasks involved in speech processing and their significance.
- Learn about Automatic Speech Recognition (ASR), including model design principles and evaluation metrics.
- Explore the concept and methodologies of Speech Translation
- Investigate Pre-trained Speech Transformers, such as Wav2vec 2.0 and Whisper Speech Encoder, and their contributions to improving speech processing tasks.
- Learn about SUPERB Evaluation Benchmark

Week 11 - Advanced Topics in Large Language Models 1 (In-context Learning, Instruction Tuning, Preference Alignment)

### **Overview:**

- Prompting
- In-context zero- or few-shot prompting
    - Emergent in-context abilities
    - Chain-of-thought Prompting
- Instruction-Tuning Large Language Models
- Reinforcement Learning with Human Feedback for LLMs (PPO and DPO)
- System 1 and System 2 Modes of Reasoning
- Inference-Time Scaling
- Deepseek R1 (GRPO, rule-based reward, distillation)
- Open challenge

### **Learning Outcomes**

By the end of this week, you'll be able to:

- Understand the concept of prompting and its significance in eliciting desired responses from Large Language Models (LLMs).
- Explore in-context zero- or few-shot prompting and the emergent abilities that enable LLMs to understand and respond with limited prior examples.
- Learn about Chain-of-Thought prompting and how it aids LLMs in breaking down complex tasks into solvable steps.
- Learn about the process of Instruction-Tuning Large Language Models for better adherence to specific instructions.
- Learn about basics of Reinforcement Learning with Human Feedback (RLHF) for LLMs
- Examine a systematic limitation of LLMs in reasoning

Week 12 - Advanced Topics in Large Language Models 2 (Tool/RAG Augmentation, Self-Correction, Language Agents)

### **Overview:**

- LLM Augmentations
    - Tool Augmentation   
        
    - Retrieval Augmentation (RAG)
    - Self-Refine
- Language Agents
    - ReAct
    - Critic
    - Reflexion
    - Language Agent Tree Search
- Language Agents Fine-tuning
    - FireAct
    - Tora
- Agentic Workflow
- Some Interesting Research Ideas

### **Learning Outcomes**

By the end of this week, you'll be able to:

- Understand the concept of LLM Augmentations and how tools, retrieval, and self-refinement enhance the capabilities of LLMs
- Explore the architecture and functionalities of Language Agents and their potential in automating complex interactions and decision-making processes.  
    
- Analyze novel approaches and frameworks for fine-tuning Language Agents
- Explore the latest research trends in testing and enhancing LLM

# Assessments 

1. Artefact
    Threshold hurdle
    Weighting: 25%
2. Artefact
    Threshold hurdle
    Weighting: 25%
3. Exam
    Threshold hurdle
    Weighting: 50%

# Recommended Resources


- Speech and Language Processing (3rd ed. draft), Dan Jurafsky and James H. Martin, Draft chapters in progress, October 16, 2019. The PDF can be obtained here: [https://web.stanford.edu/~jurafsky/slp3/](https://web.stanford.edu/~jurafsky/slp3/)
- Foundations of Statistical Natural Language Processing, Chris Manning and Hinrich Schütze, MIT Press. Cambridge, MA: May 1999. The book's website: [https://nlp.stanford.edu/fsnlp/](https://nlp.stanford.edu/fsnlp/)
- Introduction to Natural Language Processing, Jacob Eisenstein, MIT Press. Cambridge, 2019.