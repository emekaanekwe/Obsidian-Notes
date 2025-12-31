the main diﬀerence between classiﬁcation
and regression tasks is the type of output variable, which is discrete for classiﬁcation but continues for
regression.

**input space**: all inputs represented in vector space
**decision boundary**: the line at which the weight is orthogonal to every vector lying within the surface. divides the class labels

**linear separability**: datasets whose classes can be separated by linear decision surface

#### Types of classification models

##### Discriminative

**Perceptron Algorithm**


##### ProbabilisticDiscriminative

**Logistic Regression**
##### Probabilistic Generative

**Naive Bayes**

- **Model the Joint Distribution**: Instead of just a boundary, they learn the full picture of how data looks within each class.
- **Class-Conditional Density**: Estimate
    p(x|y)p open paren x vertical line y close paren
    
    𝑝(𝑥|𝑦)
    
    for each class (e.g., what features look like for "cat" vs. "dog").
- **Class Priors**: Estimate
    p(y)p open paren y close paren
    
    𝑝(𝑦)
    
    (how common each class is).
- **Apply Bayes' Theorem**: Use
    p(y|x)=p(x|y)p(y)p(x)p open paren y vertical line x close paren equals the fraction with numerator p open paren x vertical line y close paren p open paren y close paren and denominator p open paren x close paren end-fraction
    
    𝑝(𝑦|𝑥)=𝑝(𝑥|𝑦)𝑝(𝑦)𝑝(𝑥)
    
    to find the probability of a class given new data.
- **Classification**: Assign the class with the highest posterior probability.
