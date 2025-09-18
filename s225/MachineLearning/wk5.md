**expecation analysis**: train as many models, 

**variance analysis:**  

**bias**: expectation of model behavior compared to ground truth

**high bias:** specifies but further from ground truth.

### Review from wk4
![[Pasted image 20250827171230.png]]


**Justification to over-fit:** low amounts of data

*Regarding Regression*
**input**: vector of n dimensions
**output:** continuous var

*Classification*
**Note on decision boundaries and hyperplanes**
- DB is the plane in which its dimension is less than the dimensions of the classifying space. 
	- in so far as n != 0
	- the decision boundary is **indirectly** perpendicular to class data. 
*Discriminative models*
specify a function to feed the data, and that funciton is a straight line.

*Generative Model*
explicitly model the underlying distribution of the data

*Perceptron Algorithm*
NOTE: step f = activation f

*maximize likelihood*
get all the points of error and multiply them

*log likelihood*
can be used to minimize the multiplication processes 
- by taking the neg of log likelihood, we are "minimizing the maximum"

*Gradient Descent*

*Learning Rate*

*how to know when GD is done*

*visualizing 2 dimensional std*
can think of it like two contour f's where the mu's are covariant depending on there link within a vector

