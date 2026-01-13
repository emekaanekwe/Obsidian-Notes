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

### transcription

Okay. We are going to start talking about classification algorithms and models from this week, and next week, we will continue that. It is the third module of the unit talking about classification topic. We discussed the regression topic in the last 2 weeks. And now we are going to build on top of it. 

So classification problems are so prevalent and important that, as we will introduce them, you'll see a lot of applications already of this problem set up in your everyday life. 

Okay. So we are going to discuss three concepts in these two lectures. By the way, the online people do you hear me? 

Yes. Okay, perfect. So we are going to discuss in this week and the next week about three topics under this module. 

So we are going to learn the collaborative models for classification. I'll decode what we mean by discriminative models. We are going to learn probabilistic generative models and probabilistic discriminative models for classification problem as well. So these are three approaches to this problem. 

But first of all, before talking about those approaches, which are more algorithmic, let me talk about some of the basic concepts so that we build the foundations and then go to the other stuff. So I'm going to compare the regression versus classification. So in regression, we are modeling the relation of the input to a continuous output. So for example, you may want to predict the house prices based on the size and the sub -air or some other features of the house, right? So you're going to predict the continuous value of the house. Or you want to predict the exam result of students, right? The exam results in theory can be any real number in theory between let's say 0 to 100 with all of the fractions and everything. So in that sense, it's a continuous target value. You may want to predict the number of movie tickets that a movie will have based on some features for example, how many times it was too needed in Twitter or X and other other features. So in all of those examples, the target is a continuous value in some range that you want to predict. That is the nature of regression problems. 

However, for classification, we are interested into associating the input to a discrete target, like a target with discrete possible discrete values. What do I mean by that? For example, you may want to classify the customers in your bank into the ones that have good credit or bad credit, like two classes, good credit or bad credit. There is no continuous value. There are these two discrete target values that you want to associate with your input customers. The other one is, you know, classifying incoming emails in your inbox into spam or non -spamp, right? So there are two possible class labels. We technically call them class labels label spam label non -spamp, right? So these are discrete possible values that you are interested into in the target. Or like if I give you an image of some numbers that are written by hand by people, 

you are interested into associating the input, I don't know, 8 by 8 or 256 by 256 pixel image into one of the digits between 0 to 9, right? And then you want to classify it to 0 to 9, right? So for the first digit of the number, for the second digit of the number, so on and so forth. So in all of these examples, the output is a discrete category or label that we are interested in. So what is a classifier? A classifier is an algorithm that implements the classification or mapping of the input data to one of the K classes. In many of the examples that I mentioned, like, you know, spam non -spamp or good credit back credit, there are two possible classes in the output. But in the digit example, there are ten possible classes, like digits between from 0 to 9. So a classifier does that mapping of the input to one of these discrete classes in the output, all right? So this is the schema that you have for your classifier. In the training time, you have access to the training data. You use it to train the and your classifier for each of the instances or examples or data points of your training data produces one of these possible labels or classes in the output, C1, C2, Cn, there are n classes. And then after you train your model, in the training time, in the test time, you use your train model or train classifier to predict the class label of your instances in your test set. Is that clear? So that's our setup. Now the question is, what is the form of the classifier that we want to learn? In other words, what are the parameters of the classifier? How we are going to parameterize a classifier? How we are thinking about it. So that's the question that we are going to answer in the most fair stuff of the classifier. And in the second half of the class, or maybe in the one -third towards the end, we are going to introduce a very simple algorithm to learn the parameters of a classifier. That is the simplest form of a neural network. So in this lecture, we are going to see how you can train a neural network with just one neuron. 

So let me set up the problem. Suppose we have some training data to predict the classification of an input object. In here, our objects are fish. And we want to classify fish into, like, whether there are tuna or bass. These are two different types of fish. So the first question that you need to ask, yourself, when you're dealing with the real problems, is that for the prediction problem that you are interested in, what properties of the input you want to measure based on which then you basically want to build your model? So in this case, what do you think, what properties of the two types of fish that you see here? important in classifying them later, in how we can distinguish based on which properties we can distinguish these two types of fish? 

The shape or the size, right? So for example, and the size can have different interpretations, right? So I can measure the length. Apparently, the top one is longer than the bottom one. And also the width, right? The top one is wider than the bottom one, right? So for this classification problem based on these input data that I have, it seems that these features are going to be important, right? You may use the color as well if you want to, right? Anything that can distinguish these two, right? And you can measure later for your test example. Is that clear? These are what we call features. Why do we call them features? Because these are properties that distinguish the objects in different classes. These are called features. 

So in here, we have some examples that are of these two types of fish. And for each one of them, number one, two, three, four, we have measured two features, okay? Someone has their kind of measurement equipment and has done these measurements, right? So for the first one, like, you know, the length is 100, I don't know, millimeter or centimeter, I don't know, the unit. And then the width is 50. And then the label of that is a tuner. Like that fish was tuner, okay? So for 20 examples, we know what is the true or ground truth label for the objects. So the first one is tuner and you have those two measurements. Those two measurements with the notes, them by X1, X2, right? X1, X2 means the two components of the first object. In other words, X1, X2 is a two dimensional vector. that represents the first object. 

So I used like a very interesting language here. So what I did is that I mentioned each instance, each example can be represented by a vector of its feature measurements. 

That's very interesting, right? In fact, the feature vector of each instance puts it in a D -dimensional vector, a space, D -dimensional space. In here, I have two features. So each instance is represented as a vector or a point in a two dimensional space. This geometric intuition, 

as we will see shortly, is extremely important. and crucial. So I used my language very carefully here when I talked about components, vectors, and two dimensional space. Is that clear? 

Okay, so the first one, the first data point, now you can understand why do we call them data point? Because we have this geometric interpretation. that these are points in some D -dimensional space in here a two dimensional space. Okay, so the first one has those feature vector values, 150, the second one, 60, 20, and the label is best and so on and so forth. Okay, so now that I have represented my problem in a mathematical setting, I represented each data point, each instance as a data point in a two dimensional space. What do I want to do with it? How I can represent my classifier? You can imagine the way that I want to proceed forward is to connect the concept of classifiers to the concept of functions. Functions are, again, some mathematical objects, that we are familiar with, right? I'm going to connect classifiers to the concept of functions that you know from mathematics. And because we embedded these data points in a two dimensional space, I'm going to geometrically visualize what these classifier functions look like. Sorry, there is a loud speaking from there. Can I please ask to lower it down or just be silent? Thank you. 

I'm going to mention a space that these instances are embedded. Our classifier, which is a function that we represent by f of x, can be three times the value of the feature one plus two times the value of the feature two minus the number 250. That is the function, the classifier function that I'm interested in. as an example. Now, if you apply, by the way, this x that you see in f of x is a vector x. And that vector x has two component x1 and x2. And then you see, like, you know, the component x1 of that x has weight 3. And the component x2 of that vector has weight 2. And there is this vector x2. This is constant, right? So this is 250. And so this is the classifier function that I've considered for now. Now, what do I do with it? How can I, how do I use this function to do the classification? It's extremely simple. So imagine the first data point, right? 150. If I put 150 that vector into that function, what I get is three times the value of x1. 100 plus two times 50 minus 250, right? So 300 plus 100 minus 250 is 150, right? So that's the value that I get. So each data point, if you put it there in f of x, you get a value back. Okay? Now, this is by classification rule. If that value is non negative, meaning that it's above zero. zero or zero, I say it's class one or class two now. If the value is negative, I say it's the other class, class bus. So that's the classification rule that I have. Okay? Just to repeat again, right? I have that classification, that classification classifier function. What I do is that each input data point, when I put, when I compute the value of that input data point based on the features, the feature vector that I get, I get, I basically get a value based on f of x. If that value is the negative, I call it class two now. If it is negative, I call it class bus. That's my classifier. Sounds good. So that is how we connected classifier. Here's two functions. 

And here is a very interesting fact. So that f of x is the linear on long linear function. 

It's a linear function with respect to x1 and x2, the feature vectors. So in the space, in the feature space, in the two dimensional feature space, that is gonna 

be a line, right? If it was a one dimensional space. But now it's a two dimensional space, it's gonna be a plane, right? In other words, each feature function in that form is a linear feature function. And it's representation in the geometrical space, like a D dimensional space, if I have D features, is a D minus one dimensional hyper plane. 

Makes sense? 

That's really what these functions are. And that's why 

the title of the lecture was linear classification models, right? Because all of the classifiers that we learn today and next week, they're gonna be linear functions at the end of the day. They're gonna be a hyper plane in some way. feature space. So that's the commonality of all of those classifiers. All good? Everything is clear? Perfect. I try to go slowly in the beginning to build the foundation. We can, we will speed up a little bit later. 

So the concept makes make a lot of sense to you, right? If you didn't have that, it would have been difficult. So input space is what I just described, right? Each data point is represented by a vector in that input space where the components of the vector are correspond to different features that you have measured, size, I mean, length, width, et cetera. Decision boundary, 

is this guy that I just represent here. So this decision boundary is the line or hyper plane that I talked about before, right? So this is the boundary of decision between the two classes. That's why we call it a decision boundary. And it's a line here because it corresponds to some, you know, f, f, f, f, f, f. linear f of x that, you know, we discussed before. Decision regions. So all of the region, all of the data points x in this side of the line 

are going to be, are going to have the property that f of x is positive for them. Okay? So they're going to be labeled as tuna. So this is the region that is labeled by tuna. And on the other hand, this is the region, the other side of the line. This is the region that all of the data points there have f of x negative. So they're going to be labeled by bus. So this is another region. Okay? And so, so good. So we call them decision regions. 

that green part, all of the points have f of x positive in the yellow part, all of the points negative. 

Where are the points for which f of x is zero? 

Where are those lines? Where are those points, sorry? For which f of x is zero? 

The line itself, right? Remember f of x is zero was the defined any equation of the line, right? So for all of those x's, f of x is zero, for all of the x's on the line. 

Now, decision boundary for linear models are linear functions or hyperplanes, as we see here, which is the focus of the module today and lecture today and next week. And there is a very interesting concept. Look at the last bullet point, linear separability. 

is a characteristic of a problem or a data set, not a model. What do I mean by that? Forget about that line that is our classifier for now. Let me just take that out of the picture. So this is a two dimensional space. 

And these data points, like the ones that are labeled positive or tuna and the ones that are labeled negative or bus. 

Correspond to examples that I have in the training set, right? All of those positives and all of those negatives, right?

you can perfectly separate these two categories of data points, types of data points. positive and negatives by a line, right? With zero error, right? If I draw any line between these two, I can perfectly with zero error separate them. In other words, this is a linearly separable problem, but not all problems are linearly separable. There are some problems for which, like, you know, for example, 

in the moon shape, right? There are some problems that the two classes of data points, you cannot separate them by a line or a linear function. Those are non -linearly separable problems. All clear? So for linear separable problems, they can be perfectly separated by a linear function. 

and those are your functions how we find them, how we discover them, is the topic of today's lecture and next week's lecture. The other thing is that the classifier, right? 

That separates the two classes perfectly well or fine for linear, like, 

linearly separable problems, right? So, the classifier is not a linear 

classifier, right? Because any line that separates these two has a zero error and is good for us. We are interested to find one of them. Is that clear? So, for linear separable problems, usually, there are infinite number of good classifiers, right? But of course, the same thing is a linear classifier, right? So, of course, good classifiers with respect to the training set, right? It results with the training set in the sense that, you know, they have training, they have zero error on the training set. But of course, not all of them have good generalization on seeing examples, right? Then the question is that from all of these, you know, zero error, you know, lines on the training set, which one or ones are going to be good as our classifier to bet on them as our classifier to get the minimum, like, you know, error on seeing examples. This is a different question that I don't answer it in these two weeks. 

Is that clear? Okay, perfect. 

So, the class, you know, green or class red, or you can say that I abstain in making any prediction. Make sense? Let me put the prediction below. Sorry. So, the segment in the prediction will be now. In this class, in this course, the unit, we actually, for the cases that there are zero, we take them the green one. Well, generally, you have other options as well, if you want. That means the function should say greater than, should be greater than or equal to zero, not greater than. You know, anything above, above zero or zero is green and anything negative is red. Yeah, that's right. Okay. Yeah. Okay. 

So, the decision boundary, as I said, is, you know, those data points x where, like, you know, for which f parameterized by w on those x's is zero. And because our f or function in this, in this unit is a linear function, then we are going to have something like this. So, this is the kind of, function f's that we are dealing with, like, you know, w zero, right, which we call the bias or the threshold plus w one times the first component of the feature vector, w two times the second component of the feature vector, all the way to w d times the d component of the feature vector equals zero, right. So, this is the general form of the linear classifier functions that we consider in this week and the next week. And it assumes that, you know, we have a D dimensional representation of the data points. Yeah. Each data point leaves in a D dimensional space. Yeah. 

So, the W's, now, very interesting. The W's, W zero, one, two, all the way to w d. These are, the parameters of your classifier and the whole purpose of learning algorithms that we see today on next week is to learn these W's. 

Make sense? So, when I say learning the parameters of the model, classifier model, I mean these W's. Okay. And let me repeat again. W zero, we treated, like, you know, especially. And we call it, the bias term or the threshold term. W one, two, W d, we call it, you know, we denote it by this bold W in here, the weight vector. So, we have a weight vector and a bias term that we want to learn. 

Okay. 

So, I think, yeah, this is what I just mentioned before that, like, you know, if you're giving me an X, I compute like f of x, right? The value of the function f for the input x. And then if that value is positive or zero, it's class label plus one. If the value is negative, it's class label minus one. Is that clear? So, the, f of x, 

if you basically look at the slides, in the compact form, I can write it as W zero plus the inner product or the dot product of the weight vector and the feature vector x. Okay. So, this is, like, in a, this is written in a compact form, right? So, if you think about it's exactly the same thing that you see here. Okay. The other comment here is that we use the sign function here. The sign function operates on its input argument and if it, it's input argument is negative, it produces minus one. And if it is zero or positive, it produces positive one. So, it's the sign function, right? It basically gives you the sign of the value. and then with the note, the result of the sign function applied on f of x by y of x. y is your class label. 

This is a bit of notation to remember. 

Okay. So, sign function applied on this green part gives you the same thing. So, this is the same thing. This is the same thing. on this green part gives you plus one applied on this yellow part gives you minus one applied on the line itself gives, gives you plus one as well for the question that was asked before. 

Now, very interesting. So, suppose I have this, this, this problem, right? And I have this line. as my classifier, for example. This line is corresponds to three times feature length that you see here plus two times feature width that you see here minus 250 equals zero. This is the line corresponding to this algebraic form that you see here. Okay. Now, the interesting thing is that if I, for example, change the weight of the length from the length feature from three to two, what happens? Right? So, what happens is that this line changes, right? And now, the algebraic form becomes two times length plus two times width minus 250 equals zero. And then the visualization would be something like this. Now, there is equal weight for the two parts for the two features. Right? So, that's interesting. So, by changing the double use, you, I mean, double you want or double you change the slope of the line. 

Now, how about changing the bias there or the threshold? What happens? If you change the bias there, the slope doesn't change, but the line shifts, you know, with respect to the origin. So, what it means is that. So, for example, if I change the threshold, from minus 250 to minus 150, 

this is going to be my updated algebraic form. And this dash line is going to be shifted to here because you change the bias there. So, now you understand why we distinguish the bias there from the other weight, parts of the weight vector, right? So, the other parts of the weight vector, if you change them, the slope changes. for the bias there, 

the shifting happens without changing the slope. Clear? So, this is just to visualize and mention why we have separated them. Now, there is a very interesting concept here. 

So, if you think about the function W0 plus the inner product, product of the weight vector W and X, that function, that function, we call it, Voy of X in the regression lecture, right? It was our regression function, right? 

And what we did in the previous slide was to apply an F function, which is the sign function on W0 plus W, you transpose X, the inner product of weight vector and X. We applied a sign function F on our regression function to get the Voy of X as a classifier. So, there is this really interesting connection between regression and classifiers, right? Basically, it seems that, you know, you can turn any regression function to a classifier, right? So, we are applying the sign function on the result of the regression function. This is just a connection, right? Now, 

we call such classifiers generalized linear models. You generalize them from being a purely regression function to a classifier through this what we call the link function, F is the link function, okay? In a statistics. 

So, there are different types of link functions or activation function is another name for them. 

If we use a sign function or a step function, there are the same here. or we use them interchangeably here that you saw before. We get the classifier that I just mentioned, right? So, if F is a step function or a sign function that you apply on, you know, your regression function, you get the classifier as I discussed in the previous slide. And this is just a heads up. This is, you know, a very, very old classification classifier model in machine learning in, I think in 50s or so by Rosenblatt. This is called perceptron. 

So, perceptron is nothing than a linear function 

passing through a step function to give you the class tables. That's called the perceptron. How you learn the parameter, of the perceptron function or classifier for a given data set will cover at the end of the lecture. All good? So, this is perceptron. 

And notice that perceptron gets the input x and at the end gives you whether it's class plus one or class minus one. Gives you one of these two classes. However, in many real -world examples, we cannot be 100 % sure about the class level of an input data point. So, for example, if I give you the images of digits, right? Sometimes I don't know, maybe some of these handwritten digits are close to each other. So, you cannot tell whether it's exactly this digit or the other digit, right? So, it would be good to give the model this opportunity that it can actually give and provide an output with some uncertainty saying that maybe 20 % this class, 80 % the other class, right? So, it would be good to have such a model. So, this can be done by another type of link function that we call the logistic function. So, the logistic function, what it does if you look at the bottom plot is it squashes, like the real axis into numbers between zero and one. And you can treat those numbers between zero and one as probabilities. So, what happens is that with that logistic function applied on the linear model that you have, you can get the probability of class plus one and a class minus one. So, you can get the probability of class plus one and a class minus one. and say, like, you know, with this probability is class plus one, with one minus that probability is class is the other class. So, it allows you to express uncertainty over the classes. And this is something that we call that we cover in the next lecture. And the classifier that is built based on this activation function, which is called the logistic classifier. 

So, the sign function or a step function is an example of a discrimative model that assigns the input to one of the output classes. There is no probability, there is no uncertainty, right? So, it's like that. And the logistic class function or classifier, is an example of a probabilistic discriminative class model that gives you the probability of each class in the output. So, this is the difference between discriminative models and probabilities, probabilistic discriminative models. I want to make another distinction between discriminative models and generative models. If you remember, like in the beginning of the lecture, I mentioned like three types of approaches to build classifiers, right? One of them was generative models. And it should remind you of generative AI, like you know, LLMs as well. So, in generative models, 

we actually build a model for both the input x generating the input x and its label. Okay? And then, like you know, after we build that model, for generating the input x as well as its label, then we basically use the base rule to produce the probability of each class for that input. 

This is what generative models do. But in discriminative models, the assumption is that input x is given and don't need to be modeled by the model. Your model doesn't have a mechanism to generate x. It only generates the label for the given x. That's the discriminative probabilistic discriminative models. But for generative models, your model can actually generate synthetic data of input and the label. Okay? So, 

that is enough for now. We will get to that in the next lecture. 

Okay. So, we have to go into discriminative models. 

Okay? So, this is our running example. I think we don't need to cover much. We have talked about it a lot. So, we mentioned that, you know, we have this feature vector x. f of x is built by, in this example, three times the feature length plus two times the feature width minus 250. right? equal zero. So, this is, and then this is the decision boundary. And then these are the decision regions. 

And the class label algebraically can be found, you know, by looking at the sign of the value. Okay. So, these are some, you know, geometry relationships that might be interesting for some people. So, the W, that you see here, W transpose times x, like the inner product of the W and x plus W zero equals zero, right? You might be wondering why, you know, what quantities in this, in this figure, my represent is these components. 

first of all, the line, the decision boundary is which is this red line is orthogonal to this W that you see here. So, in fact, this green direction is the direction of W, which is orthogonal to this red thing. Okay. So, if you didn't give me this red thing and you just gave me this thing, the first thing that I would do is to look at the vector W. and then W, and then I would consider the, you know, my red line to be a line orthogonal to that. Okay. That's the first thing. But then the question is that how much you shifted from the origin. That is determined by this W zero, right? So, basically, you look at minus W zero divided by the length of the W on this direction, and then you draw the line. And that is guaranteed to be that the line for that equation that you see up there. Makes sense? You can look at this, you know, at home, and it is a very interesting and good figure, because it gives you intuition about where these numbers appear. Okay. But the most important thing is that this decision boundary is orthogonal to this W vector. that you see here on the green. 

And also this, the amount of shift is determined by this minus W zero divided by norm W. All good? 

Okay. Now we go to your question. Now we want to generalize from two class classification problems to K class classification problems. Right? So, we mentioned that, you know, for the digit recognition task, the input is a let's say 8 by 8 or 256 by 256 image pixels of the image. And then the output should be one of the digits between zero to nine, right, 10 possible classes. How can we, so far, I talked about two class classification problems. How can we go to 10 class classification problems, or K class classification problems? is that clear what we want to do? Yeah? Okay. So here, I'm going to mention two ideas. And then I'm going to mention why those ideas are problematic. And then we're going to see the third idea of which works. Okay? So the first idea is to translate this multi class classification problem into binary class classification. But binary classification problem as follows. 

So one way that I can do it, 

I can actually build a classifier for each class versus the rest of all other classes. 

Right? So I can build a classifier for the class bus. by treating class bus to be class plus one. And the data points of all the other classes to be class minus one. Now I have a two class, a data set with two classes. And I learn a classifier for that. 

Okay? This is that one. You can do the same thing for tuna. You can build a classifier where tuna is class plus one. And all other classes are class minus one. Right? And then you build that classifier. You can do the same thing for salmon as well. Right? 

But there are, if you do that, there are regions that belong that may be vague. That may belong. to more than one classes. And it's, there's a confusion about those data points. 

So that's the problem with it. So here is another idea. 

I have three classes here, right? I pick any pair of classes. Okay? I build a classifier for that. Like for example, I build a classifier. for a classifier for best versus tuna and a best versus salmon, for example, and salmon versus tuna. Right? So all possible pairs of classes I pick and build a classifier for that pair. Now if you give me a data point. I apply all of those classifiers. And each of those classifier classifiers tell me like a class predicted for that data point. 

to look at the class that has the majority vote. And announce it as my prediction. Clear. 

So we pick each pair. If I have K classes, like 10 classes in the digit example, digit recognition example that I mentioned, there are 10 digits, right? 0, 1, 2, all the way to 9. Right? Each pair of classes, how many pairs of the are there? 10 times 10 minus 1 divided by 2. 45 pairs. So you need to build 45 classifiers. Now if you give me an image. In the test time. I apply all of those 45 classifiers. And each of those 45 classifiers tell me will tell me. you know, this is my prediction about one of the classes. Right? Then I select the class which has the highest vote from these 45 classifiers. 

It's like the US election. Right? You pick your electorals. And then they vote for the winning class. Makes sense? 

because again, there is a weakness. So in this example. Here are the three. I mean, three class problems are a special case because the number of classifiers is going to be three as well. Right? So these are the three classifiers that you build. Now if I have an X here by this box black box. Right? The more than one. 

class label that you can assign to that thing. Right? 

And for this region, actually there is a confusion. 

More than one class have the same vote. Like two lines tell that this belongs to something. So it's okay. So in other words, like, you know, the decision regions. Produce. by this method doesn't cover the full space. And it is not good. Okay? We need to do another. Another thing. So here is what we do. So what we do is extremely extremely simple. 

So. Like before. Like the first method. Like before we are going to learn a classifier. for. We're going to learn a function. Not the classifier. A function f. A linear function. F for each of the classes. So for green class, I have I learned a linear function f one. For blue class, I learned a linear function f two that you see by lying blue. And for the red class, I learned a linear function f three. represented by the red line. Now here is my decision rule. Right? As to what is the predicted class for an input? 

If you give me an X. I apply f one. Which is the green one. I apply f two, which is the blue one. And I apply f three, which is the red one on X. And I get three values. 

I then announce the winning label to be the label for which this value is the highest. 

So if I do that, then what would be the decision boundary? That decision boundary. If I do that would be something. very interesting. So first of all, these are the lines that we learned for these three classes. Like I would call them scoring functions. Right? Because each of them say that this is my score for. You know, being for this data point to be from my class. So these are scoring functions. Right? And the class for which the score is the highest is winning and produces the output label. Now. 

Here is the interesting part. I wish I had the annotation, but annotation here is turned off. So I cannot use this thing. This guy here. But we try to do it. So if you look at this area. That you see here. For this area. I mean, this area. Let me in general. You see my mouse. Right? One of this this area. The green function. The green scoring function. The screen on this side of the green line. It's gonna. You know, the scoring for green is gonna be high. Right? 

Have over. If you consider the red one. The score of red is gonna be high on this side as well. Meaning that in this area. like a mixing of the scoring between. The red and green. Right? So we need to see which one is bigger. Okay? Anyway. So let me put it this way. So each data point on the surface that you see here. Is gonna be classified according to its distance to the line. Right? The line which is closest to it. 

I have the highest score and wins the class. The classification rule. Right? So for example, if you consider this data point here. It's closest to the green function. Compared to the blue and red. So the scoring. Function. The. The scoring. Of. For this data point. The scoring of the green function is gonna be the highest. So this guy is gonna be predicted as green. for this one. This one is closer to the red red line. So it is gonna be predicted as red eventually because the scoring of red is gonna be higher than blue and green. Right? So it seems that each data point. Is gonna be. Labeled according to the distance of the nearest line to it. That's the decision rule. That this classifier. Will. Boys down to. is that clear? Now we want to see what are the decision regions. Right? So all of the guys data points that are. 

If you consider this line. Did this figure on the left? What we need to do. Is to kind of. 

Consider this line that passes through the cross. You know, this this this this this joint of these two lines. And has. That goes. With equal angle between the two. Right? So if I consider this line. Right? This separates all of the points for which this green. All of all of the points that are closest to this. This separates all of the points that are closest to the green and red function. Right? So all of the points on this line that you see here. Are gonna be equidistance all of this guys. Equidistance. To the red and green. All of the points that you see here are gonna be equidistance to blue and red. And similarly in here. Right? Many that all of the points here. Will have the green. Function value to be. The highest are closest to the green function. and the points here are gonna be closest to the blue. And all of the points here are gonna be closest to the red. In other words, if you look at the decision boundary, it's something that you see here. 

Right? This is visually, you know, what the decision boundaries are. And this is very nice because it has no come like, you know, ambiguous region. And it covers the full space. 

you get to some linear, like, you know, partitioning of this space like this. Based on algebraic manipulation as well. If you look at algebra, you get to the same intuition. In an intuitive reason that we had here as well. So long story short, if, you know, we go according to the third approach, right? We get, we learn three separate linear scoring functions and our decision rule. To be. 

producing the label for which the scoring function is the highest, you get to something, you get to something very nice. In terms of your decision regions. Is that clear? 

An intuitively it makes sense as well. Right? Every data point is gonna be classified according to the scoring line that it is closest to. 

All right. So far so good. now let me fulfill my promise and talk about the perceptron algorithm. So far, let me recap. So far what we discussed is actually what are the linearly separable problems, how we can build what are the classifiers and how we can build classifiers by applying generalized linear functions or activation functions on top of regression functions. Right? We talked about those. and I mentioned about the notion of binary classification problems and. A k classification problems where k is bigger than two. And how can we build k classification k. Very classification models based on two way classification models. One of the other important things that we discussed was. this geometric viewpoint to classifiers and classification models, which was extremely important. Right? So we embedded. We considered actually the problem as a geometric problem in some space by embedding the data points in that space. And then we represented our linear classifiers as hyperplanes. And then the rest of this story. Right? So this is what we have seen so far. But I haven't told you so far. how we can learn the parameters of of a classifier. Now this is our first. First method or approach. Perceptron algorithm. So perceptron is. 

You can consider it a neural network with just one neuron. 

And. It's a discriminative classifier. It's not the probabilistic discriminative. It's a discriminative classifier. 

Let's see that. So perceptron as I said before. Is. Giving you. Y of X or the label in the output. Which is the result of applying an activation function f. On. That. Linear function that you see here. W transpose times X the weight vector. The inner product of the weight vector and the feature functions. The feature vector. The. the feature vector of the input. Plus this by a star. So this is. What perceptron is. Whereby in perceptron like this F. Is gonna is actually this sign function or step function. Okay. Now the parameters are this W 0 which is the biased term or the threshold term. Plus this W. Vector. That you see here that those are the parameters that we want to learn. 

Let me give you a pictorial representation of perceptron, which mimics neural network. 

You can consider the feature vector as these circles here, x1, x2, x2, all the way to xn. These are the n -dimensional feature vector that you have for the input x. And then we put a feature component 1 there as well. This is 1. And then the bias there corresponds to this feature 1, 

value 1. And then w1, w2 to wn. These are the weights in the weight vector, I mean gonna be correspond to x1, x2 to xn. So what happens is that our classifier builds a value in this orange box, where that value is the sum of the weighted combination of the input 

with the weights on them. So for example, if you multiply 1 by w0, you get w0 because 1 times everything is anything is that thing. Then x1 times w1 plus x2 times w2 to xn times wn. So you get the weighted combination of the input with the weights that connect them to the, to this red to this orange box. So that is what we compute first. And then on top of it, we apply a step function. 

And this thing is actually a perceptron. This is called a neuron. A neuron takes the weighted combination of the input signal with the weights connecting that input to the neuron. And then we get the input then in linear way, the orange. And then applies a nonlinear function on top of it, which is the yellow thing here as the step function. This is the perceptron neuron. And then it produces the output class at the end. 

So this red thing is 1 neuron. 

Now our goal is to learn the parameters of this model. What are the parameters? The parameters are these weights that you see here. w0, w1, w2, all the way to wn. So these are the things that you want to learn. Based on what? Based on the examples in the training set. Okay? 

Remember that we defined the error function and in the regression lecture. 

And then tried to learn the parameters of our regression model by minimizing some error function. This is what you want to do here. I'll go a step by step. 

Okay. So the problem set up. We have everything, right? The question is what is the error function to minimize? To learn the parameters of the model. Okay? 

if I have a perceptron like and suppose I have set the parameters. 

The way that I want, right? Maybe randomly or whatever, right? If I apply my perceptron to predict the label of the examples in the training set. And there is, let's say no error, right? There is no error for the examples in the training set. Then that perceptron is going to be a good model, right? Because it's error on the training set is zero. 

However, if it has some error on the training set, meaning that it misclassifies some of the examples in the training set. I need to think whether I can change the parameters. Such that those misclassified examples become classified correctly later, right? So in other words, what if I use as my error function the number of misclassified parameters, right? examples in the training set and try to minimize it. Make sense, right? 

Any questions? I'm going to be very slow because in here it's very important. 

What you want to do is to learn double use such that we minimize the number of misclassified examples in the training set. set. So I hear some noise from there.

Compare to the linear function which is zero. Very obvious. How does that? Okay, I see. I see. So the question is that the question is that if I have a perceptron, right? And like if somebody gives me an X, right? I can apply the weighted combination of the feature vectors and the parameters. and then apply the sign function and predict the label where the error come from. So the error come from the fact that for the training set we have the ground truth label. I know that whether this fish is bass or whether this fish is actually. What was the other one? Salmon or tuna, okay, tuna, right? So for the training set, I know the correct label. Now if my prediction doesn't match the ground truth. is error. 

Meaning that my model is not still a good model and I need to do more training. 

For the training set we know the ground truth label, right? We know the T. And if you know I apply my classifier at the moment that I've learned so far. I'm going to turn the example on the feature representation of the training example. I get the prediction, right? If that prediction doesn't match the ground truth there is error. If it does make sense? Yes. The fact makes sense. But I just, I don't know how and I only need to match the matically whole. 

in classify that although you're being a function. I mean, I mean, multiple features, 

but you still are starting a value to it. I mean, it's very straightforward. Because they're all smaller. I know I have a class. Like I understand like the way that you're saying. So when to want to know. There's error. Yes. I don't know on the knees. I'm not saying. I show you an example later. I think that's a good question. I show you an example later. Yeah. Yeah. That's a very good one. We get to that. Yeah. So. Okay. So is there any other questions? Everyone is with me because we are going to mention something extremely important. I want to make sure that everyone. 

Is good. Yeah. Okay. So we want to minimize the number of misclassified examples. But how we can do that. Okay. So here is what we do. 

So here is what I just said. And I think part of the question, right was this. So what is a misclassification condition? Right. So what is the condition if that condition. 

is a misclassification happening. Okay. This is really neat and elegant. 

So the misclassification condition is when the Royal X predict the label predicted by the model is not equal to T of n, which is the ground truth label for that X in the training set. Okay. So that's the question. So what is the condition, right? The way of X predicted by the model is not equal to the ground truth label T n given in the training set. 

So far so good, right? But how can I mathematically formalize the situations where Y of X is not equal to T of n? How can I form check for that, right? 

One way to do that is I wish I had the right. So one way to do that is to simply compare Y of X to T n, right? That's simply what we can do, right? So I have X. I have my perceptron model, right? I give X to my perceptron model. Which computes the weighted combination of X and the weight vector apply the sign function. I get Y of X and then I compare Y of X to T of X. If they're not equal, there is error. If they're equal, there is no error. That's how I can check. Yeah. So far so good. This is the condition, right? 

I mean, an equivalent way of the same condition that the justice is described, but it is mathematically nicer is the following. 

Something equivalent to what I just said. If I compute the inner product of W and X 

and then multiply it by T n. Now notice that let me mention a few things. Notice that in here I represent the classes by plus, label plus one and label minus one. Okay. So the classes are label plus one and label minus one. And what I do is that let me mention to my perceptron model first and then I mentioned that. So the input is X and the parameter vector is W. right? And then the decision boundary is W times X equals zero. So you may ask where is the bias there in here, right? What we did, as I mentioned before in the figure that has the circles, right? What we did, we have done is that we have added a dummy feature, which is always one to the vector X. And that they'll use the value of the 

negative and minus one is the inner product is negative. Now I want to learn the W vector. Is that clear? That's what we want. That's what we're interested. Now, before I go to the next slide, let's look at when there is error. When there is error, it means that the target label T is not the same as W is not the same as the size of this sign, right? So for example, the prediction is plus one. 

The prediction is plus one, but the ground truth label is minus one. That's the error situation. So if the prediction is plus one, it must have been the case that the inner product is the same. product is non -negative, right? Now, 

if I multiply the ground truth label, which is minus one, to this value, W transpose terms X, I have a negative one as my ground truth label times this value, which is positive, and at the end, I get the negative value. Right? 

I hope that I can have... 

Tn times 

W dot product X is negative, then there is an error. Make sense? 

Let me call it T. 

Right? So if T is negative one, and W X is positive, it means that the prediction of the model is going to be positive, so there is error, right? 

So if T is... If T times this thing is negative, then it must have been an error, right? Let's look at it more closely. 

There are two cases. T is negative one, W X is plus one, T is plus one, W X is negative. In both cases, if you think about the sign of W X and the T, there is an error. Make sense? 

So instead of applying the sign function on W X and compare that to T, I can just look at the product of T times W X, and if that's the same, product is negative, I automatically immediately say that there is going to be an error. Otherwise, there is no error. Make sense? 

This is really important. 

Okay? So now, 

when there is an error, and I want to change my weight vector, what I do is that I actually try to minimize this error condition that I mentioned in the previous slide. I want to minimize this error condition. So if you think about the summation over all of the data points 

for which this is negative, right? I want to actually... 

specify all of those... Is this a specify all of those examples that are misclassified, right? All of those examples that... I mean, the score that I have for the misclassified examples, right? So for all of the examples N that are misclassified, I try to minimize this summation as my error function. Okay? And we mentioned that... this summation for those cases that have error, this summation is going to be negative. So the negative... We have a minus sign here. So the negative of this negative quantity is going to be positive. So this is a positive error that you want to bring down to zero. So when it gets to zero, we don't have any misclassified examples and everything is good. All good? So the reason that I mentioned this trick is that this trick is important. and we are gonna do similar tricks later in other parts of the unit. So sometimes the odd is nuances in machine learning that you may kind of skip when you read a book and don't pay enough attention to, but these are really important because everything... I mean, not everything, but a lot of things depend on them. Anyway, let me move on. So this is the perceptron learning algorithm. Why did I mention this thing here? This error function. Before going to the perceptron algorithm. This error function that you see here is a very nice error function because you can actually use the gradient descent algorithm to minimize it. Right? You can take the gradient with this way to W, which is just x and times tn, and then move in the... I mean, negative x and tn, and then move in the opposite of the gradient. The basic principle that we saw before. So gradient descent algorithm works for this. That's why we love this error function. But if I had f a step function in Wx and then tn, it could be an error function, but it wouldn't be differentiable. And then I couldn't do gradient descent. 

So that's why we tried to do a little bit of manipulation of the terms to get rid of this f in this error function. We don't like that f function, like the step function, to be in our error function because it is not differentiable. And it prevents us from using gradient descent. 

All right. This was the crucial part. The rest is just mechanics. Okay. This is the perceptron algorithm. So we initialize the weight vector W randomly. 

And then we repeat the following steps until some condition. For each data point x in your training set, we do the following. We compute the result of this function. of the classifier, perceptron classifier. 

If the perceptron classifier, the predicted label from by perceptron for that data point, which is y, the prediction is y, is the same as the ground truth label t, 

the current weight vector for the perceptron works correctly for that data point. So we don't change it. it. We do nothing. Make sense? However, if there is a mistake, we do something. 

If there's a mistake, we do something. What do we do? If there's a mistake, we update the W vector W 

by the following rule. The updated W is the old W. plus some learning rate eta, which is similar to the gradient descent algorithm, 

times t and xn. t and is the ground truth label for the data point x. And xn is its feature vector. 

Okay. So t and is either positive or negative, depending on the ground truth label. So this is the perceptron algorithm. And you may want their where it comes from. I hinted in my previous in the previous slide. Actually, it comes from the gradient descent algorithm. So if you think about this update rule, 

it's applying the gradient descent on that objective function, error function. So if you look at the gradient of this guy with respect to W, for one data point, it goes away and it's xn t nt. Right? So that's the gradient. I mean, minus distinct. And then in the gradient descent algorithm, you have the rule is as follows. The current weight vector minus learning rate times the gradient minus by minus is plus. Right? And then you have plus t. t and xn. 

Okay. Great. So this is based on the principles that we saw before. 

the interesting thing is that as you go through the loops of this algorithm and visit your data points one by one. 

you may change the weight like on an example. And then it may lead to misclassification of a correctly classified example in the past. So that's why you need to have multiple iterations of this algorithm until convergence. 

We will mention some properties of the algorithm later. But let me answer. that question. Right? So your question now. We have really nice 

illustration here. Suppose I have this data point, this data set of four data points, ABCD. Each data point has two feature values x1, x2. And then the classes are minus one and plus one. So you see those examples data point. in the 2D space. Okay. ABCD. Right? A and C are negative and B and D are positive examples. We start by random initialization of the weight vector. 

And suppose this is our classifier that we get. Right? So in this classifier all of the points below the decision boundary as minus one, all of the points above that are plus one. Right? And you can now we want to process from, let's say, the first data point in the data set process A. Right? If you look at the data point A, I mean by picture you see that it's misclassified. Right? But let's look at the algebra. Right? So we compute like, you know, the inner product of W and x, which feature vector x of the point A. 

And we get positive, like positive point 12. So the sign function applied on that score is going to give us label one. Whereas the correct label is minus one. So there is an error happening here. So we need to do something. Yeah? So what do we need to do is to change the weight vector. Right? According to the perceptron algorithm. Suppose eta is one. Like the learning rate is one. right? So what we do is that. 

So the first component of the weight vector W one is going to be the current value plus t times x one. Okay? So minus point three minus point six. Like y minus point six because the ground truth label is minus one. And the feature value is point six is minus minus point six. And you get this value. and then for W two, you do the same thing and you get the updated updated parameter here as well. So after the correction, after the update, not the correction after the update based on the data point A, we are going to have this weight vector like point minus point nine and point one. Okay? And this is the. line corresponding to it. And all of the points on the right hand side are minus one. All of the points on the other side are plus one. Label plus one. Now this is our updated line. And we process the second data point, which is B. From the picture, you see that B is correctly classified. So it's in the area, which is. 

QCova. That means L. of the top quantity. Car cas Porter with X. If you change 45, which is positive, 

or what should it be like? What most Draw Democrats think. 

have before, the next data point is C. C is in here. 

Basically, it should have been a little bit under the right direction. But this is, this C is, we are going to do the algebra, and we'll see that it is misclassified. So here is what we do. So for C, we have this feature vector. In a product with W, it's 0 .03, which is a positive thing. So the predicted level is plus one. But the grant rule's level is minus one. So it's basically on the error. So it triggers the update. And when we do the update based on the perceptron rule, we are going to get a different line or classifier. 

So that line is going to be this one. 

Now we are going to process the data point D. 

And that data point D is correctly classified. So we don't make any updates. And at this point, if you look at like, you know, if you're doing B and D, or they are on the positive side, and points A and C are on the negative sides, right? All of the data points are correctly classified. So the training error is zero. We finish the algorithm, right? But in reality, you may have multiple iterations over the data points of your data set to reach to that zero error. If your problem is linear separable. If your problem is not linear separable, you don't get to the, to the

are very statistical. 

Correct, correct. We go to all of them if all of them are correctly classified. We finish the algorithm. If during the checking, some of them are not classified correctly, we make the updates. And then we do another what we call epoch over the training set. All good? Perfect. We are almost there. 

Important remarks. If the data is linearly separable, perceptron is guaranteed to find a perfect weight vector, a perfect weight vector meaning that the one that has zero -thringer. However, it may need too many iterations over the data to converge to a perfect weight vector. So the time to get there might be very long. Another remark is that perceptron may not converge if the data is not linearly separable. In this case, it may cycle through some weight vectors without stopping infinite loop. So you need to have, in addition to the error, it may be you need to have another condition as well. 

Perceptron is sensitive to initialization. Two runs of the algorithm may converge to two different perfect weights if they visit the training examples in the training set in different orders. 

intuitively makes a lot of sense as well based on the example that we just saw. 

These are important remarks. 

Now, we want to generalize perceptron from two class classification problems to k class classification problems. 

What we are going to do is to maintain a set of k weight vectors where k is the number of classes. For each class, we have a weight vector. We treat that weight vector as a scoring function for that class. As we saw before, if you give me an x, I compute this core of each class label based on the weight vector for that class label. 

Then I pick the class which has a class as a high school. The question is how can we learn the weight vectors for our scoring functions? 

Here is what we do. For each, it is extremely similar to the algorithm that we saw before. We have an outer loop, repeat until some convergence condition is met. In the inner loop, we have a loop over the data points in your training set. For each data point xn, we do the following. We predict the class whose weight vector produces the highest score and call it yn. For the data point xn, using the current classifier parameters, I predict the class label for that data point and call it yn. 

Suppose the ground through this label is tn for that data point. If tn and yn are the same, meaning that the prediction is the correct label, I do nothing. Current classifier is good. But if they are different, if tn is different from yn, which is the prediction, I do the following. I reduce the score of the wrong answer using this. 