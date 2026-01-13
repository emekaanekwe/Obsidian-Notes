
$$E_d[(y(x,D)-h(x)^2]=$$
$$E_d[[(y(x,D)-E_D(y(x,D)] + E_D[y(x,D)]-h(x)]^2]$$
	what be entailed below is:
	$$E_d[[y(x,D)-E_dy(x,D)]^2+[E_d[y(x,D)-h(x)]^2-2(y(x,D)-E_D[y(x,D)]) =$$
	Thus:
	$$E_D[[y(x,D)-E_D[y(x,D)]^2]+E_D[[E_D[y(x,D)]-h(x)]^2]-2E_D(y(x,D)-E_D[y(x,D)] scalarproduct\ E_D[E_D[y(x,D)-h(x)=$$
	
it is minus, thus whole will be zero 

Where,
$E_D[(y(x,D)-E_D[y(x,D)]^2]$ is the *Variance* +
$[E_D[y(x,D)]-h(x)]^2$ is the *Bias*^2

TAKE NOTE: where the calculation of the bias-var is  **connected to the calculation above--much like an integral performed over the function and the variance**. This can be repsented via the graph:
![[Pasted image 20250820185254.png]]


Note: When dealng with **multiple** training for a model, that is where bias is considered. This also helps with **over-fitting and under-fitting**
![[Pasted image 20250820172654.png]]


![[Pasted image 20250820172722.png]]

Note: bias won't fix the problem

You want to train multiple models.

# Activity

## irreducibility of error
![[Pasted image 20250820175709.png]]

$E[L] = \int {y(x) − E[t|x]} p(x) dx + \int [E[t|x] − t]^2 \ p(x) \ dx.$


the left of addition is the *function we seek to determine* with the right being *the variance of the distribution, averaged over x*



NOTE: when dealing with over-fitting situations where the high bias, high variance with var higher, then we would want to optimize using variance as preference. 

---
# Lecture Transcription

FIT5201 CL S2 Class -audio
Aug 30, 2025, 7:23 PM

 Today we are gonna touch a little bit of mathematics for bias and variance in a very (13:16) applied way, so that you can use it for diagnosing your models for, you know, overfitting, underfitting, (13:26) and these kind of things that we discussed before.
So this gives you another perspective to diagnose (13:32) your models and fix them if something is wrong. So we are going to talk about generalization error, (13:42) the mathematical form of it, and how generalization error can be decomposed into two terms. One is (13:52) called, or we call it, bias, and the other one we call the variance.
And you see, like, how (14:02) mathematically the generalization error is related to bias and variance. And we use linear regression (14:11) models as our vehicle to introduce this material, a very foundational material. However, these (14:23) approaches, I mean, the approach that we take to decompose the generalization error into bias and (14:30) variance can be applied to other forms of machine learning models as well, like classifiers and (14:38) other forms as well.
So remember that we use linear regression as a vehicle. Although when you apply (14:47) it to other type of models and problems, there is going to be a little bit of differences, but the (14:56) main backbone of the argument is the same. So far so good? So this is the plan for today, right? So (15:02) we are going to make it concrete.
What do we mean by generalization error, and how it is related to (15:08) the notions of bias and variance in the statistical sense of them? We will see shortly. (15:33) Okay, so these are the learning objectives, and throughout the lecture we use this very fun (15:41) example of dart throwing, and you'll see, you know, the connection of that to the bias and variance (15:47) shortly. So what do we mean by bias? Maybe before that, why do we need to understand bias and variance, (15:59) and how to understand the bias and variance in machine learning is what we are going to (16:04) delve into.
So basically bias indicates the, you know, like is an indicator of the accuracy of (16:13) your model. The higher is the bias, the lower is the accuracy. The variance is an indication of (16:22) consistency of your model predictions.
The higher is the variance, the less consistent is your model. (16:32) Of course, we want model that are consistent and are accurate. Why do we care about bias and variance? (16:43) As I said before, because it's an angle to diagnose your model and avoid mistakes of overfitting or (16:53) underfitting, and use, you know, some mechanisms to fix them, for example, regularization.
Let's look (17:01) at the conceptual definition. So the bias component of the error is the difference between the expected (17:13) prediction of a model and the correct value of the true model or the target. Okay, so the bias is the (17:26) difference between the expected prediction and the ground truth value.
And why we talk about the (17:36) expectation or the average of the prediction of a model? Because of the uncertainty notion that we (17:43) saw before. We said that, you know, the model that we build is highly dependent on the data set that we (17:51) obtain, right? So what we did to reduce the dependency of the model to the data set in the last (18:01) lecture, if you remember, was this bootstrap sampling, right? So we bootstrap sampled a bunch of (18:09) data sets. For each data set, we fitted a model.
And then if you give me an x in the test time to make a (18:18) prediction for, what I do is that I make a prediction for that x by each of these models. And then I take (18:25) those predictions, and the average prediction is what I output at the end. So for that test data (18:36) point x, we make n predictions, where n is the number of models that we have fitted on the bootstrap (18:44) sampled training data sets.
All good? So that's why we talk about the notion of average prediction, okay? (18:55) And bias is the average prediction of these different models, right? The average of that minus the, or the (19:07) distance of that average to the correct value. The higher is the distance, the more bias is there, okay? So (19:18) this is the bootstrap sample process that we discussed before, and that's why we have n predictions, because (19:27) we build n models, each of which for corresponding to one of those bootstrap data sets. What does the bias (19:36) measure? How far, in general, these multiple models predictions are from the correct value? This is (19:44) what, you know, the bias measures.
How about variance? Now that we have multiple predictions for that (19:54) particular input, we can talk about how spread out those predictions are, right? You can basically measure (20:05) the variance to represent the, you know, the spread of those predictions. So this is where variance comes (20:14) into play, okay? So variability of the model's predictions for that given data point. What does variance (20:23) measure? Again, imagine you repeat your model learning process many times, right? Each of those times (20:34) correspond to one bootstrap sample.
The variance is how inconsistent the predictions of those models are, right? (20:47) So the more inconsistency in the predictions of those models, the higher is the variance, okay? Let me give you (20:56) the graphical definitions corresponding to what I verbally mentioned. We use this dot example to represent, you (21:05) know, the concepts of variance and bias and variance. So let's say if a model predicts correctly, it is closer to the (21:22) heart of this dot board, okay? So the heart of the dot board shows the accuracy of the prediction, okay? Now, I have n (21:34) models, and the prediction of each of those models is represented by those blue dots that you see there, okay? So again, (21:44) the red center is the correct or the ground truth value, the correct value.
And the blue dots are the values predicted by each of (21:56) those models that I built on top of my bootstrapped training datasets. Now, here is a very interesting thing. So you see four (22:06) configurations of dots that are thrown to the dartboard.
And each of those configurations correspond to a combination of bias and (22:20) variance. So the columns show low variance, high variance, the rows show low bias, high bias. Now, whenever in these boards, the (22:36) predictions or the dart throws are spreaded out, it is high variance or low variance? High variance, okay? So whenever you see the (22:50) spreading of predictions, it's an indication of high variance, okay? So that's why in this column, the column on the right, on your right, (23:06) you see the high variance configurations.
And in the column on the left, you see low variance configurations. Low variance (23:14) configurations are those models that make predictions that are close to each other, okay? So that's low variance. Now, let's look at the (23:24) rows.
When you look at the average prediction, and the average prediction is close to the ground truth, to the red, to the center, it's low (23:38) bias, okay? So the average prediction, the average of the blue dots, look at the average of the blue dots. In the top left, the average of those (23:50) blue dots is very close or, you know, overlapped with the red center, with the ground truth. So the bias is low.
Again, on the top right, if you (24:02) look at the average prediction of those scattered predictions done by those models, the average actually is under red as well. So it's a low bias. In the (24:18) bottom left, the average prediction of those blue dots is far from the correct value, the ground truth value.
So it's a high bias. Similarly, in the (24:31) bottom right, the average prediction of the blue dots is far from the ground truth. So you see these different, four different combinations of bias and (24:40) variance in these examples, okay? So this is a, like, you know, indication of a good model.
The top left is the ideal scenario. The bottom left is a (24:58) model that has low bias, but, sorry, that has high bias, but low variance. So this is not a good configuration, right? Those models are not accurate, (25:18) although they are consistent, they have consistent predictions.
This configuration is also not good, because it has high variance, although the bias is low, (25:31) right? The high variance is not good, because, you know, it shows uncertainty, unreliability for the model. And then this is the worst of the worst, right? The (25:46) bottom right, because it has high bias and high variance. Okay, so far, so good.
Any questions? Okay. This is an excellent model. Like, if you look at, if you have (26:08) access to a range of models, and like, you know, for each model, you represent the predictions of your bootstrap samples based on different versions of the (26:25) model.
So this is an excellent model. This is a model that is not good, in the sense that it has high bias. Probably it's an underfitting scenario, like, you know, your (26:41) model is not expressive enough to capture, you know, accurately the predictions.
And this is probably a complex model, right, which is, in average, is good, but it has high (26:58) variance. And this is an example of a model that has maybe, you know, it doesn't have the right form, and it also has a lot of complexity. So these three conditions, except the top left, the other (27:19) conditions are not good.
Okay, so that was the introduction to these concepts. So far, I haven't told you why, you know, this bias and variance, I mean, how the bias and variance (27:38) related to the error. I just intuitively mentioned what the bias is, what the variance is, and I kind of argued, like, intuitively, that, you know, high bias models are not good, and high variance models are not good, but I didn't provide any mathematical (28:02) reasoning for it.
We will get into the mathematical reasoning shortly. So, in other words, what I want to do is that I want to mathematically define the error, and then I want to do some (28:16) algebraic manipulation of the error, and then come up with two terms, and I'm going to show you those two terms. One of them corresponds to the bias, and the other one corresponds to the (28:28) variance, the way that they are defined in statistics.
So far, so good. So now we are going to see the proof. Here is the setting.
So we assume that we have a training data set, which has n examples. Each (28:50) example, xn, tn, has an input x and output target t. We assume that, you know, we have a distribution p of x over the space of possible inputs. So those x1 to xn are distributed according to p of x. We have a function, a target function, h of x, which for every x gives us the target (29:19) value.
Of course, in the training time, in the machine learning training time, we don't have access to h of x. We are aiming to build h of x, because we don't have it. But for the sake of the definitions of the bias and variance, we assume that it exists, and we denote it as h of x. (29:39) y of x is the prediction of the model that we have built. Okay, so it's our model gives the output y for the input x, and it is built based on the data set D. Now, the generalization error. the error for short is this expectation that you see here. Let's look at this expectation. 
This is an integral if I can write here right this is an integral which basically is. 

the you know if your input the space is discrete you have a summation if it is continuous you have an integral so it's it's for that. So it's an expectation over the distribution of the input of the target value minus 

the predicted value squared. So this is the definition of the error in the regression problem. So remember in the regression problem we said that. 

this is the value predicted by the model and the ground truth squared is the error. So basically you take the expectation of that error over your input space and that's the error. That's the generalization error. Perfect. So this is what we want to measure right this generalization error is what we want to measure and I'm going to show how it is related to the bias invariance. okay if we could compute this we didn't need to do any machine learning because you know it assumes that we have access to the input distribution p of x and we have access to the target function that we want to approximate which is or layer which is h of x. Of course we don't know both h of x and p of x right. So we don't know. 

and but if we knew them then the definition then the definition of the error would have been this right now that we don't know these these quantities right we need to kind of see if we are only given the training data said D how we can approximate this generalization error based on the data set that we are given. Makes sense. So I am not given p of x. I am not given h of x but I am given the data set D right now given the data set D I want to approximate this generalization error right and then show that show how my approximation to the generalization error is related to the bias invariance. Is it good? Yeah. Hi doctor. I have a question about the Y of x being a linear basis function. Yeah. Yes. Yeah. Because I thought in the week three slides it says like it can be Gaussian or other functions like could you explain it. Oh it can be like you know the I mean your basis functions can be Gaussian basis functions right and then you can still have a linear Gaussian linear basis functions linear Gaussian model where your basis functions are Gaussians. Does make sense. Sorry I don't understand. Right right. So if you look at the lecture like the last lecture right we said that a linear model can have any basis function right those basis functions can be non linear transformations of the input that non linearity of the basis function is not is not the reason that we call these models linear models. The reason that we call these models linear models. models is that they take those basis functions and linearly combine them. That's what that's why we call them linear basis functions. Now your basis function can be a non linear transformation of the input for example it can be a Gaussian basis function and we define the Gaussian basis function in the last lecture if that makes sense. Yeah. Understand it. I'll think. Yeah. No worries. There was a question on the back. 

Sorry can you repeat. I can't call. We don't know. P of X. I see. Yeah. That's a good question. So imagine that you are in a hospital and you basically want to build a classifier for predicting brain tumor. Right. So the input is. the MRI images right and then you want to build a classifier. So your X is a MRI image. 

So. 

P of X in that example is the distribution overall you know possible MRI images that we can have for. You know MRI images that can possibly be. existing right but we don't know that P of X but instead what we have is you know if you go to any hospital. The MRI images of some patients right so that MRI images of some patients give you the distribution of. Possible MRI images based on the sample that you have but not all possible samples. Makes sense. 

okay great. So this is the mathematical framework that we work within which we work right now. We are going to see. 

How we can. Approximate this generalization error and. Get rid of the fact that. We don't have H of X and we don't have P of X. Okay. So this is what we want to compute this generalization error. based on a given sample. 

Okay. 

So. 

Now we start from the bootstrap idea. So given the data set D. What we do is that in fact. 

the bootstrap idea is a is a mechanism that we can. 

Handel the absence of the P of X. What we do is that. We you know sample a training data set based on the data set that we are given. And then we fit a model based on that. Bootstrap sample. Right. And then that model has some. 

data set here. Okay. So that's actually. Y of X right. However. This W in. You know Y of X and W. Is dependent on the data set that we have bootstrap. And used to train our model to show that dependency. We. 

have bootstrap sample. 

Now. 

The generalization error for a given data point X just one data point X. Is. 

the model. So let me show it this way. 

So we have built data set one data set to two data set N. True. Bootstrap. 

the build a model Y of. X. 

And D. One. Based on this sample we build another model Y of X and D. Two. And based on this one the build a model Y of X and D. N. So far so good. 

and then we set that. We take the average of these values. 

And. Output that number right. So. The error in this case when we take the average of these values. And output that one. The error is the difference between. 

the error. Yes. So basically the error. Is. Maybe I should do it that way. So the error is. Basically. 

For each output. We look at the difference of that output and H of X. 

Right. So this is. 

the. Let me see if I can. I think I needed. To. Yes. So we look at this minus. H of X. 

This is the error based on the. First model. We look at this minus H of X. This is the error based on the second model. This is. 

the last. You know the last model. And then you know the average out these errors and this is going to be the expectation of the error. Right. So this is actually the. What you see here. That we have. The expectation. Of this error. 

Over the choice of the D like over the distributions that are. Over the distribution over the possible. data. That is coming from the bootstrap sampling. Anyway so this is. Our approximation to the error that I mentioned in the previous. Slide. Okay. 

So in this approximation. We don't need the ground truth function. We just need for each data point. The target value for that data point which we have from the 

the data point. Because you know we have used the bootstrap sampling to. You know approximate. The. You know the space based on this. Now. 

Let's go to the next slide. 

So this is the error. Right. This is. 

the expectation of the error where the error is for particular X. Y of X. And D which is the. prediction of the model. Built based on the data set D. Minus the target value H of X. Okay. So let's see. So what. We do is that. We just add a term here. And. 

this term that you see here is this term. If you look at this term that you see here. It's this term. Right. In between these two terms. We have added. The expected. Y. And subtracted that. Right. So it's zero. Value that we have added to the thing. Right. So we have. We have. X. And D which is the prediction. prediction minus the average prediction plus the average prediction minus the ground truth value. All of them. Squares. So far. So good. This is. Like basic. Very basic stuff. Right. Then. You have. Like if you look at the inside. This expectation. 

is. You have. 

This block. To the power of two. Right. So what you do is that. You consider this as one. This as another term. So you have. This. 

Now the interesting thing is that. 

You can prove that two times. The first term times the second term is zero. And I put Y is Y in there. At home figure out why it is zero. Okay. 

And then. 

that. 

So we just put in here. Okay. 

Then. 

You notice that this is an expectation. Of a. Constant value. Like this. H of X which is the target is constant. This expectation. Of. 

So. 

Essentially this is a constant. So expectation of a constant is that constant. So you can get rid of this expectation. And that's why you see the difference is from the. This line to the. Line next to it. Now you. We are now left with these two terms. This is the expectation. The first term is the expectation. Of. 

the. The expectation of the Y of X. A squared. 

And. The second term is. The expectation of Y of X minus the target. A. Square. The first term is what we call the variance. And the second term is what we call as the. 

the number from the statistics. Right. A variance. You know variance of a. You know a set of. Numbers. Right. Is. 

The sum of the differences of those numbers minus the average. Divided by the number of numbers. By the size of the set. This is exactly what that green term is. This is why we call it the variance. It's exactly the definition of variance in. variance. The same number of the statistics. And the second term is by us based on the definition that I mentioned before. So the bias is we call it as the. The distance between the average prediction and the ground truth. Value. And you see exactly that in here. Right. So based on our definition of the bias. This is bias squared because we have this square here. 

the green term is the variance. Now it's very interesting so if you look at the first line and the last line. What we did is that. The error. For you know that the error on. You know on. X. Can be decomposed as two terms. Where the first term is the variance and the second term is a bias. Square. 

beautiful and then by line so that we see the derivation. In. 

beautiful, and then bind by line so that we see the derivation. 

In the core of it, these derivations are extremely simple. If you go home and then try to look at this another time, these are extremely, extremely simple. The only line that needs more, some thinking is this line that I highlighted that this is going to be 0. That's a little, that needs a little of thinking. The other ones are really simple. 

Okay, so the math exists. Okay, let's not go into details. Okay, let's not get stuck in the details. Okay, so there is some math exists, which shows that the generalization error can be decomposed into two terms where one of those terms is the so called variance. The one is the biases squared. That's what I want you to learn from this slide for this lecture, for now, right, that we are in class. Don't get into the proof for now. The math exists by which we have shown that the error is the sum of variance and biases squared. Why it's important, okay? Because it shows that if you have high variance, you are going to have high error. error. If you have high bias, you are going to have high error. So to minimize your error, you need to minimize your variance, your bias, or both of them at the same time. This is the important message of this formula. Okay, and if you think about those four configurations of the dot board, that's exactly what we saw there as well. it was intuitive. In here, we kind of showed mathematically which one of those settings from four settings is good, which ones are bad based on this. Okay? 

All right. Okay, sounds good. Now, as I said, this is what I just said, right? Our goal is to minimize generalization error. 

And actually, there is a trade off usually. that a very flexible models, like the models that have many, many parameters, like these J &AI models, like GPT -5, or et cetera. These models have very low bias, but they have high variance. Usually, when you want to make one of those terms good, the other terms, the other term becomes bad. So if you have a very flexible model, you are going to get usually a high variance for that model. So you need to kind of look at 

your eventual gain by forming these bias and variance terms and adding them and see whether in aggregate, that model is going to be good for your case or not. Meaning that the benefit that you get for bias is not is not made, you know, irrelevant by having a very high variance. On the other hand, relatively rigid models have high bias, but low variance. Again, low variance is good, but high bias is bad, right? So when you basically make one of them, good, right? The other one goes up, which is bad. Okay. 

All right, so that's the mathematics. So far, let me mention what we did so far, right. So we started from the intuitive notions of bias and variance with the dark example. And then we delved into the mathematical definitions and truth that shows like, you know, how do you solve the error is related to the bias and variance? Okay. So we did the math as well. Now we want to see, we want to look at some examples. Okay. Of these terms and how they interrelated with the notion of model complexity that we saw before. 

I want to mention something, but I keep it for later. Okay. So this is the setup. 

So the setup is as follows. I have a function, which is sign of two pi x. This is the unknown ground truth function that I want to generate some samples from it and then pretend that I don't know this function. And then I want to do my machine learning on this, on the data set that coming from, that is coming from this, this function. I want to see like how good I can approximate that function, right? And then I want to see, like, how good I can approximate that function. Right. So we have to say, that's the same. So for that purpose, I sample 100 data points from this function. And then from that 100, okay. I actually, let me revise it. I build 100 data sets, 

I use a basically a linear regression model with 24 Gaussian basis functions. Right. And then I use a L2 regularization, which we call the reach regression. Right. So this is the setup that we have. So the number of data sets that I built, build is 100. Each data set has some 25 examples. I build a linear regression model for that sample. eventually I will have 100 models, thereby each model corresponds to one of these data sets. Okay. Now, what I can do is that I can choose different regularization parameters. Right. Remember that in the regularization, we have the error term plus lambda, the regularization term, times the complexity of the model. Right. The complexity of the model is the norm. norm two of the weights. Right. So when you choose different landers, in fact, you adjust the trade -off between the complexity and the empirical error in your objective function when you do a ridge regression. If you remember the last lecture. Right. 

Now, lambda, the higher is the lambda, the more important is the regularization term. Right. So you are trying to learn a, you know, less complex model, like a simpler model. So the higher lander means less complex models. The lower lander means more complex model. So far so good. So this is the highest lander. This is the, you know, this is the bottom one is the lowest lander. And the one in between the the mean is, you know, between them. So this is gonna be the, this is gonna resolve in the simplest model. 

Let me see. 

This one, this value of lander is gonna resolve in the simplest model. 

So this is gonna be the most complex model. So far so good. Yeah. So this is, you know, without doing the regression, I know this for fact, just based on the value of lander. Right. Now, look at the bottom figure. 

What you see on the left box is that, 

for those 100 data sets that I have built, for each one of them, 

I have fitted a model, a linear regression model, where the value of lander is minus 2 .4. 

Right. Or 2 .1, I can't see. And then I have the basically plotted those 100 functions that I've fitted. 

And what you see here is those in the, in the box on the bottom left is those 100 functions. 

Is it clear? So I have constructed 100 data sets. And to each data set, I have fitted a linear regression model with the Gaussian basis functions. And then after, where the regularization term is, this value. And then I have plotted those functions, 100 functions. And you see this vaguely, plot in here. Okay. Now, these functions vary a lot. Right. Means that meaning that they have high variance, right, by just looking at them. Right. That's interesting. Now, if you look at the average of these functions. Okay. So I have 100 functions. I can take the average. And then I can plot the average. Right. This is the bottom right plot. The red curve is the average of those 100 curves. The green curve, which is very close to it, I hope that you can see that, is the ground truth function, sign of 2 pi x that we saw before. Okay. And you see that, the average function, of those 100 functions is very close to the green curve. 

Meaning that it has low bias, because the average is, is close to the ground truth. 

Yeah. But 

those functions have high variance. Right. Now, look at, think about lambda again. Right. Land of was, various, we chose it to be a very small. value. Meaning that beforehand, we knew that the model complexity is going to be high. Right. So for high model complexity, we observed from this experiment that the variance is high among the fitted functions. But the bias is low, because the average of those functions is very close. So this is exactly what we mentioned before. But we cooked up at, machine learning problems. problems in the lab. And then we investigated, or confirmed exactly the same ideas that we saw before. Now, conversely, look at the very top two figures. Okay. In the top figure, look at the one on the left. You see those 100 functions. Or by the way, the only difference between the top and the bottom figures is the value of lambda. Right. The value of lambda in the top figure is much higher than, the bottom figures. Right. Now, those 100 functions, when we plot them, we get the top left figure. And so they don't very much. Right. So the variance is low. Right. But if you look at the average of them on the top right, which is the red curve, it's far away from the green curve, which is the ground. through function. Okay. Which is the classical symptom of high bias. Okay. So interesting. So when lambda is high, we learn simple functions, which have low variance, but high bias. This is exactly what we discussed mentioned before, which is confirmed by this example. And between these two examples, extremes, you have the middle figures. Right. So a lambda, which is between those extremes, give you like a, you know, a still look at a good approximation to the ground truth function. Like I'm looking at the middle. Right. If you look at the average of those functions is almost close to the green function, but not as close as the bottom one. But reasonably close. But in the middle left, you see like the variance is, 

is also, you know, higher than the top figure, but lower than the bottom figure. Right. So this is very interesting. You can do these fun examples at home, if you want to. 

Now, what I want to show here is, let me get rid of these writings. 

Okay. What I want to show here is, um, looking at the mat again, the equations again. Okay. So let's do a bunch of, um, like, you know, the bunch of, remind you of a bunch of definitions. So, suppose I have, um, Buddhist trapped L training data set. So the, the L that you see here is the number of, uh, training data sets that you have constructed, maybe by Buddhist trapped sampling. And then for a given X. Right. Um, you get the value of that X based on the, uh, models that you build based on these, based on these constructed data sets. So the predictions of each of these, models on X is Y superscript L of X, where superscript L denotes, you know, one of those L models that you've built. 

So the average of those predictions of the models for a particular X is Y bar of X. Okay. So that's a notation. Right. So this is the average. Um, then based on our, the 

average, the average minus the ground truth for a data, for a data point X, like the average minus the ground truth to the power of two. Right. Now if you have n data points in your, um, let me call them. So this is your data set. So this is the data set. set. So if you have n data points, this, this all the way to this, for each data point, you have computed the bias of your models for that particular data point. So basically, you take the average of those biases over your n data points and you see you get the biases squared. So these, this n ranges over the data points in your data set. Right. Um, this is, um, the that's the squared. Then, um, the variance is interesting. So this is the variance over one data point and then you have n data points, you get the average of those variances and you call them the variance. Right. So the average of one data point is what we saw before. Right. So it's basically the average of the distance of each prediction minus the average prediction squared. 

Makes sense. 

the variance over the red part is the variance over one data point. And we take the average of those variances over all data points we call it the variance. 

And then the test error is what you see in the bottom. 

So the test error is the average over the models and data points of the distance between the prediction of a model on a data point minus the ground truth value for that data point squared. So this is the test error for this data set based on the model that we have built. 

Okay, so this is these are the formulas, right? 

And now we are going to see something interesting. So by the way, these formulas are written just to reflect the definitions that we saw before. Okay, so the test error from the bottom to the top shows the average of a prediction of a model over a data point, right? So this is this is the test error. And that average is for, you know, the distance between the prediction minus the ground truth value squared. That distance is average over the models and the data points, right? So that's the test error. The variance as we saw is the average of the variances of over the data points where the variance of the data point is calculated based on the predictions of L models for that data point. And the boyos squared is what you see there. Now, what we want to do is the following. I want to compute 

for the setup that we saw in the previous slide, right? I want to compute these four quantities or the three quantities like the bias two variance and the test error 

for each value of the lambda. Okay, I'm going to fix the value of lambda, the regularization. parameter in your ridge regression, right? And then I'm going to fit like, you know, L models, right? 100 models because we construct the 100 data sets. 

And then for a particular lambda, I can compute the boyos and the variance for that model for the models that I built based on that particular lambda. Right? So if you're familiar to the previous slide, okay? So if you fix lambda, you build your 100 models and then you can basically, for example, if you fix your your your your your your lambda to be 0 .5 or 0 .8, you can then compute fit your models and measure or compute. 

this variance and then you can compute the boyos and then you can basically add them up to get the boyos two plus variance, which is the pink here. And basically you can build these, you can actually build these three curves like the boyos and the variance and the sum of the two as I just mentioned. So you built those constructed, you built and constructed those one had data sets. You just changed the value of lambda for each value of lambda, you build your 100 models and then you compute the boyos and variance and boys plus variance for those models. Right? And then if you do that for this range of values, you get these three curves. Okay. So the red curve is the variance. The blue curve is the bias and the pink curve or the bias squared whenever I say bias, I mean, vice squared. And the pink here is by the squared plus variance, the sum of the other two curves. Okay. That's very interesting plot. Don't look at the black here on the top. For now, don't look at that. Okay. Those three curves are very interesting. So in those curves, 

as you increase lambda from left to the right, as you increase lambda, the red curve, which is the variance goes down, which is expected because as we increase lambda in our ridge regression, we penalize more complex models more. So the models that we learn are tend to be simpler. Right? So that's exactly what you see there. Right? So you see that, and then when those models are simpler, the variance goes down because there is not a lot of variability. So you see that the variance, as we go from more complex models on the left of the figure to the less complex models or simpler models to the right of the figure. So complex, less complex. Yeah. So we go from left to right, we see that the variance goes down and the bias goes up, which is exactly what is expected. This is interesting, right? And then you see that if you look at the bias plus, bias squared plus variance, the pink curve, which is there, as I mentioned before, is our proxy for the georization error. You see that, you know, the georization error increases gradually and then the decreases, and then increases again, increases. Right? So there is a sweetest spot, like around here, there, it seems to be the right regularization value. Lander, right? If I choose my lambda in this spot, I'm going to have a good trade -off between the bias and the variance. 

The sum of the two is going to be at that point, it's going to be at the same time. going to be lower than other areas. So that's a good trade -off, right? Now, 

keep that aside for now. Now, let's look at the black curve. The black curve is the actual generalization error. Okay? The black curve is the actual generalization error that I have computed based on the formula on the bottom. Okay? Now, if you look at that curve, right? 

Which is built for models trained based on different land of values, right? If you look at that curve, the behavior of that curve, we just make to the x -axis, land of values, or the trade -off, the complexity of the model. 

The behavior of that is very similar to the behavior of the pink curve. In other words, the black curve also instruct me to choose a similar region of for land of value as the pink curve. Okay? So the recommendation of the pink curve coincides with the recommendation of the black curve, okay? Which is a good thing. Because if I don't have the black curve, I just go ahead with the pink curve, I'm going to be good in terms of the generalization error. Is it clear? So that's again a confirmation of what we discussed before that the bias to plus variance term is a good proxy for the generalization error. You see it in this example as well, pictorially. This is nice. I stop here, 

questions? 

All right. 

Yep. Yes, yes. Is to find optimal land in practice? 

But we also want to show that the math that we did works. The math that we did connected the generalization error to the bias plus variance. So what we did is we cooked up a machine learning problem and for that machine learning problem, we 

constructed data -staring data sets and fitted redrawation models over a range of values for land. And then we looked at the behavior of variance, bias, variance plus bias and the generalization error. And then we saw like a confirmation of our math. Our math is actually correct. Right? Many that the behavior of the pink here, which is the bias plus variance, is actually very similar to the or in parallel to the behavior of the black curve, which is the actual generalization error. 

Here is generalization. So the most in practice, most data set will behave like this. Yes. Yes. That's right. That's right. That's right. Does it mean if we find a plus point where the bias and variance crossing that point, that's where there's a good trade -off between the two. So that's optimal land. Right. Remember like the land that we had in the in the organization for the linear regression models, traded off the empirical error and the complexity. But we said that we don't know usually what value lander should have to trade off. This is one way to do the trade -off. Yeah. Yeah. 

Oh, just by visual thing. this point is lower than the other. 

Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. 

Yeah. Yeah. Yeah. Yeah. Yeah. Yeah. I mean, the optimal value means that, you know, if you look at all of the values in the on the pink here, the minimum. We call it the minimum, we call the minimum optimal. Right. The minimum happens here around here. I think I need to 

find the value for this part because you want to know ways to test that. So the lowest means that, you know, I want on this curve to find the value to find the value of lander on which the error on which the minimum on the pink here happens. So there is no integral. This is just the minimum. Okay. If that makes sense. 

the integral of the dx and stuff like that, we approximated by this summation because we have access over the data points. 

It was the proof that we saw before. Yeah. No, no. That's a good question. So each data point here, each a point here on these curves. is one of these summations. Okay. And these summations are approximation to the integrals. Yeah. Because there are based on the data set that is given to us. A sample and data set is a sample of the full distribution. 

Yeah. 

Yeah. 

All right. Sounds good. Okay. Let me show you another example. So this is good. 

So this is another example in spirit. Very similar to what we saw before. So again, we have this function sine 2 pi of x. And in here, we add a little bit of noise to our, you know, our data points that we sample from this curve. Just to mimic the noise. So we have 100 sampled data points generated by the function h. and now instead of having like, you know, a linear regression model with Gauss and basis functions and playing with the lambda value playing with the lambda value for the complexity. Instead of that, we are going to have different polynomials whereby each of these polynomials has a different complexity based on the degree that it has. So the simplest one is the zeroth order polynomial. Now then the first order third order and then the 15 order is the most complex one. Right. So here is another setting where we have four class of functions from simplest to the more complex. 

And then we do a similar thing that as we did before. Right. So these are the functions that, you know, we fit. 

here of the bias and variance of each of these four functions. I would say four types of functions. Right. Okay. So these are the data points that we are given. And this is the ground truth function. Right. These are the data points that are sampled from this ground truth function. And the reason that these data points are not and that function is because of 

the noise that we've added. Right. Now, okay, this is the setup. Of course, we pretend that we haven't seen this black function. Right. And we look at the functions that we fit and compare them with this ground truth black function. Okay. Let's first start from the zeroth order. So this is basically the zeroth order function that we can train fit on the data points, which is a very bad approximation. Right. This is the degree one better than the zero degree, but again, not a good approximation. This is a good one. This is a third order function that if we fit, we get a good approximation. The green one is a 15 order polynomial that is, you know, good approximation in some parts that we have data points, but in the parts that we don't have data points, it's vaguely and. wild. 

Now, 

I want to basically for I want to fit like each of each types of these functions on some booty step samples, some sample data sets booty step from this data set. Okay. So from this full data set that you see here, I'm going to boot a step a bunch of them and fit the degree zero to start with. if I do that, I get these lines. Each of these lines is what I fitted on one of these booty strapped training data sets. Okay. So the variance is low among these functions, the variability is low. 

But if you look at the average of those functions, it's far from the ground truth. 

So high bias, not the good class of functions. 

If I look at the degree one polynomials, which are lines, and I fit one, I fit, I fit them to my booty strapped train data sets. I see that collection of lines. The variance is again, relatively low. but the bias is also the bias is high, like the average of those lines is far from the ground truth. 

If I look at the degree three polynomials, 

right, the variance seems to be really, really low. It is such low that all of them almost overlapped with each other. So you don't see the variation, right, the variance is low. the bias is also relatively low, which is really good, right? It's almost close to the ground truth and that data point. And similarly to other data points as well. But we look at the bias and the bias over that one data point for now. And if you look at the degree 15 polynomials, it's weakly, like, you know, the variance is high. And if you look at the average, 

the distance of the average to the ground truth is low, right? So the bias is low, but the variance is high if you look at the degree 15 polynomial. So the degree 15 polynomial have low bias, but because they have high variance overall their error is going to be high. Because the error is a summation of variance in bias. And the best polynomials seems to be the degree three polynomial. Is that clear? It is simpler to see than the the lambda in the ridge regression. Like here we have four class of functions and then the best class of function seems to be the degree three polynomial. But essentially the exercise is similar to the one that we saw before. Now for these four class of polynomials, I can measure the bias squared and the variance as well, right? Similar to based on the formula that we saw before. And if I look at the bias squared, the bias for the zeroth or their polynomials is going to be the highest. 

And for the 15 order polynomial is going to be the lowest. Okay? So in that case, 15 in terms of the bias, 15 order polynomial win, right? They have very low bias. If you look at the variance, the variance of the 15 order polynomial is the worst is the highest. And the variance of the third order polynomial is the least. So this wins, right? But we said that the geolization error is the summation of the bias and variance. So if you look at the summation of the bias and variance, right? The third order polynomial has the lowest value. 

wins. And the zero order polynomial has the worst. Okay? So based on the bias squared and variance, third order polynomial is the best. Okay? Now if you look at the actual, you know, error, if we calculate the actual error, the actual error also shows that the third order polynomial is the best and zero order is the worst. Okay? So again, the behavior of the error of these different class of functions concluded based on the bias variance analysis coincides with the ground truth, right? With the ground truth error analysis that we saw there. By the way, I upload this updated version of the slope. Right? Because the version that you have doesn't have this one column. Yeah. 

All right. Sounds good? Any questions? This is another example again to kind of confirm our theoretical analysis. All right. So the useful guides and practices that, you know, this points and variance basically give you some keys to two, manage the complexity of your model. So let's assume that you have a model which has high error on a test set. And you ask yourself, do I need to train my model on a larger data set, right? To answer that question. When your model has a high variance, right? 

Probably that helps. Right? If you increase the size of the data set, you reduce the variance. 

So your model, you know, is complex for the data set that you have, right? So it has much more parameters than the number of data points that you have, right? That's why you see these variance, right? Now if you increase the data set size, your model, you know, freedom increases compared to the number of data set. points that you have. So you will see less variance. So this is, and then the error is going to go down. 

However, if your model has a high bias, it will not fix the issue. Okay? 

Adding more data points doesn't help. Right? The model has already, 

the model underfits your original data set. already. So if you increase the data set, it will underfit even more. Okay. So let's let consider another situation. Do I need to train my model with a smaller set of features, right? To answer that question, if you are in a situation that you have high error, and you're asking whether I need to add, I need to remove some of the features, right? If your model has a high variance, probably this is good. Because very, very, very well, when you remove some of the features, you are making your model less complex, right? And the variance decreases. 

But if your model has a high bias, it doesn't help. Right? So mean, it means that, you know, the model was already a simple model. If you remove more features, it's going to be even worse, it's going to be even simpler. So, and the error would be worse. It doesn't help. So to answer these questions, you see that, you know, I'm using the language of the model. of bias and variance. So if I have bias and variance, I can answer these questions. 

Do I need to obtain new features? When your model has a high bias, usually this works very well. Your model is, the reason that it has high bias is that it's too simple. So by adding two features, you make it more flexible. And that helps. So this is good, I think we are close to the end of the lecture. So we can get more of your time back and give you that time to study more for 20 minutes or half an hour. So in the tutorial, in this week, you see the bias and variance in regression. 

And you play with it. Similar to the things that we have. So in this lecture. And from the next week, we are going to start, like, our part on the part A on classification. And we're going to see linear models for classification. This is a kind of follow up on the linear models for regression that we saw in this lecture and the pre -slecture. Thank you so much. And see you next week. 