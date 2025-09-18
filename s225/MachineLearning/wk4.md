
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
Of course, in the training time, in the machine learning training time, we don't have access to h of x. We are aiming to build h of x, because we don't have it. But for the sake of the definitions of the bias and variance, we assume that it exists, and we denote it as h of x. (29:39) y of x is the prediction of the model that we have built. Okay, so it's our model gives the output y for the input x, and it is built based on the data set D. Now, the generalization error