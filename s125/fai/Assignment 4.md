### CONFUSION MATRIX

$$
Accuracy = \frac{TP+TN}{TP+FP+TN+FN}
$$
as the number of samples correctly classified out of all the samples present in the test set,
$$
Precision = \frac{TP}{TP+FP}
$$
as the (positive case) number of samples actually belonging to the positive class out of all the samples that were predicted to be of the positive class by the model,
$$
Recall = \frac{TP}{TP+FN}
$$
as the number of samples predicted correctly to be belonging to the positive class out of all the samples that actually belong to the positive class,

$$
F1-Score = \frac{2*Precision*Recall}{Precision+Recall}
$$
as the harmonic mean of the precision and recall scores obtained for the positive class,
$$
Specificity = \frac{TN}{FP+TN}
$$
as the number of samples predicted correctly to be in the negative class out of all the samples in the dataset that actually belong to the negative class.

### Entropy & IG

$$
InformationGain(S, v) = H(S)-\sum_{v \in [b,o,x]}\frac{|s_v|}{|s|}*H(s_v)
$$

First, calculate the **entropy** -- *measures the total data's amount of uncertainty* 
	with a higher value meaning more uncertainty and a lower value meaning more homogeneous data:
$$
H(S)=-\sum^{n}_{i=1}Pr(x_i)log_2Pr(x_i)
$$
So first we calculate the H(S) of the split to identify the *entropy before the split*: 
$$\begin{matrix}
H(S)=-\frac{120}{157}log_2(\frac{120}{157}) - \frac{37}{157}log_2(\frac{37}{157}) = 0.788 \\
\end{matrix}
$$
and *after the split* (where it splits in three places form top-left-square):

### J48

	Builds a tree-like model of decisions based on feature values (nominal) to classify data.

**Use when**:
1. rules are easy to visualize
2. for more categorical data
**Limits**
3. prone to overfitting

### Naive Bayes

	assumes all variables are independent, calculates probabilistic outcomes

**Use when**
1. working with higher dimensional data
**Limits**
2. not all features may be independent

### KNN

	classifies based on instances that are nearest their neighbors

**Use when**:
1. analysis using distance heuristics
2. no assumptions needed about data
3. non-linear data sets

**Limits**
1. slow for large data sets
2. sensitive of noise
3. needs clear K-target


---

# Part 1

## 1 Pre-Analysis

### (a) Use WEKA to identify the board positions that most strongly predict a win or a loss.(hint: if you were the“x” player, where would you put your first cross and why?)

By using "Visualize All", we can see all the relevant variables, and try to predict the board positions:

![[Pasted image 20250521165924.png]]

with the first bar being b=blank, then o=O placement, then x=X placement, and the blue section of the bars representing positive and red being negative. The positions that most strongly correlate to a win or a loss are the middle-middle square, the top-right square, and the bottom-right square. Playing X having the most positive instances in the top-right, middle-right, and bottom-right, as those have the highest number of instances for X to play. But if we were to maximize our chances of winning, we would want to play X in the middle:

|     |     |     |
| --- | --- | --- |
|     | X   |     |
|     |     |     |

### b) Infer whether or not the first player (x winning more often) has an advantage

If we look at the attribute instances and the histogram, it would seem that the first player has a higher chance of winning because if we were to correlate the positive and negative instances like we did above, we would get:

| O   | X   | O   |
| --- | --- | --- |
| X   | O   | X   |
| O   | X   | O   |
With O being the winner. But if X were to go first and pick middle-middle, then it is more likely X will win.
## 2. Classifier Training & Evaluation
	Reference: https://www.youtube.com/@WekaMOOC
### 2.1 J48 (Decision Tree)

#### i) Train J48 with `minNumObj=2` (default) AND at least one other value. 

Here are the results of the training:
![[Pasted image 20250606172339.png]]
![[Pasted image 20250606172409.png]]
![[Pasted image 20250606172426.png]]
![[Pasted image 20250606172439.png]]

#### i.i) Examine the tree and identify the main variables

The main variables are shown in the "Preprocess" tab:

![[Pasted image 20250521204411.png]]
There are 9 attributes; each one corresponding to a square:

	 1. top-left-square: {x,o,b}
     2. top-middle-square: {x,o,b}
     3. top-right-square: {x,o,b}
     4. middle-left-square: {x,o,b}
     5. middle-middle-square: {x,o,b}
     6. middle-right-square: {x,o,b}
     7. bottom-left-square: {x,o,b}
     8. bottom-middle-square: {x,o,b}
     9. bottom-right-square: {x,o,b}
with the class attribute included at the end
#### i.ii) Then report the main splitting attributes in the tree.

By looking at the results, we see that there are 87 leaves and 130 nodes. The main splitting attributes are:

middle-middle-square=b
middle-middle-square=x
middle-middle-square=o

with all other splits going off of the first 3, creating 87 leaves.

#### ii) Trace the Decision Tree in the game below. 
![[Pasted image 20250521133607.png]]

Tracing the classifier rt above, we have the following:
middle-middle-square = o
|
	top-left-square = x
	|
		top-right-square = b: **negative, 6.0**

#### ii.i) Predict the outcome for the board state. 
middle-middle-square = o
|   top-left-square = x
|   |   top-right-square = b
|   |   |   bottom-left-square = b: negative (6.0)

=== Summary ===

Correctly Classified Instances         898               93.737  %
Incorrectly Classified Instances        60                6.263  %

#### Does the prediction make sense?

It depends. If we assume both players are rational in a zero-sum game of tic-tac-toe, then the sensibility of the prediction will depend on who goes first.

If it is player X's turn, then this would conflict with the predicted outcome. If player O's turn, then the outcome would have predicted accurately. So, it seems that the data file trains by assuming that there is a 60% chance player X *will not* make a play on the top-right corner.



#### iii) Identify the first split in the tree, and calculate Information Gain  (show hand calculations).

The first split in the tree is middle-middle-square = b, we first get a count of the differing instances and identify the positive ones (going from top to bottom):
![[Pasted image 20250606192842.png]]

and we want to identify the information gain based on the formula:
$$
IG(S, v) = H(S)-\sum_{v \in [b,o,x]}\frac{|s_v|}{|s|}*H(s_v)
$$
but in order to do that, the H(S) function is an *entropy* function that measures the total data's amount of uncertainty with a higher value meaning more uncertainty and a lower value meaning more homogeneous data:
$$
H(S)=-\sum^{n}_{i=1}Pr(x_i)log_2Pr(x_i)
$$
So first we calculate the H(S) of the split to identify the *entropy before the split*: 
$$\begin{matrix}
H(S)=-\frac{120}{157}log_2(\frac{120}{157}) - \frac{37}{157}log_2(\frac{37}{157}) = 0.788 \\
\end{matrix}
$$
and *after the split* (where it splits in three places form top-left-square):
$$\begin{matrix}
H(S_{TL,b})=-\frac{20}{28}log_2(\frac{20}{28}) - \frac{8}{28}log_2(\frac{8}{28}) = 0.655  \\ \\
H(S_{TL,o})=-\frac{32}{54}log_2(\frac{32}{54}) - \frac{22}{54}log_2(\frac{22}{54}) = 0.975 \\ \\
H(S_{TL,x})=-\frac{68}{78}log_2(\frac{68}{78}) - \frac{10}{78}log_2(\frac{10}{78}) = 0.552 \\ \\
H_{weighted}(S) = (\frac{28}{157}*0.655) + (\frac{54}{157}*0.975)+(\frac{78}{157}*0.552)= 0.728
\end{matrix}
$$

With the entropy calculations, we can calculate the Information Gain to get the sum of averages over all positive and negative instances for each attribute:
$$
IG(S,v) = H_{weighted}(S)\sum_{{v \in b,o,x}}= 0.788 - (\frac{28}{157}*0.655) + (\frac{54}{157}*0.975)+(\frac{78}{157}*0.552)= 0.5980891 
$$

#### iv) Report accuracy and interpret the confusion matrix results.

There is a lot of data that the confusion matrix provides that will be valuable throughout this assignment:

| **True Positive (TP)**  | **False Positive (FP)** | **true positive** +**false positive = Total Positive Samples**  |
| ----------------------- | ----------------------- | --------------------------------------------------------------- |
| **False Negative (FN)** | **True Negative (TN)**  | **false negative** + **true negative = Total Negative Samples** |

(with the rows being the actual values and the columns being the predicted)

We have the following methods of analysis that we can get from the confusion matrix:

$$
Accuracy = \frac{TP+TN}{TP+FP+TN+FN}
$$
as the number of samples correctly classified out of all the samples present in the test set,
$$
Precision = \frac{TP}{TP+FP}
$$
as the (positive case) number of samples actually belonging to the positive class out of all the samples that were predicted to be of the positive class by the model,
$$
Recall = \frac{TP}{TP+FN}
$$
as the number of samples predicted correctly to be belonging to the positive class out of all the samples that actually belong to the positive class,

$$
F1-Score = \frac{2*Precision*Recall}{Precision+Recall}
$$
as the harmonic mean of the precision and recall scores obtained for the positive class,
$$
Specificity = \frac{TN}{FP+TN}
$$
as the number of samples predicted correctly to be in the negative class out of all the samples in the dataset that actually belong to the negative class.

Now, if we turn our attention to the classifier's decision tree, we have:

 a   b   <-- classified as
 242  90 |   a = negative
  76 550 |   b = positive

with the analysis values as:
![[Pasted image 20250606212404.png]]

This suggests a high rate of accuracy, and precision with fewer negative instances. The recall is solid for the instances belonging to he positive class on average, with a good F1 score (harmonic mean) for the model results 
### 2.2 Naive Bayes
	
#### i)  Calculate (by hand) win/loss probabilities for the board state in from WEKA's output in 2(a.ii) using WEKA’s output.

Assuming the board configuration, we calculate the predicted probability of a win and the predicted probability of a loss by taking a look at the output:

Base Probability Negative = 124/225 = 0.35
Base Probability Positive = 296/629 = 0.65

We first calculate the posterior probability for the positive and negative instances for each attribute split rounding up to three decimals:

| Attribute              | Letter                  | P(Value,negative)                  | P(Value,positive)                  |
| ---------------------- | ----------------------- | ---------------------------------- | ---------------------------------- |
| `top-left-square`      | `x`                     | 124/335​≈0.370                     | 296/629≈0.471                      |
| `top-middle-square`    | `o`                     | 102/335≈0.304                      | 230/629≈0.366                      |
| `top-right-square`     | `b`                     | 64/335≈0.191                       | 143/629≈0.227                      |
| `middle-left-square`   | `x`                     | 154/335≈0.46                       | 226/629≈0.359                      |
| `middle-middle-square` | `o`                     | 193/335≈0.576                      | 149/629≈0.237                      |
| `middle-right-square`  | `x`                     | 154/335≈0.46                       | 226/629≈0.359                      |
| `bottom-left-square`   | `b`                     | 64/335≈0.191                       | 143/629≈0.227                      |
| `bottom-middle-square` | `x`                     | 154/335≈0.46                       | 226/629≈0.359                      |
| `bottom-right-square`  | `o`                     | 147/335≈0.439                      | 190/629≈0.302                      |
|                        | **approximate product** | $$\prod Pr(V,n)\approx 0.0000353$$ | $$\prod Pr(V,p)\approx 0.0000191$$ |
Using normalizaion:
$$\begin{matrix}
\prod Pr(V,n)+\prod Pr(V,p)\approx 0.0000544 \\ \\
Pr(negative) = \frac{0.0000353}{0.0000544}\approx 0.649 => 54.9\% \\ \\
Pr(positive) = \frac{0.0000191}{0.0000544}\approx 0.351 => 35.1\%
\end{matrix}
$$

#### ii) Report the classifier's accuracy

Based on the WEKA report:
![[Pasted image 20250602171345.png]]

we can analyze the accuracy from the confusion matrix:
$$
Accuracy = \frac{TP+TN}{TP+FP+TN+FN}
$$
giving us,
![[Pasted image 20250606212920.png]]
![[Pasted image 20250606212910.png]]

and thus we can see that overall accuracy is about 70%, which is lower than J48 Decision Tree classification of 84%. However, the classifier's precision is around 42%, suggesting that the number of positives classified *as* positive were low.

Using Naive Bayes reasoning, we care about comparing likelihoods within the data. So, we care about which squares have the most predictive power in a sense. Based on our analysis, we can see that 
 middle-middle-square=- = 193/335≈0.576 has the highest level of influence (which makes sense because it is in the middle of the tic-tac-toe board. In many cases, whoever is able to place their letter their first, has the most options for winning the game).
#### ii.i) analyze the confusion matrix.

By analyzing the confusion matrix:

   a   b   <-- classified as
 142 190 |   a = negative
 101 525 |   b = positive
$$
a: 142 + 525 = 667 \ \ \ 
b: 190 + 101 = 291
$$
which gives us 
$$
a = \frac{142}{667} \approx 0.213, \ \ \ b = \frac{190}{291} \approx 0.653
$$

#### ii.ii) What is WEKA's NAIVE Bayes prediction for the game in 2(b,i), and the probability of the prediction?

Other ways to analyze the prediction for the game is to look at the report's visualizations:

![[Pasted image 20250606213610.png]]

This visualization shows the distribution of instances with positive and negative. After increasing the jitter for easier viewing, we can see that there are more positive instances that negative, which matches our earlier analyses. 

As for the winner of the game, it is likely to be player X because our Naive Bayes analysis reports an accurate prediction for the target (in this case our x) 70% of the time.

### 2.3 KNN

#### Find three instances in the dataset that are similar to the game in item 2(a)ii, and use the Jaccard coefficient, combined with a distance metric, to calculate (by hand) the predicted outcome for this game. Show your calculations.

We ave the classifier results:
![[Pasted image 20250606220818.png]]

and 3 instances that are similar to the game can be seen on these row numbers

222	x	x	x	b	o	o	x	b	o	positive
797	x	b	o	o	x	b	o	x	x	positive
213	x	x	x	b	o	o	x	o	b	positive

In order to identify the Jaccard Coefficient, we need the confusion matrix:

   a   b   <-- classified as
 323   9 |   a = negative
   1 625 |   b = positive

We use the JC formula which measures a set of similarities of a certain cardinality with a certain class against a predicted class:
$$
JC = |\frac{A \cup B}{A\cap B}|
$$

In this case, we would calculate:
$$
JC=Precision\cup Recall
$$
For the Jaccard Coefficient: 
Precision = 0.997, Recall = 0.973 → Jaccard ≈ 0.9843.

323/323+9+1 = 0.97
625/625+9+1 = 0.984

Which says that X will win the game.

### 3. Draw a table to compare the performance of J48, Naı̈ve Bayes and IBk using the accuracy, recall, precision and F-score measures produced by weka. Which algorithm does better? Explain in terms of these summary measures. Can you speculate why?

![[Pasted image 20250603114255.png]]

Based on the reports, KNN performs the best. It outperforms the other algorithms on all accounts, especially by looking at the average. The main reason why KNN outperforms the others is because of the *type* of data that we are analyzing. KNN focuses on the similarities of categorical features, such as "this one is x, and this one is o, and x is closer to the target" lends itself much easier to using a KNN classifier.



#### ii) Test different KNN and distance Weighting settings (4 total variations).

---




# Part 2: Consider the postoperative-patient dataset

## use the weka visualization tool to analyze the data, and report briefly on the types of the different variables and on the variables that appear to be important.


![[Pasted image 20250603115223.png]]

By looking at the .arff file through an basic text editor,  can glean some important information about the nature of the variables. in "Relevant Information", the classification task is for deciding to what area post-operative patients should be sent to, due to hypothermia being a significant issue in recovery. The variables below correspond to different body temperatures:
L-CORE - core body temperature
L-SURF - surface temperature
L-O2 - oxygen saturation
L-BP - last measurement of blood pressure
SURF-STBL - surface temperature stability
CORE-STBL - stability of patient's core temperature
BP-STBL - stability of patient's blood pressure
decision - discharge decision


## Run J48 (=C4.5, Decision Tree), Naı̈ve Bayes and IBk (k-NN) to learn a model that predicts whether a patient should be discharged. Perform 10-fold cross validation, and analyze the results obtained by these algorithms as follows.

### Explain the meanings of these parameters. You should report on performance for at least two variations in total of the operational parameter minNumObj for J48, and at least two variations of each KNN and distanceWeighting for k-NN (four variations in total for k-NN).

The minimum object parameter for J48 controls the minium number of instances required  in a leaf node of the tree. This parameter can used to prevent overfitting, and helps to simplyfy the tree. 

The distanceWeighting parameter determines the distances to neighboring instances influence their "voting power" (influence) when predicting the class of a new instance. It is especially useful when neighbors are not equally relevant due to varying distances. 

The vote of a neighbor is weighted by 1/distance​. If two instances are identical (distance = 0), the weight is set to 1/small_constant​ to avoid division by zero.

distanceWeighting-1 - behaves as linear weighting where closer neighbors contribute more, but the influence drops linearly with distance. 

**J48 minNumObj=2**
![[Pasted image 20250603130453.png]]

**J48 minNumObj=3, Unpruned**
![[Pasted image 20250603130643.png]]
![[Pasted image 20250603130704.png]]

**KNN Weighting 1/Distance, KNN Obj 2**
![[Pasted image 20250603131012.png]]

**KNN Weighting 1/Distance, KNN Obj 4**
![[Pasted image 20250603131037.png]]

**KNN Weighting 1-Distance, KNN Obj 2**
![[Pasted image 20250603131113.png]]

**KNN Weighting 1-Distance, KNN Obj 4**
![[Pasted image 20250603131148.png]]

## J48 
### i. Examine weka’s output (e.g., Decision Tree), and indicate which are the main variables. 

Based on the results, we have 86 instances with 8 attributes:
L-CORE - core body temperature
L-SURF - surface temperature
L-O2 - oxygen saturation
L-BP - last measurement of blood pressure
SURF-STBL - surface temperature stability
CORE-STBL - stability of patient's core temperature
BP-STBL - stability of patient's blood pressure
decision - discharge decision
### ii. What is the accuracy of the output produced by weka (e.g., Decision Tree)? Why is it different from the accuracy you would expect by considering only the majority class? Explain the results in the confusion matrix.

The accuracy of the output of J48 is about 71% for classification. For the reports, we have a precision of an average
$$
 \frac{0.517 +0.66}{2}=0.9
$$
compared to the majority class A:
$$
 \frac{0.750 +0.718}{2}=0.734
$$
As alluded to previously, Class A is the dominant class in both reports
    
**J48 1**: 69.77% (60 correct).
        
**J48 2**: 70.93% (61 correct).

We can conclude that higher accuracy does not mean a model is better if it ignores minority classes. 

For the confusion matrices, we have:

**J48 1**
  a  b   <-- classified as
 61  1 |  a = A
 24  0 |  b = S

61/61 = 1
24/25 = 0.96

**J48 2**
  a  b   <-- classified as
 54  8 |  a = A
 18  6 |  b = S

54/60=0.9
8/26 = 0.308

We can see the that there is high precision for the classification, but 24 flase positives for A, and always fails on S. What is likely happening is that there is an overfitting of the data. 

## Naive Bayes

### Explain the meaning of the “probability distributions” in weka’s output, illustrating it with reference to the BP-STBL attribute.

From the report:
![[Pasted image 20250603163203.png]]
![[Pasted image 20250603163218.png]]

The probability distribution in the report represents the confidence levels of the classifier in terms of predicting whether a patient will be classified as **A (e.g., "stable")** or **S (e.g., "unstable")**, based on the observed feature likelihoods and prior probabilities. 
$$
P(A)=0.72, \ \ P(S)=0.28
$$
Suppose a patient object instantiates CORE-STBL, BP-STBL, and SURF-STBL. We could see the classifier output something like:
$$
P(A|features)=0.85, \ \ P(S|features)=0.15
$$
which corroborates with our previous observations that A is the majority class.

We can visualize this analysis using the test options in WEKA:
![[Pasted image 20250603164721.png]]

Here is with the entropy evaluation metric added
![[Pasted image 20250603165018.png]]

We can use these visualizations to more easily see the misclassification numbers, and where the model works or fails. In the visualizer, we can see that STBL sub-attributes are strongly correlated with A, but that could be due to bias. I don't think that is strong enough justification for actually taking the action of deciding where the patient is discharged or not. Especially since we have a lot of "overlapping" right in the middle of stable or unstable (the blue and red).

### Calculate (by hand), from the probability distributions in weka’s output, the probability that a person with the following attribute values would be discharged, and the probability that they would remain in hospital. Show your calculations.

For this section, we first calculate the posterior probabilities for each value (A:= Discharged, S:= Hospitalized) using the WEKA report for the marginal probabilities for each attribute, and then normalize:

|                       |              |              |
| --------------------- | ------------ | ------------ |
|                       | Pr(Value\|A) | Pr(Value\|S) |
| L-CORE = mid          | 42/65≈0.646  | 16/27≈0.593  |
| L-SURF = low          | 17/65​≈0.262 | 8/27≈0.296   |
| L-O2 = good           | 33/64≈0.516  | 14/26≈0.538  |
| L-BP = high           | 23/65​≈0.354 | 7/27≈0.259   |
| SURF-STBL = stable    | 32/64​=0.500 | 13/26=0.500  |
| CORE-STBL = stable    | 60/65​≈0.923 | 22/27≈0.815  |
| BP-STBL = mod-stable* | 18/65​≈0.277 | 5/27≈0.185   |
	* transferred from excel

Then,
$$
P_{posterior}(A|feature)\propto P(A)* \prod P(feature|A)
$$

For class A:
$$\begin{matrix}
Pr(A)*Pr(mid|A)*Pr(low|A)*...*Pr(mod-stable|A) = \\
0.72×0.646×0.262×0.516×0.354×0.500×0.923×0.277 \approx 0.0028
\end{matrix}
$$
For class S:
$$\begin{matrix}
Pr(A)*Pr(mid|A)*Pr(low|A)*...*Pr(mod-stable|A) = \\
0.28×0.593×0.296×0.538×0.259×0.500×0.815×0.185 \approx 0.0005
\end{matrix}
$$
Normalizing by using the reciprocal value, we have:
$$
N = \frac{0.0028}{0.00335} = 85.1\%, \ \ \frac{0.0005}{0.00335} = 14.9\%
$$

### iii. What is the accuracy of the Naı̈ve Bayes classifier? Explain the results in the confusion matrix. What is the prediction of weka’s Naı̈ve Bayes classifier for the patient in item 2(b)ii, and the probability of this prediction?

The accuracy is 69.746% of correct classifications.

the confusion matrix, or contingency table, demonstrates the Naive Bayes model performance. Based on WEKA's manual, 
"The True Positive (TP) rate is the proportion of examples which were classified as class x, among all examples which truly have class x, i.e. how much part of the class was captured. It is equivalent to Recall. In the confusion matrix, this is the diagonal element divided by the sum over the relevant row"

  a  b   <-- classified as
 58  4 |  a = A
 22  2 |  b = S
$$\begin{matrix}
A = 58+2=60, \ \ \
\frac{58}{62}\approx 0.935 \\
S = 4+22 = 26, \ \ \frac{4}{24}\approx 0.167
\end{matrix}
$$

What this says is that the classification of the patients being discharged (A) was correct 93% of the time concerning the discharged cases, and misclassified 16% of the time for non-discharged.


We know that the instance is there (as demonstrated in the previous question), and then we check the results using output to plain text:

![[Pasted image 20250604113714.png]]
![[Pasted image 20250604113730.png]]

with WEKA predicting A, and the probability being 61%

## KNN

### i. Find three instances in the dataset that are similar to the patient in item 2(b)ii (you can do this visually), and use the Jaccard coefficient, combined with a distance metric, to calculate (by hand) the predicted outcome for this patient. Show your calculations.

Three instances that are similar to the patient in item 2(b)ii are (organized by mod-stable first)

Original Instance
![[Pasted image 20250604115954.png]]

Similar Case 1
![[Pasted image 20250604120020.png]]

Similar Case 2
![[Pasted image 20250604120109.png]]

Similar Case 3
![[Pasted image 20250604120149.png]]

Summaraized in the table below:

|     | l-core | l-surf | l-o2      | l-bp | surf-stbl | core-stbl | bp-stbl    | decision |
| --- | ------ | ------ | --------- | ---- | --------- | --------- | ---------- | -------- |
| 1   | mid    | low    | good      | high | stable    | unstable  | mod-stable | A        |
| 2   | mid    | low    | good      | high | unstable  | stable    | mod-stable | A        |
| 3   | mid    | low    | excellent | high | stable    | stable    | mod-stable | A        |

To predict the outcome using the Jaccard coefficient and distance metric, we can refer to the similarity formula:
$$
J(A,B)=\frac{∣A∪B∣}{∣A∩B∣​}
$$
where A and B are instances of the attributes. Given that the attributes are categorical, we can calculate the distance by the complement:
$$
D= 1-J(A_i,B_i)
$$
$$
Ed \ x_i,x_j=\sqrt{\sum_{k=1}^{n}(f_{ik}-f_{jk})^2}
$$
We first count the number of members which are shared between the instances by lining them up:

Instance 1: {mid, low, good, high, stable, unstable, mod-stable}
Instance 2: {mid, low, good, high, unstable, stable, mod-stable}
Instance 3: {mid, low, excellent, high, stable, stable, mod-stable}

and we want to compare instance 1v2, 1v3, and 2v3
$$\begin{align}
J(I_1, I_2) = \frac{5}{7}\approx 0.714 \\
J(I_1,I_3)= \frac{5}{7}\approx 0.714 \\
J(I_2,I_3)=\frac{5}{7}\approx 0.714  \\
= 0.714*3=2.142
\end{align}
$$

Then we calculate distance to get the predicted outcome:
$$
1-2.142=1.142
$$


### ii. What is the accuracy of the k-NN classifier for different values of k (kNN)? Explain the results in the confusion matrix.

We have the KNN classifier, where KNN=1:
![[Pasted image 20250604161029.png]]

KNN=2
![[Pasted image 20250604161057.png]]

KNN=3
![[Pasted image 20250604161119.png]]

**confusion matrix**



KNN=1
  a  b   <-- classified as
 53  9 |  a = A
 24  0 |  b = S

![[Pasted image 20250606223851.png]]


KNN=2
  a  b   <-- classified as
 61  1 |  a = A
 24  0 |  b = S
![[Pasted image 20250606223916.png]]


KNN=3
  a  b   <-- classified as
 60  2 |  a = A
 24  0 |  b = S
![[Pasted image 20250606223939.png]]

## 3 Draw a table to compare the performance of J48, Naı̈ve Bayes and IBk using the accuracy, recall, precision and F-score measures produced by weka. Which algorithm does better? Explain in terms of these summary measures. Can you speculate why?

![[Pasted image 20250606224008.png]]

Based on what I see, I think that both J48 and KNN with parameters set to 2 are practically tied with J48 only slightly being higher in terms of Recall. I believe the reason is because the J48 is a model that works well with categorical data, or data that does not have numeric values as instances. Decision trees tield discrete outputs, and tree traversal can be easily performed using nominal data.  Moreover, since KNN is simply finding similarities, it is no surprise that it can perform the same.

# Part 3


![[Pasted image 20250604200408.png]]
![[Pasted image 20250604200429.png]]
## 1.  What is the resultant regression function?

$$
UR = (-0.0014 * AllOrdsIndex) + (-0.2452 * HousingLoanInterestRate) + 13.7286
$$
## Using the resultant regression function, calculate by hand the Absolute Error for the year 1986.

We have the following data:

"No.	Year	Immigrants-ks	All-Ords-Index	Housing-Loan-Interest-Rate	Exports-ms	Unemployment-Rate
6	    1986.0	204.4	1779.1	15.5	77547.0	8.4"

Plugging to the correct values into the regression function, we have
$$
UR=(-0.0014*1779.1)+(-0.2452*15.5)+13.7286 = 7.43726
$$
We want to find:
$$
MAE=\frac{1}{n}\sum^{n}_{i=1}|y_i-\hat y_i|
$$
We first compute the absolute error from the original data, and the output from the function and plug them into the formula above:
$$
MAE=\frac{1}{n}\sum^{n}_{i=1}|8.4-(7.43726)| = 0.9627
$$

## Calculate (by hand) the Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) obtained by the regression function between the years 1986 and 2006 (to avoid “?” values). You can use the excel spreadsheet abs.xlsx provided on moodle. How is MAE different from RMSE? (do these functions emphasize different aspects of performance?

We have the two formulas:
$$\begin{matrix} \\
MAE=\frac{1}{n}\sum^{n}_{i=1}|y_i-\hat y_i| \\
RMSE = \sqrt {\frac{1}{n}\sum^{n}_{i=1}}(y_i-\hat y_i)^2
\end{matrix}
$$
With the following data from 1986 to 2006:

| year   | immigrants | all-ords-index | housing-loan-interest | exports  | unemployment-rate | UR(x)  |
| ------ | ---------- | -------------- | --------------------- | -------- | ----------------- | ------ |
| 1986.0 | 204.4      | 1779.1         | 15.5                  | 77547.0  | 8.4               | 0.9627 |
| 1987.0 | 242.3      | 1585.3         | 13.5                  | 84101.0  | 7.7               | 0.4989 |
| 1988.0 | 249.9      | 1527.7         | 17.0                  | 84981.0  | 6.5               | 0.9214 |
| 1989.0 | 231.9      | 1508.8         | 16.5                  | 88950.0  | 6.0               |        |
| 1990.0 | 236.4      | 1504.9         | 13.0                  | 98970.0  | 8.5               |        |
| 1991.0 | 234.2      | 1652.7         | 10.5                  | 108328.0 | 11.1              |        |
| 1992.0 | 203.7      | 1722.6         | 9.5                   | 116355.0 | 11.8              |        |
| 1993.0 | 207.4      | 2040.2         | 8.75                  | 127362.0 | 11.3              |        |
| 1994.0 | 238.5      | 2000.8         | 10.5                  | 132979.0 | 9.3               |        |
| 1995.0 | 262.7      | 2231.7         | 9.75                  | 146225.0 | 8.8               |        |
| 1996.0 | 261.0      | 2662.7         | 7.2                   | 162144.0 | 8.9               |        |
| 1997.0 | 265.4      | 2608.2         | 6.7                   | 169720.0 | 8.6               |        |
| 1998.0 | 271.9      | 2963.0         | 6.5                   | 172977.0 | 7.8               |        |
| 1999.0 | 305.1      | 3115.9         | 7.8                   | 189827.0 | 6.8               |        |
| 2000.0 | 348.6      | 3352.4         | 6.8                   | 205369.0 | 6.6               |        |
| 2001.0 | 353.4      | 3241.5         | 6.55                  | 203964.0 | 7.0               |        |
| 2002.0 | 373.8      | 3032.0         | 6.55                  | 204334.0 | 6.4               |        |
| 2003.0 | 401.3      | 3499.8         | 7.05                  | 206761.0 | 5.8               |        |
| 2004.0 | 426.9      | 4197.5         | 7.3                   | 213985.0 | 5.2               |        |
| 2005.0 | 457.4      | 4933.5         | 7.55                  | 219678.0 | 4.9               |        |
| 2006.0 | 513.4      | 6337.6         | 8.05                  | 228442.0 | 4.5               |        |


For the MAE:
$$
MAE= \sum \frac{Absolute Errors}{n} = |\frac{0.962+ ... +1.162}{21}| = \frac{3.762}{21} = 1.179
$$


For the RMSE:
$$
RMSE=\sqrt \frac{\sum (y_i - \hat y_i)^2}{n} = \sqrt \frac{(0.9627)^2+ ... + (2.288)^2}{21}= 1.281
$$

The difference lies in understanding the error size: 
MAE shows that the prediction is about 7 points away from the target value, but real world data suggests that unemployment is usually between 4% and 12%. So the model's errors are higher than the average variations in unemployment. The causes could be due to not taking into account other important predictors, like the immigrants column.
RSME shows that, since the large errors are penalized more, there are no extreme outliers as the value is close to the MAE.


### Use your model to predict the Unemployment-Rate for the year 2010.

First I created a supplied test data set that WEKA could use and ran it through the model. I then selected "re-evaluate moel on current test set", and got the results:
![[Pasted image 20250605105704.png]]

So based on the model it looks like 20210's unemployment rate will be 5.566

### How would you impute missing values for the All-Ords-Index for the years 1981-1983 and for the Housing-Loan-Interest-Rate for the years 1981-1985? Justify your answer. (Answers without justifications will receive no marks) Rerun weka to build a new regression model (using your imputed values). How does the new regression model compare to the previous one? What is the RMSE and MAE of the new model?

We could impute the missing values by first taking a look at data in the wild.

I looked into two sites: https://www.rba.gov.au/statistics/interest-rates/, and https://www.rba.gov.au/statistics/tables/ , but could not find more historical data that dated far enough.

I then looked into  Wikipedia https://en.wikipedia.org/wiki/All_Ordinaries:

- **1980:** The All Ordinaries Index closed at 713.50.
- **1981:** The index closed at 595.50, a decrease of 16.54%.
- **1982:** The index continued to fall, closing at 485.40, a further decrease of 18.49%.
- **1983:** The index saw a significant rebound, closing at 775.30, a jump of 59.72%.

the RBA https://www.rba.gov.au/publications/annual-reports/rba/1982/eco-fin-developments.html#:~:text=Rates%20charged%20on%20loans%20by,per%20cent%20in%20June%201982.:
- **1981:** Interest rates reached a peak of 21.0% in November 1981.
- **1982:** Rates declined to 20.5% in June, then further to 18.5% in March.

and into ABS https://www.abs.gov.au/AUSSTATS/abs@.nsf/Lookup/6202.0Main+Features1Jul%202007?OpenDocument=#:~:text=2%2C700%20to%20143%2C800.-,UNEMPLOYMENT%20RATE,unemployment%20rate%20remained%20at%204.8%25.:
- **2007:** The Australian unemployment rate in July 2007 was 4.3%. 
- **2008-2009:** The Great Recession, triggered by the 2008 financial crisis, caused significant job losses and increased unemployment globally. 
- **2010:** The Australian unemployment rate remained at 5.1% in June 2010, marking a slight increase from the previous year. In the US, the unemployment rate peaked at 10.0% in October 2009 and took several years to return to pre-recession levels.

![[Pasted image 20250605150716.png]]


With this data, we can generate new results:
![[Pasted image 20250606225857.png]]However, I think it would be crucial to consider immigration rates and other factors such as the ebb and flow of contracted employment to get a more accurate imputation.



