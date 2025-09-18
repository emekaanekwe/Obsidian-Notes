## Books
1. PRML pg 4-57
2. PRML exercises pg 58 
3. PRML pg 137-140
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


