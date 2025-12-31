# Keywords

### Marginal Pr

	the sum over the joint probability distro

### Joint Pr

	the Pr of two or more events occurring

$$
Pr(X intersectY)=Pr(X)*Pr(Y)
$$

### Posterior Pr

	Derived from Bayes, where a belief/confidence in an event is revised after observed data

$$
Pr(A|B)=\frac{Pr(B|A)}{Pr(B)}
$$
### Conditional Pr

	the Pr that event A occus given that B occurs

$$
Pr(A|B)=Pr(A \ intersect \ B)
$$

### Bayes's Theorem

	allows for the inversion of conditional Pr to calculate cause given effect

$$
Pr(A|B)= \frac{Pr(B|A)*Pr(A)}{Pr(B)}
$$
### D-Separation

	Criterion used to determine if two variables are independent
	If a trail is active, then not D-Separated


## Inference Methods

### enumeration
	distribution of subset of collection
	marginal probabilities
	joint probabilities

**Steps**
1. Use Bayes
2. 
## Calculating from Joint Pr Distro

### steps
	get the posterior probability

$$
Pr(Q|E_1 = e_1,...e_n)
$$

	then get most likely explanation
	
$$
\begin{align}
\text{for a random varriable Q and evidence E: }\\
argmax(Pr(Q | E=e,...,e_n ))
\end{align}
$$


---

### conditional prob tables

each row contain the CP of node val for each possible combo in parent nodes
![[Pasted image 20250515202145.png]]

	given that lung cancer is true, there are a series of combos, and the same for cancer is false 

---

