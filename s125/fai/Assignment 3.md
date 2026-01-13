
# Part 1

I first want to mention that I completed sections of the assignment at different times, and I did not have enough time to make every section look stylistically identical. I apologize for that.

Also, I want to note that in some cases, I will name the nodes using the following abbreviations: 
$$\begin{matrix} \\
Pr(\phi|\psi) = c, \ \text{ where }\phi \ {\text{ and }}\psi \text{ are random variables (nodes) and c is some constant.}\\  \\
example: Pr(S|-F)= 1\ \ means \ \text{  "the probability that it is true that there is smoke} \\ 
\text{given that it is false there is fire is 1."} \\
\end{matrix}
$$
## a) 

"*The probability of smoke when there is fire is 0.9
and the probability of smoke when there is no fire is 0.01*"

This states that Fire precedes Smoke, and there is nowhere else that states that Smoke precedes any other node. So Fire -> Smoke

*"When the fire alarm goes off  ... you will have to evacuate ... But if there's no alarm, there's never an evacuation"*

These clauses form two conditional statements: if the Alarm node is true, then Evacuate is true, as as ll as the converse. So Fire -> Alarm -> Evacuate

*"If there is an evacuation ... thenewspaper will report it. If there is no evacuation ... the newspaper won’t report it."*

This also has the form of a conditional statement like above. So Fire -> Alarm -> Evacuate -> Report

Given that we know the following nodes with their respective probabilities:

|                            | P(alarm=true \| tampering, fire) | P(alarm=f \| tampering, fire) |
| -------------------------- | -------------------------------- | ----------------------------- |
| tampering=true, fire=true  | 0.5                              | 0.5                           |
| tampering=true, fire=false | 0.85                             | 0.15                          |
| tampering=false, fire=true | 0.99                             | 0.01                          |
| tampering=false, fire=true | 0                                | 1                             |

We can add in the additional nodes Smoke, Evacuation, Report with the corresponding probabilities added into their tables in each node:

![[Pasted image 20250523131227.png]]

With the final CPT Table below:

![[Pasted image 20250523212106.png]]


## b)
### i) What is the marginal probability that your smoke detector has been tampered with?

We want to essentially calculate the joint probability distribution starting with Tampering, however, the only node in question is Tampering. Thus, 

$$
Pr_{marginal}(Tamering=true)= \sum_{e \ \epsilon \ {(t,w)}} Pr(Tampering=e | Prior=e)
$$
and since there is no prior probability, we can simply conclude that the answer is:
$$
Pr(T=t)= 0.02
$$
### ii) ii. What is the marginal probability that there will be a news report tomorrow?

And by using Netica and observing the True row, we can deduce that the marginal probability that there will be a report is 0.387.

### iii. You have observed smoke in your apartment. What is the posterior probability that there will be a news report tomorrow?

We have observed the following:
![[Pasted image 20250523183822.png]]


We want to find the posterior probability. More generally,
$$
Pr(\theta|X)= Pr(X|\theta)*Pr(\theta)
$$

Where the posterior distribution of theta *after* observing the data is the distribution of the data dependent on theta times the prior distribution of theta *before* observing the data.

Using Netica's automated feature, we can see that the answer is 0.32

---

### iv) You have observed no fire, and saw a news report about your apartment. What is the posterior probability that your smoke detector has been tampered with?

We have the diagram:

![[Pasted image 20250523183958.png]]

In this case, we have to consider two scenarios:
$$
Pr(T)= Pr(-F \land R) =x
$$
Using Netica, we can see that the answer is 0.535


So we calculate,

$$
Pr(R|S) = \frac{Pr(S|R)*Pr(R)}{Pr(S)}
$$
then, by the following formula:

P(R) = 32/100
P(S)= 100
P(R|S)=
1/2
$$
Pr(R) = \sum_{}^{}Pr(E|A)*Pr(R|E)*Pr(T)*Pr(S|F)*Pr(A|F,T)
$$

Plugging in the appropriate values for the joint probabilities:

$$
0.9*0.01*0.02*0.5*0.88*0.75 +, ... + \ n
$$

We have 0.329

which concerns two dependent variables. This requires the application of Baye's rule to Tampering, Fire and Report:

P(T|F) = independ
P(T=1|R=.51)

$$
Pr(T|-F,R)=\frac{Pr(R|T,-F)*Pr(T|-F)}{Pr(R|-F)}
$$


We normalize to constrain the data so that it sums to 1:

$$
\frac{Pr(R|T,-F)*Pr(T|-F)}{\sum Pr(R|-F)} =\sum(\sum_{A,E}Pr(R|E)*Pr(E|A)*Pr(A|-F)Pr(T))
$$
This gives us close to 1. Thus, it is very likely that the alarm has been tampered with.

### v) 

We have the visual:

![[Pasted image 20250523190323.png]]

Again using Netica, we can see that the posterior probability that the detector has been tampered with is 0.02.

To find the correct independence property we need to find a case in which the following is true:

$$
 \forall x,y (Pr(x,y) = Pr(x)Pr(y)) \models Pr(A,B) = Pr(A)Pr(B)
$$

We know that Fire, while not dependent on Tampering, is an option that shares the same Alarm parent. Due to this, we can try:

$$\begin{matrix}
1. \ Assume: Smoke=false \\
2. \ Smoke \implies Fire \\
3. \ Pr(-S|F) = 0.99, Pr(-S) = 0.1, Pr(F) = 0.1 \\
4. \ Pr(-S,F) \not = Pr(-S)Pr(T)
\end{matrix}
$$

But that does not give us the answer we want. So, instead we can try:

$$\begin{matrix}
1. \ Assume: Smoke=false \\
2. \ Smoke \implies Fire \implies Alarm \implies Evacuation \implies Report \\
3. \ Pr(-S,T) = Pr(-S)Pr(R) \\
\end{matrix}
$$

We know from the data that the possible truth value of *Report* is **independent** from *Smoke* by way of a conditional independence property. Thus, *Report* would be a good option in finding x.


### vi) 

We have the visual:

![[Pasted image 20250523190727.png]]

Which gives us the posterior probability of 0.519

We have the following:
$$\begin{matrix}
 \ R \ \land -S \\
 \ Pr_{posterior}(T) = x \\
 \ R=true
\end{matrix}
$$


The reason why the absence of smoke has an effect on whether or not there was tampering is due to the variable being dependent. Fire has an effect on Smoke, Fire and Tampering have a common effect Alarm which leads all the way to Report. And since we are taking a "reversal" approach, then if Smoke were true, then it increases the chances of Fire, which then increases the chances of Alarm, and since the increase of one dependent variable increases/decreases the chances of the other dependent variable in a v-structure (Tampering), we can safely conclude that Smoke has an effect on Tampering. 

### vii) 

We have the visual:
![[Pasted image 20250523191110.png]]

Using Netica, we can see that the posterior probability is again 0.535. 

And if we were to consider the case that smoke is not observed, we get the same result for Tampering:
![[Pasted image 20250523191810.png]]

The reason for this is that the variables are D-separated where there has been a break in the connection between the two variables due to observed information, specifically Fire.

## C) 

By using both visual and analytic solutions, we can check if marginal and/or conditional independence holds for each hypothesized instance. 
$$\begin{align}
Pr_{marginal}(X,Y) = Pr(X)*Pr(Y)  \\
Pr_{conditional}(X,Y|Z)=Pr(X|Z)Pr(Y|Z) \\
P_{totalprobability}(X=True)=T,F∑​P(A=True∣T,F)⋅P(T)⋅P(F)  \\
\end{align}
$$

with 2^n permutations where n=[true, false].

And we want to establish whether they contain the the following properties:
$$\begin{matrix}
Pr_{commonCause}(C|A \land B) = Pr(C|B) \equiv A\perp C|B  \\
Pr_{commonEffect}(A| C \land B) = Pr(A|B) \equiv \lnot(A\perp C|B)\\
Pr_{causalChain}(A,B,C) = Pr(C|B)Pr(B|A)Pr(A) \equiv A\perp B|C
\end{matrix}
$$

**Tampering ⊥ Evacuation** 

For the visual solution:
![[Pasted image 20250524162730.png]]
Since observing Alarm would change the probabilistic outcomes, then Alarm blocks the path.
Since there is not clear observation of Alarm and that the statements are connected, we must conclude 
that this statement is false. 

The analytic solution:
$$\begin{align} 
Pr(T,E)=Pr(T)*Pr(E) \\ \\
Pr(A|T,F)=0.5 \\
Pr(E|A)=0.88\\  \\
P_{totalprobability}(A|T)=\sum_{T,F}​P(A∣T,F)*P(T)* P(F) \\ \\
0.5*0.02*0.01= 0.001 \\
0.85*0.02*0.99=0.01683 \\
0.99*0.98*0.01=0.00972 \\
0*...=0 \\
0.001+0.01683+0.00972+0=0.026632 \\
0.88*0.026632+0 = 0.0234 \\ \\
Pr(T,E)=∑​P(T)*Pr(F)*Pr(A|T,F)
0.02×0.01×0.5×0.88=0.000088 \\
0.02×0.99×0.85×0.88=0.0148 \\
0.02*0.234=0.000486 \\ \\
Pr_{joint}(T,F)=0.0148 \not = Pr(T)*Pr(F)= 0.00486 \\ 
\therefore \text{independence fails}
\end{align}
$$


**Tampering ⊥ Evacuation | Alarm** 

For the visual solution:
![[Pasted image 20250524152621.png]]

Via Netica and the above visual aid, we can see that there is a causal chain from Tampering to Alarm to Evacuation, and that Alarm being observed determines whether or not the Tampering and Evacuation are independent.
The statement is D-separated since the condition from Evacuate to Alarm  and Tampering to Alarm settle on the same node.
There is no inbetween path.
The statement has the property of a causal chain, and Alarm does break the connection if observed
This shows that the statement is true.

Even Netica itself throws an inconsistency error:


Using the analytic solution:
$$\begin{align}
Pr(T,E|A)= Pr(T|A)*Pr(E|A) \\ \\
Pr(T|A)=\frac{Pr(A|T)Pr(T)}{Pr(A)} \\
P(A|T)=0.5*0.01*0.85*0.99=0.8465 \\
\frac{0.02660*8465}{0.0266}​≈0.636 \\ \\
Pr(E|A)=0.88 \\ \\
Pr(T|A)=Pr(T,A,E) =  \\
\sum_{F=true} P(F)*P(A∣T,F)*P(E∣A)= \\
0.02*0.01*0.5*0.88=0.000088\\
\sum_{F=false} P(F)*P(A∣T,F)*P(E∣A)= \\
0.02*0.99*0.85*0.88≈0.0148 \\
0.000088+0.0148≈0.0149 \\
Pr(T,E|A)=\frac{0.0149}{0.0266}\approx 0.56 \\
Pr(T|A)*Pr(E|A)=0.636×0.88≈0.560 \\
\therefore \text{independence holds}
\end{align}
$$






**Tampering ⊥ Evacuation | Smoke**

For the visual solution:
![[Pasted image 20250524162759.png]]

Netica allows for one to change the values of Smoke even if Tampering and Evacuate are assigned a value.
We can see that both are parents of Alarm, and causal effect. 
Unless we observe Alarm, then they are D-separated.
Alarm blocks the path.
The statement is true.

For the analytic solution:
$$\begin{align}
Pr(T,E|S)=Pr(T|S)*Pr(E|S)\\ \\
Pr(T,E,S) \\
0.9*0.02*0.99 \approx 0.0189 \\
Pr(T|S)=\frac{0.0289*0.02}{0.0289} \approx 0.02 \\ \\
Pr(E|S)=\sum_{Alarm}Pr(E|A)*Pr(A|S)= \\
Pr(A|S)=\frac{Pr(S|A)Pr(A)}{Pr(S)} \\
Pr(E|S)=0.88*0.5+0=0.44 \\ \\
Pr(T,E|S)=Pr(T)*\sum_{F,A}*Pr(A|T,F)*Pr(S|F)*Pr(E|A) \\
0.02*0.01*0.05*0.9*0.88 \approx 0.0000792 \\
0.02*0.01*...*0=0 \\
0.02*0.99*0.85*0.01*0.88\approx 0.000148 \\
0.02*0.99*...*0=0 \\
0.000148+0.0000792=0.000227 \\
\frac{0.000227}{0.0189}\approx 0.012 \\
Pr(T,E|S)=0.02 \not =Pr(T|S)*Pr(E|S)=0.0088 \\
\therefore independence fails
\end{align}
$$


**Tampering ⊥ Fire**

For the visual solution
Netica throws an error here as well:
![[Pasted image 20250524155050.png]]
The properties are D-separated, and form a common effect.
The blockers of their paths would be Alarm, Evacuation, Report, depending on what is observed.
The statement is true.

And for the analytic solution:
$$\begin{align}
Pr(T,F)=Pr(T)Pr(F) \\ \\
0.02*0.01 = 0.0002\\ \\
\text{no other depdencies}
\therefore \text{independence holds}
\end{align}
$$





**Tampering ⊥ Fire | Alarm**

For the visual solution
![[Pasted image 20250524162554.png]]
Netica allows Alarm to be of different values when Fire and Tampering are observed.
The variables are connected since both point to Alarm, making it a common effect.
There is no clear blocking.
The statement is false since they both depend on Alarm

And for the analytic solution:
$$\begin{align}
Pr(T,F|A)=Pr(T|A)*Pr(F|A)\\ \\
Pr(A|T,F)
0.02*0.01*0.5=0.0001 \\
\frac{0001}{0.0266}\approx 0.00376\\  \\
Pr(T|A) = 0.05*0.01+0.85*0.99=0.8465 \\
\frac{0.8465*0.02}{0.0266}\approx 0.636 \\ \\
Pr(F|A) = 0.5*0.02+0.99*0.98=0.9802 \\
\frac{0.9802*0.01}{0.0266}\approx 0.368 \\
0.368*0.636\approx 0.234 \\ \\
0.00376 \not = 0.234 \\
\therefore independence fails
\end{align}
$$


**Alarm ⊥ Smoke**

For the visual solution:
![[Pasted image 20250524162520.png]]

Netica allows the instantiation of the nodes for all combinations combined with Fire and Tampering.
The variables are connected, as they share a common cause which is Fire.
They are D-separated only if Fire is an observed event, otherwise no.
Assuming Fire is unobserved, then the statement is false.

For the analystic solution:
$$\begin{align}
 Pr(A|S)=Pr(A)*Pr(S)\\ \\
Pr(A,S)=\sum_{T,F}Pr(T)*Pr(F)*Pr(A|T,F)*Pr(S|F)= \\
0.02*0.01*0.5*0.9=0.00009 \\
0.02*0.99*0.85*0.01\approx0.000168 \\
0.98*0.01*0.99*0.9\approx 0.008732 \\
0.98*0.99*0*...=0 \\
0.00009+0.000168+0.008732+0=0.00899 \\ \\
Pr(S)=\sum_{F}Pr(S|F)*Pr(F)= \\
0.99*0.01*2=0.0198 \\
0.0266*0.0198\approx 0.00052668 \\
0.00052668 \not = 0.00899
\therefore \text{independence fails}
\end{align}
$$

**Smoke ⊥ Report**
![[Pasted image 20250524162455.png]]
Playing around in Netica allows us to observe a variety of scenarios once Smoke and Report are assigned.
The variables represent a causal chain since both depend on Fire.
There is no observed variable where blocking occurs.
The statement is false.


For the analytic solution (this requires 2^3 since we must marginalize over Fire, Alarm and Evacuate):
$$\begin{align}
 Pr(S,R)=Pr(S)*Pr(R)\\  \\
Pr(S,R)=\sum_{F,A,E}Pr(F)*Pr(S|F)*Pr(A|F)*Pr(E|A)*Pr(R|E)= \\
0.9802*0.01*0.9*0.88*0.75\approx 0.00582 \\
0.99*0.01*0.017*0.88*0.75\approx 0.000111 \\
0.000111+0.00582=0.00593 \\ \\
Pr(S)=\sum_{F}Pr(S|F)*Pr(F)= 0.0189\\
\sum_{E}Pr(E|A)*Pr(A)\approx 0.0234 \\
0.0234*0.75+0.1*0.9766\approx 0.0186 \\
0.0186*0.0189\approx 0.000351\\  \\
0.000351 \not = 0.00593
\therefore \text{independence fails}
\end{align}
$$
**Smoke ⊥ Tampering**

For the visual solution:
![[Pasted image 20250524162430.png]]
Netica again allows various instantiations of the other nodes when smoke and Tampering are observed
Similar to Tampering ⊥ Fire, the variables form a common effect.
the path is Blocked at Alarm and are D-separated at that junction.
This statement is true.

For the analytic solution:
$$\begin{align}
 Pr(S,T)=Pr(S)*Pr(T)\\ \\
Pr(S,T)=\sum_{F,A}Pr(T)*Pr(F)*Pr(A|T,F)*Pr(S|F)= \\
0.02*0.01*0.9=0.00018 \\
0.02*0.99*0.01=0.000198 \\
0.00018+0.000198=0.000378\\ \\
P(S)=\sum_{F}Pr(S|F)*Pr(F)= \\
0.9*0.01+0.01*0.99=0.0189 \\
0.0189*0.02=0.000378
\therefore \text{independence holds}
\end{align}
$$
**Smoke ⊥ Tampering | Alarm**
![[Pasted image 20250524162408.png]]

In Netica, with Smoke and Tampering observed, Alarm instantiation does not block off the instantiation of other nodes.
The two variables have a common effect to Alarm, thus activating the path, which are not D-separated.
The statement is blocked via Alarm.
The statement is false.

For the analytic solution:
$$\begin{align}
 Pr(S,T|A)=Pr(S|A)*Pr(T|A)\\ \\
Pr(S,T,A)=  \\
0.02*0.01*0.5*0.9=0.00009 \\
0.02*0.99*0.85*0.01\approx 0.000168 \\
0.000168+0.00009\approx 0.000258 \\
Pr(S,T|A)=\frac{0000258}{0.0266}\approx 00097\\ \\
Pr(S|A)=\sum_{T,F}Pr(T)*Pr(f)*Pr(A|T,F)*Pr(S|F) \\
0.02*0.01*0.5*0.9=0.00009 \\
0.02*0.99*0.85*0.01\approx 0.000168 \\
0.98*0.01*0.99*0.90\approx 0.008732 \\
0.98*0.99*...*0=0 \\
\frac{0.00899}{0.0266}\approx 0.338\\ \\
Pr(T|A)=\frac{(\sum_{F}Pr(A|T))*Pr(T)}{Pr(A)} \\
0.5*0.01+0.85*0.99\approx 0.8465 \\
\frac{0.8465*0.02}{0.266}\approx 0.636\\ \\
0.636 \not = 0.215
\therefore \text{independence fails}
\end{align}
$$
**Smoke ⊥ Tampering | Report**

For the visual solution:
![[Pasted image 20250524165916.png]]
I Netca, we can see that there is at least one combination of instantiations that throw the inconsistency error, but it doesn't seem to match data suggesting that the statement is false.
We know that the variables Smoke and Tampering have a common effect with Report as a descending node, and are explained away since Report is observed.
Both Tampering and Smoke influence each other with a collider at Alarm.
So it seems that this statement is false.

For the analytic solution:
$$\begin{align}
 Pr(S,T|R)=Pr(S|R)*Pr(T|R) \ => Pr(S,T,R)\\ \\
Pr(S,T,R)=\sum_{F,A,E}Pr(T)*Pr(F)*Pr(A|T,F)*Pr(S|F)*Pr(E|A)*Pr(R|E) \\
summarizing: 0.02*((0.01*0.05*0.9*0.88*0.75)+ \\
(0)+ \\
(0.99*0.85*0.01*0.88*0.75)+ \\
(0)) \\
\approx 0.000170 \\
Pr(S,T|R)=\frac{0.000170}{Pr(R)=0.0186}\approx 0.00914 \\
Pr(S|R)\approx 0.318: Marginalize \\
Pr(T|R)\approx 0.0638: Bayes \\
0.318*0.0638\approx 0.202\\ \\
0.202 \not = 0.00914
\therefore \text{independence fails}
\end{align}
$$

## d)

Let us assume the following instantiations and see if there are any conflicting commitments we must have about evacuating:
![[Pasted image 20250524172759.png]]

Since we know:
Smoke ⊥ Alarm
Smoke ⊥ Tampering
Tampering ⊥ Fire
Tampering ⊥ Evacuation | Smoke

and we also know that the Evacuate node is dependent on its parent nodes, one of them being Tampering. If we were to assume Smoke=false, Fire=false, but Tampering=true, it would still be rational to evacuate since the chances of the alarm going off increases. 

Intuitively, even if there is no fire, nor smoke, but the alarm goes off, it would be rational to still assume that the alarm is *not* indicating a false positive and has *not* been tampered with. These are factors that can be investigated at a time *after* evacuation.

Thus, I believe that the building should be evacuated.
# Part 2 

## a) Design a BN accordingly. Justify your design.

My design of the Bayesian network, based on the details of the brief, is as as follows:
![[Pasted image 20250524174044.png]]

And the CPT Table:

![[Pasted image 20250524174152.png]]

*"After a summer of backyard cricket, the lawn is in poor shape .. could also damage the*
*new lawn."*
	this tells me that the purpose of the network is not only to model the effects on Ron's lawn, but to see whether or not it will grow. Thus, a Grow variable was added.
	**Grow=true/false**
	This node will cause D-separation because condition 3 holds:
	"Neither Z nor any descendant of Z(Grow) is in E(Set of Nodes) and both path arrows lead to Z (common effect)"

*"Without rain before summer, it will*
*be very hard to grow new grass"*
	Here I decided on another variable called Watered and Rain since the lawn's health is conditioned upon it receiving water either from Ron himself (him watering the lawn), or from Rain.
	**Pr(Rain, Watered, Grow)**
	

*"... if there is no rain, the authorities could increase the level of water restrictions, meaning that Ron will be unable to water his lawn at all."*
	Evidence from the brief suggests that a variable Restriction should also be considered, but it is a child of rain.
	**Pr(Rain, Restrictions)**

*"there is a small chance that the*
*area could experience another frost before the weather warms up... "*
	I also considered a Frost variable since it specifies the kind of weather Ron will have. Using Weather is too broad, and using Summer is redundant since the brief states that Summer is guaranteed to be the next season.
	**Frost=true/false**

*"The area has been in drought for the previous 12 months."*
	I used this piece of information to come up with the chance that it will rain the next day. We know that there are 365 days in a year, and 30 days per month. Assuming we are considering Ron's lawn on Day D_1, I simply calculated, *certis paribus*, the chances of rain occurring the next day:
	12 months in a year with 1/12 chance of year having rain: 8.3%
	30 days in a month with 1/30 chance of month: 3.3%
	8.3 * 3.3 = 27.39, rounding down to keep a cleaner number, Pr(Rain=true) = 2.73, Pr(Rain=false) = 7.27
	I kept the restrictions at a 0.5 chance f being either true or false, since "*the authorities could increase the level of water restrictions*", does not give us sufficient justification to consider it more or less likely they will impose a restriction if there is no rain. So, Pr(Restrictions=true) = 0.5, Pr(Restrictions=false) = 0.5.
	For the values Pr(Frost=true) = 0.05, Pr(Frost=false) = 9.5, I decided to consult the Australian Bureau of Meteorology (ABM), which has a map of frost periods based on month. If we were to assume that *"Winter is coming to a close*, and the summer periods for Australia are typically December, January and February, and it is true that we do not know where exactly Ron lives, then we could infer that it might be the month of November for Ron at any location in the continent. Moreover, since it stated that, *"there is a small chance that the area could experience another frost ... "*,  then I figured it would fit with the following map from  ABM:
![[Pasted image 20250523125203.png]]
(Source: http://www.bom.gov.au/jsp/ncc/climate_averages/frost/index.jsp?period=nov&thold=lt2deg#maps)
### 1. There is no evidence.

If nothing has been observed, then the probability that the lawn will grow is a 0.63 
### 2. There is no rain, and water restrictions have been applied. Explain your results compared to the previous case.

We need to calculate,

$$
Pr(\lnot Rain \land \lnot Restrictions) = x
$$
We can first identify that Pr(-Rain)= 0.73 and Pr(-Restrictions)= 0.63, and Pr(Restrictions | Rain) = 0.5.

By simply calculating the probabilities of no rain and restrictions, we have Pr(-Rain, Restrictions) = 0.73* 0.5 = 0.365

Multiplying that to Pr(Grow), we have 0.23

### 3. There is frost, but it has rained. Explain your results compared to the previous 2 cases.

We need to find Pr(Frost, Rain | Grow). First, we marginalize the outcomes:

$$
Pr(Grow=true)=\sum_{Rain, Restrictions, Watered, Frost}
$$
and expand the joint probabilities by calculating combinations:
$$\begin{align}
Pr(Rain)*Pr(Restriction|Rain)*Pr(Watered|Restrictions) =>\\
Pr(Grow=true)=1*0.05+1*0.95=1 
\end{align}
$$
Taking into account that Pr(Rain,Frost):
$$\begin{align}
0.27*1*1=0.27 \\
0+0.73*0.5*1=0.365 \\
0.27+0.365=0.635 \\
\therefore Pr(Grow=true)=0.635
\end{align}
$$

# Part 3

In this section, I will use the following abbreviations:

Interest_Rate: I
Housing_Prices: H
Property_Location: P
Tenant: T
Rent_charged: R
Desirable_Investment: D

Here is the CPT Table for the network:

![[Pasted image 20250523125734.png]]

## a)


The goal here is to figure out under what conditions would observing something about the interest_rate change the belief about rent_charged.

To do this, we first consider the case:
$$
InterestRate \not \perp RentCharged | e
$$
where the path is active (not D-separated).

prob of event => P(t)= .5/1

intersection of events => i is high, t is yes or no => p(i & t) = 0.5/1

conditional pr => p(i|t) = p(i&t)/p(t) = 0.5

Let us assume that
$$
Pr(I=high) = 0.5
$$
We can find the intersection of additional events and calculate the conditional probabilities using housing prices:

![[Pasted image 20250518164107.png]]
$$
Pr(H=high|I=high) = \frac {P(H,I)*Pr(0.5)}{Pr(0.5|H=high,low)} = 
$$


![[Pasted image 20250518164753.png]]
Using Netica, we can confirm that having observed interest rates to be high, it only has a *small* impact on the housing prices, with high being 49.0 and low being 51.0. This means that there is likely another factor that is involved in marginalizing over this part of the CPT: the property location.

Suppose we set the property location to good:

![[Pasted image 20250518164949.png]]

This leads to higher housing prices and increases the chances of a tenant being present. Here is where we have a causal chain to an extent.

So, we can conclude that Property_Location is a better node to instantiate since the change in Housing Prices is low.

But suppose we set Interest_Rate to low?
![[Pasted image 20250518170149.png]]

We can see a much more significant shift in the housing price, where high=79. So, it would be better to have Housing_Prices as a node to instantiate.

In the same fashion, since we know that Tenant is an effect of Rent_Charged, we can say that the node Tenant is certainly one that must be instantiated. Especially since in either observation of Rent_charged, there is a significant change.
![[Pasted image 20250518170439.png]]
![[Pasted image 20250518170531.png]]
If rent is high, then the chances of there being a tenant is 67.5, whereas if the rent is low, then the chance 86.
## b) 

This time we want to find a backwards propagation:

$$
DesirableInvestments \not \perp HousingPrices | evidence
$$

To do this, we would have to assume the path is not blocked, which would mean that Tenant is not observed since it is the next step in the path to Housing_Prices.

However, if we were to observe Tenant, then we come across something interesting:

![[Pasted image 20250518171756.png]]
![[Pasted image 20250518171808.png]]

This means that Tenant would not be the final node towards the path--we are still left with explaining the effect. 

When considering an observation on Property_Location *and* Interest_Rate, it is then that we have a stronger link to Housing_Prices:

![[Pasted image 20250518172549.png]]

with Rent_Charged not having any effect:

![[Pasted image 20250518172734.png]]

But this can lead to seemingly contradictory results. Suppose the investment in a set of housing properties was enacted, there were no tenants, and the property location is fair, then it seems that housing prices would typically be lower than higher. However, we can see below that such is not the case:

![[Pasted image 20250518174815.png]]

Based on the evidence we gathered from a), we can conclude that to make any headway that makes sense, the nodes we may want to instantiate are Property_Location and Interest_Rate, with reservation.

## c) 



Concerning **arcs** (linking the nodes where we formally encode conditional independence) in this model, we can take a Markovian approach, i.e., assume the Markov property:

$$\begin{matrix}
\text{For some Process P, P is predicated by the markov property iff}  \\
\text{P is stochastically discrete and memoryless wihin a finite state space}
\end{matrix}
$$

paraphrased on the context of our case, the Baysian Network node connections would assume away the past and future events within a finite series of nodes (without this constraint, we could add n-nodes to justify anything in the network.)

Assuming Housing_Prices -> Tenant:

![[Pasted image 20250518174105.png]]

This makes propagating from Desirable_Investment to Housing_Prices much more simple. No node except Desirable_Investment would need to be instantiated in order to justify the backwards propagation, and if we were to instantiate Tenant, then we would run into similar problems as the last question. 