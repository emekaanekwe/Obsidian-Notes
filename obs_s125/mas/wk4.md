**Sutton, chapter 6.5**
**Shoham, chapter 7.4**

- We develop Q-Learning, the foundation of most reinforcement learning algorithms. 
- Q-Learning will form the foundation of the first part of your project.
- 
## Review

### Monte Carlo Methods
**Key features**:
1. uses only experience (sample sequences of states, actions, and rewards with an environment) 
2. solves the problem of average sample returns by **getting the expected value of the returns** 
	1. treats **each episodic state like a bandit problem**
		1. bandit problem: 
			1. a bandit is a single state MDP 
			2. multiple agents sharing the same resource where each are trying to maximize gains
			3. formally as **Expected Values**:  $$
			 q_*(a)=E[R_t|A_t=a]
			$$
			with the action at t with its corresponding reward R
			4. REMEMBER reward does not consider the actions, just the Pr
3. 
4. 

## From model-based prediction to model-free

---
---
# Lab

## Review
markov chain: {S, P}
S: finite states
P: matrix P[S_t+1 = s' | S_t = s']

Rewards (MRP) 

**Be aware that your transition matrix will affect what your reward vector will be, so be careful**

Regarding bellman, we will be suing *closed form* 




## 1
$$
S_n = {\text{home, city, market, shelter, dead}}
$$
* see python code
## 2
$$
\begin{pmatrix}
H\\
C\\
M \\
S \\
D
\end{pmatrix} = \begin{pmatrix}
0 \ 0 \ 0 \ 0 \ 0 \\
0 \ 0 \ 0 \ 0 \ 0 \\

\end{pmatrix}
$$
## 6.1 
How to capture fact that cat cannot actively influence situation:
	transition matrix of shelter state non-linear (independent) 
		so policy has not effect, just random

note: that there a equal states given the Pr of the S_t

## 6.2 
death as absorbing state. so Pr(D -> D) = 1`
	solution attemept: more negative rewards to influence agent

## 7

consider the expectation equation (from slide 22):
	qπ(s, a) = [Rt+1 + γqπ(st+1, at+1) ∣ St = s, At = a]

so the matrix would expand from 5x5 to 2x5x5.

$$
\begin{pmatrix}

\end{pmatrix} = \begin{pmatrix}
0 \ 0 \ 0 \ 0 \ 0 \\
0 \ 0 \ 0 \ 0 \ 0 \\
\end{pmatrix} multipled--current--state--action(run) \begin{pmatrix}

\end{pmatrix}
$$
MDP is like a modeling problem

**very important is how you reward the agent!!**

Question 9 is a high level model

Question 10 is a lower level

VALUE ITERATION
suppose V_0 = [1,2,3,4,5] where V_0 is random/arbitrary
	we calculate 
	
	Here is where you would (in a loop) use a stack DS in programming, and NOT a counter 

# bellman expect eq

$$
V=R+\gamma PV \text{ with matrix form: } V=(I-\gamma P)^{-1} R
$$

# V = R+gammaPV

# matrix: V = (I-gammeP)^-1R

