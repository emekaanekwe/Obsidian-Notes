# Model-Free Control

	aim is to have a model free policy through which the agent learns
	
## Greedy Q Policy

	this allows for state -> value mapping

$$
\pi '(s)=\underset{x \in X}{\mathrm{argmax}}Q(s,a)
$$
So if we want our greedy policy, we maximize our Q values Q(s,a):= "for each state, how good is it to take each different action?"
	this somewhat "caches" the vals of actions

![[Pasted image 20250525110420.png]]

By getting to the top dot, we run epsodes using a <mark class='red'>monte carlo/q-something policy evaluation</mark> -> got estimates of the policy -> take the mean value of each state action pair and choose the best one

**But this poses a problem since it's possible that the better option is not chosen**

to fix this, we consider ***epsilon greedy*** for continual exploration.
	epsilon greedy := improves main policy

	for all actions m are tried with a Pr value
	if 1-epsilon -> choose greedy action
	if epsilon -> make a random decision

$$
\pi (s,a)=\begin{cases}\epsilon/m+1-\epsilon, \ \ \ if \ a^*=\underset{a in A}argmaxQ(s,a) \\ \epsilon/m, \ \ \ \ otherwise\end{cases}
$$
like flipping a coin where H=epsilon, T=1-epsilon
	if heads, then chose a random action
	if tails, maximize

This **guarantees greedy within the limits of infinite exploring**

### epsilon greedy proof
![[Pasted image 20250525112653.png]]

expectation: epsilon policy (probability that action taken)* (value of the action) 

### On Increasing Efficiency

	having the mean value of return updated only for the visited states -> get best policy possible

Episodes, i.e., *entire sequence of going from start to goal*:
$$
\text{for episode }e\in E, \text{entails a sequence of updates for policy }\pi
$$

Greedy Exploration:

$$
\lim_{k-> \infty} N_k(s,a)= \infty
$$
so we can converge on:
$$
\lim_{k-> \infty} \pi _k(a|s)=1(a=\underset{a' \in A}{\mathrm{argmax}}Q(s',a'))
$$
which is needed to *satisfy the bellman equation*
$$
Q(S, A) ← R + γ max Q(S′, a′)
$$

which implies the expectation equation:
$$
qπ(s, a) = [Rt+1 + γqπ(st+1, at+1) ∣ St = s, At = a]
$$

with the goal of Bellman optimality via decomposition:

![[Pasted image 20250525143552.png]]

Temporal Difference (Improved from Monte Carlo)Algorithm Example:
![[Pasted image 20250525143934.png]]

Q-Learningfor off-policy:
![[Pasted image 20250525144057.png]]

## Policy Updating

![[Pasted image 20250526131758.png]]

	For each state and action in each episode, the policy gets updated.

policy is then improved based on the *new* action-value function.

**GLIE** - the policy *eventually* becomes greedy *only if* sufficient exploration

**Off-Policy Learning = Q-Learning**
learn the optimal Q function *independent* on the behavior policy

**Updating**
uses max(Q(s,a))

**GLIE Q-Learning advantages**
Outperforms standard Q-learning:

- **Delay policy updates** (e.g., update Q 1000x before syncing to policy)
    
- **Use optimistic initial Q-values** to force early exploration
    
- **Dynamic ε decay**: Faster/slower based on state-space coverage


### Algorithmic Approach

**Policy Evaluation Step**
1. samples current policy by
	1. an episode (agent goes north)
	2. generates data (state, action, reward)

**Policy Improvement Step**
note: in code, pi isn't stored, instead Q
This is where the Pr Distro of e-greedy comes in
1. update action val by *counting how many times k(s,a) pair has been seen*
2. get the mean of the sum of updates
3. 
***what do q values look like? could there be more optimal ones in gridworld?***

	on to control (optimization)

**Temporal difference learning**
benefits: lower variance, online, sequences incomplete

1. this takes the role as the policy evaluator for Q(s,a)
	1. also called *SARSA* (SARSA update):
$$
Q(S,A) = Q(S,A)+\alpha(Reward+\gamma Q(S',A')-Q(S,A))
$$
	![[Pasted image 20250526175152.png|200]]
(S,A) - call the pairwise function
| R - sample environment to analyze reward
(S') - this leads us to new state
| - Sample estimate of on-policy (policy used to make the decision)
(A') - next action

**SARSA Algorithm for On-Policy Control**
![[Pasted image 20250525143934.png]]

### SARSA & Convergence

	key is the ensure the step size is within expected parameters

Theorem: Convergence if

$$\begin{align}
\text{assume GLIE }\pi_t(a|s) \\
Q(s,a) -> q_*(s,a) \ iff: \\ \\
\sum^{\infty}_{t=1}\alpha_t=\infty\ \ and \ \sum^{\infty}_{t=1}\alpha^{2}_t <\infty
\end{align}
$$

### Step-by-Step SARSA

![[Pasted image 20250527102004.png]]

Note that the discount modifier to the reward comes after the first step. 

- **n-step SARSA**: wait for **n steps**, then update using the sum of **n rewards** and the Q-value from the state at step t+nt + nt+n.
    

So:

- In 1-step SARSA, you update _right away_ (like a reflex).
    
- In n-step SARSA, you gather experience for n steps before updating.
    

It’s still happening **within** an episode — the episode might last 100 steps, and you’re doing many n-step updates inside that.

#### Q-Return
$$
G^{(n)}_t​=R_{t+1}​+γR_{t+2}​+⋯+γ_{n−1}R_{t+n}​+γ^nQ(s_{t+n}​,a_{t+n}​)
$$
So the **Q-return** is the **bootstrapped estimate** of future rewards — what we expect to earn starting from now.

It happens **within episodes**, during updates. It's not just the return of the whole episode; it's computed at each learning step using nnn-step rewards + a Q-bootstrapped tail.

### SARSA-Lambda Algorithm

SARSA(λ) is a **bridge between 1-step SARSA and Monte Carlo methods**, using what's called **eligibility traces**.

It's like saying:

> "Let’s not update only the last action we took — let’s also give credit to recent past actions, decaying their influence the further back in time they are."

This decay is controlled by **λ ∈ [0, 1]**:

- λ = 0 → behaves like **1-step SARSA** (only most recent step counts)
    
- λ = 1 → behaves like **Monte Carlo** (entire episode return counts)

This will result in *exponential decay*:
$$
G​=(1−λ)\sum_{n=1}^{∞}​λ^{n−1}G_{t}^{(n)}​
$$
- Each Gt(n)G^{(n) is the **n-step return** starting from time t
    
- You weight them with λn−1\lambda^{n-1}λn−1 and combine them
    
- This gives you a **weighted average of all future n-step returns**
    
This tells us: rather than committing to just 1-step or 5-step or full-episode return, **we blend all of them** — shorter returns get higher weight.

**Q-Values ARE what's bootstrapped** appear *inside every n-step return*
Example: 
$$\begin{matrix}
\text{10-step return} \\
G_{t}^{(10)}​=r_{t+1}​+γr_{t+2​}+⋯+γ^9r_{t+10​}+γ^{10}Q(s_{t+10}​,a_{t+10​})\\ \\
note: \ Q(s_{t+10}​,a_{t+10​})=E[r_{t+11}+\gamma r_{t+12}+..] \\
\text{multiply by }\gamma ^{10}: \\
\ \gamma ^{10}Q(s_{t+10}​,a_{t+10​})=\gamma ^{10}(r_{t+11}+\gamma r_{t+12}+..)= \\
(\gamma ^{10}r_{t+11}+\gamma ^{11}r_{t+12}+..)
\end{matrix}
$$

The final Q-term is really just a shorthand for *all rewards starting after the 10th step, weighted by discounting*. It is the *sum of all future expected rewards after n-steps* 
## Sampling

**Sampling**: collecting actual transitions (state, action, reward, next state, next action) by interacting with the environment.

**Policy Evaluation**: estimating how good a policy is (usually via value functions), *often using those samples*.

## Training

*Within* an episode *after every step/action*, **Q-Learning**
$$
Q(s,a)←Q(s,a)+α[r+γ\underset{a'}{max}​Q(s',a')−Q(s,a)]
$$


# Q-Tables
## Where Q-Tables Are Implicitly Covered

#### **Lecture 4 – Model-Free Prediction**

- Introduces **Monte Carlo (MC)** and **TD learning** methods to estimate **state values**.
    
- At this point, the focus is still on V(s)V(s)V(s), not Q(s,a)Q(s, a)Q(s,a).
    

#### **Lecture 5 – Model-Free Control**

- This is the key lecture where the concept of **Q-values Q(s,a)Q(s,a)Q(s,a)** begins.
    
- Silver explains:
    
    - How to estimate Q(s,a)Q(s,a)Q(s,a) using **MC** and **TD methods**.
        
    - **SARSA** and **Q-Learning** algorithms, both of which use Q-values directly.
        

##### → Around **26:30 onward** in Lecture 5:

> “We want to learn action values… We maintain estimates Q(s,a)Q(s,a)Q(s,a) for each state-action pair…”

This is where Q-tables are conceptually introduced. In **tabular methods**, this implies having a **Q-table** with one entry per (s,a)(s,a)(s,a) pair. While Silver doesn't explicitly say "Q-table", it's exactly what he's referring to when he talks about “storing the Q-values” for each pair.
## So what is a Q-table?

In practical terms:

- A **Q-table** is just a matrix or dictionary storing the value of Q(s,a)Q(s,a)Q(s,a) for each (s,a)(s,a)(s,a) pair.
    
- It’s used in **tabular Q-learning**, where the environment has **discrete** state and action spaces.


# Set up

gridworld

| 1   | 2   | 3   |
| --- | --- | --- |
| 4   | 5   | 6   |
| 7   | 8   | 9   |

transition Pr

| 0.3 | 0.5 | 0.2 |
| --- | --- | --- |
| 0.8 | 0.1 | 0.1 |
| 0.6 | 0.2 | 0.2 |

q-table
[0. 0. 0. 0.

| 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 |
| 0 | 0 | 0 | 0 |
