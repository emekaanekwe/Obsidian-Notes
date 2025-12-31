# Slides


# Vocab

### Pure Strat
	predetermined strat regardless of other players
### Mixed Strat
	Pr distro over a set of pure strats
### Shapely Values
	.
### Normal Form
	a game that can take the form of a matrix that has ALL player's pure strats
### Bi-Matrix
	a strictly two player with the player's choices represented as sparated matrices
### Symmetric Games
	all players ahve the same options









| #     | Page | Topic      | Sub-Topic |
| ----- | ---- | ---------- | --------- |
| wk2.2 | 8-17 | mark chain |           |
|       |      |            |           |
|       |      |            |           |
# wk 6
![[Pasted image 20250626175039.png]]

# wk 7
![[Pasted image 20250626175015.png]]

# Glossary
1. Markov Chains
2. Markov Reward Process
3. Markov Decision Process
4. Bellman Expectation
5. Bellamn Optimization
6. Monte Carlo Policies
7. Time Difference-Updates
8. Epsilon-Greedy
9. Greedy within the Limit of Infinite Exploration
10. SARSA
11. Reinforcement Learning
12. Q Values
13. On-Policy
14. Off-Policy
15. Q-Learning


# State Transitions

## ![[Pasted image 20250624170356.png]]

## Solution

Given:
M1​=[21​41​​21​43​​]

$$\begin{matrix}
M1​=\matrix{\frac{1}{2} \frac{1}{2} \\
​​\frac{1}{4} \frac{3}{4}}, \ \ \ M2​=\matrix{X \ Y \\
​​\frac{5}{16} \frac{11}{16}}
\end{matrix}
$$
We compute M2=M12M2​=M12​:

$$
M2=M1×M1=[(12×12)+(12×14)(12×12)+(12×34)(14×12)+(34×14)(14×12)+(34×34)]M2​=M1​×M1​=[(21​×21​)+(21​×41​)(41​×21​)+(43​×41​)​(21​×21​)+(21​×43​)(41​×21​)+(43​×43​)​]
$$
Now, let's compute each entry:

1. **Top-left entry (X)**:
    
$$
X=(12×12)+(12×14)=14+18=38X=(21​×21​)+(21​×41​)=41​+81​=83​
$$
2. **Top-right entry (Y)**:
    
$$
Y=(12×12)+(12×34)=14+38=58Y=(21​×21​)+(21​×43​)=41​+83​=85​
$$
3. **Bottom-left entry (given as 516165​)**:
    

(14×12)+(34×14)=18+316=516(matches given)(41​×21​)+(43​×41​)=81​+163​=165​(matches given)

4. **Bottom-right entry (given as 11161611​, not 116611​)**:
    

(14×12)+(34×34)=18+916=1116(41​×21​)+(43​×43​)=81​+169​=1611​

_(Note: 116611​ is likely a typo—probabilities must sum to 1 in rows.)_

### **Final Answer**

The correct values are:

X=38,Y=58X=83​,Y=85​






## Two-state Markov Chain


# Classic Non-cooperative Game Theory


# Myopic Best Response
## Normal Form Games
![[Pasted image 20250625163734.png]]


# Population Games

## Evolutionary Stable Strategies
![[Pasted image 20250625164021.png]]
## first iteration of the replicator evolution
check nash article

## Pure Nash in ESS
## state the Nash equilibrium point/rest point of this replicator equation

# Coalition

## Core and Shapley Vals

## Social Choice and Welfare Functions
![[Pasted image 20250625165011.png]]

## compute the top preference (the winner) of the complete population of agents

## the Condorcet condition

## the Borda count condition

## Coalition Game characteristic function



# Reinforcement Learning

![[Pasted image 20250624171329.png|500]]
![[Pasted image 20250624171409.png|500]]


