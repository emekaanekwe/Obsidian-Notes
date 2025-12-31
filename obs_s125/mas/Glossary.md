## Terms

| Category                                     | Term                                         | Definition                                                                  | Examples              | Use Cases               | Problems                                | Fixes |     |     |
| -------------------------------------------- | -------------------------------------------- | --------------------------------------------------------------------------- | --------------------- | ----------------------- | --------------------------------------- | ----- | --- | --- |
| Agents, Analysis                             | analytic solutions                           | mathematical representations of agents and relations                        |                       |                         | math becomes too complicated, can break |       |     |     |
|                                              | agent-based simulations                      | computational contructs that predict events with high levels of flexibility | ODE, LLMs             | Ant Colony Optimization | heavily parameter-dependent             |       |     |     |
|                                              | numeric solutions                            | a combining f analytic and simulation, one of the roots of MAS              |                       |                         |                                         |       |     |     |
| Macroscopic Models, Deterministic-Stochastic |                                              |                                                                             |                       |                         |                                         |       |     |     |
| NOTE: We are concerned with *discrete cases* | SIR models                                   | capture rates of infection                                                  | (See Notes for Model) |                         |                                         |       |     |     |
|                                              |                                              |                                                                             |                       |                         |                                         |       |     |     |
| Markovian Analysis                           | Markov Assumption                            |                                                                             |                       |                         |                                         |       |     |     |
|                                              | Markov property                              |                                                                             |                       |                         |                                         |       |     |     |
|                                              | state sequence                               |                                                                             |                       |                         |                                         |       |     |     |
|                                              | transition matrix                            |                                                                             |                       |                         |                                         |       |     |     |
|                                              | role of rows                                 |                                                                             |                       |                         |                                         |       |     |     |
|                                              | role of columns                              |                                                                             |                       |                         |                                         |       |     |     |
|                                              | calculating a (stochastic) transition matrix |                                                                             |                       |                         |                                         |       |     |     |
|                                              |                                              |                                                                             |                       |                         |                                         |       |     |     |
|                                              |                                              |                                                                             |                       |                         |                                         |       |     |     |
|                                              |                                              |                                                                             |                       |                         |                                         |       |     |     |
|                                              |                                              |                                                                             |                       |                         |                                         |       |     |     |
|                                              |                                              |                                                                             |                       |                         |                                         |       |     |     |
|                                              |                                              |                                                                             |                       |                         |                                         |       |     |     |
|                                              |                                              |                                                                             |                       |                         |                                         |       |     |     |
|                                              |                                              |                                                                             |                       |                         |                                         |       |     |     |
|                                              |                                              |                                                                             |                       |                         |                                         |       |     |     |
|                                              |                                              |                                                                             |                       |                         |                                         |       |     |     |
|                                              |                                              |                                                                             |                       |                         |                                         |       |     |     |
|                                              |                                              |                                                                             |                       |                         |                                         |       |     |     |
| wk3?                                         | non-linear function                          |                                                                             |                       |                         |                                         |       |     |     |
|                                              | monotone                                     |                                                                             |                       |                         |                                         |       |     |     |
|                                              | ODE system                                   |                                                                             |                       |                         |                                         |       |     |     |
|                                              | initial conditions (models vs code)          |                                                                             |                       |                         |                                         |       |     |     |
|                                              |                                              |                                                                             |                       |                         |                                         |       |     |     |
|                                              |                                              |                                                                             |                       |                         |                                         |       |     |     |
|                                              |                                              |                                                                             |                       |                         |                                         |       |     |     |
|                                              |                                              |                                                                             |                       |                         |                                         |       |     |     |


[REPRESENT AS MUCH AS POSSIBLE IN CODE]
# Markov Chains

# Markov Reward Process

# Markov Decision Process

# Bellman Expectation

# Bellamn Optimization

# Monte Carlo Policies

# Time Difference-Updates

# Epsilon-Greedy
# Greedy within the Limit of Infinite Exploration

# SARSA

# Reinforcement Learning

# Q Values
# On-Policy

# Off-Policy

# Q-Learning

# Deep Q-Learning (ignore)

# Game Theory
## use to provide an "interpretation" of how the agents are behaving

# Pareto Dominance

# Pareto Optimality

# Nash Equlibrium

# Mixed Nash

# Correlated Equilibrium

---

[STRUCTURE OF CODE]

# Architecture
## initialization
	gridworld
	agent positions
	configuration 
	state space 

# Agents
## Properties
	policy
	move options

# Learning
	RL, Q(S,A)

# Decision Making (Game Theoretic)
## Strategies
	paret optimality
	nash equilibria
	

## Math

---

### 1. **Matrix Transposition**

Matrix transposition is often used to handle state transitions and adjacency matrices in MAS:
$$
(A^T)_{ij}=A_{ji}, \text{where A is an mxn matrix}
$$



$$
\begin{align}
x_i=\frac{1}{\lambda}\sum_{j} A_ijx_j \\
 
\text{where A is the adjacency matrix, } 
\lambda \text{ the largest eignevalue, and x the centrality score}
\end{align} 
$$



---

### 2. **Markov Chains (State Transition)**

Markov chains model the evolution of a system of agents over discrete time steps. The state transition matrix PP defines the probability of moving from one state to another:

P(Xt+1=j∣Xt=i)=PijP(X_{t+1} = j \mid X_t = i) = P_{ij}

where PP is a **stochastic matrix** (rows sum to 1).

The steady-state distribution π\pi (if it exists) satisfies:

πP=πand∑iπi=1\pi P = \pi \quad \text{and} \quad \sum_i \pi_i = 1

---

### 3. **Expected Utility in Game Theory**

Agents often aim to maximize expected utility in strategic environments:

U(a)=∑s∈SP(s∣a)R(s)U(a) = \sum_{s \in S} P(s \mid a) R(s)

where:

- U(a)U(a) = expected utility of action aa
    
- P(s∣a)P(s \mid a) = probability of state ss given action aa
    
- R(s)R(s) = reward for being in state ss
    

---

### 4. **Nash Equilibrium**

In a two-player game, a Nash equilibrium occurs when no player has an incentive to unilaterally change their strategy:

ui(si∗,s−i∗)≥ui(si,s−i∗)u_i(s_i^*, s_{-i}^*) \geq u_i(s_i, s_{-i}^*)

for all strategies sis_i of player ii and s−is_{-i} of the other player.

---

### 5. **Bellman Equation (Reinforcement Learning)**

Used to compute the optimal value function in dynamic programming and multi-agent reinforcement learning:

V(s)=max⁡a[R(s,a)+γ∑s′P(s′∣s,a)V(s′)]V(s) = \max_a \left[ R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V(s') \right]

where:

- V(s)V(s) = value of state ss
    
- R(s,a)R(s, a) = reward from taking action aa in state ss
    
- γ\gamma = discount factor
    
- P(s′∣s,a)P(s' \mid s, a) = transition probability
    

---

### 6. **Flocking Behavior (Boid Model)**

Flocking in MAS is often modeled using vector-based alignment, separation, and cohesion:

v⃗i(t+1)=v⃗i(t)+kaa⃗i+kss⃗i+kcc⃗i\vec{v}_i(t + 1) = \vec{v}_i(t) + k_a \vec{a}_i + k_s \vec{s}_i + k_c \vec{c}_i

where:

- v⃗i\vec{v}_i = velocity of agent ii
    
- a⃗i\vec{a}_i = alignment term
    
- s⃗i\vec{s}_i = separation term
    
- c⃗i\vec{c}_i = cohesion term
    

---

### 7. **Social Influence Model (DeGroot Model)**

Models how agents’ opinions evolve over time:

xi(t+1)=∑j∈Niwijxj(t)x_i(t + 1) = \sum_{j \in N_i} w_{ij} x_j(t)

where:

- xi(t)x_i(t) = opinion of agent ii at time tt
    
- wijw_{ij} = weight of agent jj’s influence on agent ii
    
- NiN_i = neighborhood of agent ii
    

---

### 8. **Eigenvector Centrality**

Used to measure the importance of an agent in a network:

xi=1λ∑jAijxjx_i = \frac{1}{\lambda} \sum_j A_{ij} x_j

where:

- AA = adjacency matrix of the network
    
- λ\lambda = largest eigenvalue
    
- xx = eigenvector representing the centrality score
    

---

### 9. **Gradient-Based Learning in Multi-Agent Reinforcement Learning**

Policy gradients for learning agent policies:

∇θJ(θ)=E[∇θlog⁡πθ(a∣s)Q(s,a)]\nabla_\theta J(\theta) = \mathbb{E} \left[ \nabla_\theta \log \pi_\theta(a \mid s) Q(s, a) \right]

where:

- πθ\pi_\theta = agent’s policy
    
- Q(s,a)Q(s, a) = action-value function
    

---

### 10. **Consensus Algorithm**

Consensus algorithms ensure that agents reach agreement on a shared state:

xi(t+1)=xi(t)+∑j∈Nicij(xj(t)−xi(t))x_i(t+1) = x_i(t) + \sum_{j \in N_i} c_{ij}(x_j(t) - x_i(t))

where:

- cijc_{ij} = consensus weight between agents ii and jj
    

---

These should cover most of the math you'll encounter in multi-agent systems. Let me know if you’d like to dive deeper into any of these! 😎