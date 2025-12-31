

---

# 1)

### If Messi has the ball (the system is in the Messi state), what action should we choose to maximize our reward in the next state: pass or shoot? Assume we are using the values for U after iteration 2.

Given,
$(U_2): messi = −2, \ suarez = −0.6, \ scored = 2.$

pass: $( \mathbb{E}[U_2] = U_2(\text{suarez}) = -0.6 )$
    
shoot: $( \mathbb{E}[U_2] = 0.2\cdot U_2(\text{scored}) + 0.8\cdot U_2(\text{suarez}) = 0.2\cdot 2 + 0.8\cdot(-0.6) = 0.4 - 0.48 = -0.08 )$
    
Since $(-0.08 > -0.6)$, we ought to shoot in order to maximize our reward.


---

# 2) 
### Complete the values of these states for iteration 3 using value iteration. Show your work.

With Bellman optimality:  
$(U_{3}(s) = R(s) + \max_a \mathbb{E}[U_2(s')\mid s,a]).$

**messi**: we use the max we just computed (-0.08) (shoot).  
$(U_3(\text{messi}) = -1 + (-0.08) = \mathbf{-1.08})$
    
**suarez**:  
    pass: $( \mathbb{E}[U_2] = U_2(\text{messi}) = -2 )$
    shoot: $(0.6\cdot 2 + 0.4\cdot(-2) = 1.2 - 0.8 = 0.4) → \text{take the shot}.$  
$(U_3(\text{suarez}) = -2 + 0.4 = \mathbf{-1.6}).$
    
**scored**: since only transition is $return → messi (1.0)$:  
$(U_3(\text{scored}) = 3 + U_2(\text{messi}) = 3 + (-2) = \mathbf{1}).$
    

We can say that the Iteration 3 row contains:  
messi **−1.08**, suarez **−1.6**, scored **1**.

---

# 3) 
### Consider the following initial policy table, with discount factor γ = 0.8:

| iteration | $\pi(messi)$ | $\pi(suarez)$ | $\pi(scored)$ |
| --------- | ------------ | ------------- | ------------- |
| 0         | pass         | pass          | return        |
| 1         |              |               | return        |
| 2         |              |               | return        |

We evaluate **under this fixed policy**, starting from (V_0\equiv 0).
    

Policy transitions:  
messi → suarez; suarez → messi; scored → messi.

Borrowing Bellman again: $(V_{k+1}(s) = R(s) + \gamma \sum_{s'} P_\pi(s'|s)V_k(s')).$

**Iteration 1** from (V_0=0):
    
$$(V_1(\text{messi}) = -1 + 0.8\cdot V_0(\text{suarez}) = -1)$$
        
$$(V_1(\text{suarez}) = -2 + 0.8\cdot V_0(\text{messi}) = -2)$$
        
$$(V_1(\text{scored}) = 3 + 0.8\cdot V_0(\text{messi}) = 3)$$
        
**Iteration 2**:
    
$$(V_2(\text{messi}) = -1 + 0.8\cdot(-2) = \mathbf{-2.6})$$
        
$$(V_2(\text{suarez}) = -2 + 0.8\cdot(-1) = \mathbf{-2.8})$$
        
$$(V_2(\text{scored}) = 3 + 0.8\cdot(-1) = \mathbf{2.2})$$
        

---

# 4) 
### What is the difference between Sarsa and Q-Learning?

- **SARSA** (on-policy):  
$$(Q(s,a) \leftarrow Q(s,a) + \alpha\Big[r + \gamma,Q(s',a') - Q(s,a)\Big])  $$
    Uses the **action actually taken next** under the behavior policy (e.g., (\epsilon)-greedy).
    
- **Q-learning** (off-policy):  
$$(Q(s,a) \leftarrow Q(s,a) + \alpha\Big[r + \gamma,\max_{a'} Q(s',a') - Q(s,a)\Big])  $$
    Uses the **greedy target** $(\max_{a'})$, independent of the next action executed.
    

Result: SARSA is typically **more conservative** in stochastic/risky settings; Q-learning **learns the greedy policy** even while exploring.

---

# 5) 
### Assume the following Q-table, which is learned after several episodes:

| state  | pass | shoot | return |
| ------ | ---- | ----- | ------ |
| messi  | -0.4 | -0.8  |        |
| suarez | -0.7 | -0.2  |        |
| scored |      |       | 1.2    |
### In the next step of the episode, from the state ‘Suarez’, Suarez passes the ball to Messi. Show the Q-learning update for this action using a discount factor γ = 0.9 and learning rate α = 0.4.

We need an immediate reward (r). In model-free problems this comes from the environment; and since it’s not specified, the **standard assumption for non-terminal “pass”** is (r=0) (don't score). We compute with (r=0) and note how to swap (r)

- Target for **Q-learning**:  
$\text{target} = r + \gamma \max_{a'} Q(\text{messi},a')$ 
$= 0 + 0.9 \cdot \max(-0.4,-0.8)$  
$= 0.9 \cdot (-0.4) = -0.36.$  

    
- Update:  
$Q_{\text{new}}(\text{suarez},\text{pass})$     
$= -0.7 + 0.4 \big( -0.36 -(-0.7)\big)$ 
$= -0.7 + 0.4\cdot 0.34      = \mathbf{-0.564}.$  
    

---

# 6) 
### Consider again being in the state ‘Suarez’, Suarez passes the ball to Messi and then Messi decides to shoot. Show the SARSA update for the Pass action using a discount factor γ = 0.9 and learning rate α = 0.4 and assuming a’ (the next action to be executed) is Shoot. Compare to the Q-learning update. What is different?

Same transition (suarez, pass → messi), parameters $(\gamma=0.9,; \alpha=0.4)$, and again assume (r=0).

- **SARSA** target uses the **actual next action** $(a'=\text{shoot})$:  
$\text{target} = r + \gamma,Q(\text{messi},\text{shoot})$
$= 0 + 0.9\cdot(-0.8)$  
$= -0.72.$  
    
    
- Update:  
    
$Q_{\text{new}}(\text{suarez},\text{pass})$      
$= -0.7 + 0.4 \big( -0.72 - (-0.7) \big)$  
$= -0.7 + 0.4\cdot (-0.02)$      
$= \mathbf{-0.708}.$

**Comparison:**

- **Q-learning** moved $(Q(\text{suarez},\text{pass}))$ **up** to −0.564 (it looked ahead to the **best** action at messi, i.e., pass with −0.4).
    
- **SARSA** moved it **slightly down** to −0.708 (it looked ahead to the **chosen** next action, shoot with −0.8).  
    This is the classic on-policy vs off-policy difference: SARSA’s update reflects the _actual exploratory_ behavior; Q-learning’s target reflects the _greedy_ value regardless of what you really do next.
