# Markov Games

# Nash Equilibrium

# MiniMax-Q



# Aikun's Summary

---

### **Slides 4–5: Individual vs. Collective Perspective & Pareto Optimality**

**Explanation**:

- These slides distinguish between evaluating outcomes from a _collective_ versus an _individual_ lens.
    
- **Pareto Optimality**: An outcome is _Pareto optimal_ if no agent can be made better off without making someone else worse off.

**NOTE**: it is crucial to make sure to go through the motions of the calculating dominance in a zero-sum game.

    

**Critique**:

- Pareto efficiency is silent on fairness or equity — multiple Pareto optimal points can exist, some being highly inequitable.
    
- In multi-agent settings, especially with self-interested agents, achieving Pareto optimality may require coordination or external incentives, which aren't always feasible.
    

---

### **Slides 6–9: Pareto Examples (Traffic, Battle of the Sexes, Matching Pennies, Prisoner's Dilemma)**

**Explanation**:

- Illustrates Pareto optimal vs. dominated outcomes using standard games.
    
- **Important insight**: Not all Nash equilibria are Pareto optimal — _especially in Prisoner’s Dilemma_, where the only equilibrium is socially suboptimal
- ONLY viewed as one match

[TERMS]
Pure strategy - a single action is selected and made


**Critique**:

- Matching Pennies has no Pareto-optimal outcome in pure strategies — a reminder that zero-sum games often defy cooperative reasoning.    

---


**Explanation**:

- A **dominated strategy** is always worse than another strategy, regardless of what the other player does.
    
- Rational agents prune such strategies — this is the beginning of _iterated dominance_ elimination.
    

**Critique**:

- Real-world agents (e.g., humans, bounded-rational agents) often do not eliminate dominated strategies.
    
- Dominance elimination doesn’t always lead to a unique solution or equilibrium.
    

---

### **Slides 14–15: Morra Game (Zero-sum)**

**Explanation**:

- This is a strictly competitive game where each player's gain is the other’s loss.
    
- The payoff matrix is shown for the **row player** only, assuming zero-sum.
    

**Critique**:

- Real multi-agent environments often aren't strictly zero-sum; extending reasoning here to cooperative or general-sum games requires care.
    

---

### **Slides 16–17: Best Response and Strategic Form Games**

**Explanation**:

- Defines **best response**: given what everyone else does, what is your best move?
    
- Introduces action profiles and how to isolate an individual’s view of the game.
    

**Critique**:

- Not practical in high-dimensional or continuous action spaces. Also, best responses assume full knowledge of others’ strategies — rarely available in open MAS.
    

---

### **Slides 18–20: Nash Equilibrium (Pure Strategy)**

**Explanation**:

- A **Nash equilibrium** is a stable strategy profile where no player can improve unilaterally.
    
- It’s essentially mutual best responses.

	pareto: these are the best options
	nash: these are the sisble outcomes

` think of nash equilibrium as a decision procedure that goes from one state to another`

[NASH EQ]
A strategy proﬁle s = (s 1 , . . . , s n ) is a Nash equilibrium if, for all agents i, s i is a best response to s −i .
[STRICT NASH]
Deﬁnition 2.2.3 (Strict Nash). A strategy proﬁle s = (s 1 , . . . , s n ) is a if, for all agents i and for
all strategies s i = s i , u i (s i , s −i ) > u i (s i , s −i ).
[WEAK NASH]
Deﬁnition 2.2.4 (Weak Nash). A strategy proﬁle s = (s 1 , . . . , s n ) is a if, for all agents i and for
all strategies s i = s i , u i (s i , s −i ) ≥ u i (s i , s −i ), and s is not a strict Nash equilibrium.


**Critique**:

- Doesn’t guarantee efficiency or fairness.
    
- In MAS, we often seek _mechanism design_ to steer agents toward socially desirable equilibria — Nash alone isn’t sufficient.
    

---

### **Slides 21–27: Examples of Nash Equilibria**

**Explanation**:

- Walks through Traffic Game, BoS, PD, and Matching Pennies.
    
- Shows how in each case, the Nash equilibrium is either efficient (Traffic) or not (PD), and deterministic strategies fail in zero-sum games like Matching Pennies.
    

**Critique**:

- These examples are pedagogically good but simplistic.
    
- Real systems have asymmetric information, dynamic strategies, and uncertainty — beyond static matrix games.
    

---

### **Slides 28–31: Mixed Strategies and Expected Payoffs**

**Explanation**:

- A **mixed strategy** is a probability distribution over actions.

`pure strategy: only one action is played with positive probability

`mixed strategy: more than one action is played with positive probability`

`mixed nash: best reponse picked based on player's Pr distribution`
- where no pure strategy equilibrium exists.
    
[NASH'S EQ]
for any game G with a fininte number of pure strategies, G has at least one NASH Eq in mixed strategies.

**NOTE**: make sure to keep in mind that we are not concerned with *how the player reaches nash eq*



**Critique**:

- Finding optimal mixed strategies is tractable for small games, but explodes combinatorially.
    
- For real-world agents, approximations (e.g., reinforcement learning, regret minimization) are used instead of full-blown Nash computation.
    

---

### **Slides 32–36: Finding Mixed Nash Equilibria**

**Explanation**:

- Uses the **support-matching method**: assume what strategies might be used (support) and find probabilities that equalize payoffs.
    
- For two-player games like BoS, this reduces to solving linear equations.

To compute nash eq, use the following formula below:
    ![[Pasted image 20250416201504.png|500]]
    if we start with the column player:
    $$
q +0(1-q)=0q+2(1-q)
$$
$$
q=\frac{2}{3}
$$

If the support of the mixed Nash Equilibria is known,
the problem reduces to a system of linear equations.

- The set of supports for a finite ngame is a combinatorial space (size 2 for n actions)

- Finding mixed in general Nash Equilibria is a combinatorial problem.

---

### **Slides 37–40: Nash as Linear Programs; Minimax Theorem**

**Explanation**:

- Any **two-player zero-sum game** can be transformed into a linear program (LP).
    
- Minimax Theorem: each player’s max-min = min-max.
    
- Duality in LP proves Nash existence for zero-sum cases.
    ![[Pasted image 20250416202445.png|450]]



the technique of re-making the equatin linear is to place the constraint in the equation and add z' which will be equal to the max of the constraints.

[GAME THEORY UNDERPIN]
![[Pasted image 20250416204119.png]]


    

---

### **Slides 41–44: Price of Anarchy**

**Explanation**:

- **Price of Anarchy** (PoA): how much worse the worst-case equilibrium is compared to the optimal coordinated solution.
    
- E.g., PD has PoA = 2/6 = 1/3 — a loss of social utility due to selfish behavior.
    

**Critique**:

- Critical for MAS engineers: shows how equilibrium behavior can be detrimental.
    
- Suggests designing incentives (e.g., taxes, rules, communication channels) to reduce PoA.
    

---

### **Slides 45–48: Correlated Equilibrium**

**Explanation**:

- A **correlated equilibrium (CE)** allows players to condition their actions on shared random signals.
    
- Strictly more general than Nash: every Nash is a CE, but not vice versa.
    
- Traffic light analogy shows CE can achieve coordination without central enforcement.
    

**Critique**:

- Hugely important and under-taught.
    
- Mechanism design in MAS often implicitly builds CE (e.g., shared norms, protocols).
    
- CE allows decentralized coordination — highly relevant in real MAS.
    

---

### **Slide 49: Take-home Lessons**

**Highlights**:

- Nash equilibrium solves a game’s circular dependencies.
    
- Mixed strategies, CE, and LP-based solutions all generalize and extend the concept.
    
- Yet, finding equilibria is often hard, and many real-world problems are NP-complete to even reason about.
    

**Critique**:

- This wrap-up is solid.
    
- But in practical multi-agent systems, we also care about **dynamics** — how agents learn strategies over time (e.g., via reinforcement learning, no-regret algorithms). These are absent from static equilibrium theory but critical for functioning MAS.
    

---

### 🧠 Summary (for next week's class)

Before your lecture:

1. **Understand the formal definitions** (Nash, Pareto, Best Response).
    
2. **Practice identifying dominated strategies and computing best responses.**
    
3. **Critically think about what assumptions make these models unrealistic**:
    
    - Complete info?
        
    - Rationality?
        
    - Synchronous moves?
        
4. **Be ready to discuss how equilibrium concepts apply to real agent systems** — like robot coordination, distributed AI, or social networks.
    
5. **Read ahead on Correlated Equilibria** — the most practically relevant extension for MAS.
    

---

Would you like a cheat sheet summarizing definitions and formulas from this lecture? Or some practice problems to prepare?