

## Aikun's Summary

---

## **Slides 4–6: Motivation**

### **Core Idea:**

So far, agents were **logical but brittle** — now they’ll handle **uncertainty** with **probability theory**.

### **Real-world example:**

Imagine a self-driving car:

- **Logic-based** agent: “If I see green, go.”
    
- **Probabilistic** agent: “If the light _looks_ green and it’s not foggy, there's a 95% chance it's safe to go.”
    

### **Critique:**

No error, but it could be clearer: uncertainty includes _sensor noise_, _incomplete information_, and _model assumptions_.

---

## **Slides 7–9: Key Terms**

### **Core Terms:**

- **Experiment**: Rolling a die.
    
- **Sample space**: {1,2,3,4,5,6}.
    
- **Event**: {3} or {2,3,4} (e.g., “roll between 2 and 4”).
    
- **Random variable**: `X = die outcome`.
    
- **Domain of X**: {1,2,3,4,5,6}.
    
- **Probability function**: `Pr(X=3) = 1/6`.
    

### **Real-world example:**

- `D = commute time`
    
- Domain: [10, 100] minutes
    
- `Pr(D=20)` = 0.2 (20% chance your commute is 20 min)
    

### **Critique:**

Correct, but the term “random” could mislead — these aren’t always _truly_ random. They're _used to model uncertainty_, not just randomness.

---

## **Slides 10–12: Probabilistic Inference**

### **Key Idea:**

We calculate things like `Pr(A | B)` — the **probability of A given evidence B**.

### **Simple example:**

- `Pr(on time | no accidents) = 0.9`
    
- `Pr(on time | no accidents and 5 a.m.) = 0.95`
    
- `Pr(on time | no accidents, 5 a.m., raining) = 0.8`
    

You can think of this as **gradually refining** your belief as more evidence comes in.

---

## **Slide 13–14: Models and Distributions**

### **Core Concept:**

##### *A **probability distribution** maps outcomes to likelihoods.*

### **Simple Example:**

- `Pr(Weather)` =
    
    - sunny: 0.6
        
    - rain: 0.3
        
    - fog: 0.1
        

### **Critique:**

The G.E.P. Box quote is good, but the notion that we use models to "predict, explain, and decide" should be more emphasized. That’s the real power here.

---

## **Slide 15–19: Joint & Marginal Distributions**

### **Core Concepts:**

- **Joint**: `Pr(Weather=sunny AND Temp=hot)`
    
##### **Marginal: `Pr(Weather=sunny)` — sum over temperature**
    

### **Example Table:**

| Weather | Temp | Pr  |
| ------- | ---- | --- |
| sun     | hot  | 0.4 |
| rain    | hot  | 0.1 |
| sun     | cold | 0.2 |
| rain    | cold | 0.3 |

- **Marginal Pr(sun)** = 0.4 + 0.2 = **0.6**
		meaning that marginal is concerned with a set of weather temps
    

---

## **Slide 20–23: Conditional Distributions**

### **Key Idea:**

To get `Pr(A | B)`, we use the formula:

> `Pr(A | B) = Pr(A ∧ B) / Pr(B)`

### **Simple Example:**

From the table above:

- `Pr(rain | cold)` = Pr(rain ∧ cold) / Pr(cold) = 0.3 / (0.2+0.3) = 0.3 / 0.5 = **0.6**
    

---

## **Slide 24: Normalization Trick**

This trick is **essential**:

1. Pick all joint entries consistent with the evidence.
    
2. Divide each by their total to make them sum to 1.
    

> Think of it as:  
> "You have a bunch of beliefs. You just learned something. Now make sure your beliefs still add up to 100%."

---

## **Slide 25–27: Definitions and Applications**

Solid definitions of:

- **Marginal**: Total for one variable.
    
- **Joint**: Combo of two variables.
    
- **Conditional**: Probability of one given the other.
    

### **Example:**

- `Pr(sun | summer)` = 0.4 / 0.5 = **0.8**
    

---

## **Slide 34–35: Inference by Enumeration**

This is brute-force:

- **List all possible combinations**, then
    
- **Sum out** irrelevant variables, and
    
- **Normalize**.
    

### **Critique:**

Accurate, but **computationally infeasible** for large systems. This is where **Bayesian Networks** or **factor graphs** become essential.

---

## **Slides 36–39: Product & Chain Rule**

### **Product Rule:**

> `Pr(A ∧ B) = Pr(A | B) * Pr(B)`

### **Chain Rule:**

> Break any joint into a chain of conditionals:
> 
> `Pr(A, B, C) = Pr(C|B,A) * Pr(B|A) * Pr(A)`

### **Example:**

Rain → Umbrella → Traffic

- `Pr(Traffic | Umbrella, Rain) * Pr(Umbrella | Rain) * Pr(Rain)`
    

---

## **Slide 40–42: Bayes’ Rule**

### **Bayes’ Theorem:**

> `Pr(H | E) = [Pr(E | H) * Pr(H)] / Pr(E)`

This flips conditionals and is the engine of **diagnostic reasoning**.

### **Simple Example (Breast Cancer):**

- `Pr(cancer) = 0.01`
    
- `Pr(+ive | cancer) = 0.8`
    
- `Pr(+ive | no cancer) = 0.1`
    
- **Pr(cancer | +ive)** = 0.008 / (0.008 + 0.099) ≈ 0.075
    

**Despite the positive test, only ~7.5% chance you have cancer.** This is _counterintuitive but correct_.

---

## **Slide 45–47: CJD Problem**

### **Critique:**

Good demonstration, but a better way to teach this is with **natural frequencies**:

Imagine 100,000 people:

- 1 has CJD → eats burgers → 0.99
    
- 99,999 don't have CJD → 10% eat burgers → ~10,000
    

So among 10,001 burger eaters, **only 1** has CJD.

> **Real chance: 1 in 10,001 ≈ 0.01%**

---

## **Slide 48–55: Independence and Conditional Independence**

### **Simple Independence:**

Two coin flips. Pr(H1,H2) = Pr(H1)*Pr(H2)

### **Conditional Independence:**

Knowing `Cavity` renders `Toothache` and `Catch` independent.

> **Analogy**:  
> If I already _know_ it’s raining, seeing an umbrella tells me _nothing new_ about whether traffic is bad.

---

## **Final Thoughts**

This slide deck does a great job introducing core probability concepts for AI. But here’s how I’d improve it as an MIT professor:

### **Key Pedagogical Fixes:**

1. Add **simple, real-life examples** early and often (like we just did).
    
2. Clarify that **probability is about belief under uncertainty**, not randomness alone.
    
3. Emphasize the **computational costs** of full enumeration.
    
4. Make sure **Bayes’ rule** is taught with both math and natural frequencies — the intuition is essential.
    
---
# LAB

Exercise 1: Estimating probabilities In a game, you will be given $5 if you draw a red jelly bean from a bowl, without looking. You may draw only one jelly bean. You may choose which of two bowls to draw from: a large bowl has 7 red jelly beans and 93 black ones, and a small bowl has 1 red jelly bean and 9 black ones. Which bowl do you wish to draw from?

X and Y

| Bowls | X   | Y   |
| ----- | --- | --- |
| Red   | 7   | 1   |
| Black | 93  | 9   |

Pr(X=red) => 7/100 => .07
pr(Y=red) => 1/9 => .1
pr(R=red) -> $5
$$
Pr(A|B)=
$$

In a school, 14% of students take drama and computers, and 67% take a computer class.
What is the probability that a student taking computers also takes drama?

pr(A|B)
A: drama
B: computers

$$
Pr(A|B)=Pr(D|C) * Pr(C|D)

$$

$$
\text{assuming independence: } \ Pr(C) = .67
$$
pr(c) = .67
pr(d) = x