I want to apologize in advance for the following paragraph. It is unnecessary and may work against me, but for some reason I feel extremely uncomfortable if I do not mention it.

---

I do think I agree with the use of the phrase "possible worlds" which is often employed by Russell and Norvig. In the field of logic, mathematics, and philosophy, 

"The idea of possible worlds raised the prospect of extensional respectability for modal logic, not by rendering modal logic itself extensional, but by endowing it with an extensional semantic theory — one whose own logical foundation is that of classical predicate logic and, hence, one on which possibility and necessity can ultimately be understood along classical Tarskian lines. Specifically, in _possible world semantics_, the modal operators are interpreted as _quantifiers_ over possible worlds, as expressed informally in the following two general principles:
	![[Pasted image 20250424090636.png|600]]

		(Source: https://plato.stanford.edu/entries/possible-worlds/#ModalLogicAndPWs)
			(You can also use the contemporary philosopher David Lewis as a reference)



Instead, I will simply use,
$$
\begin{align}
\text{for some proposition p and a complete formal system F, } \\ \text{ p is true iff p }\epsilon \ F \text{ interpreted under a model-theroretic semantics M}
\end{align}
$$

---



Now on to the assignment

# Part 1

## a)

In the first section, I write out the derivation process in tabular format with three columns: the row number, the proposition, and the row number of the proposition used with the inference rule, respectively. For example,

| 1   | A -> B |                                 |
| --- | ------ | ------------------------------- |
| 2   | A      | 1, assume for conditional proof |
| 3   | B      | 1,2 Modus Ponens                |
As you can see, the inference rule *Modus Ponens*, is invoked from lines 1 and 2, hence "1,2 Modus Ponens."

---

 For the following propositional statements,
$$
\phi_1 \equiv A \implies ((A \implies B) 
\lor \ (B \implies C))
$$
We can use a truth assumption for this statement, since the core form of this statement is X -> Y, we can assume the truth of X and check that Y is true. As we know, attempting to assume the falsity of A would make the statement necessary pre-proof, thereby making it tautological.. Using the laws of derivations for sentential logic with A assumed, we should expect B and C to follow. What is below demonstrates this statement's validity.

|     | $$<br>\phi_1 = A \implies ((A \implies B) <br>\lor \ (B \implies C))<br>$$ |                              |
| --- | -------------------------------------------------------------------------- | ---------------------------- |
| 1   | A                                                                          | Assume for Conditional Proof |
| 2   | (A -> B) V (B -> C)                                                        | 1, 2 Modus Ponens            |
| 3   | A -> B                                                                     | 2 Simplification             |
| 4   | B                                                                          | 1, 3 Modus Ponens            |
| 5   | B -> C                                                                     | 2 Smplification              |
| 6   | C                                                                          | 4, 5 Modus Ponens            |
The above proves that all sentences can be inferred from the expression. Therefore, the expression is valid.

---

For the next statement,
$$
\phi_2 = (A \ \& \ B) \iff ((B \implies C) 
\implies ((C\  \& \ A) \lor A))
$$
We can use a similar method for this next statement, except that we must first deconstruct the biconditional,

| 1      | $$\phi_2 = (A \land B) \iff ((B \implies C) \implies ((C \land A) \lor A))$$ |                                     |
| ------ | ---------------------------------------------------------------------------- | ----------------------------------- |
| 2      | (A & B) -> ((B -> C) -> (C & A) v A))                                        | Assume for Bionditional Proof       |
| 3      | ((B -> C) -> (C & A) v A)) -> (A & B)                                        | Assume for Biconditional Proof      |
| 4      | -((A & B) v ((B -> C)) v (C & A) v A))                                       | 2 Conditional Disjunction           |
| 5      | -((A & B) v ((B -> C))                                                       | 4 Simplification                    |
| 6      | -(A & B) & -(B &C)                                                           | 5 Demorgan's                        |
| 7      | (-A v -B) & (-B v -C)                                                        | 6 Demorgan's                        |
| 8      | (C & A) v A                                                                  | 4 Simplification                    |
| 9      | A                                                                            | 8 Simplification                    |
| 10     | -((B -> C) -> (C & A) v A)) v (A & B)                                        | 3 Conditional Disjunction           |
| 11     | -((B -> C) -> (C & A) v A))                                                  | 10 Simplification                   |
| 12     | --(B -> C) v ((C & A) v A)                                                   | 11 Conditional Disjunction          |
| 13     | (B -> C) v ((C & A) v A)                                                     | 12 Double Negation                  |
| 14     | B -> C                                                                       | 13 Simplification                   |
| 15     | -B v C                                                                       | 14 Conditional Disjunction          |
| **16** | **-B**                                                                       | 15 Simplification                   |
| 17     | -((B -> C) v ((C & A) v A)) v (A & B))                                       | 3 Conditional Disjunction           |
| 18     | -(B -> C)                                                                    | 17 Simplification                   |
| 19     | -(-B v C)                                                                    | 18 Conditional Disjunction          |
| 20     | --B & -C                                                                     | 19 Demorgan's                       |
| 21     | B & -C                                                                       | 20 Double Negation                  |
| **22** | **B**                                                                        | 21 Simplification **Contradiction** |

Since a contradiction is found, there is no model M where all statements can be true. Ergo, statement 2 it is not valid. Even so, there is an M where statement 2 is satisfiable. Specifically, when A=F, B=T, C=F.

---

The last statement has a core form of X & Y, in which no antecedent or consequent is present,
$$
\phi_3 \vDash (A \iff B) \ \land \ ( \neg( \ A \land \neg C) \lor (B \iff D))
$$
 So, what we can do is consider a case in which the statement would be invalid and check if we can derive a contradiction. It is important to note that since we have a biconditional on the left, negating it would give us tautological results. Let us instead place more focus on the propositions. Specifically, let us assume that A=T, B=T, C=F, D=F,
$$
\phi_3 \vDash (A \iff B) \ \land \ ( \neg( \ A \land \neg \neg C) \lor (B \iff \neg D))
$$
Focusing on the right,

| 1     | $$\neg( \ A \land \neg \neg C) \lor (B \iff \neg D)$$ | Simplification                      |
| ----- | ----------------------------------------------------- | ----------------------------------- |
| 2     | -(A & --C)                                            | 1, Simplification                   |
| 3     | -A v ---C                                             | 2, Demorgan's                       |
| 4     | -A v -C                                               | 3, Double Negation                  |
| **5** | **-A**                                                | 4, Simplification **Contradiction** |

As we can see, **statement 3 is not valid**, since A (by deconstructing the biconditional) and -A. Even so, statement 3 is **satisfiable** if we  assumed that A = T, B = T, C = F, D = T.

## b)

I will use derived statements constructed using the derivations done in the previous section for the relevant expression.

For the first statement, its CNF is as follows:

$$
\phi_1 \equiv A \implies ((A \implies B) 
\lor \ (B \implies C))
$$
becoming 
$$
-A  \lor  (-A \lor B) \lor (-B \lor C) 
$$
which can be converted further to,
$$
-A \lor B \lor -B \lor C
$$
From the above statement, we can also infer that the expression contains the property of a horn clause, since it has at most one positive literal.

For \phi_2,
$$
\phi_2 = (A \ \& \ B) \iff ((B \implies C) 
\implies ((C\  \& \ A) \lor A))
$$

We can do the following,

$$
((A∧B)→((B→C)→((C∧A)∨A)))∧(((B→C)→((C∧A)∨A))→(A∧B))
$$

By transforming the biconditional into conditionals and further reduce to reach  conjunction of clauses,
$$
\begin{align}
¬A∨¬B∨¬C\neg A \lor \neg B \lor \neg C¬A∨¬B∨¬C \\

¬B∨C∨A\neg B \lor C \lor A¬B∨C∨A
    
¬A∨B\neg A \lor B¬A∨B
\end{align}
$$

By breaking up the clauses, we can see that the first and last clause are horn clauses,

$$\begin{align}
¬A∨¬B∨¬C\neg A \lor \neg B \lor \neg C¬A∨¬B∨¬C \\
    
¬B∨C∨A\neg B \lor C \lor A¬B∨C∨A \\
    
¬A∨B\neg A \lor B¬A∨B
\end{align}
$$

# Part 2


In this next section, we convert the following into Propositional Logic Form:

$$\begin{align}
\text{If a person is likely to vomit and looks } \text{pale and is thirsty, then is sick.}
\end{align}
$$
$$
(V\  \& \ P\  \& \ T) -> S
$$

$$
\begin{align}
\text{Always, a person does not have a high } \text{temperature or has slept well or is thirsty.}
\end{align}
$$
$$
\neg H \lor \neg NS \lor T
$$
$$\begin{align}
\text{If a person does not look pale } \text{, then feels well or has slept well.}
\end{align}
$$
$$
\neg P -> (\neg NF \lor \neg NS)
$$

$$\begin{align}
\text{If a person has a fever, then their temperature is high.}
\end{align}
$$
$$
 F -> H
$$
$$
\begin{align}
\text{It is known that when a person looks pale, then they should drink water.}
\end{align}
$$
$$
P -> W
$$

$$\begin{align}
\text{If a person has a high temperature and does not feel well, then is likely to vomit.}
\end{align}
$$
$$
(H \  \& \ NF) -> V
$$
$$
\begin{align}
\text{It is always the case that a person does not feel well or does not have a fever.}
\end{align}
$$
$$
NF \lor F
$$

$$
\begin{align}
\text{If a person has slept well, then is not exhausted.}
\end{align}
$$
$$
\neg NS -> \neg E
$$


Here, we check if the following statement can be derived from the set of propositions above,


$$
\text{A person is sick if one assumes both that the person has a fever and that the person is exhausted.}
$$


Converting into logical form we have,

$$
(F \ \& \ E\ ) -> S
$$

Just like the technique employed above (Question 1 \phi_1), we will assume the antecedent to be true,
$$
(F \ \& \ E\ )
$$

We can now establish the AGENDA for this this proof

$$
AGENDA=(F,E)
$$
Following the **forward chaining** approach, we maintain the AGENDA while we use the INFERRED table to mark whether a symbol has already been INFERRED. The table below demonstrates the details:

![[Pasted image 20250423203655.png|500]]

Thus demonstrating that the statement can be inferred.

---

 Using **Backward Chaining**, we can do the following steps:

1. To prove S, need V, P, T
2. V: needs H and NF
    - H: via F→H
    - NF: via NF v ¬F, given F
3. T: from ¬H ∨ ¬NS ∨ T deduced via:
    - F→H 
    - E→N
    - So ¬H=F, ¬NS=F ⇒ T must be true
4. P: from ¬P→(¬NF v ¬NS)
    - Both NF, NS are true ⇒ **RHS** is false ⇒ ¬P=F⇒P       
Thus:
F, E ⊢ S


# Part 3

In this section, we consider the goal statement,
$$
Goal: \ \ \forall x. \ P(x)
$$
and use the **resolution method** to check if the goal can be proved from statements 1), 2), and 3),
$$\begin{aligned}
1. \ (\exists z. \ R(z)) \implies \exists w. \ T(w)
\end{aligned}
$$
$$
2. \ \forall x \exists y ((\forall z. \ P(x) \lor R(z)) \ \land \ (\exists w. (T(w) \implies Q(y))))
$$
$$
3. \ \forall w \exists y ((\forall x. (P(x) \implies T(w))) \land (\forall x. (Q(y) \implies P(x))))
$$
---
$$\begin{align}
\neg P(a) \\
\neg Q(h(w)) \\
f(x) = h(w) => \neg T(g(a)) \\
T(g(a)) = g(a) => \neg P(x) \\ \\

\text{resolve forwarding} \\ \\
R(z) \\
T(w) \\
Q(f(x)) \\
P(x) \\ \\
resolving \ \neg P(x) \\ \\
contradiction

\end{align}
$$

Since we have arrived at a contradiction, we can conclude that **the goal is provable from the statements**.

---
The list of the most used Most General Unifiers are as follows:
$$\begin{align}
 1.\ x:=ax \text{ to match } P(x) with \neg P(a) \\

2. \ Q(f(a)) = Q(h(w)) \text{ with unifier } f(a) := h(w) \\
3.\ T(g(a)) = T(w) \text{ with unifier } g(a) \\
4.\ R(z) and \neg R(z) \text{ identity}*
\end{align}
$$

										\*MGUs from ChatGPT
										