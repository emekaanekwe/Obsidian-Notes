
#### Knowledge representation Topics (ch 7-11)

| Done | task                          | Source | Due |
| ---- | ----------------------------- | ------ | --- |
|      | read knowledge representation | R & N  | /   |
| O    | Lab Work                      | Moodle | /   |
|      | A1                            |        | 9/4 |
|      | -- q1a                        |        |     |
- Knowledge-based agents
- Logic -- models, entailment and inference
 d
    - definitions
    - syntax and semantics
    - validity and satisfiability
    - backward and forward reasoning
    - resolution-refutation systems

---
# Lecture
| Category | Term                    | Definition                  | Complete? | Optimal? | O-Complexity | Use Cases | Problems | Example |
| -------- | ----------------------- | --------------------------- | --------- | -------- | ------------ | --------- | -------- | ------- |
| Ch 7     | Grounding               |                             |           |          |              |           |          |         |
|          | Completness             |                             |           |          |              |           |          |         |
|          | Soundness               |                             |           |          |              |           |          |         |
|          | validity                | S is true in all models     |           |          |              |           |          |         |
|          | satisfiability          | S is true in some models    |           |          |              |           |          |         |
|          | implication             |                             |           |          |              |           |          |         |
|          | entailment              | S entails P iff S implies P |           |          |              |           |          |         |
|          | conjunctive normal form |                             |           |          |              |           |          |         |
|          | horn clause             |                             |           |          |              |           |          |         |
|          | backward chaining       |                             |           |          |              |           |          |         |
|          | forward chaining        |                             |           |          |              |           |          |         |
|          |                         |                             |           |          |              |           |          |         |

![[Pasted image 20250331130208.png]]
#### Assignment 1

#### Review 
(Russell ch 6)
#### Quiz
---

# Lab

## 1

| B   | v   | C   |
| --- | --- | --- |
| F   | F   | F   |
| T   | T   | T   |
| T   | T   | F   |
| F   | T   | T   |

| -A  | v   | -C  | v   | -C  | v   | -D  |
| --- | --- | --- | --- | --- | --- | --- |
| T   | F   | T   | F   | T   | F   | T   |
| F   | T   | F   | T   | T   | F   | T   |
| T   |     | F   |     | T   |     | T   |
| F   |     | T   |     | T   |     | T   |
| T   |     | T   |     | T   |     | T   |
| F   |     | F   |     | F   |     | F   |
| T   |     | F   |     | F   |     | F   |
| F   |     | T   |     | F   |     | F   |
| T   |     | T   |     | F   |     | F   |
| F   |     | F   |     | T   |     | F   |
| T   |     | F   |     | T   |     | F   |
| F   |     | T   |     | T   |     | F   |
| T   |     | T   |     | T   |     | F   |
| F   |     | F   |     | F   |     | T   |
| T   |     | F   |     | F   |     | T   |
| F   |     | T   |     | F   |     | T   |

## 2
$$

\exists (p)((C_p) -> R_p \ \ \& \ \ E_p \  V \ \ notC_p ) 
$$

## 3

$$
Food \ \implies \ Party \ V \ Drinks \implies Party \ \implies [(Food \ \& \ Drink) \implies Party]
$$

## 4

