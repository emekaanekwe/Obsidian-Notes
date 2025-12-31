# Topics
- **Russell and Norvig, Chapter 5 (5.1, 5.2, 5.3, 5.4)**
- **Also recommended: Sections 5.5 to end of chapter.**
- Adversarial search
    - Models and motivations
    - Minimax
    - Alpha-beta
    - Heuristics  
        
    - Monte-Carlo Tree Search

## C Satisfaction Problems


A CSP is an **assignment of values to variables** where:
$$
\text{there is a set of variables X where} x\epsilon (x_i,...,x_n) \tex{a the set of domains } d\epsilon D (D_i,...,D_n) \text{and a set of contraints that consist of pair <scope, relation> where scope is a tuple of X and relation defines the values that most variables can take on} c\epsilon C () 
$$
D is the set of domains
C is the set of constraints

CSP has a <scope, relation> pair where scope is a tuple of X and relation defines the values that most variables can take on.

### Visualization of CSP


![[Pasted image 20250325093818.png]]

- Nodes WA, NT, etc represent to the variables
- Edges represent the constraint relation of any two variables
### Constraint Propagation
- To reduce the number of legal values for a variable

The process of reducing is the use of:

### Local Consistency
There are three ypes

#### Node
- X is node-consistent if all D of X are injective (for every input there is one and only one output)
- X is arc-consistent if for X, ther is another variable Y where there is some value in D that satisfies the binary constraint <X,Y>, i.e., surjective (for every paired input there is at least one output)

- X s path consistent if there is an additional (third) constraint on a pair of variables

- X is k-consistent if k-1 entails node consistency, k-2 entails arc consistency, and k-3 entails path consistency
	- X is strongly k-consistent if it is (k-1, k-2,...,k-n) consistent

## MinMax Algorithms

#### Alpha and Beta Values
$$
\alpha \text{ of a MAX node := lower bound for a }
$$
**α (alpha) value** – Represents the **best value** that the MAX player can guarantee at that point or above (i.e., the lower bound of max player)
	example: Pacman would want to get the max value (most pellets)

**β (beta) value** – Represents the **best value** that the MIN player can guarantee at that point or below (i.e., the upper bound of min player)
	example: a ghost would want to minimize the time it takes to get to pacman