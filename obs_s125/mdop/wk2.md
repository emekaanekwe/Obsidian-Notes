# Problem Paradigms


# Sets

Sets are a data structure that allows it to be broken up into subsets and an empty set. 

*Consider the variable declaration `'var set of {8,9}: x;'`.* 
*How many different values can x possibly take?*

	it can take 4, {}, {8}, {9}, {8,9}

### When maximizing
```
solve maximize sum(i in MOVES)(power[i] * bool2int(occur[i]));
```

## Set Cardinality

We can get cardinality by manipulating them in arrays, while ensuring that the array keeps the set caridanlity.
```
int: nSpots;

set of int: SPOT = 1..nSpots; % spot set

array[SPOT] of int: damage;
constrinat forall(i,j in 1..nSpots where i<j)(damage[i] != damage[j])

```
	a set of s = {1,2,3} has card of 3, but a corresponding array does note represent it a = [1,2,3]

One difference between sets and arrays is that an element can only occur once in a set but multiple times in an array. Which one of the following is another difference?

	Arrays are ordered whereas sets are not

### Making sure to have one solution representation

there are possible solution representations here:

```
[0,2,0], [2,0,0] ... with card = 2 % wrong

```
	so we make use of arraysand combine strict and non-strict ordering:
	`forall(i in 1..size-1)(x[i] >= (x[i] != 0) + x[i+1]);

so every variable set has exactly one proper representation:
```
var set of {1,2,3}: x;

array[1..3] of var 0..3: y;
```

	with 0 representing the null element

What does **bounded cardinality** mean in this context?

	the number of selected elements from a set  of possible elements must be less or equal to that bound.

```


## Casting a Bool to an Int

```
array[MOVES] of var bool: occur;

constraint (sum(i in MOVES)(duration[i] * xbool2intx(occur[i]))) <= timeBound;
```
	putting of var bool: occur, the bool2int can be removed bc MZ already does it implicitly

# Q&A

1. is the code linear?
```
var 0..1: x
var 0..1: y

constraint x+5*y >= 20

solve maximize x*y
```
no, because the obj funct is not

2. What is a fact about Enums?
 each enum type is mapped to an int value

3. the diff btw model and instance?
model is parametic (work for many diff problems of same class), and instance is a concrete problem

4. why need arrays?
can write models where num of objects are not fixed, but rather as a param

5. for this result [| 1,2,3,4| 5,6,7,8|], what is the correct expression?
`array2d(1..n, 1..n, [ n*j+n-i | i,j in 1..n])`

6. what expression is equivalent to `alldifferent(x)` here x is an array of int vars with index set 1..n?
`forall (i in 1..n-1, j in i+1..n) (x[i] != x[j])`

# --Topics--

- Express problems with parameters and decision variables
- Express common arithmetic constraints
- Use enumerated types
- Create parametric models and instantiate them with data
- Model objects using collections of arrays
- Create and manipulate arrays using comprehensions
- Use global constraints to capture common problem sub-structures

# Enums

### Allow for the definition of custom sets of possible values

### Function: solving problems involving categories or states

# Operators

# Graph Labels

###### *Find a solution where all nodes are assigned one of the values and no adjacent nodes have consecutive values*

![[ML_AI - fit5216_mdop.png]]

#### one could *try* to brute force the problem... but that is not efficient.

## What is an efficient solution?

##### try cutting the map in half (DFS or BFS) and find all possible solutions, then continue on with the rest

![[Pasted image 20250311160333.png|200]]![[Screenshot from 2025-03-11 16-05-17.png|400]]

# Lab
## Minizinc Approach
### 1. run an empty model to check the solvers

*remember to keep in mind that the domain can be defined within the variable!!*

### Clarify (use cases)
1. int: n = length(readings);

```

# Graph Label
```python
include "globals.mzn";

var 1..8: a;
var 1..8: b;
var 1..8: c;
var 1..8: d;
var 1..8: e;
var 1..8: f;
var 1..8: g;
var 1..8: h;


constraint abs(a-b) >= 2 /\ abs(a-c)>=2 /\ abs(a-d)>=2;

constraint abs(c-b)>=2 /\ abs(c-d)>=2 /\ abs(c-e)>= 2 /\ abs(c-f)>2 /\ abs(c-g) >=2;

constraint abs(f-b) >=2 /\ abs(f-d)>=2 /\ abs(f-e)>=2 /\ abs(f-g)>=2 /\ abs(f-h)>= 2;

constraint abs(h-e) >=2 /\ abs(h-g) >=2;

constraint alldifferent([a,b,c,d,e,f,g,h]);

solve satisfy;
```

# Power Gen

model a comparison of energy expected and energy oduced by a power plant, but take into account current

  % if expected is 25 while current is 18, then we need 7 more to meet demand

    % N=4, C=1, S=1

    % N + S + S + S = 7 + 18  = 25

      % demand met


```python
constraint forall(energy in a, n in N, c in C, s in S)(N[energy] = 2);
```
I don't understand why N[energy] works.

