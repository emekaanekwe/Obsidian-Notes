# 3 Parts

## Parameters

## Decision Vars

## Constraints

# Variable Dec

there are:

	unbounded variables

```var int: x```

	vars within a range

```var 1..10: y```

	arrays

```array[1..5] of var int: z```

# Modeling Process

## Identify Decision Vars

## Define What Values Vars Can Take

## List Constraints (what must be true/false)


# Patterns

##### Pick at most k items from a list

```python
constraint card(S) <= k;


# boolean version
array[P] of var bool: x;
constraint sum(p in P)(bool2int(x[p])) <= k;
```

##### "Maximize total value of selected items"

```python
constraint value = sum(p in S)(v[p]);
solve maximize value;

# boolean version
constraint value = sum(p in P)(v[p] * bool2int(x[p]));

```

##### "The average of some quantity must be ≥ X"

$$
\frac{total}{count} >= l => total >= l*count
$$
	rewrite as
```python
constraint count > 0 -> total >= l * count;

```

##### "Consider distance/conflicts for all pairs of items in a set"
```python
sum(i, j in S where i < j)(some_expr(i,j))

```

##### "Each item has a type or label"
```python
enum Colors = {red, green, blue};
array[1..n] of var Colors: color;

```

## Sets & Selection

| English Phrase            | Purpose              | MiniZinc Code               | Notes                     |
| ------------------------- | -------------------- | --------------------------- | ------------------------- |
| “Choose up to k items”    | Limit cardinality    | `constraint card(S) <= k;`  | `S` is `var set of int`   |
| “Choose exactly k items”  | Fixed selection size | `constraint card(S) = k;`   |                           |
| “Choose at least k items” | Minimum selection    | `constraint card(S) >= k;`  |                           |
| “Item i is selected”      | Track with bool      | `array[P] of var bool: x;`  | Use `x[i] = 1`            |
| “If item i in S...”       | Set membership       | `constraint i in S -> ...;` | Or use `bool2int(i in S)` |
## Pairwise Combinatorics
| English Phrase                    | Purpose              | MiniZinc Code                                      | Notes                                 |
| --------------------------------- | -------------------- | -------------------------------------------------- | ------------------------------------- |
| “Every pair must be...”           | All pairs            | `forall(i, j in S where i < j)(...)`               |                                       |
| “Distance between selected pairs” | Pairwise interaction | `sum(i, j in P where i < j)(x[i] * x[j] * d[i,j])` | Used in clustering, facility location |
| “Conflict if both selected”       | Binary conflict      | `x[i] + x[j] <= 1;`                                |                                       |


# Solutions Discovered
| Week | {Problem, Discovered} |
| ---- | --------------------- |
| 1    |                       |
| 2    |                       |
| 3    |                       |
| 4    |                       |
| 5    | {reindeer, y}         |
| 6    |                       |
| 7    |                       |
| 8    |                       |

# Arrays & Enums

## Core Function

#### Arrays are like 1-D spreadsheets that are indexed by a set

the rows are the **entries with index *i***,  where ***i* is in some set**
```latex
array[1..3] of int: primes = [2,3,5];
```

	can be visualized like:

| index 1 | index 2 | index 3 |
| ------- | ------- | ------- |
| 2       | 3       | 5       |

---
## Array with Enums

use **enums** to assign a name to an index of the array, which allows for **easier updating**
```c++

enum DAY = {Mon, Tue, Wed};

array[DAY] of int: temp = [10, 20, 40]; // strictly has length SHIFT, i.e. 3
```

```c++

temp[Mon] + temp[Wed] >= temp[Tues]
```
	the array model lets you "tell a story" about the different temps for the days

---
## Looping

For the set up using enums as the indices and a 3d array:

```c++

enum PERSON = {Vor, Xavier, Serena} //makes a set with 3 vals

array[PERSON, PERSON] of bool: rivals = [
|| false, true, false
   | true, false, true
   | false, true, false |]; // makes 3 column spreadsheet
```

the array "spreadsheet" will look like:

|        | Vor | Xavier | Serena |
| ------ | --- | ------ | ------ |
| Vor    | f   | t      | f      |
| Xavier | t   | f      | t      |
| Serena | f   | t      | f      |
Now we can **tell a story** about the data:

```c++

constraint rivals[Xavier] = rivals[Vor] // true
```

With loops:

```c++

constraint forall(i in PERSON, j in PERSON where i != j)(not (rivals[i,j] /\ rivals[j,i]));

```


---
# Sets
## Core Function
##### Sets are containers for values that help organize your model

| Container A | Container B |
| ----------- | ----------- |
| 1           | 2           |

### Uses
define the **length** of arrays  
specify **what is allowed** in the array

```
set of 1..7: y;
```
	input: {2} = 2..2
	input: {2, 4} = 2..4

```latex
set of int: days = 1..7; % set could be of card(1), card(4) or card(7), etc.
```
	returns 1..7

```latex
set of int: PRIMES = {2, 3, 5}; % this set only accepts 2,3, and 5
```
	returns {2, 3, 5}

assign to a **decision variable**

```latex 
set of int: VALUES = 1..10;
var VALUES: x; % x can be any num between 1 and 10
```
	returns 1

## Understanding Cardinality

Example: *make a model that represents a set of nums 1..1000 of cardinality between 3 and 10*
```c++
array[1..10] of var 0..1000: x;
```

Example: *make a model to represent a set of nums from 1 to 10 of a cardinality AT MOST 1000*
```c++

var set of 1..10: numsss;

```
	give the nature of the requirement, there is no need to store N elements, since there is no lower bound 

Example: *make a model that represents a set of num from 1 to 1000 of cardinality btw 99 and 150*
```c++

array[1..1000] of var bool: x;
```
	this is valid because you can define a SUBSET of the entire length and when 99 <= i <= 150 = true

	Like this:
	
| 1..   | ..50.. | ..99.. | ..130.. | ..150.. | ..151.. | ..200.. |
| ----- | ------ | ------ | ------- | ------- | ------- | ------- |
| False | Flase  | True   | True    | True    | False   | False   |
# Using forall()

## Core Function
$$
\text{for some set S and for every element s } \epsilon \ S \text{, apply constraint C}
$$

Example: *all elements in array must be grater than 0*
```c++

array[1..4] of var 1..10: x;

constraint forall(i in 1..4)(x[i] > 0);
```

Example: *no person gets seat 1*
```c++

enum PERSON = {Vor, Xavier, Serena};
array[PERSON] of var 1..3: seat;

constraint forall(p in PERSON)(seat[p] != 1);
```

	Notice the syntactical structure:

enum <mark class="blue">PERSON</mark> = {}

array[<mark class="blue">PERSON</mark>] of var 1..3;    <mark class="green">seat</mark>

constraint     forall(      <mark class="cyan">p</mark>    in    <mark class="blue">PERSON</mark> )(   <mark class="green">seat</mark>[<mark class="cyan">p</mark>] != 1    )

"for every person, they are not in seat 1"

Example: *all values must be different*
```c++

array[1..4] of var 1..10: values;

constraint forall(i,j in 1..4 where i < j)(values[i] != values[j])
```


# Flattening

### Flattening Rules:

- Rightmost index moves fastest
    
- Middle index moves after
    
- Leftmost index moves slowest
    

---

### Flattening 2D array `[i,j]`

|Order|Indexes|
|---|---|
|1|(i1,j1)|
|2|(i1,j2)|
|3|(i1,j3)|
|...||
|n|(i2,j1)|

### Flattening 3D array `[i,j,k]`

| Order | Indexes    |
| ----- | ---------- |
| 1     | (i1,j1,k1) |
| 2     | (i1,j1,k2) |
| 3     | (i1,j2,k1) |
| 4     | (i1,j2,k2) |
| 5     | (i2,j1,k1) |
| 6     | (i2,j1,k2) |

# Minizinc Conditions

|Natural Language Condition|MiniZinc Form / Strategy|
|---|---|
|**"Each element must..."**|`forall(i in I)(...)`|
|**"There exists an element such that..."**|`exists(i in I)(...)`|
|**"Sum of values..."**|`sum(i in I)(expr(i))`|
|**"Count how many..."**|`sum(i in I)(bool2int(condition(i)))`|
|**"Exactly one..."**|`sum(i in I)(bool2int(condition(i))) = 1`|
|**"At most one..."**|`sum(i in I)(bool2int(condition(i))) <= 1`|
|**"At least k..."**|`sum(i in I)(bool2int(condition(i))) >= k`|
|**"If A then B..."**|`A -> B`|
|**"A if and only if B..."**|`A <-> B`|
|**"x is in a set"**|`x in myset`|
|**"x is not in a set"**|`not (x in myset)`|
|**"The maximum of..."**|`max([expr(i)|
|**"The minimum of..."**|`min([expr(i)|
|**"All different"**|`alldifferent(array)`|
|**"All equal"**|`forall(i in I)(array[i] = array[1])`|
|**"Consecutive values"**|Often `abs(x[i] - x[i+1]) = 1`|
|**"Group contains these items"**|Use `subset`, `set union`, `intersect`|
|**"Apply condition only when..."**|Use `let` blocks or `if ... then` inside expressions|


# When Modeling

## 1. Understand Clearly the word problem 

## 2. Check output
check to see what the vars will output to help give you an idea of what data the solver is working with


## If each spot needs to be filled
##### constraint forall(d in DIRN)(exists(i in 1..6)(DIRN[i] = d));
## Are any array-out-of-bounds for solver?
Remember that writing constraints such as 
##### constraint forall(i in <mark class="red">1..n-1</mark>)(abs(x[i] - x[i+1]) > d); 
	where the red text accounts for the bounds

















![[Pasted image 20250425112852.png]]



![[Pasted image 20250425114904.png]]
# Questions
1. is ther eany diff btw show() and \(x)?
2. what does constraint x **mod** 5 = 2 mean
3.  why do we need a dummy val here:
	 array[1..10] of var 1..1000: x is instead
	 array[1..10] of var **0**..1000: x;
4. Can I get a solution for the most separated set problem?
5. 















# Keywords & Phrases

"So that each node is labelled by a different number **from 1 to 8**, and **each pair of nodes that share an edge** have labels which are at **least two apart**."
		abs values and quantity relations

