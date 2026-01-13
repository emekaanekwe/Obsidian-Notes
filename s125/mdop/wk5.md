
| Done | Task               | Source             | Due |
| ---- | ------------------ | ------------------ | --- |
|      | modeling with sets | wk 3 prep material |     |
|      | comprehensions     | w 3 lab            |     |
|      | reindeer problem   | wk 5               |     |
|      | hidato problem     | wk 5               |     |
|      | fix A1             |                    |     |
#### Multiple Modeling
- Identify different viewpoints for the same problem
- Use channeling constraints to combine multiple viewpoints
- Express assignment and permutation problems using multiple viewpoints
- Select an appropriate viewpoint based on which constraints can be expressed in it

#### Week 4 Videos

#### Week 5 Videos

# Reindeer
	Permutation Problem

### constraints hold
```c++

include "globals.mzn";

  

set of int: POS = 1..4; % type int

enum REINDEER = { Lancer, Quentin, Ezekiel, Rudy };

array[REINDEER] of var POS: x; % type int

  

constraint alldifferent(x);

  

% not {Lancer, Ezekiel}

%constraint x[Lancer] > x[Ezekiel] + 1;

  % fixed lancer > ezekiel by AT LEAST 2

constraint abs(x[Lancer] - x[Ezekiel]) > 2;

% {Quentin...Rudy} or {Lancer...Rudy}

constraint (x[Rudy] > x[Quentin]) \/ (x[Rudy] > x[Lancer]);

  

output [

"x = array1d(REINDEER,\(x));\n",

"REINDEER: \(REINDEER[1])\n",

"POS: \(POS[1])\n",

];
```


## Hidato

# 
#### Fix Assignment 1

---
# Workshop

## lb(x) 
- lowerbound function, gets the lowest possible value, useful for writing predicates
 a reflection function

 not guaranteed to get a specific value
poll - global card closed

---
## Clarification

what is the **clear** difference between the global cardinality constraints?
*Core of CP is modeling when I need the constraints I need and if there are inverse constraints that make the modeling easier*

# Permute

| 1   | 2   | 3   | 4   | 5   | 6   | 7   | 8   | 9   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     |     |     | 2   |     |     | 1   | 2   |
