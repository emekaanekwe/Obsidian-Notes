

# Option Types (IGNORE)

		describes the equivalence of a structure of code

```
fotall(i in x)(p[x >= 0]) 

forall([if i in x then p[x] >= 0 else true endif) | in 1..10)
```


# New Techniques

## configurations
	output solving statistics
	output objective statistics

```
table([x,y,z], [| 0,0,0| 1,2,3 | 4,2,0|])
	% versitile constraint
	% used to represent path in a graph

```



## Missing Solutions

	table([x,y,z], [| 0,0,0| 1,2,3 | 4,2,0|])
	
	when path is unknown, make educated guess on max possible length of path

```
int: maxstep;
var int: step; % the ACTUAL num of steps
set of int: STEP = 1..maxstep;
array[STEP] of var NODE: path;

% add an edge at destination (keep looking, but stay at a index)
```

	can use a table constraint to force 2 consecutive path positions to be connected by an edge 

	can add a soluton to the problem by constructing small instance of problem.

	can also "relax" the constraints.

	for example, flattening the edges and ordering


### Using a Sliding Sum 
	sliding_sum(n, m)

	enforces properties of a seq, such as for some sequence S, the summation of every subsequence is greater or less than lowerB and less than or equal to the upperB

```
x = [1,4,2,0,0,3,4] 
	% cut each piece into 4
		% [1,4,2,0], [4,2,0,0], [0,0,3,4]

y = [1,4,3,0,1,0,2] % cut each piece into 4

silding_sum(4,8,4,x)

% here, y fails because there is an instance that sums up to 3

```

## Resting?

	"not to be concerned with now"


## Too Many Solutions

	when problems has many, even though the constraint specifies that it shouldn't show. they key is precendence of operators
		fixes: add another constraint specification, or add a specific objective funct

```

Where does MiniZinc put (implicit) parentheses on the following expression?

a <-> b \/ c -> d = e

1. ((a <-> b) \/ c) -> (d = e)

2. a <-> ((b \/ c) -> (d = e)) % this one

3. (a <-> (b \/ c)) -> (d = e)

4. ((a <->) b) \/ (c (-> d) = e)
```


## Assertions
		return: bool
		
	a check if whether or not a param holds
		can use within forall() constraints and can output as well
	
*"Any time that you make assumptions about your model, add assertions"*

## Trace

	return: outputs val of string-exp
```
trace(<string-exp>, <exp>)
```


# Problem Paradigms
## Cavalry Wedge Problem
	"involves determining the optimal formation of cavalry units to maximize their impact on an enemy formation, typically represented as a triangle or wedge shape"

# Workshop

initial output
price = [12, 25, 17, 12, 9];

buy = [1, 2];

free = [1, 1];

how = [1, 1, 1, 1, 1];

cost = 0;

price of pizza = [12, 25, 17, 12, 9];

buy voucher = [1, 2];

free vouchers = [1, 1];

how = [2, 1, 0, 2, -1];

cost = 26;