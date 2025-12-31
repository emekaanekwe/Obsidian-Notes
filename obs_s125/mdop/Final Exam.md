
# Syntax That is Intuitive

## Predicates

I am often tempted to write:
```Python
int: target_sum = 15;

# cells of a grid
var 1..9: a; var 1..9: b; var 1..9: c;
var 1..9: d; var 1..9: e; var 1..9: f;

# define the row
var int: row1 = a+b+c;
var int: row2 = d+e+f;

# make sure the sum of the rows equal 15
constraint row1 = 15 /\ row2 = 15 /\ row3 = 15;

```
But *this throws an error*. 


A better way is to **use predicates**:
```python
int: dimensions = 3; 

%% Decision variables
var 1..9: a; var 1..9: b; var 1..9: c;
var 1..9: d; var 1..9: e; var 1..9: f;

%% Predicates (reusable constraints)
	%% where you put the parameter of "must equal 15"
predicate row_sum(var int: x, var int: y, var int: z) = 
    (x + y + z = 15);
    
% Main constraints
constraint all_different([a, b, c, d, e, f, g, h, i]);

% Rows
constraint row_sum(a, b, c);

```

	Predicates can contain a large block of subconstraints in them

### Condense alldifferent()

```python
predicate valid_box(array[int,int] of var int: grid, int: row_start, int: col_start) =
	# condenses into one
  all_different([grid[i,j] | i in row_start..row_start+2, j in  col_start..col_start+2]);

% Usage:
constraint valid_box(grid, 1, 1);  % Top-left box
constraint valid_box(grid, 1, 4);  % Next box (for 6x6 Sudoku)
```
### Predicates with Conditionals

```python
predicate task(var int: start_time, int: duration, var bool: has_break) =
	# here uses if-else to toggle constraints
  if has_break then
    start_time + duration + 1 <= end_time  % Adds a 1-hour break
  else
    start_time + duration <= end_time
  endif;

% Usage:
constraint task(start1, 5, true);  % Task with break
constraint task(start2, 3, false); % Task without break
```

### Condense global constraints

```python
predicate schedule_jobs(array[int] of var int: starts, 
                        array[int] of int: durations, 
                        array[int] of int: resources) =
    # using global constraint here
  cumulative(starts, durations, resources, max_resources);

% Usage:
constraint schedule_jobs(starts, [2, 3, 1], [4, 1, 2], 5);  % Max 5 resources
```

### Symmetry Break with Predicates
#### Symmetry Breaking

**Symmetry Breaking:**
	the process of eliminating duplicate solutions

Suppose graph coloring:
```python
array[1..4] of var 1..3: colors;  % 4 nodes, 3 colors
constraint colors[1] <= colors[2] <= colors[3] <= colors[4];
```

	this results in suplicated outcomes

##### Fix

#### Lexigraphical Ordering

**Lexigraphical Ordering**
	the ordering of items letter by letter, or number by number (like "apple" < "banana")

```python
array[1..4] of var 1..4: queens;  % Queen positions
# lex_leesq checks if something is lexicogrpahically related to another thing
constraint lex_lesseq(queens, [queens[i] | i in 1..4]);  % No mirrored solutions
```
constraint lex_lessq(x,  [x[i] | i in n..m]) <- array comprehension








```python
predicate ordered_colors(array[int] of var int: colors) =
  forall(i in 1..length(colors)-1) (colors[i] <= colors[i+1]);

% Usage:
constraint ordered_colors(colors);  % Eliminates equivalent color permutations
```
# Predicates
![[Pasted image 20250617122225.png]]
![[Pasted image 20250617122240.png]]

n1 is the number of positions in the array c1 taking a value in the array c2, and n2 is the number of positions in the array c2 taking a value in the array c1.

Consider the constraint:
$$
\text{common(var int: n1, var int: n2, array[int] of var int: c1, array[int] of var int: c2)}
$$
Examples:
$$
common(3,4, \ [1,9,1,5], \ [2,1,9,9,6,9])
$$
with the union of the arrays being:
$$
{1,2,5,6,9}
$$
common(3) refers to 3 positions in the first array that are in the set above where you can make it size common(4): {1,2,6,9}
### Purpose
	decision vars used when repeatedly building constraints of same form. Encapsulation

### Syntax
```Python
# like a method, pred name(param) = constraint

# set up
predicate <name>(<booloean evaluated expression>) = <constraints/rules>

# use
constraint <name>(variable)
```

### Examples
```Java
var int: x; var int: y; var int: z;
constraint x != y;
constraint y != z;
constraint x != z;

# can be condesned into

predicate diff(var int: a, var int: b, var int: c) =
    a != b /\ b != c /\ a != c;

var int: x; var int: y; var int: z;
constraint diff(x, y, z);

```

### Answer 

Consider the constraint:
$$\begin{align}
\text{common(var int: n1, var int: n2, array[int] of var int: c1, array[int] of var int: c2)}
\end{align}
$$
So:
```python
common(
	# declare two decision vars
	var int: n1,
	var int: n2,
	# assigns two arrays to variables
	array[int] of var int: c1,
	array[int] of var int: c2,
)
```

Examples:
$$
common(3,4, \ [1,9,1,5], \ [2,1,9,9,6,9]): True
$$

what are the input vars?
1. the length of the arrays
what are the decision vars?
2. the two deciding nums
3. the two vars 

linking:

n1 is the number of positions in the array c1 taking a value in the array c2, and n2 is the number of positions in the array c2 taking a value in the array c1.

n1 -> c1 when n1 = card(c1)
n2 -> c2 when n1 = card(c2)

var int: n1; 
var int: n2;
array[int] of var int: c1; 
array[int] of var int: c2;

### 1.2.a Answer

	count the # of common elements between the 2 arrays
```Python
predicate common(common(var int: n1, var int: n2, array[int] of var int: c1, array[int] of var int: c2) = 

# puts the valid ones together
n1 = sum(i in index_set(c1))
# returns the valid set of indices
(exists(j in index_set(c2))(c1[i]=c2[j])) 
/\ n2 = sum(i in index_set(c2))(exists(j in index_set(c1))(c2[i]=c2[j]));
```

#### index_set()

**index_set() def**
	returns usable indices/positions in an array

**Uses**
	do not need to hard code ranges like 1..5

# Flatzinc & Solvers

**Flatzinc**
	the assembly language of CP, allowing solvers like Geocode to run well

1. it simplifies

