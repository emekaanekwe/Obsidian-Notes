# Viewpoints and Representations



# Channeling

```
forall(o in OBJ)(x[i] = o in s)

```

# Multiple Modeling

	Some models have more that one way to write it 
	Model constraints as Functions: Consider  f: A -> B, with the inverse being B -> A and you can write constraints to account for both.

# Permutations

	These problems ALWAYS have two viewpoints
1. example problems
	1. traveling salesmen
	2. langford

# Array Out of Bounds Constraints

	see video 1.4.2 inverse constraints
# Modeling & Data Structures

### A sequence of variables that have certain values




















---
# Minizinc Methods

# Minizinc Order of Operations

| Symbol(s)   | Assoc. | Highest to Lowest |
| ----------- | ------ | ----------------- |
| `<->`       | left   | 1200              |
| `->`        | left   | 1100              |
| `<-`        | left   | 1100              |
| \/          | left   | 1000              |
| `xor`       | left   | 1000              |
| /\|left     | 900    |                   |
| `<`         | none   | 800               |
| `>`         | none   | 800               |
| `<=`        | none   | 800               |
| `>=`        | none   | 800               |
| `==`,       |        |                   |
| `=`         | none   | 800               |
| `!=`        | none   | 800               |
| `in`        | none   | 700               |
| `subset`    | none   | 700               |
| `superset`  | none   | 700               |
| `union`     | left   | 600               |
| `diff`      | left   | 600               |
| `symdiff`   | left   | 600               |
| `..`        | none   | 500               |
| `+`         | left   | 400               |
| `-`         | left   | 400               |
| `*`         | left   | 300               |
| `div`       | left   | 300               |
| `mod`       | left   | 300               |
| `/`         | left   | 300               |
| `intersect` | left   | 300               |
| `^`         | left   | 200               |
| `++`        | right  | 100               |
| indent      | left   | 50                |

# Output for Debugging

```Minizinc
%%% DEBUG OUTPUT
int: d = ceil(log10(m+n+1));
output[show_int(d,x[r,c]) ++ " " ++ if c = m then "\n" else "" endif | r in ROW, c in COL];
```

## Built-in Functions & Operations

Comparison Builtins
These builtins implement comparison operations.

In this section:   ~!=, ~=.
**‘!=’, ‘<’, ‘<=’, ‘=’, ‘>’, ‘>=’,**

Arithmetic

In this section: ‘/’, ‘^’, ***‘div’,*** ‘mod’, ***abs***, arg_max, arg_min, ***count***, max, max_weak, min, min_weak, pow, product, sqrt, ***sum***, ~*, ~+, ~-, ~/, ~div.

Exponential and logarithmic builtins
These builtins implement exponential and logarithmic functions.

In this section: exp, ln, log, log10, log2.

Coercions
These functions implement coercions, or channeling, between different types.

In this section: bool2float, bool2int, ceil, enum2int, floor, index2int, int2float, round, ***set2array***.

Array operations
These functions implement the basic operations on arrays.

In this section: ‘++’, ‘in’, array1d, ***array2d***, array2set, array3d, array4d, array5d, array6d, arrayXd, col, has_element, has_index, index_set, index_set_1of2, index_set_1of3, index_set_1of4, index_set_1of5, index_set_1of6, index_set_2of2, index_set_2of3, index_set_2of4, index_set_2of5, index_set_2of6, index_set_3of3, index_set_3of4, index_set_3of5, index_set_3of6, index_set_4of4, index_set_4of5, index_set_4of6, index_set_5of5, index_set_5of6, index_set_6of6, index_sets_agree, ***length***, ***reverse***, row, slice_1d, slice_2d, slice_3d, slice_4d, slice_5d, slice_6d, array_check_form.

Logical operations
Logical operations are the standard operators of Boolean logic.

In this section: ***‘->’, ‘/’, ‘<-’, ‘<->’***, ‘\/’, ‘not’, ‘xor’, bool_not, clause, ***exists, forall***, iffall, xorall.

Set operations
These functions implement the basic operations on sets.

In this section: ‘..’, ‘..<’, ‘<..’, ‘<..<’, ‘diff’, ‘***in***’, ‘intersect’, ‘subset’, ‘superset’, ‘symdiff’, ‘***union***’, ..<o, ..o, <..<o, <..o, array_intersect, array_union, ***card, max, min***, o.., o..<, o<.., o<..<, set_to_ranges.

String operations
These functions implement operations on strings.

In this section: ‘++’, concat, file_path, format, format_justify_string, join, json_array, json_object, outputJSON, outputJSONParameters, output_to_json_section, output_to_section, show, show2d, show2d_indexed, show3d, showJSON, show_float, show_int, string_length, string_split.

Functions for enums
In this section: enum_next, enum_of, enum_prev, to_enum.

## Global Constraints

4.2.2.1. All-Different and related constraints
***all_different***
all_different_except
all_different_except_0
all_disjoint
all_equal
nvalue
symmetric_all_different
4.2.2.2. Lexicographic constraints
lex2
lex2_strict
lex_chain
lex_chain_greater
lex_chain_greatereq
lex_chain_greatereq_orbitope
lex_chain_less
lex_chain_lesseq
lex_chain_lesseq_orbitope
lex_greater
lex_greatereq
lex_less
lex_lesseq
seq_precede_chain
strict_lex2
value_precede
value_precede_chain
var_perm_sym
var_sqr_sym
4.2.2.3. Sorting constraints
arg_sort
decreasing
increasing
***sort***
strictly_decreasing
strictly_increasing
4.2.2.4. Channeling constraints
int_set_channel
***inverse***
inverse_in_range
inverse_set
link_set_to_booleans
4.2.2.5. Counting constraints
among
at_least
at_most
at_most1
***count***
count_eq
count_geq
count_gt
count_leq
count_lt
count_neq
distribute
exactly
***global_cardinality***
***global_cardinality_closed***
4.2.2.6. Array-related constraints
element
member
write
writes
writes_seq
4.2.2.7. Set-related constraints
disjoint
partition_set
roots
4.2.2.8. Mathematical constraints
arg_max
arg_max_weak
arg_min
arg_min_weak
arg_val
arg_val_weak
maximum
maximum_arg
minimum
minimum_arg
piecewise_linear
range
sliding_sum
sum_pred
sum_set
4.2.2.9. Packing constraints
bin_packing
bin_packing_capa
bin_packing_load
diffn
diffn_k
diffn_nonstrict
diffn_nonstrict_k
geost
geost_bb
geost_nonoverlap_k
geost_smallest_bb
knapsack
4.2.2.10. Scheduling constraints
alternative
cumulative
cumulatives
disjunctive
disjunctive_strict
span
4.2.2.11. Graph constraints
bounded_dpath
bounded_path
circuit
connected
d_weighted_spanning_tree
dag
dconnected
dpath
dreachable
dsteiner
dtree
network_flow
network_flow_cost
path
reachable
steiner
subcircuit
subgraph
tree
weighted_spanning_tree
4.2.2.12. Extensional constraints (table, regular etc.)
cost_mdd
cost_regular
mdd
mdd_nondet
regular
regular_nfa
table
4.2.2.13. Machine learning constraints
neural_net
4.2.2.14. Deprecated constraints
at_least
at_most
exactly
***global_cardinality_low_up***
***global_cardinality_low_up_closed***
# Tips


---
---

# Convert Arrays to Sets

```
enum OBJ = {a, b, c};
array[OBJ] of var 0..1: x;

var set of OBJ: s;
```


# Inverses
```
enum PIVOT = {A, B, C}
array[PIVOT] of var POS: order;
array[POS] of var PIVOT: route;

route[1] = first;

% inverse method
inverse(order, route);


```

# Enumerate Types

A key behaviour of enumerated types is that they are automatically coerced to integers when they are used in a position expecting an integer

# Sets


# Combining Models
	Why?
### Inverse approach
	
# Bounds

## Upper Bound
	greater than or equal to sum of all el in  set
# Lower Bound
	less than or equal to sum of all el in set


### --------


### **Useful Libraries**
```minizinc
% ensure vars have all different values
include "alldifferent.mzn" 

% an import for a group of libs
include "globals.mzn"

```

### 1. **Basic Variables**
```minizinc
var int: x;     % Declare a variable x
var 1..10: y;   % Declare y in range 1 to 10
var float: z;
var int: <name> % makes a decision variable
int: <name>	% parameter variable
```

### 2. **Constraints**
```minizinc
constraint x + y <= 10;  % Constraint on x and y
constraint x != y;       % x and y must be different
```

### 3. **Objective Functions**
```minizinc
solve minimize x + y;  % Minimize sum of x and y
solve maximize x * y;  % Maximize product of x and y
```

### 4. **Arrays**
```minizinc
array[1..5] of var int: arr;  % Declare an array
constraint arr[1] + arr[2] <= 10; % Constraint on array elements
```

### 5. **Loops & Comprehensions**
```minizinc
constraint forall(i in 1..5)(arr[i] >= 0);  % All elements non-negative
constraint sum(i in 1..5)(arr[i]) <= 20;   % Sum of elements constraint
```

### 6. **Sets**
```minizinc
set of int: S = {1, 3, 5, 7};
var set of 1..10: T;
constraint 3 in T;   % 3 must be in set T
```

### 7. **Functions & Predicates**
```minizinc
predicate is_even(var int: x) = x mod 2 = 0;
constraint is_even(x);  % Apply predicate
```

### 8. **Solve Strategies**
```minizinc
solve satisfy;   % Find any feasible solution
solve minimize x; % Find optimal solution
```

### 9. **Advanced Modeling**
```minizinc
array[1..3, 1..3] of var int: matrix;
constraint forall(i, j in 1..3)(matrix[i, j] >= 0);
```

This cheatsheet provides a quick reference for essential MiniZinc features. Each section includes simple and advanced examples to assist with modeling discrete optimization problems.
