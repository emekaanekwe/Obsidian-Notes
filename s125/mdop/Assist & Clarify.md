$$
\text{for some state } S^i, \text{there are agents }\ a \ \epsilon  \ A, \text{where }\ a \ \ \text{has position }\ P^i{_{nm}}
$$
---
"*Person A wants to make a music composition where each note is different, and te differences between each pair of adjacent notes are also different. Given notes 1 to n, arrange them in a sequence so that the diff betw adjacent notes are all diff
the 
Data for problem: n = num of notes*"
# NOTE: this model satisfies all constraints
```
include "globals.mzn";

int: n;
array[1..n] of var 1..n: order;
array[1..n-1] of var 1..n: diffs;

constraint forall(i in 1..n-1)(diffs[i] = abs(order[i+1] - order[i]));

constraint alldifferent(diffs);
constraint alldifferent(order);

solve satisfy;
```

---
---

# Needs Clarification

---
### How to understand this?
```
%digit_copy = m*(d-1)+c
set of int: DIGCOP = 1..1;
array[POS] of var DIGCOP: dc;
```
---
### What does this mean?
```
constraint forall(i in 1..u-1)(party[i] >= party[i+1] + (party[i] != dummy));
```
---

### What is removing the optionality problem?

```
% this removes optionality problem

var int: minhonor;

constraint minhonor = min([honor[i] | i in NEGOTIATOR where i in party]);
```

---
### What's the purpose of `|`?
```
% this removes optionality problem

var int: minhonor;

constraint minhonor = min([honor[i] | i in NEGOTIATOR where i in party]);
```
### why does this prevent summing up values twice?
`constraint sum(i,j in party where i < j)(joint[i, j]) >= 1;`

```
enum NEGOTIATOR;

NEGOTIATOR: dummy; 

  

  

int: l; % minimum party size

int: u; % maximum party size

int: m; % minimum joint ability

  

array[NEGOTIATOR] of int: honor;

array[NEGOTIATOR,NEGOTIATOR] of int: joint; 

  

% --DECISION VRIABLES--

  

% "To do this he must select a set of negotiators to make up the party"

% we need to know what set of negotitors to take

var set of NEGOTIATOR: party;

  

% "The party must be made up of between l and u negotiators."

% we need upper and lower bounds

% NOTE: these bounds can be identified using cardinality

constraint card(party) >= l;

constraint card(party) <= u;

  

% "The total negotiation strength of the party is the sum of the negotiation strength of all the pairs in the party"

% need sum of all negotiators

  

% summing up both index_ij for 2d array joint

## why does this prevent summing up values twice? ##
constraint sum(i,j in party where i < j)(joint[i, j]) >= 1;```

```


```
- check to see output
- 