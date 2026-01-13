
# Most Separated

int: <mark class="green">n</mark>;                % number of points
set of int: <mark class="purple">P</mark> = 1..<mark class="green">n</mark>;  % points 
array[P,P] of int: d;  % distance matrix
array[P] of int: v;    % value matrix
int: <mark class="orange">k</mark>;                % size limit for chosen set 
int: l;                % average distance lower bound 


set of int: <mark class="yellow">P0</mark> = 0..<mark class="green">n</mark>; % just give us the num of points to solver
array[1..<mark class="orange">k</mark>] of var <mark class="yellow">P0</mark>: Sx; % defines the card of set with the possible vals being <mark class="yellow">P0</mark>


var int: value;

solve maximize value;




## Scenario

 Which declaration is best to represent a set of numbers from 1..1000 of cardinality between 99 and 150?

array[1..1000] of var 99..150: x;

var set of 99..150: x;

array[99..150] of var 1..1000: x; 

array[1..1000] of var bool: x; <- this does not satisfy cardinality constraint necessarily

array[99..150] of var bool: x;

if we have bounded cardinality set, we need a dummy var to control what we will consider*

# Team Select

## Confusions

1. 
```python
set of PLAYER: xavier;

set of PLAYER: Zena;

set of PLAYER: yuri;
```
why are we making a set for each captain. can we not just use the array[captain, pos]?
2. how do you know when to (use x intersect y)? That option never came to mind when doing the problem.
```python
  

enum PLAYER = {Ant, Bee, Chu, Deb, Eve, Fin, Ged, Hel, Ila, Jan, Kim};

  

enum CAPTAIN = {Xavier, Yuri, Zena};

  

set of PLAYER: goalies = {Ant, Bee};

set of PLAYER: offense = {Chu, Deb, Eve, Fin};

set of PLAYER: defense = {Ged, Hel, Ila, Jan, Kim};

  

% make set for each captain

var set of PLAYER: xavier;

var set of PLAYER: zena;

var set of PLAYER: yuri;

  

set of int: POS = 1..6;

  

array[CAPTAIN,POS] of var PLAYER: team;

  

% higher num compare bc of reserve 

constraint card(xavier) = 6 /\ card(xavier intersect goalies) >=1 /\ card(xavier intersect goalies) <= 2;

constraint card(xavier intersect offense) <= 3 /\ card(xavier intersect offense) >= 2;

constraint card(xavier intersect defense) <= 3 /\ card(xavier intersect defense) >=3;

  

constraint card(yuri) = 6 /\ card(yuri intersect goalies) >=1 /\ card(yuri intersect goalies) <= 2;

constraint card(yuri intersect offense) >=2 /\ card(yuri intersect offense) <= 3 /\ card(yuri intersect defense) >=2 /\ card(yuri intersect defense) <=3;

  

constraint card(zena) = 6 /\ card(zena intersect goalies) >=1 /\ card(zena intersect goalies) <= 2;

constraint card(zena intersect offense) >= 2 /\ card(zena intersect offense) <= 3;

constraint card(zena intersect defense) >=2 /\ card(zena intersect defense) <=3;

  

constraint card(xavier intersect zena) <= 2 /\ card(xavier intersect yuri) <=2;

  

% xav xena <= 2 same members

% xav yuri <= 2 same members

  

solve satisfy;

output ["xavier = \(xavier);\nyuri = \(yuri);\nzena = \(zena);\n"];
```
