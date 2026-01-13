
## arrays


```
array[1..3] of var 1..n: y; # boundary set to the length of the array (this case length of y is 3)
```
items:  n  n  n
array : __ __ __


```
array[1..n] of var 1..3: z; # boundary set to what the variables can be (this case the values array can take are only 1,2 or 3)
```
items: 1/2/3    1/2/3     1/2/3
array:   ___        ___          ___     ___ ...  n spots


## enumerated types

```
enum CHARS = {A, B, C};
```
most basic structure, no conversions

```
var CHARS: chars;
```
assign to a variable

```
array[1..n] of var CHARS: word;
```


```

array[1..3] of var DAR: schedule;
```
```
enum E = {A, B, C};
```
enumerated types are discrete symbolic values

```
enum KIND = { E, S, O };
array[ESSTANZA] of KIND: kind = [ E | i in 1..k ] ++ [ S | i in 1..l ];
```
# Convert Arrays to Sets

```
enum OBJ = {a, b, c};
array[OBJ] of var 0..1: x;

var set of OBJ: s;

## bijection (inverse)






# Constraint Modeling