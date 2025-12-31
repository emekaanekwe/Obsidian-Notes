# Looking Up Possible Values of Vars

	identify lower and upper bounds of a var

```Java

var int: x

constraint x >= 1 /\ x <= 10;

lb(x)
```
```
## Purpose

allows you to check the properties of vars in the model

## Syntax

```Java
array[]
```

# Predicates

	Note: A predicate that encapsulates the common Bool expressions in the constraints is the following:

## Purpose

	puts repeated constraints into one constraint

	allows you to make custom alldifferent() constraints
## Analogy

 Imagine you're baking cookies. A predicate is like a recipe step - *"cream butter and sugar"* might be one predicate, *"mix dry ingredients"* another. You can **reuse** these steps in different recipes.
## Types

Predicates have type **var bool** 
can be used anywhere a var bool can be used

## Syntax

```predicate name(param inputs) = constraint body```


```Java

enum ENUM;
set of ENUM: B;
set of ENUM: C;
set of ENUM: D;

var set of SET: A;
constraint card(A intersect B) >= 1;
constraint card(A intersect C) >= 2;
constraint card(A intersect D) >= 2;
constraint card(A) = 6;
```

	can transform to
```Java

predicate form(var set of ENUM: x) = 
card(A intersect B) >= 1 
/\  card(A intersect C) >= 2 
/\ card(A intersect D) >= 2 
/\ card(A) = 6;

```

# Let-in

## Purpose

allows you to **define** **local** **variables** within the building of a parameter/constraint

--Remember that the var will *only* be used within the in bracket
## Analogy

Imagine you're solving a math problem and say "Let x be 5, then calculate x + 3". The `let` part defines x, and the `in` part uses it.

## Syntax
```let {define loca var} in (expression using var)```

Example
```Java
enum ENUM;


```
```Java
constraint let {
    var int: radius = 5;
    var float: pi = 3.14159;
    var float: area = pi * radius * radius;
} in (
    total_area >= area
);
```