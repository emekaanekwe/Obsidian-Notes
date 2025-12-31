
# Workshop

unification 

substitution - 

**unary**
s = {v_1| t_1, v_2, ...}
v|t  :=  t subtitutes the var v
	example: P(x,y) {x|A, y|B} => P(A,B)

**n-ary**

s1 = {z|g(x,y)}           s2 = {x|A, y|B, w|C, z|D}

s1s2 = {z|g(A,B), x(A,B), x|A, y|b, w|C}



# Lab

## Glossary

MGU
Proof By Refutation (for FOL)

# Styling Required by FAI


## 1
a)
$$
\exists x \exists y \ P(x,y) \ \& \ \exists y \exists x \ P(y,x) \implies False
$$

b)
$$
\exists x {Px \ \& \ Q(x)}\ \& \ \exists y \ P(y) \ \& \ \exists x \ Q(x)
$$

$$
\text{suppose:  } \\ 
\exists x ({Px \ \& \ Q(x)})\ \& \ \exists y \ P(y) \ \& \ \exists x \ Q(x) \implies True
$$
c)
$$
inclass
$$

## 2

a)  p(x, f(x)) & p(f(a), f(a))

unify{p(x, f(x)), p(f(a), f(a)}
*p entails subst(theta, p)*

subt(0,p1) & ... & subt(0,p) -> subt(0, q)

subst(0)

$$
P(x, \ f(x)) \ \& \ P(f(A),f(A))
$$
unify(p,q)  = 0 iff ubt(0,p) = sub(0,q)

Fails

P(x)
P(f(x)) & P(f(A))
	     P(f(A))

b)

$$
P(x,f(x),f(x)) \ \& \ P(y,z,f(y))
$$
$$
\theta = \{y \ -> x, \  z \ -> \ f(x)\}
$$
UNIFY(x,y) iff SUBST(p,x) = SUBT(0,q)

## 3

a) 
$$
\forall x\exists y(ISFOOD(x) \  -> \ LIKES(y) y = John )
$$
$$
\forall x (EATS(x) \ \& \ ALIVE(x) \ -> FOOD(x))
$$
$$
\exists x \exists y (EATS(x) x=Peanuts \ \& \ ALIVE(y) y =Bill )
$$
$$
\exists x (ALIVE(x) x = Bill)
$$
	resolution

*CORRECTION: if you see statements that don't "contain" variables, then just write predicate calculus form*
		 ---I don't understand this---


 c) 
 not-ISFOOD(x, y) V LIKES(x,y)
 
$$
\forall x\exists y(ISFOOD(x) \  \& \ LIKES(y) y = John )
$$

$$
\forall x (EATS(x) \ \& \ ALIVE(x) \ -> FOOD(x))
$$
$$
\exists x (LIKES(x) x = Bill)
$$
$$
\exists x (ALIVE(x) x = Bill)
$$
#### How FAI Wants You To Write It
$$
\exists x (ALIVE(x) x = Bill)
$$
instead:
				EATS(Bill, x)
				
$$
\exists x (LIKES(x) x = Bill)
$$
instead:
				ALIVE(Bill)


$$
\neg ISFOOD(x) \ \lor \ LIKES(John,y) -> EATS(x,y) \lor F(x,y)
$$

$$
\neg(\neg EATS(x) \lor \neg ALIVE(x)) \lor FOOD(X)
$$
**thus: resolution**
**we can equal Bill (assumption) y = x**
	**thus, John likes peanuts**		



$$
\exists x (ALIVE(x) x = Bill)
$$
instead:
				EATS(Bill, x)
				
$$
\exists x (LIKES(x) x = Bill)
$$
instead:
				ALIVE(Bill)


$$
\neg ALIVE(BILL)
$$
$$
\neg(\neg EATS(x) \lor \neg ALIVE(Bill)) \lor FOOD(X) \text{where } \neg (\neg EATS(x) \lor \neg ALIVE(Bill) 
$$
	refutation

(convert to CNF) 
suppose:
	not-ISFOOD V not-LIKES -> John doesn't like Peanuts

## 4
$$
vars: x .. n := natural \ nums 
$$
$$
GE(\phi, \psi) := x>=y
$$



One (overly simple) solution
E(b,p) & -E(bill, p) V F(y) 



$$
\neg LIKES(j,p)
$$
$$

$$
EATS(x) \ \& \ ALIVE(x) \ -> FOOD(x))

x >= y

