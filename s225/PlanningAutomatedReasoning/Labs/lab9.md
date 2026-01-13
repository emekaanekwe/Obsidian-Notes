
### structure

you have a set of actions
you have a set of states (facts. this includes goals)
there are constants (objects)
predicates (modifiers/coefficients)


## Q1. The Spare Tire Problem
Consider the problem of *changing a flat tire*. You have accidentally **run over a bump** in the
road and consequently have a flat tire. In order to change the tire you **need to get the spare**
**tire from the trunk**, **remove the flat tire from the axle** and **put the spare tire on the axle**. **If** you do **not have a spare** tire you would have to **leave your car unattended overnight to go get one**. We assume that the car is parked in a particularly bad neighborhood, so that the effect of **leaving it overnight is that the tires disappear**.

**initial state** 

$$S=s\in S| s= (Spare, Flat, Tire, Axle, Neighborhood)$$
precondition(object, location)
	effects: not precondition()

**goal state** 
tire is replaced
**operators** 

Initialize: $Pre(x) \subset S$
change: $Swap(x)$, or $move(x,y)$, where x and y are locations and y
perimeter: 


#### 1. Formally model the problem in STRIPS. What is the initial state, goal state, and what are the possible actions with their respective preconditions and effects?
#### 2. Draw the state transition model for the problem.
#### 3. Write a pseudocode solution for the problem.


