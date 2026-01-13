# Lab

1. 4-connected: admissible & consistent
	1. MD: admisible & consistent 
	2. Straght Line: admissible & not consistent
	3. Octile: admin & consist
2. 8-connected gridmap
	1. MD: admiss & consistent
	2. Straight line
3. Road Network
	1. MD: 
	2. SL:
	3. OCT:

|       | MD                             | OD                            | SLD                    |
| ----- | ------------------------------ | ----------------------------- | ---------------------- |
| 4-map | Admissible, consistent         | Admissible, consistent        | Admissible, consistent |
| 8-map | admissible, not consistent     | admissible, consistent        | admissible, consistent |
| Road  | not admissible, not consistent | not admisible, not consistent | admissible, consisten  |
when looking at a 8 grid map, we can cal the MD by taking the square root f the path, for exmaple.

**note: when designing the algo, you want the function to perform as close as possible to getting to the TRUE DISTANCE. Also important to note that the largest value is the easiest to measure since it is the closest to the true distance**

**note: the power of a-star is that for each expansion, it is always optimal**
source --- --- ---> target nodes > 1

a.) A*, but a good solution. 
b) upper bound - where the num of expansions is less than or equal to the 
upper bound := worst case
c) h(n) 
d) if all are targets, then an informed search is useless

3. Asks why UCS is solveable with only using queue instead of priority queue 
	1. it can, it doesn't matter usually, but queue is generally cheaper
		1. but with A*, we *need* to use priority queue

Queue
{1,2,3,4}
pop()
1
{2,3,4}
pop()
2
{3,4}

4. Random tie breaking
	1. advantages - can allow search to complete, reduces time complexity. can remove worst case
	2. disadvantages - risk of falling into worst case and exceeding upper bound. also inconsistent.

5. Q
	a) -
	b) can use w = $\frac{C}{h(n)}$
		could set w to  large value to find the solution really quickly. 
		note: can create multiple bounds, and we know we are bounded my the linear factor of w

# Code Exercises



