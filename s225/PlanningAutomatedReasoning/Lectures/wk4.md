
![[Pasted image 20250819103353.png]]

![[Pasted image 20250819103530.png]]

##### Consider this for Q3 of Assignment
**differential heurisitic F**
	memory-based techniques used in pathfinding algorithms like A* to improve search performance. the algorithm uses the precomputed distances to these canonical states to estimate the distance to the goal
	
$$|d(s,p)-d(t,p)|\le d(s,t)$$

**K-way Heap **

	merging k sorted arrays into a single sorted array. Upper bound of O(n log n)

where A* is $O(b^d)$

***For A-star expansions:** $O(n\ log \ n)+O(e\ log\ n)$


## Differential Heuristic
### What a Differential Heuristic Is

- In grid or graph-based pathfinding, a **differential heuristic** is a way to speed up A*-like searches.
    
- Instead of computing the full distance between **current node and goal** each time, it **precomputes distances from all nodes to a set of landmarks**.
    
- Then, the heuristic h(n)h(n)h(n) is:
    

$h(n)=max⁡landmarks l∣d(l,goal)−d(l,n)∣h(n) =$ 
$\max_{\text{landmarks } l} |d(l, goal) - d(l, n)|h(n)=$ 
$$landmarks lmax​∣d(l,goal)−d(l,n)∣$$
Where:

- $d(l,goal)d(l, goal)d(l,goal)$ = precomputed shortest distance from landmark lll to the goal
    
- $d(l,n)d(l, n)d(l,n)$ = precomputed shortest distance from landmark lll to current node nnn
    

**Intuition**: using *triangle inequality* to guarantee admissibility:  
$∣d(l,goal)−d(l,n)∣≤d(n,goal)|d(l, goal) - d(l, n)| \le d(n, goal)∣d(l,goal)−d(l,n)∣≤d(n,goal)$


## Code Tricks

get the grid dims in gridmao_h.py
```python
print("grid height", domain.height_)
print("grid width", domain.width_)
```

grid map implementation at `lib-piglet/domains/gridmap.py`

