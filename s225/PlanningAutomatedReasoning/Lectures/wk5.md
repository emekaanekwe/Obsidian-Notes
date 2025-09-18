***NOTE: THI SLAB IS IMPORTANT FOR THE ASSIGNMENT Qs 2 & 3***
## Different types of temporal problems
The addition of time gives rise to several distinct path planning variants.
### Earliest arrival: 
find a path from s to t which departs immediately and minimises travel time k
### Latest departure: 
find a path that departs from s as late as possible but reaches t before a deadline time k.
 
 *Each of these objectives can appear in different problem settings:*
![[Pasted image 20250826131749.png]]

## No-Passing Property
Practitioners often assume a no passing property which guarantees arrival times are **monotonically** (a function between ordered sets that preserves or reverses the given order) **increasing**.

## Reservation Tables

![[Pasted image 20250826132139.png|500]]

	What other data structures could we use to store and efficiently check if a proposed move is collision-free?

## 1. **Sparse 3D Arrays/Matrices**
```python
# Using dictionary of dictionaries
reservation_table = {
    time_step: {
        (x, y): agent_id for occupied cells
    } for time_step in range(horizon)
}

# Query: O(1) average case
def is_free(x, y, t):
    return (x, y) not in reservation_table.get(t, {})
```


## 5. **Hash-Based Approaches with Bloom Filters**
```python
# Use multiple hash functions for probabilistic checking
from bloom_filter import BloomFilter

bloom_filters = [BloomFilter() for _ in range(time_horizon)]

def reserve(x, y, t):
    bloom_filters[t].add(f"{x},{y}")

def might_be_occupied(x, y, t):
    return f"{x},{y}" in bloom_filters[t]  # Possible false positives
```

## Performance Comparison:

| Data Structure | Memory | Query Time | Best For |
|----------------|---------|------------|----------|
| **3D Array** | O(X×Y×T) | O(1) | Small grids, short horizons |
| **Sparse Set** | O(occupied) | O(1) avg | Sparse occupancy |
| **Interval Trees** | O(cells × agents) | O(log n) | Long durations in same cell |
| **Trajectory Lists** | O(agents × path_len) | O(agents × path_len) | Few agents, long paths |
| **Bloom Filters** | O(T) | O(k) | Probabilistic, memory-constrained |
| **Hybrid** | Variable | O(1) for recent | Mixed time horizons |



This gives you O(1) checks for specific (x,y,t) while maintaining reasonable memory usage for sparse occupancy scenarios typical in pathfinding.

---

## SIPP

Safe Interval Path Planning (SIPP) is a powerful algorithm for solving single-agent pathfinding problem when the agent is confined to a graph and certain vertices/edges of this graph are blocked at certain time intervals due to dynamic obstacles that populate the environment.

#### Key Insight
represents time dimensions based on the **safe interval rule**. 
	These safe intervals are determined by analyzing the predicted trajectories of obstacles.

#### Searching
constructs a search space where nodes are these (configuration, safe interval) states. This effectively merges multiple states (one for each timestep) within a safe interval into a single, more compact representation,
#### Critiques
	problematic for narrow cases
	not effective for multi-agents



# Lab Activity 1

### 1
$$\exists x(x=k_{time}) \ where \ arrival <= k. $$
	want to find a path by backtracking with the search node. have agent come into destination, and then have agent back-propagate. 

### 2
in the 3D array implementation, it is a lot of space i order to store. So, the reservation table could be a hash function. 

### 3
SIPP is powerful because we can concatenate time into blocks. storing x,y, time step it is free, and time step that it is no longer free. 
*note: nodes are determined by intervals of time that are free*

***BEWARE TO NOT USE ALGOS LIKE CBS TO FOR ATTMEPTING TO SOLVE Q 3 IN ASSIGNMENT.***








### Readings

  
There are no textbook readings this week. Instead we recommend that you pick ONE of the following papers to read:

- Felner, A., Goldenberg, M., Sharon, G., Stern, R., Beja, T., Sturtevant, N.R., Schaeffer, J. and Holte, R. 2012. Partial-Expansion A* with Selective Node Generation. In Proceedings of AAAI  
      
    
- Standley, T.S., 2010, July. Finding Optimal Solutions to Cooperative Pathfinding Problems. In Proceedings of AAAI  
      
    
- Sharon, G., Stern, R., Felner, A. and Sturtevant, N.R., 2015. Conflict-based search for optimal multi-agent pathfinding. Artificial Intelligence, 219, pp.40-66
