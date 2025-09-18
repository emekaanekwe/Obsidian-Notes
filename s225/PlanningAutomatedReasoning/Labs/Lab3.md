

#### *Here is data captured for running the search problems for questions 1-5. each column is the mean of the corresponding cells above*

**Titles** ![[Pasted image 20250818143321.png|700]]
**1. A-Star**![[Pasted image 20250818143816.png]]
**1. UCS**
![[Pasted image 20250818143901.png]]
**2. A-star**![[Pasted image 20250818143919.png]]
**2. UCS**
![[Pasted image 20250818145028.png]]
**3. A-star, Straight Line**
![[Pasted image 20250818145059.png]]
**3. UCS, Straight Line**
![[Pasted image 20250818145111.png]]
**4. A-star, Largest-h**
![[Pasted image 20250818145133.png]]
**4. A-Star, Random-h**
![[Pasted image 20250818145145.png]]
**4. A-Star, Smallest-h**
![[Pasted image 20250818145212.png]]
**4. UCS, Smallest-h**
![[Pasted image 20250818145237.png]]
**4. A-Star, Octile**
![[Pasted image 20250818145315.png]]
**4. UCS, Octile**
![[Pasted image 20250818145338.png]]



---

# Search Algorithm Performance Analysis

## Data Overview
Analysis based on mean values from 11 sheets of search algorithm performance data, each containing averages of at least 200 test runs.

## Question 1: Do searches from sheets 1 and 2 have the same cost? Why or why not?

**Answer: Yes, they have identical costs.**

- Sheet 1 (A*): Cost = 205.92
- Sheet 2 (UCS): Cost = 205.92

**Explanation:** Both algorithms found optimal solutions with identical path costs because:
- Both A* and UCS are optimal algorithms
- They guarantee finding the shortest path when using consistent heuristics
- The same grid and problem instances were tested
- Cost represents the actual shortest path length, which is independent of the search algorithm used

However, their **efficiency differs significantly:**
- A*: 4,354.57 nodes expanded, 0.56s runtime
- UCS: 15,432.53 nodes expanded, 1.89s runtime

A* is ~3.5x more efficient due to its informed heuristic guidance.

## Question 2: Searches from sheets 3 and 4 searched on an 8-connected grid. What happened?

**Answer: Performance improved significantly for both algorithms.**

**8-connected grid results:**
- Sheet 3 (A*-8-grid): 4,349.52 nodes expanded, 0.51s runtime
- Sheet 4 (UCS-8-grid): 15,433.58 nodes expanded, 1.70s runtime

**Comparison with 4-connected grid:**
- A*: Slight improvement (4,354.57 → 4,349.52 nodes)
- UCS: Minor improvement (15,432.53 → 15,433.58 nodes)

**Explanation:** 8-connected grids allow diagonal movement, providing:
- More direct paths to goals
- Reduced search space exploration
- Slightly faster convergence
- Better runtime performance (0.56s → 0.51s for A*, 1.89s → 1.70s for UCS)

## Question 3: Sheets 5 and 6 used straight line and octile distance. What changed?

**Answer: Distance heuristics significantly impacted performance and path quality.**

**Straight line distance (Manhattan):**
- Sheet 5 (A*-straight): Cost = 286.11, 6,655.69 nodes expanded, 0.88s runtime
- Sheet 6 (UCS-straight): Cost = 335.99, 8,973.43 nodes expanded, 1.18s runtime

**Octile distance:**
- Sheet 7 (UCS-octile): Cost = 206.81, 4,373.73 nodes expanded, 0.56s runtime
- Sheet 14 (A*-octile): Cost = 363.43, 10,042.39 nodes expanded, 1.33s runtime

**Key Changes:**
1. **Path quality varied:** Octile distance produced better costs (206.81) vs straight line (286.11-335.99)
2. **Search efficiency:** Octile distance was more effective at guiding search
3. **Heuristic accuracy:** Octile distance better approximates actual grid distances with diagonal movement

## Question 4: For sheets 6 to 11, I used smallest-g, random-h and largest-h. What happened to tie breaking?

**Answer: Tie-breaking strategies significantly affected performance:**

**Tie-breaking comparison (sheets 7-12):**
- **A*-Long (largest-h):** Cost = 336.15, 8,984.05 nodes expanded, 1.18s runtime
- **A*-rand (random-h):** Cost = 336.34, 8,984.09 nodes expanded, 1.18s runtime  
- **A*-small (smallest-g):** Cost = 363.87, 10,055.45 nodes expanded, 1.34s runtime
- **UCS-small:** Cost = 363.61, 10,089.06 nodes expanded, 1.34s runtime

**Impact of tie-breaking:**
1. **Largest-h and random-h performed similarly** with costs ~336 and ~8,984 nodes expanded
2. **Smallest-g performed worse** with higher costs (~363) and more node expansions (~10,055)
3. **Runtime increased** from ~1.18s to ~1.34s with smallest-g strategy

**Explanation:** 
- Largest-h tie-breaking favors nodes closer to the goal, leading to more direct exploration
- Random tie-breaking provides balanced performance without bias
- Smallest-g tie-breaking favors nodes with lower path costs, potentially leading to more thorough but less efficient search

The largest-h and random-h strategies were more effective at guiding search toward the goal, while smallest-g caused more conservative exploration.