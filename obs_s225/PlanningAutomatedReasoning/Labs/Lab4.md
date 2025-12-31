## 1. "At what point does it stop paying off to add more pivots?"

  Theoretical Analysis:
  - 2 pivots: Captures diagonal relationships across the grid
  - 4 pivots: Adds horizontal/vertical extremes, significant improvement
  - 8+ pivots: Diminishing returns due to:
    - Computational overhead: More distance calculations per heuristic call
    - Memory overhead: More precomputed distances to store
    - Limited improvement: Additional pivots in grid interiors provide less
   discriminating power

  Expected breakpoint: Around 4-8 pivots for typical grid sizes. Beyond
  this, the computational cost outweighs heuristic accuracy gains.

## 2. Differential vs Perfect-Distance Heuristic Comparison

  Perfect-Distance Heuristic Advantages:
  - Optimal: Always returns true shortest path distance
  - Maximum informativeness: h(n) = h*(n) for all states
  - Guaranteed optimality: A* with perfect heuristic expands minimal nodes

  Differential Heuristic Limitations:
  - Underestimation: Often h(n) << h*(n), especially in grid centers
  - Pivot dependency: Accuracy depends on pivot placement relative to
  start/goal
  - Variable quality: Better near grid boundaries, weaker in interior
  regions

  Expected Results:
  - Perfect heuristic wins decisively in node expansions and search time
  - Average DH error: Likely 30-60% of optimal distance for typical grid
  problems
  - DH error pattern: Higher error for start positions equidistant from
  multiple pivots

  Why DH has high error:
  - Triangle inequality gives loose lower bounds
  - Grid centers are far from corner pivots
  - Multiple paths between pivots dilute discriminating power

  The perfect heuristic should significantly outperform differential
  heuristic in all metrics except preprocessing time.
