
the lower bound is the least f cost of any currently open node and is increased when all noced with the smallest f-cost are expanded. 





This week we look at the Problem Model: an approach that allows us to describe sequential decision-making problems in general using tools such as STRIPS and the Planning Domain Definition Language (PDDL). We will look at key features of these languages, study their syntax, evaluate examples and compare between them to understand their relative strengths.

### **Learning Objectives:**

- Learn the syntax and key features of PDDL and STRIPS
- Understand how to formulate problems using the PDDL and STRIPS planning languages
- Evaluate a toy problem example and a real world example
- Compare the advantages of PDDL the syntax to the STRIPS representation

# Article Summaries

---

# **Conceptual Map**

- **Anytime algorithms (ARA*, AHS):** Optimize _time-quality tradeoff_.
    
- **CPDs:** Optimize _speed of heuristic evaluation via precomputation_.
    
- **Theta*:** Optimize _path realism by relaxing grid constraints_.
    
- **Navmeshes:** Optimize _path optimality by moving to continuous space_.

## 1. Hansen & Zhou (2007) – _Anytime Heuristic Search_

**Summary**

- Introduces the concept of _anytime algorithms_ for heuristic search.
    
- An anytime search returns a valid solution quickly, then continues to improve solution quality as time allows.
    
- Key algorithms: Anytime Repairing A* (ARA*), Anytime Heuristic Search (AHS), etc.
    
- Focus: balancing _solution quality_ vs _time constraints_.
    
- Applications: robotics, planning, and real-time systems where timely but improvable solutions are critical.
    

**Contribution**

- Established a foundational framework for balancing _optimality_ with _practicality_ in heuristic search.
    
- Emphasized _flexibility under uncertainty_ (time/resources).
    

---

## 2. Bono et al. (2019) – _Path Planning with CPD Heuristics_

**Summary**

- Introduces _Compressed Path Databases (CPDs)_ as heuristics.
    
- CPDs precompute and store optimal paths between regions, then use this compressed knowledge for fast online queries.
    
- Strength: tradeoff between precomputation space and extremely fast lookups during search.
    
- Demonstrates improvements over classical heuristics like Manhattan distance or pattern databases.
    
- Target domain: grid and road networks, where repeated pathfinding queries are common.
    

**Contribution**

- Shows how _precomputation + compression_ can yield near-instant heuristic evaluations.
    
- More efficient than traditional heuristics for large-scale, repeated planning problems.
    

---

## 3. Nash et al. (2007) – _Theta*_: Any-Angle Path Planning

**Summary**

- Extends A* to _any-angle pathfinding_ on grids (agents can move in continuous straight lines, not just along grid edges).
    
- Uses line-of-sight checks to connect nodes directly if possible, reducing path length.
    
- Produces shorter, more natural-looking paths than A* while maintaining efficiency.
    
- Important for robotics, navigation meshes, and simulations where agents should move realistically.
    

**Contribution**

- Bridged the gap between discrete (grid-based) and continuous (geometric) planning.
    
- A practical middle ground: _efficient like A_ but produces smoother, more realistic paths*.
    

---

## 4. Cui et al. (2017) – _Compromise-free Pathfinding on Navigation Meshes_

**Summary**

- Works with _navigation meshes (navmeshes)_ rather than grids.
    
- Proposes algorithms that avoid the “compromise” of discretizing continuous space.
    
- Guarantees paths that are both _shortest_ and _valid_, without trade-offs common in grid-based approaches.
    
- Efficiently handles complex polygonal environments.
    
- Applications: video games, robotics, 3D simulations.
    

**Contribution**

- Moves beyond grids entirely by embracing _continuous, geometry-based search_.
    
- Stronger guarantees of optimality than grid-based or approximate methods.
    

---

## **Comparison Across the Four Papers**

| **Aspect**       | **Hansen & Zhou (2007)**                         | **Bono et al. (2019)**                              | **Nash et al. (2007)**              | **Cui et al. (2017)**                       |
| ---------------- | ------------------------------------------------ | --------------------------------------------------- | ----------------------------------- | ------------------------------------------- |
| **Problem**      | Real-time decision making under time constraints | Fast repeated path queries with better heuristics   | Natural, efficient paths in grids   | Optimal, compromise-free paths in navmeshes |
| **Core Idea**    | Anytime search (improve solution over time)      | Precomputed compressed path databases as heuristics | Any-angle grid pathfinding (Theta*) | Geometry-based continuous pathfinding       |
| **Domain**       | General heuristic search                         | Large-scale, repetitive pathfinding                 | Grids, robotics, navigation         | Continuous environments, games, robotics    |
| **Tradeoff**     | Quality vs time                                  | Preprocessing space vs fast queries                 | Grid efficiency vs realism          | Complexity vs accuracy                      |
| **Contribution** | Framework for anytime heuristic search           | New heuristic class (CPD)                           | Practical any-angle algorithm       | True optimality in continuous spaces        |

---

## **Skeptical, Forward-Looking Take**

- **Hansen & Zhou** remain foundational, but anytime search now faces competition from neural-based heuristics and reinforcement learning planners that adapt online without hand-crafted tradeoffs.
    
- **Bono et al. (CPDs)** reflect a “data-driven heuristic” approach—anticipating modern trends where _learned heuristics_ or _policy networks_ replace static databases.
    
- **Nash et al. (Theta*)** was an early step toward realism in motion planning, but newer sampling/planning hybrids (e.g., RRT*, PRM*) and learning-based planners have gone further.
    
- **Cui et al.** arguably represent the _future direction_: abandoning grids altogether in favor of geometry-aware, mesh-based, or neural implicit representations of space.
    

---

👉 In short:

- Hansen & Zhou focus on _time-quality tradeoffs_.
    
- Bono et al. focus on _fast heuristics via precomputation_.
    
- Nash et al. focus on _path realism in grids_.
    
- Cui et al. focus on _continuous, optimal geometry-based planning_.
    

---

# 1. **Anytime Repairing A*** (ARA*) – _Hansen & Zhou, 2007_

**Core Idea**

- Start with a _fast, suboptimal_ solution.
    
- Iteratively improve it until time runs out or optimality is reached.
    
- Uses a _weighted A*_ search (inflating heuristic by factor ε > 1).
    

**Structure**

1. **Initialization**: Run weighted A* with weight ε (e.g., 2.5).
    
    - $f(n) = g(n) + \varepsilon \cdot h(n)$
        
    - Returns a quick, but possibly suboptimal path.
        
2. **Improvement Loop**:
    
    - Reduce ε gradually (e.g., 2.5 → 2.0 → 1.5 → 1.0).
        
    - Reuse previous search results to avoid starting from scratch.
        
    - Refine path towards optimal.
        
3. **Termination**: Stop when ε = 1 (path is optimal) or time runs out.
    

**Key Concept**: _Bounded suboptimality_ — at any point, ARA* guarantees a solution within factor ε of optimal.

---

# 2. **Anytime Heuristic Search (AHS)** – _Hansen & Zhou, 2007_

**Core Idea**

- Generalization of ARA*: not tied to weighted A*.
    
- Provides an anytime framework where search algorithms (A*, IDA*, etc.) can be adapted to anytime use.
    

**Structure**

1. **Initial Solution**: Quickly generate a feasible solution (not necessarily optimal).
    
2. **Improvement Cycle**:
    
    - Continue search beyond the first solution.
        
    - Use _solution-cost bounds_ (upper = current solution, lower = best heuristic estimate).
        
    - Narrow the gap iteratively.
        
3. **Interruptibility**: At any interruption, algorithm returns best-so-far solution.
    

**Key Concept**: AHS emphasizes _solution quality improvement over time_ rather than fixed ε guarantees.

---

# 3. **Compressed Path Databases (CPDs)** – _Bono et al., 2019_

**Core Idea**

- Precompute all-pairs shortest paths but compress the storage.
    
- Online search = instant heuristic lookup (fast guidance).
    

**Structure**

1. **Offline Precomputation**:
    
    - Partition graph into regions (e.g., grid cells, road network segments).
        
    - Compute optimal path “choices” between regions.
        
    - Store compressed representation (not full path, but decision rules).
        
2. **Compression Techniques**:
    
    - Remove redundancies (many paths share common prefixes/suffixes).
        
    - Encode only _first move_ toward destination.
        
3. **Online Query**:
    
    - Given start & goal, decompress path decisions step by step.
        
    - Lookup is _O(1)_ per step, much faster than running A*.
        

**Key Concept**: CPDs transform heuristics from _computed online_ to _looked-up offline_.

---

# 4. **A* → Any-Angle Pathfinding (Theta*)** – _Nash et al., 2007_

**Core Idea**

- A* is restricted to grid edges (N/E/S/W or diagonals).
    
- Theta* allows straight-line (“any-angle”) shortcuts if there’s line-of-sight.
    
- Produces shorter, more natural-looking paths.
    

**Structure**

1. **Like A***:
    
    - $f(n) = g(n) + h(n)$ where $h(n)$ = Euclidean distance.
        
    - Expands nodes from start to goal.
        
2. **Line-of-Sight Relaxation**:
    
    - If parent of node $p$ has line-of-sight to successor $s$, then:  
        $g(s) = g(p) + \text{dist}(p, s)$
        
    - Instead of stepping grid-cell by grid-cell, it “cuts corners”.
        
3. **Result**: Path is closer to Euclidean optimal than A* grid path.
    

**Key Concept**: _Visibility-based relaxation_ of grid constraints.

---

# 5. **Navigation Meshes (Navmeshes)** – _Cui et al., 2017_

**Core Idea**

- Represent traversable space with polygons (triangles, convex regions) instead of grids.
    
- Agents can move freely inside each polygon, not restricted to discrete cells.
    

**Structure**

1. **Mesh Construction**:
    
    - Environment divided into convex polygons (triangulation, Voronoi, etc.).
        
    - Each polygon is “free space” — agent can move anywhere inside.
        
2. **Graph Abstraction**:
    
    - Navmesh polygons are nodes, adjacency between polygons = edges.
        
    - Search runs on this graph.
        
3. **Pathfinding**:
    
    - A*-like search finds polygon sequence from start to goal.
        
    - Then, compute _string-pulling_ / _funnel algorithm_ to create smooth continuous path.
        
4. **Compromise-Free Claim**:
    
    - Unlike grids (approximate paths), navmesh ensures _true shortest path within continuous space_.
        

**Key Concept**: Continuous, geometry-based planning → no discretization artifacts.

---


### **Activity 1 Questions and Exercises**

**1. What are sequential decision-making problems? How are they different from one-shot decision-making problems?**

**Answer:**
*Sequential decision-making problems* - agent must make a sequence of decisions over time to achieve a long-term goal. each decision (or action) influences the subsequent state of the world, which in turn affects the options and outcomes of future decisions. 
	**Goal**  find a **policy** that maximizes some notion of cumulative reward or utility.

*One-shot decision-making*  involves a single, isolated decision where the outcome is independent of any past or future decisions 

---

**2. What are the key components of a sequential decision-making problem?**

**Answer:**
The key components are formally defined in the Markov Decision Process (MDP) framework:
1.  **States (S):** A set of all possible situations or configurations the agent and environment can be in.
2.  **Actions (A):** A set of all possible moves or decisions the agent can make from a given state.
3.  **Transition Model (T(s, a, s')):** The probability that taking action `a` in state `s` will lead to state `s'`. This defines the dynamics of the environment.
4.  **Reward Function (R(s, a, s')):** The immediate reward (or cost) the agent receives for transitioning from state `s` to state `s'` by taking action `a`.
5.  **Discount Factor (γ):** A number between 0 and 1 that determines the present value of future rewards. It quantifies the agent's preference for immediate rewards over long-term ones.
6.  **Policy (π(s)):** A function that specifies what action the agent will take in any given state. The goal is to find the optimal policy (π*).

---

**3. What is the Planning Domain Definition Language (PDDL)? What is its purpose?**

**Answer:**
The Planning Domain Definition Language (PDDL) is a standardized formal language used to model and define **planning problems**. Its purpose is to separate the general logic of a domain (the "rules of the world") from a specific problem instance within that domain.

This separation allows for:
*   **Generality:** A single planning **domain** file (e.g., a "Blocksworld" domain) can be reused for countless different specific **problem** files (e.g., different initial tower configurations and goal states).
*   **Abstraction:** It allows AI researchers and practitioners to focus on high-level task specification (what needs to be achieved) without worrying about the low-level implementation details of the planning algorithm.
*   **Standardization:** It provides a common language for the international planning competition, allowing different automated planners to be tested and compared on the same problems.

---

**4. What are the key components of a PDDL domain description?**

**Answer:**
A PDDL domain description (`domain.pddl`) contains the following key components:
1.  **(`define (domain <name>)`)**: Declares the name of the domain.
2.  **(`:requirements`)**: Specifies which optional PDDL features are needed (e.g., `:strips`, `:typing`, `:equality`).
3.  **(`:types`)**: (Optional) Defines a hierarchy of object types in the domain.
4.  **(`:predicates`)**: Defines the logical predicates that describe the state of the world. These are Boolean properties that can be true or false (e.g., `(on ?x ?y)`, `(clear ?x)`).
5.  **(`:action`)**: For each action the agent can take, it defines:
    *   **`:parameters`**: The variables the action operates on.
    *   **`:precondition`**: A logical expression that must be true for the action to be legally executed.
    *   **`:effect`**: A logical expression that describes how the state changes after the action is executed (what becomes true and what becomes false).

---

**5. What are the key components of a PDDL problem description?**

**Answer:**
A PDDL problem description (`problem.pddl`) contains the following key components:
1.  **(`define (problem <name>)`)**: Declares the name of the problem.
2.  **(`:domain <domain_name>)`)**: Links this problem to its corresponding domain file.
3.  **(`:objects`)**: Lists all the specific objects that exist in this problem instance and their types (e.g., `a b c - block`).
4.  **(`:init`)**: Defines the initial state of the world by listing all the predicates that are true at the start. Everything not listed is assumed false.
5.  **(`:goal`)**: Defines the goal state that the planner must achieve. It is a logical expression that must be satisfied for the problem to be solved.

---

**6. What is the Stanford Research Institute Problem Solver (STRIPS)? How is it related to PDDL?**

**Answer:**
The Stanford Research Institute Problem Solver (STRIPS) is one of the **earliest automated planning systems**. Its historical significance lies in its underlying formal representation for actions, which became a foundation for modern planning.

**Relation to PDDL:**
PDDL is a direct descendant and formalization of the STRIPS representation. The most basic version of PDDL is called "STRIPS PDDL," meaning it uses the core ideas from the STRIPS system:
*   **States** are represented as sets of grounded predicates (facts).
*   **Actions** are defined by their preconditions (a set of facts that must be present) and effects (a set of facts to add and a set of facts to delete).
This "add-list" and "delete-list" model is the core innovation of STRIPS and is a fundamental part of the PDDL standard. In essence, **PDDL is the modern, standardized language that grew out of the ideas pioneered by the STRIPS planner.**

---

**7. What are the key components of a STRIPS action description?**

**Answer:**
A STRIPS action description consists of three key components:
1.  **Action Signature:** The name of the action and its parameters (variables).
2.  **Precondition (PRE):** A conjunction of positive literals (predicates) that must all be true in the current state for the action to be applicable.
3.  **Effect (EFF):** Describes how the action changes the state. It is divided into:
    *   **Add List (ADD):** A set of literals that become true after the action is executed. These are added to the state.
    *   **Delete List (DEL):** A set of literals that become false after the action is executed. These are removed from the state.

**Example (from Blocksworld):**
```
Action: PickUp(b)
PRE: clear(b), ontable(b), handempty
DEL: ontable(b), handempty
ADD: holding(b)
```

---

**8. Describe the basic idea behind the STRIPS planning algorithm.**

**Answer:**
The basic idea behind the STRIPS planning algorithm is **goal stack planning** and **backward search (regression planning)**.

1.  **Goal Stack:** The planner maintains a stack of subgoals (predicates that need to be true) that need to be achieved.
2.  **Backward Search (Regression):** Instead of starting from the initial state and simulating actions forward (which can lead to a huge search space), STRIPS starts from the **goal state** and works backward.
3.  **Process:**
    *   It looks at the current goal.
    *   It selects an action whose **effect** (add list) can achieve that goal.
    *   It then takes the **precondition** of that action and sets it as a new subgoal to be achieved *before* the action can be applied.
    *   This process continues recursively, adding new subgoals to the stack until all subgoals are satisfied by the initial state.
4.  **Linear Plan:** Once a path is found from the initial state to the goal state by chaining these actions together through their preconditions and effects, the plan (the sequence of actions) is popped from the stack in the correct order.

This approach is efficient because it focuses the search only on actions that are directly relevant to achieving the final goal.

---

# Lab Activity 1

### **Activity 1 Questions and Exercises**

##### 1. Pre-computed heuristics work well in dynamic settings where costs can never fall below a given floor. 
**What happens to the properties of these heuristics if fixed obstacles suddenly disappear? Is the search still complete?**

####  **Answer:** 

***think of a case in which it was node N was reachable and another case in which it is not*** 
- The heuristic values become **inadmissible**, overestimate the cost of path. 
- Search is likely to remain **complete** if the underlying algorithm (e.g., A*) is complete,
- It is no longer **optimal** because the heuristic is no longer admissible, which is a requirement for A* to guarantee an optimal solution.

##### 2. CPD Search terminates and returns an optimal path when the minimum f-value on the OPEN list is >= UB.  (Anytime Algorithms)
**How could we modify the termination condition to instead return a bounded suboptimal solution less than w * Optimal? What if we wanted a solution not more than some value delta+Optimal?**

*CPD: use actual cost of solution as lower bound*
*   **Answer:**
    *   **For bounded suboptimal solution (w * Optimal) is greater or equal to the optimal cost C*:** The termination condition could be modified to `OPEN >= UB/w`
    *   **For additive bound is simply delta + above answer:** The termination condition could be modified to `OPEN >= UB - delta`. This ensures that when we terminate, `UB ≤ OPEN + delta ≤ C* + delta`.

##### Anytime Weighted A* (AWA*) and other anytime algorithms rely on the process of the Upper Bound (UB) and Lower Bound (LB) crossing in order to terminate search, and to prove optimality. What can we say about this process:

**a. When can we update the Upper Bound during search?**
*   **Answer:** The Upper Bound (UB) can be updated **whenever a new goal state is found** with a path cost (`g_{goal}`) that is lower than the current UB. The *UB represents the cost of the best solution found so far.*


**b. Does the LB change during search? Why or why not?—justify your answer.**
*   **Answer:** The LB **in fact does change** **Lower Bound (LB) can change**. The LB is typically the minimum `f`-value on the OPEN list (`min_f`), which is a lower bound on the cost of any solution that might be found by expanding the current frontier. As nodes are expanded and new nodes are generated with different `f = g + h` values, the minimum value on OPEN (`min_f`) can increase, thus raising the lower bound.

**c. In AWA*, why can nodes from CLOSED return to OPEN?**
*   **Answer:** Nodes can be **re-opened** (moved from CLOSED back to OPEN) when a better path to them is found. In AWA*, this happens specifically when the Upper Bound (UB) is improved (i.e., a cheaper solution is found). A node's `g-value` might have been based on a previous, higher UB. When UB decreases, the `f = g + w * h` value for a node might become lower than its previous value, making it promising again. Since the search is weighted, a node that was previously considered suboptimal might now lie on a path that could potentially lead to a better solution than the new UB, warranting re-expansion.

**4. AWA* terminates when the OPEN list is exhausted or when time runs out. Why is it necessary to drain the OPEN list before optimality can be proved? HINT: Remember each expanded node tracks two values: f=g+h and f=g+w*h.**

*   **Answer:** It is necessary to drain the OPEN list to **prove that no better solution exists than the current UB**. The `f = g + h` value for a node is its estimated total cost with an admissible heuristic. The condition for optimality is that the minimum `f = g + h` value on OPEN (`min_f`) is greater than or equal to the current UB. Only by processing (expanding) all nodes on OPEN can we be sure that we have fully explored all paths that could potentially yield a solution better than UB. If we terminate early, there might be a node on OPEN with `f = g + h < UB` which could lead to a better, undiscovered solution. The `f = g + w*h` is used to guide the search quickly to a first solution but is not used for the final optimality proof.

**5. String pulling is a simple post-processing technique for improving the cost of grid-paths, albeit without any guarantees about the resulting cost. Suggest a concrete procedure for implementing this idea.**

*   **Answer:** A concrete procedure for string pulling on a grid path is as follows:
    1.  Start with a path defined by a sequence of grid cells: `[S, A, B, C, ..., G]`, where `S` is the start and `G` is the goal.
    2.  Set the current point `P` to the start node `S`.
    3.  Iterate through the subsequent nodes in the path. For each node `Q` (starting from the node after `G` and moving backwards, or from the node after `P` and moving forwards):
        *   Check if there is a **clear line of sight** from point `P` to node `Q` (i.e., the straight line between the centers of `P` and `Q` does not pass through any blocked cell).
    4.  If a line of sight exists from `P` to a much later node `Q`, it means all intermediate nodes between `P` and `Q` can be "bypassed". Remove all nodes between `P` and `Q` from the path and directly connect `P` to `Q`.
    5.  Set the new current point `P` to this node `Q`.
    6.  Repeat steps 3-5 until you reach the goal `G`.
    This process effectively "pulls the string tight" between the start and goal, removing unnecessary zig-zags and creating a shorter, more direct path where possible.