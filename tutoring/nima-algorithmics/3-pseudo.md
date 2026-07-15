Here are complete answers for all questions across all sections of this problem set.

---

# Section A: Writing Pseudocode

## Q1. Precision

**(a)** `for i ← 1 to n do if A[i] > threshold then add A[i] to result`

**(b)** `while i < n do` (or any precise loop condition, e.g. `while not foundTarget`)

**(c)** `result ← A[i] + A[j]`

---

## Q2. Scaffold to Pseudocode

| Plain English                                     | Pseudocode                           |
| ------------------------------------------------- | ------------------------------------ |
| Start with a count of zero.                       | `count ← 0`                          |
| Look at each element in the list.                 | `for i ← 1 to n do`                  |
| If the element is positive, add one to the count. | `if A[i] > 0 then count ← count + 1` |
| After the loop, return the count.                 | `return count`                       |

---

## Q3. Write from Scratch

```
function FindMax(A, n):
    max ← A[1]
    for i ← 2 to n do
        if A[i] > max then
            max ← A[i]
    return max
```

---

## Q4. Modular Decomposition

**(a)** Two sub-functions:

- **`CountOccurrences(A, n, val)`** — Input: list A of length n, value val. Output: integer count of how many times val appears in A.
- **`FindDuplicates(A, n)`** — Input: list A of length n. Output: new list containing each value that appears more than once (using CountOccurrences as a helper).

**(b)** Simpler function first:

```
function CountOccurrences(A, n, val):
    count ← 0
    for i ← 1 to n do
        if A[i] = val then
            count ← count + 1
    return count
```

**(c)** Main function:

```
function FindDuplicates(A, n):
    result ← empty list
    seen ← empty set
    for i ← 1 to n do
        if A[i] not in seen then
            if CountOccurrences(A, n, A[i]) > 1 then
                add A[i] to result
            add A[i] to seen
    return result
```

---

## Q5. ADT Operations as Black-Box Calls

```
procedure ProcessStream(items):
    Q ← empty Queue
    for each item in items:
        enqueue(Q, item)        // add item to queue
    while not isEmpty(Q):
        x ← dequeue(Q)         // remove next item
        process(x)
```

**Why a queue?** A queue uses FIFO (First In, First Out) ordering, which guarantees that items are processed in exactly the order they arrive. A stack (LIFO) would process the most recently added item first, reversing arrival order and violating the requirement that processing order matches arrival order.

---

# Section B: Recursion vs Iteration

## Q6. Definitions

**(a)** Every recursive algorithm must have:

1. A **base case** — a condition under which the function returns a result without making another recursive call.
2. A **recursive case** — a call to itself on a smaller or simpler version of the problem, moving toward the base case.

**(b)** If the base case is missing, the function recurses infinitely, causing a **stack overflow** (the call stack runs out of memory).

**(c)** Each call's state (local variables, return address, parameters) is stored on the **call stack**.

---

## Q7. Trace a Recursive Algorithm

**(a)** Call tree for Mystery(5) — this computes Fibonacci numbers:

```
Mystery(5)
├── Mystery(4)
│   ├── Mystery(3)
│   │   ├── Mystery(2)
│   │   │   ├── Mystery(1) → 1
│   │   │   └── Mystery(0) → 0
│   │   │   returns 1
│   │   └── Mystery(1) → 1
│   │   returns 2
│   └── Mystery(2)
│       ├── Mystery(1) → 1
│       └── Mystery(0) → 0
│       returns 1
│   returns 3
└── Mystery(3)
    ├── Mystery(2)
    │   ├── Mystery(1) → 1
    │   └── Mystery(0) → 0
    │   returns 1
    └── Mystery(1) → 1
    returns 2
returns 5
```

**(b)** This computes the **Fibonacci sequence**. Mystery(n) = Fib(n), so Mystery(5) = 5.

**(c)**

- **Base case:** `if n ≤ 1: return n`
- **Recursive case:** `return Mystery(n-1) + Mystery(n-2)`

---

## Q8. Write Both Forms

**(a)** Recursive version:

- Base case: if the list has one element, return that element.
- Recursive case: return A[1] + sum of the rest of the list.

```
function SumRecursive(A, n):
    if n = 0 then
        return 0
    return A[n] + SumRecursive(A, n - 1)
```

**(b)** Iterative version:

```
function SumIterative(A, n):
    total ← 0
    for i ← 1 to n do
        total ← total + A[i]
    return total
```

**(c)** The recursive version adds one call to the call stack per element. For very large n (e.g. n = 10,000+), this may exceed the maximum stack depth and cause a **stack overflow**. The iterative version uses constant stack space and works for any n.

---

## Q9. Recursive DFS vs Iterative DFS

**(a)** In recursive DFS, the **call stack** (the program's execution stack) implicitly plays the role of the explicit stack — each recursive call suspends the current frame and pushes a new one.

**(b)** The line corresponding to `DFS(G, v)` is:

```
push(S, v)
```

_(pushing v onto the stack is what schedules the recursive exploration of v — the actual "visit" happens when v is later popped.)_

**(c)** The LIFO property of the stack means the **most recently discovered** unvisited neighbour is always explored next. This causes traversal to go as deep as possible along one branch before backtracking, which is the defining property of depth-first search.

---

# Section C: BFS & DFS in Python

## Q10. Pseudocode to Python Translation

|Pseudocode|Python|
|---|---|
|`visited[v] ← False for all v`|`visited = {v: False for v in G}`|
|`enqueue(Q, s)`|`Q.append(s)`|
|`u ← dequeue(Q)`|`u = Q.popleft()`|
|`for each v in neighbours(G, u)`|`for v in G[u]:`|
|`dist[v] ← dist[u] + 1`|`dist[v] = dist[u] + 1`|

_(Assumes `from collections import deque` and `Q = deque()`)_

---

## Q11. BFS Trace

Using the graph from the notes (A connects to B, C; B connects to A, C, D; C connects to A, B, E; D connects to B, F, G; E connects to C; F connects to D; G connects to D — adjust if your adjacency list differs):

**(a)** BFS from A (neighbours in alphabetical order):

|Step|Vertex dequeued|Queue after step|
|---|---|---|
|1|A|[B, C]|
|2|B|[C, D]|
|3|C|[D, E]|
|4|D|[E, F, G]|
|5|E|[F, G]|
|6|F|[G]|
|7|G|[]|

**(b)** Distances:

|v|A|B|C|D|E|F|G|
|---|---|---|---|---|---|---|---|
|d(A,v)|0|1|1|2|2|3|3|

---

## Q12. DFS Trace

**(a)** DFS visit order from A (neighbours pushed in reverse alphabetical order so smallest is popped first):

**A → B → D → F → G → C → E**

**(b)** BFS visits nodes **level by level** (all nodes at distance 1, then distance 2, etc.), producing a wide, shallow traversal. DFS follows one branch as deep as possible before backtracking, producing a deep, narrow traversal. This shows BFS is ideal for finding shortest paths, while DFS is better for exhaustive exploration or cycle detection.

---

## Q13. Shortest Path Extension

**(a)** The parent dictionary stores, for each vertex v, the vertex from which v was first discovered during BFS. After BFS completes, you reconstruct the path from target t back to source s by repeatedly following `parent[v]` until you reach s, then reversing the result.

**(b)** Shortest path from A to G (using BFS trace): **A → B → D → G**

**(c)** Updated BFS pseudocode with parent dictionary:

```
procedure BFS(G, s):
    visited[v] ← False for all v
    dist[v] ← ∞ for all v
    parent[v] ← null for all v
    visited[s] ← True
    dist[s] ← 0
    enqueue(Q, s)
    while not isEmpty(Q):
        u ← dequeue(Q)
        for each v in neighbours(G, u):
            if not visited[v]:
                visited[v] ← True
                dist[v] ← dist[u] + 1
                parent[v] ← u
                enqueue(Q, v)
```

**Extension — Python implementation:**

```python
from collections import deque

def bfs_path(G, s, t):
    visited = {v: False for v in G}
    parent = {v: None for v in G}
    visited[s] = True
    Q = deque([s])
    while Q:
        u = Q.popleft()
        if u == t:
            # Reconstruct path
            path = []
            while u is not None:
                path.append(u)
                u = parent[u]
            return path[::-1]
        for v in G[u]:
            if not visited[v]:
                visited[v] = True
                parent[v] = u
                Q.append(v)
    return []  # No path found
```

---

# Section A (Week 2): Tree Recursion & Backtracking

## Q1. Linear vs Tree Recursion

**(a)** In **linear recursion**, each call makes at most one recursive call, forming a chain. In **tree recursion**, each call makes two or more recursive calls, forming a branching tree structure.

**(b)**

- Linear recursion: `SumRecursive(A, n)` — each call makes exactly one recursive call.
- Tree recursion: `Fibonacci(n)` — each call makes two recursive calls.

**(c)** At depth d with each call spawning exactly 2 sub-calls, total calls = **2^(d+1) − 1**. For d = 4: 2^5 − 1 = **31 total calls**.

---

## Q2. Fibonacci Call Tree

**(a)** Full call tree for Fib(5):

```
Fib(5) = 5
├── Fib(4) = 3
│   ├── Fib(3) = 2
│   │   ├── Fib(2) = 1
│   │   │   ├── Fib(1) = 1
│   │   │   └── Fib(0) = 0
│   │   └── Fib(1) = 1
│   └── Fib(2) = 1
│       ├── Fib(1) = 1
│       └── Fib(0) = 0
└── Fib(3) = 2
    ├── Fib(2) = 1
    │   ├── Fib(1) = 1
    │   └── Fib(0) = 0
    └── Fib(1) = 1
```

**(b)** Fib(2) is computed **3 times**.

**(c)** This repeated computation means the naive recursive Fibonacci has **exponential time complexity O(2^n)** — it does enormous amounts of redundant work recalculating the same subproblems. This makes it highly inefficient for large n, motivating techniques like memoisation or dynamic programming.

---

## Q3. Backtracking Structure

**(a)** The three steps of backtracking in order:

1. **Choose** — make a candidate choice and extend the current partial solution.
2. **Explore** — recurse on the new partial solution.
3. **Undo** — remove the choice (backtrack) and try the next candidate.

**(b)** The undo step restores the state to what it was before the choice, allowing other branches to be explored from a clean state. In simple recursion, each call works on independent data so no state needs undoing; in backtracking, a shared mutable state is modified and must be restored so that sibling branches see an unmodified state.

**(c)**

- **(i)** If a cell is left marked after backtracking, other paths that legitimately pass through that cell will incorrectly treat it as visited and skip it, potentially missing valid solutions.
- **(ii)** If visited cells were never unmarked, the algorithm would fail to find any solution that revisits a cell via a different route — effectively treating many reachable cells as dead ends and producing incorrect or incomplete results.

---

## Q4. Decision Tree for Subset Enumeration

```
                    {}
              /            \
           {1}             {}
          /    \          /    \
       {1,2}  {1}      {2}     {}
       / \    / \      / \     / \
{1,2,3}{1,2}{1,3}{1}{2,3}{2}{3} {}
```

Leaves (all 8 subsets): `{1,2,3}, {1,2}, {1,3}, {1}, {2,3}, {2}, {3}, {}`

**Number of leaves = 2^n.** For n = 3: **2^3 = 8 leaves**.

---

# Section B: Brute-Force Design Pattern

## Q5. Definition and Characteristics

**(a)** Brute-force is a design pattern that solves a problem by systematically trying every possible candidate solution and checking each one against the solution criteria.

**(b)**

- **Strength:** Always finds the correct answer (guaranteed correctness) and is straightforward to implement.
- **Weakness:** Often very slow — time complexity is typically exponential or polynomial of high degree, making it impractical for large inputs.

**(c)** Brute-force is a correctness baseline because it is guaranteed to find the right answer by exhaustive search. More efficient algorithms can be validated by checking that their results match the brute-force solution on small inputs.

---

## Q6. Trace: Pair-Sum Search

A = [3, 7, 1, 9, 2, 5], t = 10

**(a)**

|i|j|A[i]+A[j]|Match?|
|---|---|---|---|
|1|2|3+7 = 10|✓ YES|
|1|3|3+1 = 4|No|
|1|4|3+9 = 12|No|
|1|5|3+2 = 5|No|
|1|6|3+5 = 8|No|
|2|3|7+1 = 8|No|
|2|4|7+9 = 16|No|

**(b)** Returned pair: indices **(1, 2)**, values **(3, 7)**.

**(c)** Number of pairs = **n(n−1)/2**. For n = 6: 6×5/2 = **15 pairs**.

---

## Q7. Modified PairSum — Return All Pairs

```
function PairSumAll(A, n, t):
    result ← empty list
    for i ← 1 to n do
        for j ← i+1 to n do
            if A[i] + A[j] = t then
                add (i, j) to result
    return result
```

For A = [3, 7, 1, 9, 2, 5], t = 10:

- (1, 2): 3 + 7 = 10 ✓
- (2, 6): 7 + ... wait — checking all: (1,2)=10✓, (3,4)=1+9=10✓, (5,6) — no other pairs sum to 10 checking systematically.

**All valid pairs: (1,2) → values (3,7) and (3,4) → values (1,9).**

---

## Q8. Brute-Force Largest Triple Product

**(a)** Try every combination of three distinct elements, compute their product, and keep track of the largest product seen.

**(b)**

```
function LargestTripleProduct(A, n):
    maxProduct ← -∞
    for i ← 1 to n do
        for j ← i+1 to n do
            for k ← j+1 to n do
                product ← A[i] * A[j] * A[k]
                if product > maxProduct then
                    maxProduct ← product
    return maxProduct
```

**(c)** Number of triples = **n(n−1)(n−2)/6** (i.e. n choose 3).

---

# Section C: Greedy Design Pattern

## Q9. Definition and Conditions

**(a)** The greedy design pattern solves a problem by making the locally optimal choice at each step, never reconsidering past choices, with the aim of finding a global optimum.

**(b)** A problem must have:

1. **Greedy choice property** — a globally optimal solution can always be constructed by making locally optimal (greedy) choices.
2. **Optimal substructure** — an optimal solution to the whole problem contains optimal solutions to its subproblems.

**(c)** This tells you the problem **lacks the greedy choice property** — the locally optimal choice at some step leads away from the global optimum, meaning a greedy approach is not guaranteed correct for this problem.

---

## Q10. Coin Change — Australian Denominations

Denominations: {200, 100, 50, 20, 10, 5}

**(a) T = 285 cents:**

- 200 (rem 85) → 50 (rem 35) → 20 (rem 15) → 10 (rem 5) → 5 (rem 0)
- Coins: **200, 50, 20, 10, 5 → 5 coins**

**(b) T = 175 cents:**

- 100 (rem 75) → 50 (rem 25) → 20 (rem 5) → 5 (rem 0)
- Coins: **100, 50, 20, 5 → 4 coins**

**(c) T = 60 cents:**

- 50 (rem 10) → 10 (rem 0)
- Coins: **50, 10 → 2 coins**

---

## Q11. Counter-Example — Greedy Failure

Denominations: {1, 15, 25}, T = 30

**(a)** Greedy trace:

- 25 (rem 5) → 1 (rem 4) → 1 (rem 3) → 1 (rem 2) → 1 (rem 1) → 1 (rem 0)
- Greedy gives: **25, 1, 1, 1, 1, 1 → 6 coins**

**(b)** Optimal solution:

- 15 + 15 = 30 → **2 coins**

**(c)** Greedy fails here because choosing 25 (the largest coin ≤ 30) forces the remaining 5 cents to be filled with five 1-cent coins. This violates the **greedy choice property** — the locally best choice (largest coin) does not lead to a globally optimal solution. The problem has overlapping substructure that requires dynamic programming to solve correctly.

**(d)** For Australian denominations, the greedy choice property holds informally because each denomination is designed so that the largest coin fitting the remainder never "blocks" a more efficient combination. No two smaller coins can ever combine to make a denomination more efficiently than choosing the next-largest coin first.

---

## Q12. Activity Selection

**(a)** Sorted by finish time:

|Activity|Start|Finish|
|---|---|---|
|A|1|4|
|F|2|5|
|B|3|6|
|C|5|7|
|D|6|9|
|E|8|11|

**(b)** Greedy trace (select activity if start ≥ last finish):

|Activity|Start|Finish|Selected?|Reason|
|---|---|---|---|---|
|A|1|4|✓|First activity|
|F|2|5|✗|Starts at 2 < last finish 4|
|B|3|6|✗|Starts at 3 < last finish 4|
|C|5|7|✓|Starts at 5 ≥ last finish 4|
|D|6|9|✗|Starts at 6 < last finish 7|
|E|8|11|✓|Starts at 8 ≥ last finish 7|

**(c)** Maximum non-overlapping activities: **3 (A, C, E)**

**(d)** Selecting by earliest start time does **not** always give the same result. Counter-example: if one activity starts earliest but runs for a very long time (e.g. start=1, finish=10), it blocks many shorter activities. Earliest-finish-time greedy correctly avoids this, while earliest-start-time would incorrectly select the long activity.

---

# Section D: Pattern Identification & Algorithm Selection

## Q13. Identify the Pattern

**(a)** **Greedy** — it sorts jobs by a criterion and makes an irrevocable locally optimal assignment at each step without reconsidering.

**(b)** **Brute-force** — it exhaustively checks every triple (i, j, k) in the list, trying all possible combinations.

**(c)** **Greedy** — this is Dijkstra's algorithm, which always extracts the minimum-distance vertex and makes irrevocable locally optimal relaxation decisions.

---

## Q14. Algorithm Selection and Justification

**(a)** **BFS (Breadth-First Search).** BFS guarantees the shortest path in an unweighted graph because it explores nodes level by level, and the first time a node is reached it is via the fewest possible edges.

**(b)** **Brute-force** (generate all permutations). With no constraints, there is no structure to exploit, so every ordering must be considered — this is inherently exhaustive with n! possible orderings.

**(c)** **Greedy (fractional knapsack greedy).** The fractional knapsack has optimal substructure and the greedy choice property — always picking the item with the highest value-to-weight ratio is provably optimal, because fractional items can be taken, leaving no "wasted" capacity.

---

## Q15. Design Task

**(a)** Brute-force — try all subsets of the list and return the smallest-sized subset that sums to exactly T:

```
function MinSubsetSum(A, n, T):
    minCount ← ∞
    for each subset S of A:
        if sum(S) = T then
            if |S| < minCount then
                minCount ← |S|
    if minCount = ∞ then return "no solution"
    return minCount
```

**(b)** Greedy fails here. Counter-example: A = [1, 5, 6], T = 10.

- Greedy (largest first): picks 6 (rem 4) → picks 1 (rem 3) → picks 1... but there's no second 1. Actually picks 6, then 1 → can't reach 10 without a second pass.
- Better: A = [1, 5, 6], T = 11 → Greedy: 6 + 5 = 11 ✓ (2 coins). That works.
- Clearer counter-example: A = [3, 4, 5], T = 8 → Greedy picks 5 (rem 3) → picks 3 → total 2 elements. Optimal is also 4+4 — but 4 appears once: 4+3+... → Actually greedy gives 5+3 = 2 elements, which is also optimal here.
- Definitive: A = [1, 3, 4], T = 6 → Greedy picks 4 (rem 2) → picks 1 (rem 1) → picks 1 — but there's only one 1. Fails to find 3+3 (only one 3). Optimal: 3+3 not possible. Picks 4+1+1 — but only one 1 available, so greedy gets stuck. Optimal is 3+... actually 2+4 isn't available. This highlights that greedy can fail to find a valid solution or produce suboptimal results when the denominations don't have the greedy choice property.

The greedy approach (always pick the largest integer ≤ remaining target) **does not work in general** — it can get stuck or use too many integers, as shown above.

**(c)** For small input (n ≤ 20), **brute-force is recommended**. With only 20 elements, there are at most 2^20 ≈ 1 million subsets — easily computable in milliseconds. It guarantees the correct minimum, whereas greedy may return a wrong answer. The simplicity and correctness of brute-force outweighs efficiency concerns at this scale.

---

# Section A (Dijkstra & Shortest Paths): Edge Relaxation

## Q1. Relaxation

**(a)** Relaxation condition for edge (u, v) with weight w: **If dist[u] + w(u,v) < dist[v], then set dist[v] ← dist[u] + w(u,v) and prev[v] ← u.**

**(b)** dist[u] = 5, w = 3, dist[v] = 9. Check: 5 + 3 = 8 < 9. **Yes, relaxation occurs.** Updated: **dist[v] ← 8, prev[v] ← u**

**(c)** dist[u] = 6, w = 4, dist[v] = 8. Check: 6 + 4 = 10 > 8. **No relaxation occurs** — the current path to v is already shorter.

---

## Q2. Dijkstra Trace

Graph edges (from diagram): A→C(2), A→B(4), C→B(1), C→D(5), B→D(3), B→E(2) _(adjust weights to match your printed diagram — the trace below uses the described 4,2,5,1,3,2 weights as labelled)_

Using edges: A-B(4), A-C(2), C-B(1), C-D(5), B-D(3), B-E(2), D-E(1) _(reading the diagram as A→C=2, A→B=4, C→B=1, B→D=3, B→E=2, C→D=5, D→E=1)_:

**(a)**

|Step|dist[A]|dist[B]|dist[C]|dist[D]|dist[E]|
|---|---|---|---|---|---|
|Init|0|∞|∞|∞|∞|
|Pop A|0|4|2|∞|∞|
|Pop C|0|3|2|7|∞|
|Pop B|0|3|2|6|5|
|Pop D|0|3|2|6|5|
|Pop E|0|3|2|6|5|

**(b)** prev array:

|v|A|B|C|D|E|
|---|---|---|---|---|---|
|prev[v]|—|C|A|B|B|

**(c)** Shortest path A to E: **A → C → B → E**, total weight = 2 + 1 + 2 = **5**.

**(d)** There is no other shortest path — D's path (A→C→B→D→E = 2+1+3+1=7) is longer, so the path A→C→B→E is **unique**.

---

## Q3. Why Dijkstra Fails with Negative Edges

Graph: A→B(3), A→C(4), B→C(−2)

**(a)** Dijkstra trace:

- Init: dist[A]=0, dist[B]=∞, dist[C]=∞
- Pop A: dist[B]=3, dist[C]=4
- Pop B: tries to relax C: 3+(−2)=1 < 4 → dist[C]=1... **but B is popped after C if dist[C]=4 < dist[B]=3 is false — actually B=3 < C=4, so B is popped first, and C is correctly updated to 1.**

_(In this particular case Dijkstra actually finds the right answer. The classic failure case needs C to already be finalised before B is processed. The graph below illustrates the issue:)_

A→B(3), A→C(4), B→C(−2): Dijkstra pops A, sets B=3, C=4. Pops B (dist=3), relaxes C to 1. Pops C (dist=1). Final: C=1. Correct here.

**(b)** In the general case with negative edges, Dijkstra **may report an incorrect (too-large) distance** for some vertex. The correct shortest distance can be lower than what Dijkstra finds.

**(c)** Dijkstra makes the **greedy assumption** that once a vertex is popped from the priority queue, its distance is finalised and cannot be improved. With negative edges, a later-discovered path through a negative edge can produce a shorter distance to an already-finalised vertex. Since Dijkstra never revisits finalised vertices, it misses these improvements and produces incorrect results.

---

# Section B: Bellman-Ford

## Q4. Bellman-Ford Concepts

**(a)** Bellman-Ford relaxes all edges **n − 1 times** (where n is the number of vertices). This is because the shortest path in a graph with no negative cycles can visit at most n − 1 edges — so n − 1 rounds of relaxation is sufficient to propagate correct distances along any shortest path.

**(b)** After the main loop, Bellman-Ford performs **one additional pass over all edges**. If any distance can still be relaxed (dist[u] + w(u,v) < dist[v]), a negative cycle exists and is reported.

**(c)** 6 vertices → n−1 = 5 iterations. 10 edges per iteration. Total relaxations = **5 × 10 = 50**.

---

## Q5. Bellman-Ford Trace

Graph: A→B(6), A→D(7), B→C(5), B→D(−3), D→C(4) _(using values from diagram: edges as labelled with weights 6,7,5,−3,4)_

Wait — the diagram shows weights 6, 7, 5, −3, 4. Let me use: A→B=6, A→D=7, B→C=5, B→D=−3, D→C=4.

**(a)**

|Iteration / Edge|dist[A]|dist[B]|dist[C]|dist[D]|
|---|---|---|---|---|
|Init|0|∞|∞|∞|
|Iter 1: (A,B)|0|6|∞|∞|
|Iter 1: (A,D)|0|6|∞|7|
|Iter 1: (B,C)|0|6|11|7|
|Iter 1: (B,D)|0|6|11|3|
|Iter 1: (D,C)|0|6|7|3|
|Iter 2: (A,B)|0|6|7|3|
|Iter 2: (A,D)|0|6|7|3|
|Iter 2: (B,C)|0|6|7|3|
|Iter 2: (B,D)|0|6|7|3|
|Iter 2: (D,C)|0|6|7|3|
|Iter 3: all|0|6|7|3|

**(b)** Final shortest distances from A:

- dist[A] = 0, dist[B] = 6, dist[C] = 7, dist[D] = 3

**(c)** Shortest path from A to C: **A → D → C** (weight 7 + ... wait: A→D=7, D→C=4 → total 11. But A→B→D→C = 6+(−3)+4=7). So: **A → B → D → C**, total weight = **7**.

---

## Q6. Dijkstra vs Bellman-Ford — Selection

**(a)** **Dijkstra.** All travel times are positive, so Dijkstra's greedy assumption is valid. It is significantly faster than Bellman-Ford (O((V+E) log V) vs O(VE)), making it the better choice for real-time navigation.

**(b)** **Bellman-Ford.** Negative edge weights (exchange rates can create negative costs) require Bellman-Ford. Furthermore, detecting negative cycles (arbitrage loops) is done by Bellman-Ford's extra relaxation pass — Dijkstra cannot detect negative cycles at all.

**(c)** **Bellman-Ford.** With mixed positive/negative edges (even without negative cycles), Dijkstra's greedy finalisation assumption breaks down. Bellman-Ford correctly handles negative edges and, with only 8 nodes, its O(VE) complexity is perfectly acceptable.

---

# Section C: Floyd-Warshall & Transitive Closure

## Q7. Floyd-Warshall Concepts

**(a)** Update rule:

```
D[i][j] ← min(D[i][j], D[i][k] + D[k][j])
```

For each intermediate vertex k, for all pairs (i, j).

**(b)** Time complexity: **O(V³)**. The algorithm has three nested loops: one over each intermediate vertex k (V iterations), and for each k, one loop over all i (V) and one over all j (V), giving V × V × V = V³ operations.

**(c)** After running Floyd-Warshall, if any diagonal entry **D[i][i] < 0**, then vertex i is on a negative cycle.

**(d)** Floyd-Warshall is preferred over repeated Dijkstra when: (1) **all-pairs** shortest paths are needed (Floyd-Warshall does this in one pass rather than V separate Dijkstra runs), or (2) the graph is **dense** (many edges), since Floyd-Warshall's constant factors can be better than V × O((V+E) log V) for dense graphs.

---

## Q8. Floyd-Warshall Trace

Graph edges (from diagram): 1→2(3), 1→3(8), 2→4(2), 3→2(? — reading the diagram: 1→2=3, 2→4=2, 1→3=8, 3→4=? — using the weights 3,8,2,1,5,4 as described):

Using: 1→2=3, 1→3=8, 2→4=2, 3→1=5 (back edge), 3→4=1 (if present), 4→3=4 _(adjust to your specific diagram)_

For a clean trace I'll use: edges 1→2(3), 1→4(8), 2→4(2), 3→2(1), 4→3(4), 3→1(5) based on the described layout:

**(a)** Initial matrix D⁽⁰⁾:

||1|2|3|4|
|---|---|---|---|---|
|**1**|0|3|8|∞|
|**2**|∞|0|∞|2|
|**3**|5|1|0|∞|
|**4**|∞|∞|4|0|

**(b)** k=1 (route through vertex 1): Check if D[i][1] + D[1][j] < D[i][j] for all i,j.

- D[3][2]: D[3][1]+D[1][2] = 5+3 = 8 > 1 → no change
- D[3][3]: 5+8=13 > 0 → no change
- No entries improve through vertex 1.

**(c)** k=2 (route through vertex 2):

- D[1][4]: D[1][2]+D[2][4] = 3+2 = 5 < ∞ → **D[1][4] ← 5**
- D[3][4]: D[3][2]+D[2][4] = 1+2 = 3 < ∞ → **D[3][4] ← 3**

**(d)** Final matrix D⁽⁴⁾ (after all k):

||1|2|3|4|
|---|---|---|---|---|
|**1**|0|3|9|5|
|**2**|11|0|6|2|
|**3**|5|1|0|3|
|**4**|9|5|4|0|

**(e)** Shortest path 1 → 4: D[1][4] = 5, via **1 → 2 → 4** (weight 3+2=5).

---

## Q9. Transitive Closure

**(a)** Transitive closure update rule:

```
T[i][j] ← T[i][j] OR (T[i][k] AND T[k][j])
```

Unlike Floyd-Warshall which tracks minimum distances, transitive closure uses boolean OR/AND — it only asks "is there any path?" not "how long is it?"

**(b)** Initial boolean matrix T⁽⁰⁾ (1 if direct edge exists, 1 on diagonal):

||1|2|3|4|
|---|---|---|---|---|
|**1**|1|1|1|0|
|**2**|0|1|0|1|
|**3**|1|1|1|0|
|**4**|0|0|1|1|

**(c)** Final matrix T⁽⁴⁾ after all k (all pairs reachable after propagation):

||1|2|3|4|
|---|---|---|---|---|
|**1**|1|1|1|1|
|**2**|1|1|1|1|
|**3**|1|1|1|1|
|**4**|1|1|1|1|

**(d)** All pairs of vertices are **mutually reachable** — the graph is strongly connected.

**(e)** Transitive closure is more useful than shortest distances in a **prerequisite/dependency system** (e.g. university courses). You only need to know _whether_ course B is reachable (directly or indirectly) from course A — not how many steps it takes. Storing boolean reachability is also more space-efficient than storing all distances.

---

# Section D: Algorithm Selection

## Q10. Comparison Table

|Property|Dijkstra|Bellman-Ford|Floyd-Warshall|
|---|---|---|---|
|**Problem type**|Single-source shortest path|Single-source shortest path|All-pairs shortest paths|
|**Negative edges**|✗ No|✓ Yes|✓ Yes|
|**Negative cycles**|Cannot detect|✓ Detects|✓ Detects (diagonal < 0)|
|**Time complexity**|O((V+E) log V)|O(VE)|O(V³)|
|**Design pattern**|Greedy|Dynamic programming|Dynamic programming|
|**Best suited for**|Large sparse graphs, positive weights|Graphs with negative edges|Dense graphs, all-pairs queries|

---

## Q11. Justify Your Selection

**(a)** **Dijkstra.** All road weights are positive, satisfying Dijkstra's requirement. It is significantly faster than Bellman-Ford for this use case, and navigation apps require real-time performance — Dijkstra's O((V+E) log V) is well-suited for large sparse road networks.

**(b)** **Floyd-Warshall.** For a 20-node network, Floyd-Warshall runs in O(20³) = 8,000 operations in a single pass. Repeated Dijkstra would require 20 runs of O((V+E) log V) each — for a dense graph this is roughly 20 × O(400 × log 20) ≈ 20 × 1,700 ≈ 34,000 operations. Floyd-Warshall is simpler to implement for this all-pairs use case.

**(c)** **Bellman-Ford.** Negative edge weights (exchange rate profits/losses) are inherent to currency graphs. Only Bellman-Ford can correctly handle negative edges and explicitly detect negative cycles via its extra relaxation pass — a negative cycle in this context directly corresponds to an arbitrage opportunity.

**(d)** **Transitive closure (Floyd-Warshall variant).** The question is purely about reachability — whether one course can reach another through any chain of prerequisites — not about the shortest chain. Transitive closure answers this directly using boolean operations and is more interpretable and space-efficient than storing full distance matrices.