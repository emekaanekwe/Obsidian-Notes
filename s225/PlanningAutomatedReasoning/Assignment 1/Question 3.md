Essentially, this is a special rule so that trains don't get stuck. Essentially, a train will TURN around once they reach a deadend (and so technically never move backwards). This ensures that you can never accidentally reach a search state that is subsequently unsolvable due to train directions.

### visualizer = True
![[Pasted image 20250913161717.png]]

![[Pasted image 20250913223351.png]]

![[Pasted image 20250921155243.png]]
![[Pasted image 20250921160947.png]]
![[Pasted image 20250921163056.png]]
![[Pasted image 20250921164826.png]]

![[Pasted image 20250921200355.png]]
![[Pasted image 20250922175846.png]]
```txt
RAILWAY GRID REPRESENTATION:

# grid 
Size: 10x10. 
# TRAIN POSITIONS AND ORIENTATIONS:
Train 0: at (5, 8) facing UP (↑) At Dead end
Train 1: at (7, 0) facing RIGHT (→) Colliding with Train 3
Train 2: at (5, 5) facing UP (↑) Colliding with Train 4
Train 3: at (3, 8) facing LEFT (←) Colliding with Train 1
Train 4: at (1, 7) facing LEFT (←) Colidding with Train 2

TEXT-BASED GRID VISUALIZATION:

Row 0: . . . . . . B . . .
Row 1: . . C . . . . . . .
Row 2: . . . . . . . 4 . .
Row 3: . . A . . . . . . .
Row 4: E . . . . 2 . . 3 .
Row 5: . . . . . . . . . .
Row 6: 1 . . . . . . . 0 .
Row 7: . . . . . . . . . .
Row 8: . . . . . . . D . .
Row 9: . . . . . . . . . .

Where The agents are 0, 1, 2, 3, 4 and their destinations are A, B, C, D, E resctively

LEGEND:
G(#) : Goal of agent 
= : Horizontal track
║ : Vertical track
╥ : Junction (vertical track connecting to horizontal)
╗ : Right-turn connection (horizontal to vertical)
→, ←, ↑, ↓ : Train with direction
┼
┴
┐
```
### valid transitions
#### agent 0
agent: 0
loc: (6, 8)
direction: 1
=====valid transitions===== [(0, 0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 1), (1, 0, 0, 1), (0, 0, 0, 1), (1, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 1, 0, 0), (1, 0, 0, 1), (0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 1)]
***why are they 4 coordinates on a 10x10 grid?***

#### agent 1
agent: 1
loc: (6, 0)
direction: 3
=====valid transitions===== [(0, 1, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 0, 0)]
#### agent 2
agent: 2
loc: (4, 5)
direction: 3
=====valid transitions===== [(0, 1, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0), (0, 1, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (0, 0, 1, 0), (0, 0, 1, 0), (0, 0, 1, 1), (0, 1, 0, 0), (0, 1, 0, 0), (0, 1, 0, 0), (1, 1, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0), (1, 0, 0, 0)]


$x=$$$\begin{bmatrix}
-1& 2& 1 \\
1.5& 0.5& 0.5 \\
2& -1 & 4 \\
-1& 1& 0.5
\end{bmatrix}
$$
$y=$$$\begin{bmatrix}
1& 2& 1& 3
\end{bmatrix}$$
Input Layer: $h^0(x)=x$
Hidden Layer 1: $h^1(x)$
Layer 2 Output: $h^2(x) -> p(x) = softmax(h^3(x))$
where $W^1 \in R^{3x5}$, $b^1 \in R^{1x5}$, $W^2 \in R^{5x3}$, and $b^2 \in R^{1x3}$

Of course! Here is a comprehensive list of text symbols (ASCII art) commonly used to represent various railway track components. These are perfect for creating maps in a terminal, text document, or any simple text-based interface.

### 1. Basic Tracks

| Component                       | Symbol       | Description                                                              |
| :------------------------------ | :----------- | :----------------------------------------------------------------------- |
| **Horizontal Track**            | `──` or `＝＝` | A simple straight track. Double lines can represent a double-track line. |
| **Vertical Track**              | `│` or `‖`   | A simple vertical track.                                                 |
| **4-Directional Track** (Cross) | `┼`          | A crossing where tracks go both ways. Trains can't typically turn here.  |

### 2. Junctions (Trains can choose a path)

| Component                  | Symbol | Name & Description                                                                                   |
| :------------------------- | :----- | :--------------------------------------------------------------------------------------------------- |
| **3-Way Junction (Up)**    | `┴`    | A junction leading **down** to left, right, and up.                                                  |
| **3-Way Junction (Down)**  | `┬`    | A junction leading **up** to left, right, and down.                                                  |
| **3-Way Junction (Left)**  | `┤`    | A junction leading **right** to up, down, and left.                                                  |
| **3-Way Junction (Right)** | `├`    | A junction leading **left** to up, down, and right.                                                  |
| **4-Way Junction**         | `┼`    | The universal junction point. (Same as the 4-directional track, but implies a switch can be thrown). |

### 3. Curves / Turns

| Component              | Symbol | Name & Description                          |
| :--------------------- | :----- | :------------------------------------------ |
| **Top-Left Curve**     | `┌`    | Connects right-going and down-going tracks. |
| **Top-Right Curve**    | `┐`    | Connects left-going and down-going tracks.  |
| **Bottom-Left Curve**  | `└`    | Connects right-going and up-going tracks.   |
| **Bottom-Right Curve** | `┘`    | Connects left-going and up-going tracks.    |

### 4. Switches (Points) - The most important symbols

A switch has one **diverging** end and two **converging** ends. The symbol points to the *diverging* path.

| Component                  | Symbol     | Name & Description                                                   |
| :------------------------- | :--------- | :------------------------------------------------------------------- |
| **Left-Hand Switch**       | `├` or `┣` | **Main route:** Straight (Left to Right). **Diverging route:** Up.   |
| **Right-Hand Switch**      | `┤` or `┫` | **Main route:** Straight (Left to Right). **Diverging route:** Down. |
| **Vertical Switch (Up)**   | `┴` or `┻` | **Main route:** Straight (Up and Down). **Diverging route:** Left.   |
| **Vertical Switch (Down)** | `┬` or `┳` | **Main route:** Straight (Up and Down). **Diverging route:** Right.  |

**Thicker line variants (`┣`, `┫`, `┻`, `┳`) are often used to emphasize the "main" route.**

### 5. Symmetrical / Double Switches (e.g., Slip Switches)

These are more complex and allow for multiple crossing paths.

| Component | Symbol | Name & Description |
| :--- | :--- | :--- |
| **Double Slip Switch** | `╪` or `╬` | A compact symbol representing two switches that allow crossing between two tracks. |
| **Single Slip Switch** | `╫` | Less common, allows a slip from one track to the other. |

### 6. Bidirectional / Universal Symbols

Sometimes you just need a simple, clear placeholder.

| Component | Symbol | Description |
| :--- | :--- | :--- |
| **Bidirectional Track Switch** | `◄►` or `⮀` | A generic symbol indicating a switch exists here. |
| **Universal Node** | `○` or `●` | A simple circle can represent any complex junction, station, or yard. |

---

### How to Use These in a Grid:

You can build your map line-by-line. For example, a simple passing siding with a train (`[T]`) might look like this:

```
      ┌────[T]────┐
═══┳═══           ═══┳═══
      └─────────────┘
```

Or a more complex crossover:

```
═══╬═══
    ║
═══╬═══
```

**Pro Tip:** Use a monospaced font (like Consolas, Courier New, or Monaco) when working with these symbols so everything aligns perfectly.