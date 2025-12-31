# Requirements

## What is shown in code

High level and low level decisions take into account the current action and strategy of the teammate agent. (Two agents should cooperate with each other, and share information with each other.)

## Description of approach

**Use this to guide report**
Well documented numerical experiments, analysing the efficiency, advantages, drawbacks of your different implementations. E.g. improvements on success rate between different strategies/attempts, runtime statistics on planning time, and appropriate reference algorithms.

The document is expertly structured in the style of a scientific report, including appropriate supporting materials that clearly improve the quality of associated discussion.

## Possible Relevant papers

# 1. Background

## Background describing what is known on the subject

**Use this for report to explain the background and goals**

**Planning** is defined as an explicit deliberation process that selects and organizes a sequence of actions to achieve a predefined goal.
*   **AI Planning** is the computational study of this process, focusing on the development of algorithms and systems that can autonomously generate such action sequences (plans).
The lecture defines Classical Planning within the **Basic State Model** (or State Transition System), characterized by the following properties:
*   **Finite and Discrete State Space**
*   **Fully Observable**
*   **Deterministic** actions
*   **Static** environment (no external changes)
*   **Attainment Goals** (achieving a target state)
*   **Sequential Plans**
*   **Implicit Time**
*   **Offline Planning** (the plan is generated before execution)
### Key Representation Languages

**A. STRIPS (Stanford Research Institute Problem Solver)**
*   **Operators** are defined by:
    *   **Preconditions**: Conditions that must be true for the action to be executable.
    *   **Effects**: Changes to the state, consisting of an **Add List** (propositions made true) and a **Delete List** (propositions made false).

**B. PDDL (Planning Domain Definition Language)**
*   Presented as a standardized, domain-independent language for describing planning problems, widely used in International Planning Competitions (IPC).

 The following reinforces the theoretical concepts with detailed examples:
*   **Blocks World**: A classic domain for testing planners, involving stacking blocks.
*   **Sliding Tile Puzzle**: Formulated in PDDL, demonstrating the use of `move` actions with complex preconditions involving relative tile positions (`inc`, `dec`).
*   **Dock Worker Robot**: A more complex logistics domain from the literature, involving robots, cranes, containers, and piles, showcasing the scalability of the representation.

### Classical Planning

#### **5. Approaches to the Control Problem**

formal underpinnings of classical planning. For an IT research paper, it serves as a concise reference for:
*   The **formal model and assumptions** of classical planning.
*   The **standard syntax and semantics** of STRIPS and PDDL.
*   **Canonical example domains** (Blocks World, Sliding Tile) for benchmarking and comparison.
*   The distinction between **domain-specific** and **domain-independent** planning, highlighting the generality of the latter.

### Panning and Heuristics
#### **1. Core Motivation: The Need for Heuristics**
*   Formally representing a planning problem (e.g., in PDDL) is distinct from solving it efficiently.
*   Blind search algorithms are infeasible for large state spaces. **Heuristic functions \( h(s) \)**—estimating the distance from a state \( s \) to the goal—are essential to guide the search.
*   **Heuristic Search Planning** involves automatically deriving these heuristics from the problem's declarative encoding, transforming planning into a guided search problem.
*   The historical success of this approach is evidenced by its dominance in the **International Planning Competition (IPC)**.
#### **2. The Core Principle: Deriving Heuristics from Simplified Problems**
The central paradigm is to generate a heuristic for a problem \( P \) by finding the exact solution to a simpler, relaxed problem \( P' \).
*   **Formal Model**: Given a planning problem \( P \), a transformation \( r \) creates a simplified problem \( P' \). The cost of the optimal solution for \( P' \), \( h'^*(s) \), serves as a heuristic estimate \( h(s) \) for \( P \).
*   **Trade-off**: The choice of simplification involves a trade-off between the **generality** of the heuristic and its **computational performance** and accuracy.

#### **3. Four Families of Heuristic Generation Methods**

1. **Abstraction**
2. **Delete-Relaxation**
3. **Critical Path Heuristics**
4. **Landmark Heuristics**

#### **4. Supporting Algorithms and Formalisms**
*   **Relaxed Planning Graph (RPG)**: A layered graph that propagates facts forward from the initial state, ignoring delete effects. It is a core data structure for efficiently computing several heuristics (e.g., landmarks, \( h^{add} \)).
*   **Regression (Backward Search)**: A search strategy that starts from the goal and applies actions in reverse to find a path back to the initial state. It is foundational for defining critical path heuristics.

### AI and Explainability
#### **1. Core Motivation: The Need for Explainable AI**
*   **Paradigm Shift**: As AI systems transition to being collaborative partners with humans, there is a growing need for them to be **explainable**, not just effective.
*   **Key Drivers**:
    *   **Contestability**: Allowing humans to question and challenge AI decisions.
    *   **Trust and Collaboration**: Building user confidence and enabling effective teamwork.
    *   **Error Detection**: Providing a mechanism to identify and correct AI failure modes.
    *   **Teaching**: Enabling humans to impart their preferences to the AI system.

## What elements are still subject to controversy


## What is the exact gap in the knowledge that your study explores

## Objective function

**Use this for the report**
maximize the number of winning games.
## Cite the exact parameter from your code you plan to measure

# Methods

## Describe what was done so that it can be reproducible

## Detail every major coding technique used

## Detail the statistical analysis

### Inclusion and non-inclusion criteria
### normally distributed data (guassian distribution)

### non-normally distributed data

### Source(s) of study data

### number of samples


# Results

## Describe observations and findings

**Use this data for report**
Map, Games, WinRate, AvgScore, AvgRedFoodLeft, AvgBlueFoodLeft
map fast capture, 2, 1.00, 1.00, 11.00, 10.00
alleyCapture, 1, 1.00, 1.00, 13.00, 12.00
bloxCapture, 1, 1.00, 1.00, 35.00, 34.00
crowdedCapture, 1, 0.00, -6.00, 58.00, 65.00
defaultCapture, 1, 1.00, 1.00, 20.00, 19.00
distantCapture, 1, 1.00, 1.00, 21.00, 20.00
fastCapture, 1, 1.00, 1.00, 11.00, 10.00
jumboCapture, 1, 1.00, 1.00, 158.00, 157.00
mediumCapture, 1, 1.00, 1.00, 43.00, 42.00
officeCapture, 1, 1.00, 1.00, 123.00, 123.00
strategicCapture, 1, 1.00, 1.00, 24.00, 22.00
tinyCapture, 1, 0.00, -4.00, 7.00, 11.00
average_scores = [1.0, 1.0, -6.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -4.0]

### Tabular data supporting description

#### Baseline characteristics
#### outcomes

#### multi-parameter conceptual variables 
# Discussion

## Interpret and explain the significance of results

## How do results fit in to the wide picture of pathing, planning, and heuristics

# Bibliography/References




