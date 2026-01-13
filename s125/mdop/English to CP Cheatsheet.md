
## Tip:
_When solving a problem, always start by writing down the decision variables, constraints, and objective in plain English. Then, translate each part step by step into MiniZinc._

---

$$
\begin{align}
\text{It's crucial to adhere to the specific strategies when approaching a problem to model. One must first check the requirements given by MiniZinc, the goal, the model structure, what can be formally implied from the word problem, and the constraints: }
\end{align}
$$

### Nathan's Recommendation
Remember, there are 4 parts of the model that need to clearly define:

1. The input variable: the problem which is known.
2. The decision variable: what we want to achieve, which is unknown in advance. A.k.a the solution
3. The constraints: the dependent between input and decision var.
4. The objective: what value to determine which solution is better than which.
 
 Then your job is to connect everything and writing it in MiniZinc language ways. If you visualise the problem correctly, then it’s not hard to you to write constraints.


### Look at Input Variables
	see what is already known to us by the problem 
		example: there is a team made up of captains and players)


## 1. Read the Word Problem
	what are the knowns in the problem?

	what are the unknowns?

## 2. Run the file
	check what requirements are needed, based on the output/log

## 3. Identify a Pattern
###### *Most constraint programming problems fit into one or more of these common patterns:*

| Problem Type         | Example                                            | MiniZinc Strategy                                              |
| -------------------- | -------------------------------------------------- | -------------------------------------------------------------- |
| **Assignment**       | "Assign each student to one class."                | Use `sum`, `forall`, and binary decision variables.            |
| **Packing/Knapsack** | "Maximize value under weight limit."               | Use `sum`, `<=`, and binary variables to model item selection. |
| **Scheduling**       | "Task A must finish before Task B."                | Use `int` decision variables and inequality constraints.       |
| **Graph Problems**   | "Find the shortest path or minimum spanning tree." | Use adjacency matrices, distances, and path constraints.       |
| **Set Covering**     | "Select a minimal set that covers all elements."   | Use `sum` and `forall` to ensure coverage.                     |
| **Sequencing**       | "Order elements to minimize conflicts."            | Use integer variables with order-based constraints.            |

## 4. Define the Input Variables

	these would be the individual objects or sets (e.g., teams, warehouses, costs)

## 5. Define the Decision Variables

	Find he unknown values that the solver will use to satisfy any constraints and thus find a solution.
		Examples: the unknown value of an order, the unknown value of a set of items,  

## 6. Create Constraints Based on the Object's Properties
   
   **Use `sum`, `forall`, `exists` for Quantifiers**
	
	what is a general limitation of the objects? 
		Example: a threshold, assignment of a to b, a summation

## 7. Model Pairwise Relationships with `where` and other conditions

## 8. Define the Objective Function

	if maximizing or minizing value

---
# Syntax for Patterns

## **1. Scheduling Problems**

### **Word Problem:** Employee Shift Scheduling

A company has **4 employees** and needs to assign **work shifts** across **7 days**. Each employee can work at most **5 shifts per week**, and each shift requires **at least one employee**. Find a valid work schedule that satisfies these constraints.

### **MiniZinc Code:**

```minizinc
int: num_employees = 4;
int: num_shifts = 7;
array[1..num_employees, 1..num_shifts] of var 0..1: schedule;

% Each employee works at most 5 shifts
constraint forall(e in 1..num_employees)(
    sum(s in 1..num_shifts)(schedule[e, s]) <= 5
);

% Each shift must have at least one employee
constraint forall(s in 1..num_shifts)(
    sum(e in 1..num_employees)(schedule[e, s]) >= 1
);

solve satisfy;
```

### **Identification:**

- **Input Variables:**
    
    - `num_employees = 4`, `num_shifts = 7`
        
- **Decision Variables:**
    
    - `schedule[e, s] ∈ {0,1}` → 1 if employee `e` is assigned to shift `s`, otherwise 0.
        
- **Constraints:**
    
    - Employees work at most 5 shifts.
        
    - Each shift has at least one employee.
        

---

## **2. Graph Problems**

### **Word Problem:** Graph Coloring

A city needs to divide **6 districts** into separate political zones. Any two neighboring districts **cannot have the same zone**. We need to assign each district one of **3 available zones**.

### **MiniZinc Code:**

```minizinc
int: num_districts = 6;
set of int: zones = 1..3;
array[1..num_districts] of var zones: district_zone;
array[1..num_districts, 1..num_districts] of 0..1: adjacency =
    [| 0, 1, 0, 1, 0, 0 |
       1, 0, 1, 1, 0, 0 |
       0, 1, 0, 1, 1, 0 |
       1, 1, 1, 0, 1, 1 |
       0, 0, 1, 1, 0, 1 |
       0, 0, 0, 1, 1, 0 |];

constraint forall(i, j in 1..num_districts where adjacency[i, j] = 1)(
    district_zone[i] != district_zone[j]
);

solve satisfy;
```

### **Identification:**

- **Input Variables:**
    
    - `num_districts = 6`, `zones = 1..3`, adjacency matrix
        
- **Decision Variables:**
    
    - `district_zone[i] ∈ {1, 2, 3}` → Assigned zone for each district.
        
- **Constraints:**
    
    - No two adjacent districts have the same zone.
        

---

## **3. Set Covering & Partitioning**

### **Word Problem:** Facility Location

A company has **3 warehouses** that can serve **5 customers**. Each warehouse has a **limited service range**. Find the minimal number of warehouses needed to serve all customers.

### **MiniZinc Code:**

```minizinc
int: num_warehouses = 3;
int: num_customers = 5;
array[1..num_warehouses, 1..num_customers] of bool: can_serve =
    [| 1, 0, 1, 0, 1 |
       1, 1, 0, 1, 0 |
       0, 1, 1, 1, 1 |];

array[1..num_warehouses] of var 0..1: open_warehouse;
array[1..num_customers] of var 0..1: customer_served;

constraint forall(c in 1..num_customers)(
    customer_served[c] = max(w in 1..num_warehouses)(can_serve[w, c] * open_warehouse[w])
);

solve minimize sum(w in 1..num_warehouses)(open_warehouse[w]);
```

### **Identification:**

- **Input Variables:**
    
    - `num_warehouses = 3`, `num_customers = 5`, `can_serve[w, c]`
        
- **Decision Variables:**
    
    - `open_warehouse[w] ∈ {0,1}` → Whether warehouse `w` is open.
        
    - `customer_served[c] ∈ {0,1}` → Whether customer `c` is served.
        
- **Constraints:**
    
    - Every customer must be served by at least one open warehouse.
        

---

## **4. Assignment Problems**

### **Word Problem:** Task Assignment

A company has **4 employees** and **4 tasks**. Each employee can do only **one task**, and each task must be **assigned to exactly one employee**. The goal is to minimize total cost.

### **MiniZinc Code:**

```minizinc
int: num_workers = 4;
array[1..num_workers, 1..num_workers] of int: cost =
    [| 10, 20, 30, 40 |
       15, 25, 35, 45 |
       40, 30, 20, 10 |
       25, 35, 15, 45 |];

array[1..num_workers] of var 1..num_workers: assignment;

constraint alldifferent(assignment);

solve minimize sum(w in 1..num_workers)(cost[w, assignment[w]]);
```

### **Identification:**

- **Input Variables:**
    
    - `num_workers = 4`, `cost matrix`
        
- **Decision Variables:**
    
    - `assignment[w] ∈ {1..4}` → Task assigned to worker `w`.
        
- **Constraints:**
    
    - Each worker gets a unique task (`alldifferent()`).
        

---

## **5. Vehicle Routing & Logistics**

### **Word Problem:** Traveling Salesperson Problem (TSP)

A salesperson must visit **5 cities** exactly **once** and return to the start, minimizing travel cost.

### **MiniZinc Code:**

```minizinc
int: num_cities = 5;
array[1..num_cities, 1..num_cities] of int: distances =
    [|  0,  10,  15,  20, 25 |
       10,   0,  35,  25, 30 |
       15,  35,   0,  30, 20 |
       20,  25,  30,   0, 15 |
       25,  30,  20,  15,  0 |];

array[1..num_cities] of var 1..num_cities: tour;

constraint alldifferent(tour);

solve minimize sum(i in 1..num_cities-1)(
    distances[tour[i], tour[i+1]]
) + distances[tour[num_cities], tour[1]];
```

### **Identification:**

- **Input Variables:**
    
    - `num_cities = 5`, `distances matrix`
        
- **Decision Variables:**
    
    - `tour[i] ∈ {1..5}` → Order in which city `i` is visited.
        
- **Constraints:**
    
    - Each city is visited exactly once.
        

---

This covers intermediate examples for each pattern with clearly identified components. Let me know if you want any modifications or more details! 🚀
### **"Every X must satisfy Y"**:
   - Translation: Use a `forall` loop to ensure that every element in a set or array satisfies a condition.
   - Example: "Every student must be assigned to exactly one class."
     ```minizinc
     forall(s in Students)(
         sum(c in Classes)(assignment[s, c]) == 1
     );
     ```

### **"No two X can be the same"**:
   - Translation: Use the `alldifferent` constraint to ensure all elements in a set or array are unique.
   - Example: "No two students can have the same schedule."
     ```minizinc
     alldifferent(schedule);
     ```

### **"If X, then Y"**:
   - Translation: Use an implication (`->`) to model conditional constraints.
   - Example: "If a student is assigned to class A, they cannot be assigned to class B."
     ```minizinc
     forall(s in Students)(
         assignment[s, A] -> assignment[s, B] == 0
     );
     ```

### **"Exactly n of X must satisfy Y"**:
   - Translation: Use a sum constraint with equality to enforce the exact count.
   - Example: "Exactly 3 classes must have more than 15 students."
     ```minizinc
     sum(c in Classes)(sum(s in Students)(assignment[s, c]) > 15) == 3;
     ```

### **"Minimize/Maximize Z"**:
   - Translation: Use the `minimize` or `maximize` keyword to define the objective function.
   - Example: "Minimize the total cost of assignments."
     ```minizinc
     minimize sum(s in Students, c in Classes)(cost[s, c] * assignment[s, c]);
     ```

### **"X must be between A and B"**:
   - Translation: Use inequality constraints to bound a variable.
   - Example: "The number of students in each class must be between 10 and 30."
     ```minizinc
     forall(c in Classes)(
         10 <= sum(s in Students)(assignment[s, c]) <= 30
     );
     ```

---

### **How to Improve Your Skills**
Here are some recommendations to help you get better at translating word problems into MiniZinc:

1. **Practice, Practice, Practice**:
   - Start with simple problems and gradually move to more complex ones.
   - Use online platforms like [MiniZinc Challenges](https://www.minizinc.org/challenges.html) or [CSPLib](http://www.csplib.org/) for practice problems.

2. **Break Down the Problem**:
   - Identify the decision variables (what you need to decide).
   - Identify the constraints (what rules must be followed).
   - Identify the objective (what you’re trying to minimize or maximize).

3. **Learn by Example**:
   - Study existing MiniZinc models for common problems (e.g., scheduling, packing, routing).
   - The [MiniZinc Handbook](https://www.minizinc.org/doc-2.7.4/en/index.html) has many examples and explanations.

4. **Use Debugging Tools**:
   - Use the MiniZinc IDE to visualize and debug your models.
   - Add `output` statements to print intermediate results and understand how your constraints are working.

5. **Collaborate and Discuss**:
   - Work with classmates or join online communities (e.g., the MiniZinc Google Group or forums) to discuss problems and solutions.

---

### **Recommended Books**
Here are some books that can help you master constraint programming and MiniZinc:

1. **"Handbook of Constraint Programming"** by Francesca Rossi, Peter van Beek, and Toby Walsh:
   - A comprehensive guide to constraint programming concepts and techniques.

2. **"Programming with Constraints: An Introduction"** by Kim Marriott and Peter J. Stuckey:
   - A beginner-friendly introduction to constraint programming.

3. **"Modeling and Solving with MiniZinc"** by Kim Marriott, Nicholas Nethercote, and Peter J. Stuckey:
   - A practical guide to modeling problems in MiniZinc, written by the creators of MiniZinc.

4. **"Constraint Processing"** by Rina Dechter:
   - A deeper dive into the theory and algorithms behind constraint satisfaction problems.

5. **"Principles of Constraint Programming"** by Krzysztof Apt:
   - A theoretical yet accessible book on the principles of constraint programming.

---

### **Online Resources**
- [MiniZinc Tutorials](https://www.minizinc.org/tutorials/index.html): Official tutorials for MiniZinc.
- [CSPLib](http://www.csplib.org/): A library of constraint satisfaction problems with solutions.
- [MiniZinc Examples](https://github.com/MiniZinc/minizinc-examples): A GitHub repository with example models.
