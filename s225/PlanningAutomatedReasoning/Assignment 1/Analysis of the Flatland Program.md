# Question 3 Implementation

### ReservationTable Class - The "Traffic Controller"

  **Purpose**: Acts like an air traffic control system, keeping track of where
  every train will be at every moment in time to prevent collisions.

  Analogy: Imagine you're managing a busy airport. You need to track:
  - Which runway each plane will use at each minute (cell reservations)
  - Which planes are **switching runways to avoid mid-air collisions (edge** 
  **reservations**)

  class ReservationTable:
      def __init__(self):
          self.cells = {}    # (row, col, time) → agent_id  
          self.edges = {}    # ((prev_pos), (new_pos), time) → agent_id

  *Two Types of Conflicts Prevented:*

  1. Vertex Conflicts (Same Cell, Same Time)

  def is_reserved(self, r, c, t, aid):
      return (r, c, t) in self.cells and self.cells[(r, c, t)] != aid
Two planes trying to land on the same runway at the same time -CRASH!

  2. Edge Conflicts (Swapping Positions)
  def is_edge_reserved(self, r0, c0, r1, c1, t, aid):
      return ((r0, c0), (r1, c1), t) in self.edges and self.edges[((r0, c0),
   (r1, c1), t)] != aid
  Analogy: Two trains approaching each other on the same track section -
  they'd collide head-on while passing!

  Safe Intervals Logic:
  def get_safe_intervals(self, x, y, horizon):
      # Returns time periods when cell (x,y) is FREE
      blocked = self.blocked_times_for_cell(x,y,horizon)
      # Convert blocked times into safe intervals
  Analogy: Finding open gaps in a schedule - "The conference room is booked
  9-10am and 2-4pm, so it's available 10am-2pm and after 4pm"

### sipp_plan_one() - The "Smart Route Planner"

  Purpose: Plans a single train's path using Safe Interval Path Planning
  (SIPP) - like a GPS that knows about traffic jams in the future.

  Analogy: You're planning a road trip, but you have a crystal ball showing
  you exactly when each road segment will be blocked by traffic. You plan
  your route to arrive at each *segment during a "safe interval" when it's*
  *clear.*

  def sipp_plan_one(agent, rail, reservation_table, max_timestep):
      start = tuple(agent.initial_position)
      goal = tuple(agent.target)

      # A* search in space-time
      while openq:
          _, g, _, (r, c, t, dir_in) = heapq.heappop(openq)

          # Goal check
          if (r, c) == goal:
              # Reconstruct and return path

          # Generate successors: WAIT and MOVE

#### Key Features:

**Space-Time Search**

  Instead of just searching (x,y) positions, it searches (x,y,t) - position
  + time
  Analogy: Instead of just asking "Can I go to Main Street?", you ask "Can I
   go to Main Street at 3:15 PM?"

  **Railway Constraints**

  valid_transitions = rail.get_transitions(r, c, dir_in)
  for action in range(4):  # N,E,S,W
      if not valid_transitions(action):
          continue  # Can't go this direction from current track piece
  Analogy: A train can't suddenly turn 90° - it must follow the tracks. If
  you're on a straight track going north, you can't suddenly go east.

  **Reservation Checking**

  # Check if destination cell is free
  if reservation_table.is_reserved(nr, nc, t_next, aid):
      continue
  # Check if edge movement conflicts with another train
  if reservation_table.is_edge_reserved(nr, nc, r, c, t_next, aid):
      continue

  Mathematical Foundation:
  - State Space: (row, col, time, direction)
  - Cost Function: g(n) = time steps taken
  - Heuristic: h(n) = Manhattan distance to goal
  - Evaluation: f(n) = g(n) + h(n) (A* formula)

### space_time_a_star() - The "Alternative Pathfinder"

  Purpose: Another implementation of space-time pathfinding, but with a
  simpler approach that doesn't use safe intervals.

  Analogy: While sipp_plan_one() is like a sophisticated GPS with real-time
  traffic data, space_time_a_star() is like a basic GPS that just checks "**Is**
   **this road blocked right now?**" at each step.

  def space_time_a_star(agent_id, start, goal, rail, max_timestep, 
  reservations):
      # Create space-time nodes: (x, y, t)
      start_node = SpaceTimeNode(start[0], start[1], 0, 0, manhattan(start,
  goal))

      while open_list:
          current = heappop(open_list)

          # Try WAIT action
          if not reservations.is_reserved(current.x, current.y, current.t +
  1, agent_id):
              # Add wait successor

          # Try MOVE actions  
          for move_dir in range(4):
              if cell_transitions & (1 << move_dir):  # Can move this 
  direction?
                  # Add move successor

#### Key Differences from SIPP:

  **Simpler Conflict Checking**

  - SIPP: Uses safe intervals - "This cell is free from t=5 to t=8"
  - Space-time A*: Point checks - "Is this cell free at exactly t=6?"

  Railway Movement Logic

  # Check if current cell allows movement in this direction
  if cell_transitions & (1 << move_dir):
      dx, dy = DXY[move_dir]
      nx, ny = current.x + dx, current.y + dy
  Analogy: Like checking if your current train car has a door on the left
  side before trying to step left.

  When to Use Each:
  - SIPP (sipp_plan_one): Better for dense environments with many agents
  - Space-time A* (space_time_a_star): Simpler, good for sparse environments

### plan_agent_paths() - The "Coordination Orchestrator"

  Purpose: *Coordinates* the planning of all agents using *Prioritized Planning*
   - like a train station dispatcher giving each train a departure slot in
  order.

  Analogy: Imagine you're the manager of a busy train station. Instead of
  letting all trains plan their routes simultaneously (chaos!), you handle
  them one at a time:
  1. Train #1 plans its route and reserves its track time
  2. Train #2 plans around Train #1's reservations
  3. Train #3 plans around both previous trains' reservations
  4. Continue until all trains have conflict-free paths

  def plan_agent_paths(agents, rail, max_timestep):
      n = len(agents)
      paths = [[] for _ in range(n)]
      reservation_table = ReservationTable()  # Shared "booking system"

      for aid, agent in enumerate(agents):
          # Plan path for THIS agent, avoiding all previous agents
          path = sipp_plan_one(agent, rail, reservation_table, max_timestep)
          paths[aid] = path

          # Reserve this agent's path so future agents avoid it
          for t in range(min(len(path), max_timestep)):
              pos = path[t]
              prev = path[t - 1] if t > 0 else pos
              reservation_table.reserve(pos, t, prev, aid)

  Why This Works:

  **Sequential Planning**

  - Agent 0: Plans with empty reservation table (easy!)
  - Agent 1: Plans avoiding Agent 0's reserved cells
  - Agent 2: Plans avoiding Agent 0's AND Agent 1's reserved cells
  - Agent n: Plans avoiding all previous agents' reservations

  **Completeness Guarantee**

  Each agent gets a path (maybe sub-optimal, but conflict-free)

  Analogy: Like booking a hotel during a busy conference - the first person
  gets their ideal room, the second person gets their ideal room from what's
   left, etc. Everyone gets a room, but later bookers have fewer choices.

  Algorithm Properties:
  - Runtime: O(n × single_agent_search_time) - linear in number of agents!
  - Optimality: Sub-optimal (later agents get worse paths)
  - Completeness: Always finds solution if one exists
  - Practical: Scales to hundreds of agents

### replan() - The "Crisis Management System"

  **Purpose:** *Handles dynamic problems* during execution - when trains
  malfunction or can't follow their planned paths. It's like an emergency
  dispatcher who reroutes traffic around accidents.

  Analogy: You're managing a choreographed dance performance when suddenly:
  - Some dancers get injured (malfunctions) and must sit still
  - Some dancers miss their cue and are out of position (failed agents)The
  replan function is like the choreographer quickly adjusting everyone
  else's moves around these problems.

  def replan(agents, rail, current_timestep, existing_paths, max_timestep, 
             new_malfunction_agents, failed_agents):

      updated = [p[:] for p in existing_paths]  # Copy existing paths
      reservation_table = ReservationTable()

      # Step 1: Re-reserve all existing paths up to current time
      # Step 2: Handle new malfunctions (force agents to wait)  
      # Step 3: Replan failed agents around existing reservations

#### Three-Step Strategy:

**Step 1: Preserve the Past**

  # Re-reserve all agents' paths up to current_timestep
  for aid, path in enumerate(existing_paths):
      for t in range(min(len(path), current_timestep + 1)):
          pos = path[t]
          prev = path[t - 1] if t > 0 else pos
          reservation_table.reserve(pos, t, prev, aid)
  Analogy: "Everyone stick to the original plan up until now - we can't
  change the past!"

**Step 2: Handle Malfunctions**

  for aid in new_malfunction_agents:
      # Get malfunction duration
      malfunction_duration = getattr(agent, 'malfunction', 0)
      # Force agent to wait at current position
      wait_len = min(malfunction_duration, max_timestep - current_timestep)
      base = updated[aid][:current_timestep + 1]
      tail = [pos] * wait_len  # Wait at current position
      updated[aid] = base + tail
  Analogy: "If a dancer is injured, they must sit in place until they
  recover. Mark that spot as 'occupied' so others don't bump into them."

**Step 3: Selective Replanning**

  for aid in failed_agents:
      if aid in new_malfunction_agents:
          continue  # Already handled above

      # Replan this agent around all existing reservations
      new_path = sipp_plan_one(pseudo, rail, reservation_table,
  max_timestep)
      updated[aid] = new_path

      # Reserve the new path
      for t in range(min(len(new_path), max_timestep)):
          reservation_table.reserve(pos, t, prev, aid)

**Why This Is Smart:**

  1. Minimal Disruption

  - Only replans agents that actually have problems
  - Keeps successful agents on their original paths
  - Prevents cascading failures

  2. Prioritized Recovery

  - Malfunctioning agents get priority (they can't move anyway)
  - Failed agents get new paths around the "frozen" agents
  - Order matters: first-come, first-served for new paths

  Analogy: When a car accident blocks a highway lane:
  1. Preserve: Cars already past the accident keep going normally
  2. Handle blockage: The crashed cars stay put until towed
  3. Selective rerouting: Only cars approaching the accident need new routes

### Summary: How They All Work Together

  Your algorithm is like a well-orchestrated transportation system:

####  The Big Picture Flow:

  1. ReservationTable = The central "air traffic control" system tracking
  all movements
  2. plan_agent_paths() = The initial dispatcher giving each train a
  departure time
  3. sipp_plan_one() = The smart GPS that finds conflict-free routes using
  safe intervals
  4. space_time_a_star() = Alternative GPS with simpler conflict checking
  5. replan() = The crisis manager who handles emergencies and breakdowns

  Mathematical Properties:

  - Completeness: Will always find paths if they exist (agents can always
  wait)
  - Optimality: Sub-optimal but fast (prioritized vs. optimal search)
  - Time Complexity: O(agents × branching^depth) - scales linearly with
  agents
  - Space Complexity: O(cells × timesteps) for reservation table

  Why This Approach Is Excellent for Flatland:

  ✅ Handles railway constraints (can only move on valid track
  transitions)✅ Prevents all collision types (vertex + edge conflicts)✅
  Scales to many agents (prioritized planning is efficient)✅ Robust to 
  malfunctions (selective replanning minimizes disruption)✅ Real-time 
  capable (doesn't recompute everything from scratch)

  Your implementation demonstrates deep understanding of both the
  theoretical foundations (A*, SIPP, multi-agent coordination) and practical
   engineering concerns (efficiency, robustness, maintainability). Excellent
   work!











1.  start by reviewing the make files or build files to see what components are being built.
2. Pick one or two key components, build in debug mode, and set breakpoints. 
3. add log statements
4. Run the code and trace the execution flow: following the path that the program takes as it runs, from start to finish, to understand how different components interact and in what order things happen. It's like creating a roadmap of the program's behavior.
5. Take notes along the way. 
6. Ask yourself "Why does it exist?" or "what problem is it solving?" 
7. break the code, and figure out why it did
8. make a valid code change and test output 
9. Gradually expand this approach to more components, and over time, you'll develop a clear understanding of the code, its purpose, and how everything fits together. This method works even when documentation is limited.

####  To understand a code base quickly, understand the why. 
1. 
#### Pick the component you think you would have the best chance of understanding.
1. Then go very specific. Pick a very small piece of this component
2. start expanding your knowledge from there.
3. If you get stuck on a particular component/step, skip it. Move on.
4. Just understand what it takes as input, and what it generates as output. Skip everything else

#### Logging & Testing
1. add log statements and run code asap
2. break the code, and figure out why it broke
3.  make a valid code change that affects the output

#### Reading Repos Approach
1. read the documentation (README's)
2. Run the project and us the debugger to follow the execution of the code
	1. Check for unit, integration, or functional tests in the project
3. Use the END Product/Service
	1. What is the goal
	2. key features of MVP
	3. what is the dev methodology

#### GPT/DeepSeek's Approach
1. start with high-level documentation
	1. review any architecture diagrams/papers
2. identify the key directories
3. identify the key components
4. tarce execution flow
5. use debugging tools
6. study any test environments made
7. 
