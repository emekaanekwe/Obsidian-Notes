

# 1) What you need to do to complete the assignment

**A. Set up & sanity-check the environment**
    
- Learn to run matches from the CLI and vary maps/seeds: `python capture.py -r staffTeam.py -b berkeleyTeam.py`, try `-l RANDOM23`, `-n`, `-q/-Q`, and `-i` to control games, output, and limits.
    

**B. Implement your team in `myTeam.py`**

- The simulator calls `createTeam` (to spawn your two agents) and repeatedly calls each agent’s `chooseAction(state)` every turn. Start from `MixedAgent` or from scratch (inherit `CaptureAgent`).
    
- Provide a **high-level planner** (PDDL) and a **low-level planner** (heuristic search or learning). The given baseline wires these together; your job is to improve/replace them.
    

**C. High-level (PDDL)**

- Convert the current game state to a PDDL problem: extend `get_pddl_state` to add objects/predicates you need; pick a goal in `getGoals`; call `getHighLevelPlan` to solve (Piglet). Keep an eye on “close/medium/long” distance constants.
    
- Decide **when to replan**: if preconditions fail or the world changes materially (e.g., you were eaten, capsule flipped scared timers), recompute the high-level plan.
    

**D. Low-level (movement)**

- Either finish `getLowLevelPlanHS` (A*/any-angle not required—grid cardinal moves only) **or** improve the provided approximate Q-learning path selector in `getLowLevelPlanQL`. Each high-level action should map to an appropriate low-level strategy.
    
- If using Q-learning: refine feature functions, rewards, and the `QLWeights` structure; train with `self.training=True`, manage `epsilon/alpha/discount`, and persist weights.
    

**E. Use the environment APIs effectively**

- Learn `GameState` + `CaptureAgent` convenience methods (`getFood`, `getWalls`, defending/attacking food getters, etc.) and the `Grid` representation. This is where most students either shine or stall.
    

**F. Test, compare, iterate**

- Regularly play vs `staffTeam.py` and `berkeleyTeam.py` on multiple maps and random seeds; profile failure modes (camping ghosts, reckless foraging, poor escapes).
    

**G. Prepare for the contest & report**

- **Contest:** you must convincingly beat the staff baseline on 49 games across 7 maps (≥28 wins) to pass Criterion 1, then accumulate victory points vs peers.
    
- **Report:** describe your strategy (HL+LL), analyze strengths/limits (complexity, guarantees), show experiments (success rates, planning time), and justify design choices. Submit `src` + PDF report; deadline **Fri 31 Oct 2025, 11:55 pm**.
    

---

# 2) The most important Pacman environment pieces to start with

**Core rules that drive design choices**

- **Role switching:** On your half you’re a **ghost**; across the border you’re **Pacman**. Points are scored only when carried food is returned home; being caught drops your backpack and respawns you. This pushes for **risk-aware foraging** and **safe return paths**.
    
- **Observation model:** You only see opponents within **Manhattan 5**; otherwise you have **noisy distance** readings with error **±6**. Plan under **partial observability**; encode predicates/features that reflect proximity beliefs and “last seen” tracking.
    
- **Capsules:** Scared timers last **40 steps**—time-bounded offensive windows and defensive retreats; use this to gate “commit vs bail” logic.
    
- **Game limits:** Ends when all but two dots are returned or after **450 agent timesteps** (i.e., 1800 actions overall). Optimize for **time-bounded scoring**, not just elegance.
    

**APIs / Files you’ll lean on first**

- `chooseAction(state)` in `myTeam.py`: the **control loop** that stitches HL and LL plans. Read it line by line.
    
- `get_pddl_state`, `getGoals`, `getHighLevelPlan`: the high-level pipeline. Extend predicates, add actions/goals in `myTeam.pddl`.
    
- `getLowLevelPlanQL` / `getLowLevelPlanHS` and `posSatisfyLowLevelPlan`: movement generation and plan reuse checks.
    
- `GameState` + `CaptureAgent` **convenience methods** (e.g., `getFoodYouAreDefending`, `getWalls`, team/score queries) and the `Grid` structure accessed as `grid[x][y]` with `asList()` for coordinates. These give you map topology, foods, and walls quickly.
    

**Simulator hooks & CLI**

- `capture.py` args: `-r/-b` teams, `-l` layout or `RANDOM<seed>`, `-n` repetitions, `-q/-Q` quiet modes, `-i` max moves. You’ll need these to run fast training batches and reproducible tests.
    

---

# 3) What an **excellent** solution looks like (and how to get there)

Think of “excellent” as: **beats staff decisively**, shows **principled HL/LL design**, **adapts online**, and demonstrates **team coordination**—with clear empirical evidence in your report. Concretely:

**A. High-level planning that’s richer than baseline**

- **More expressive PDDL domain**: Add predicates that matter under partial observability and time limits (e.g., `enemy_likely_close`, `safe_corridor`, `return_risk_low`, `capsule_window_active`, `food_cluster_k`). Extend actions beyond “attack/defend/escape” to include **“probe/peek”**, **“escort/cover”** (one agent screens corridors while the other carries food), and **“capsule-strike”** timed to the 40-step window. Implement in `myTeam.pddl` and collect in `get_pddl_state`.
    
- **Goal arbitration**: Replace static priorities with **state-contingent goal selection** (score margin, time left, noisy enemy proximity, teammate intent). Replan HL when preconditions break or when outcomes surprise you (caught, capsule taken, layout bottleneck).
    

**B. Low-level planning tailored per high-level action**

- **Distinct LL strategies per HL action**:
    
    - _Forage_: A* to **safe** food chosen by a **safety-weighted utility** (distance minus enemy risk minus corridor choke risk).
        
    - _Return_: Shortest **low-exposure** path home (penalize narrow corridors when enemies are near).
        
    - _Defend_: Greedy interception to last-eaten food or inferred opponent path using noisy distances.
        
    - _Escape_: Avoidance planner that maximizes **margin to enemy isovists**; treat capsules as dynamic waypoints when available.  
        This is explicitly called out as HD-level craft: different LL plans for distinct HL actions.
        
- If you keep learning: redesign **features** to be smooth, normalized, and aligned with rewards (e.g., distance-to-goal ∈ [0,1], choke-risk ∈ [0,1], carry-load ratio, time-to-timeout). Avoid sparse/contradictory rewards (classic pitfall). Train with exploration off for submission.
    

**C. Team coordination & information sharing**

- Have agents **avoid redundant foraging** (partition food clusters or sides), **hand off roles** (one presses offense while the other screens), and **signal intent** via shared state (class variables or lightweight shared beliefs). This is explicitly part of the top rubric tier (“decisions consider teammate strategy; cooperate & share info”).
    

**D. Robustness to partial observability**

- Maintain **beliefs** from noisy distances (±6) and last sightings to drive **defensive patrols** and **risk estimates** (even if you don’t implement full POMDP). Encode summary predicates like `enemy_likely_on_route(home_gate)` for PDDL and features like `min_estimated_enemy_distance`.
    

**E. Capsule-aware tempo**

- Build **time-windowed plays**: when a capsule is eaten, flip to a “power mode” HL plan that aggressively invades/escorts for ~40 steps; otherwise prefer conservative scoring and safe returns.
    

**F. Solid experimental evidence and reporting**

- Compare vs `staffTeam.py`/`berkeleyTeam.py` across **multiple layouts** and random seeds; report **win rate over 49-game blocks**, **average score**, **planning time per move**, and **ablation** (with/without a new predicate/action/feature). The rubric rewards a well-structured, referenced, experiment-backed report.
    

**G. Practical workflow**

- Batch runs with `-Q -n 100 -l RANDOM23` across a few seeds; disable training for final evals; persist and reset `QLWeights` responsibly (delete file when adding features). Use `-i` to stress behavior under tight horizons (e.g., 300 vs 450 steps).
    

---

## Quick “first two weeks” action plan (skeptical but forward-leaning)

1. **Get deterministic baselines running** on multiple maps; snapshot results.
    
2. **Refactor `chooseAction`** for clear HL→LL stages and replanning triggers. (Baseline code mixes concerns; make the logic auditable.)
    
3. **Add two high-impact PDDL actions** (e.g., `escort_teammate_return`, `probe_corridor`) + 3–5 predicates that capture partial-obs risks.
    
4. **Implement a heuristic LL for “return” and “escape”** (these two phases decide most games). Keep QL for “forage” if you like, but give “escape” a hand-crafted avoidance planner first.
    
5. **Introduce teammate-aware goal arbitration** (score/time/risk).
    
6. **Run ablations** to prove each piece adds win-rate/time-to-score gains.
    

If you share your current `myTeam.py` and any logs, I’ll mark up (line-by-line) where to insert the new predicates/actions and stub the LL planners for “return” and “escape” first.

# Coding in both PDDL and Python

### What goes in PDDL

- Extend or modify the **domain/actions/predicates** in `myTeam.pddl` so the planner can express richer decisions (e.g., attack/defend/escape variants, capsule plays, coordination). The brief provides a starter domain and encourages you to add advanced predicates/actions.
    

### What goes in Python

- **Glue the game to PDDL**: implement `get_pddl_state` (convert GameState→:init/:objects) and `getGoals` (choose goal sets); call the solver via `getHighLevelPlan`.
    
- **Agent control loop** in `chooseAction` (high-level plan→low-level actions; replan as needed).
    
- **Low-level planner**: either finish the provided **Q-learning** (features/rewards/weights) or implement **heuristic search** (`getLowLevelPlanHS`). Each **high-level action should have its own low-level strategy**.
    
- **Setup details** (e.g., ensuring `self.pddl_solver` points at your domain file in `registerInitialState`).
    

### Grading reality check (why both matter)

- To reach **D/HD**, the rubric expects **new PDDL actions/predicates or new goal functions**, **better goal prioritisation**, and **improved/custom low-level planning** (often distinct per high-level action). You won’t hit top bands by editing only Python or only PDDL.
    

Pragmatically: start from `MixedAgent` (PDDL HL + Q-learning LL) and evolve both layers—or, if you prefer, start from `emptyTeam.py` and build your own, but you’ll still need HL PDDL + LL Python to meet the brief.
