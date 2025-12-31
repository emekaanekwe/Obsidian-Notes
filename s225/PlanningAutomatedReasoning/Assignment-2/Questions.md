## Transforming into Ghost-Pac
### If agent A transitions to the enemy-state space
1. at what time specifically does it happen?
	1. does a time step pass for the agent to transform?
	2. Can the agent still move while transforming?

## Grid
### If the grid is divided in "half"
1. are there any cases in which the length of the grid is uneven?
### Structure
1. Is the structure of the grid constant throughout the entire competition (like a football field)?
2. Are they randomized? If so, is the game determined by a single match or a "best out of x"?
3. if it is a tournament system, does one's position in the leader boards affect the grid structure they will compete in (e.g., the higher up you go, the more complex the grid) ? 
## Scoring
### Total or Average Score?
1. is your score based on a single randomized game, or a series of games?
2.  
## Power Capsules
### If the agent eats a power capsule
1. does a time step pass while eating the capsule?
### When a power capsule is eaten
1. What does it mean for the enemies to be "feared"?
	1. is it "move semi-randomly away from empowered agent"
## Colliding with Enemies
### If opposing agents come into contact
1. who has decision priority?
2. are the decisions simply queued?
3. do we have control over that priority (e.g., can we install a game-theoretic event)?

## Learning
### Diversify Learning
1. Do we have the option to implement Q-Learning at tn and then heuristic searching at tm, depending on the situation we tell our planner?

---

Hi Emeka,

There is no transformation. It is a direct agent = Ghost in home territory, and is Pacman in enemy territory. This will happen instantaneously on the timestep that you move (and thus are in) the respective location.

Each map should be symmetric, so the half-way point should always be fair (you can check these yourself by visualising each map).

You play 49 games against opponents - I believe there is 7 games on each of the 7 maps (this is in the documentation). You play all of these maps each time you vs an opponent. Nothing changes with ladder position.

Your score is based on all games (though we have come to the conclusion this _may_ be computationally infeasible this year given the number of students... we are working on this). Nonetheless, its just a situation of "win as many games as possible", and more wins = higher ladder position. Again, a single "game" against an opponent is actually 49 games (win, tie, loss).

Eating the capsule happens the instant you step onto the capsule. Time is always simple and discrete (no hidden waiting). "Feared" enemies mean that Pacman can actually _eat and kill_ the ghosts---the same way you can in a normal Pacman game (if you have any experience). The other team still has _full control_ over their agents (they will not always run away). Evidently you want to code your own agents to not chase and kill themselves if a power pellet is currently active.

You do not have control over agent priority - but this is _given_ information (agent ID = priority). Essentially though, the four agents move one a time according to the priority. This somewhat equates to one real-world timestep.  
  
EDIT: agent id is separated by teams: one team has IDs 1,3 and the other 2,4 for fairness.

You can absolutely pick and choose how you create low level actions - and mix and match algorithms and ideas! Your strategy is up to you. Note that PDDL **is absolutely necessary to implement** for the high-level actions. Again, the resolution/control given to high/low-level actions is also up to you.