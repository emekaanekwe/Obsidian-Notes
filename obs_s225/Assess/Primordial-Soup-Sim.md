# Coding
*when it comes to coding, there needs to be a level of excitement that will not be disturbed by high amounts of debugging or looking up syntax*

**Requirements**
	some basic knowledge of the programming language

***= = = = =====IDEA======== = = = =***

#### Choose your own adventure
why not try mkaing a game where it is choose your ownd adventure high stakes puzle game that requires you to write a line or a block of code in order to pass to the next level.
EXAMPLE:
	"You step into the room with your holographic console. You see a bipedal robot with two arms and buzzsaws for hands. It starts charging towards you. What do you do?"
```python
# option 1
print("Do not attack me!")
# option 2
def deplete_energy(x):
  x = 0
  return x
deplete_energy(robot)
# option 3
from bot import controls
for i in robot.legs:
  controls.shutdown(i)
```


### Hackathons

### Find Use Case Project

# Primordial Soup Simulator

<mark class="blue">
For this project, I think it is important that I clearly identify the rudimentary conditions:<br>
1. what is the underlying desideratum?<br>
2. what is the atomic object that will be simulated?<br>
3. What constraints will be needed to ensure the simulation is as close to reality as possible?<br>
</mark> 
# Desideratum
**Objective Function**
	Make a simulated environment that models the collated observations of dynamic movement for a single prebiotic chemical (RNA)  parsimoniously
# Constraints
The movement data needs to be as close as possible to natural, undisturbed RNA movement in a living cell.
# Plan
**Get Recordings** 
	obtain recorded videos from a database upon request. 
	
**Use a Microscope**
Access a microscope to generate video recordings that minimizes risk of causing any damage to the prebiotic chemical (perhaps using something like a *laser scanning confocal microscope*)


a 3D simulated environment in python that models the collated observations of dynamic movement for a single prebiotic chemical (RNA) parsimoniously.

Tell me if something almost exactly like this has been done before. Were they aiming to make it as close to actual RNA as possible in order to get a near-perfect simulation?

 Possibly `scipy` for ODE/SDE integrators, but you can roll your own Euler–Maruyama

Help me clarify my foundational reuirements for my project. --- I consider my desiderata as 1) the simulated environment models dynamic movement of the prebiotic chemical chosen, 2) the model must be parsimonious, 3) the amount of assumptions must be minimal, 4) the computations

### refining desiderata

nucleotide and coarse-grained RNA dynamics

Rouse model: Treats RNA as beads connected by harmonic springs in a viscous medium + Brownian noise. Predicts diffusion/scaling laws and relaxation modes.

Zimm model: motion of one segment drags solvent and affects others.

Worm-Like Chain: RNA like a semiflexible filament with bending energy.

$$E=\frac{1}{2}k_BT\int^{L_0}_{0}P \cdot (\frac{d^2r}{ds^2}) ds \to x \gets y$$

