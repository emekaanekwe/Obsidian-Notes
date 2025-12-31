# Lab

## Terms

**Off-Policy** - for q-learning, epsilon greedy algo
**on-policy** - SARSA
**Q-Tables** - matrix that functions like a reward matrix where 
**Terminal State** - finished state of when goal is reached
$$
Q(S,A) = E[G_t | S_t=s \ A_t=a]
$$
$$
So, \ \
Q(S,A) + \alpha [R_{t+1} + \gamma max(Q(s',a')) - Q(S,A)]
$$
$$
= (1-\alpha)+ \alpha (R_{t+1} + \gamma max(Q(s',a')) <- TargetPolicy
$$


suppose `states`, we have qtable:

[[0.44931272 0.55503782 0.77463582 0.46507236 0.66802632]
 [0.51095329 0.68227718 0.37169457 0.41552456 0.08994431]
 [0.76056979 0.1197732  0.25348276 0.23585741 0.39891482]
 [0.55112672 0.29420197 0.66798844 0.607449   0.1177721 ]
 [0.84828077 0.2458601  0.32687851 0.09473627 0.79279672]]

and we have reward matrix:
`r_matrix = [0.5, 0.2, 0.3, 0.7, 0.0]`

we are in *home* and take action *stay* :
Q(market, stay) + learning rate(reward(home)+discount*max(q(s',a'))-q(s,a))
	1 + .1\*()
shelter      home
stay _0_      _0_
run _0_       _0_

after q(s,a) function (like home)

shelter    home

stay _0_     _0.4_
run  _0_     _0_


## project phases

1. def make_grid():
	1. return np.array()
2. class gridworld:
	1. def step(aget, a):
		1. updates agent, a
		2. return r
3. do the Q(s,a) calculation
4. follow q-learning algo with high val of epsilon => have episode epislon*epislon
	1. test by setting epsilon = 0. 

after getting accumulated reward, you have a num of episodes that looks like 

![[Pasted image 20250410155453.png]]
then set threshold

#### track if agent learning
#### method 1 
Note: have detemrinistic way of calculating every step. so optimal num of steps for case n, and can compare number of steps with optimal steps:
accumulate reward = optimal steps/accumulated reward
#### method 2
could get visual graph of all paths

#### method 3

find average pathing after many steps
