### Readings

We recommend you read ONE of the following papers, which provide further details and technical insights into today's lecture material.

Hansen, E.A. and Zhou, R., 2007. Anytime heuristic search. _Journal of Artificial Intelligence Research_, _28_, pp.267-297.

Bono, M., Gerevini, A.E., Harabor, D.D. and Stuckey, P.J., 2019. Path Planning with CPD Heuristics. In _IJCAI_ (pp. 1199-1205).

Nash, A., Daniel, K., Koenig, S. and Felner, A., 2007, July. Theta^*: Any-angle path planning on grids. In _AAAI_ (Vol. 7, pp. 1177-1183).

Cui, M., Harabor, D.D., Grastien, A. and Data61, C., 2017, August. Compromise-free Pathfinding on a Navigation Mesh. In IJCAI (pp. 496-502).

---

![[Pasted image 20250909175012.png]]

![[Pasted image 20250909172847.png]]
![[Pasted image 20250909172900.png]]


a) *If f2=f1 then it is a simple A\**  . If f2 is really similar then theres no point on implementing
the focal search. F2 has to be actually different.
b) f2=depth
It becomes a breadth ﬁrst search (I dint get why or what f2 = depth even means)
c) MAPF?
Find a path that’s maybe slightly longer (in a range of at most 1.5 times (for weight 1.5)) but
still ﬁnds a solution.
Weighted terrain?
Prioritse something else we care about
Pipe routing?
Yes, it has applications there.
Q3)W -> inﬁnite ?
Then everything will go to the focal queue. The whole open list is irrelevant. F2 may be
inadmissible so that’s a huge problem if w is really big. I need to test di>erent values of w,
and ﬁnd the best one.
Q4)
Sliding window allos tot change dynamically the bounds without the need of waiting to
exhaust the whiole focal nodes before updating them, which can save time sometimes.
Q5)
I can initialize tha paths by maybe use just prioritsed planning and just do A* or SIIP for
each agent to initialize each agents path. Or maybe an A* with a focal search and that will
be faster than a A*. The important part is to no invest a lot of time in this initialization. I
wanna something fast.
Q6)
Yes, we can continue to improve its solution quality.


FOCAL search maintains two lists of nodes during the search process. Which statement best characterizes their roles?  

Question 1Answer

a.

**OPEN list:** is ordered arbitrarily; **FOCAL list:** ensures that nodes are expanded strictly in optimal order.

b.

**OPEN list:** contains all generated search nodes similar to A*; **FOCAL list:** a subset of OPEN containing nodes within a bounded threshold of the best f1(n).

c.

**OPEN list:** contains only globally optimal nodes; **FOCAL list:** contains only suboptimal nodes.

d.

**OPEN list:** must be ordered by the admissible evaluation function; **FOCAL list:** can be order by any inadmissible function.


Why do researchers and practitioners frequently employ suboptimal MAPF algorithms rather than relying exclusively on optimal MAPF algorithms?  

Question 2Answer

a.

Although each agent may follow a suboptimal path, the combined solution is always globally optimal.  

b.

Optimal algorithms are restricted to single-agent pathfinding tasks and cannot be applied in multi-agent domains  

c.

Suboptimal algorithms disregard collision constraints to reduce computational effort.  

d.

Optimal algorithms exhibit poor scalability and become computationally infeasible for large numbers of agents.

In the lecture, we introduced LNS for MAPF. Can you identify which of the following graphs best represents the convergence behavior of LNS?

Question 3Answer

a.

![o3](https://learning.monash.edu/pluginfile.php/5253023/question/answer/10142082/4/174561037/O3.png?time=1756990134700)  

b.

![o4](https://learning.monash.edu/pluginfile.php/5253023/question/answer/10142082/4/174561038/O4.png?time=1756990177012)  

c.

![o2](https://learning.monash.edu/pluginfile.php/5253023/question/answer/10142082/4/174561036/O2.png?time=1756990102930)  

d.

![o1](https://learning.monash.edu/pluginfile.php/5253023/question/answer/10142082/4/174561035/O1.png?time=1756990065187)