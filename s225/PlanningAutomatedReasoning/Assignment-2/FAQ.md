Hi Everyone,

I just want to clarify some FAQ questions about the Pacman competition as a whole and some rule specifics.

**1. Beating the Staff Baseline:**

The staff baseline should be _very easy_ to beat when you get your implementation working (much more so than A1). Beating the baseline is an absolute expectation from students, and failing to do so will result in a FAIL for the 50 competition marks (33% of the overall). These marks are _independent_ of those for your Agent strategy implementation (50 marks), which is based on your strategic approach and effort put into improving upon the baseline.

**2. I beat the Staff Baseline _locally_, but not on the server!**

As stated elsewhere, the Staff baseline (Staffteam.py) is the exact same implementation on your local machine as well as on the server. The most common cause for this situation is a misunderstanding of how head-to-heads are conducted. You will play 49 games agains the baseline (or other players), consisting of 7 games x 7 maps.

Your team's performance needs to be _generalisable_. A good implementation needs to be able to perform well in different environments, and not 'hardcoded' for a single terrain layout. When you run your local experiments, you _need_ to check your performance across multiple maps. Use the `-l` argument to specify the layout (map), and use the `-n` parameter to specify the number of games you play.

**3. What maps are played on the server?**

The 7 maps played on the server are _not_ visible to students. Again, your agent performance needs to be generalisable! You want to develop AI agents that are robust to small changes in layouts. Your server results do not list the wins/losses in the same order of games played (i.e., it is not accessible to you which map the wins came from).

Know that the 7 maps _are_ chosen from your subset of local layouts, such that there are _no hidden maps -_ you can be confident that you can adequately test your performance across the board.

**4. Action time-outs:**

This assignment is designed to make _efficient_ decisions in a dynamic and fast-paced, real-time environment. In these types of problems, your agent must make decisions _quickly_ and with limited computational resources.

In server competitions, actions must be returned by your agent within reasonable time frames. There are _two_ conditions for instant forfeit:  
  
1) Your agent fails to return an action within **3 seconds (one time = forfeit)**  
  
2) Your agent fails to return an action within **1 second (3 times = forfeit), given warnings**

**5. Deep Q-Learning:**

We have had some students inquiring about the use of deep Q-learning for A2. To note, the assignment is _not_ designed with Q-learning in mind. Nonetheless, while (improving) PDDL is a strict requirement of the assignment, you are _free_ to implement your own methods at the low-level. We suggest and include templates for heuristic search and Q-learning, but if you have experience in other subjects using Deep Q-learning (which would be harder to implement), you are free to take this option. Take note of the limited server resources and time limits.

This option is not something our TAs directly support and you cannot expect to be helped with this approach (outside of unit content). Feel free to explore this at your own discretion.

**6. ELO Matchmaking:**

When considering matchmaking in head-to-heads, you new pacman team entry enters the leaderboard with a base 1000 EP. From here, matchmaking allocates 22 random games against teams within 256 EP (dynamically adjusting with time as your EP rises or sinks). This value has been 128 EP until very recently (adjusted based on complaints that teams were playing only a narrow band of other agents). We believe that this is only a symptom of the currently low number of agents on the leaderboard, but have increased matchmaking tolerances anyway (we may reign this in later based on observations).

**7. Do I need to use PDDL?**

Yes - see the agent strategy aspect of the rubric. Improving the baseline PDDL and high-level reasoning is _the_ key learning outcome from this assignment! The trade-off and interactions between an agent's high-level and low-level reasoning is fundamentally important for designing good AI agents. See below Q's for specifics.

**8. I have implemented PDDL (and will meet learning outcomes in the report), but my other algorithm that manually handles the low-level performs better?**

Evidently, using PDDL is a mandatory learning outcome of the assignment. If you can already demonstrate that you have a strong understanding of these learning outcomes in one version of your algorithm, make sure to include a discussion on the PDDL improvements in your report and describe your approach! This already shows that you have met learning outcomes! Note that any huge if-else statements you have in the low-level _should_ be transferable to more human-understandable high-level plans.

Otherwise, I think you are free to keep your other implementation on the leaderboard if it performs better. Just remember that you _will_ need to beat the baseline implementation with your PDDL-improved team. Having two highly different approaches to the assignment would be excellent for demonstrating _experimental discussions_, collecting numerical data from your two teams playing off against each other. Discuss and reflect on what seems to be working, and what doesn't!

**9. Why do I need to use PDDL, can't I just implement everything in the low-level?**

This has been a sentiment expressed to me by several students in-person. While this is _technically true_, I want to help spell out why PDDL is important and why it can be extremely helpful. In the general literature, PDDL retains its place as one of the foremost relevant approaches in AI planning. It allowing researches to _formally_ describe planning problems in a standardised language and to create more flexible and dynamic AI systems that are adaptable to complex, changing environments.

For your own use: first and foremost, PDDL allows you to express your strategic ideas through _human-understandable_ logic and actions, such that your overall plan is known to you, and directly controllable. If you ever choose to _change_ your strategy down the line, updating your implementation can be easier than the alternative. In the high-level, you can introduce new ideas/actions/goals by tweaking pre-conditions and predicates rather than manually tweaking X number of nested if-else statements in your code (which you will probably do wrong, and have a harder time proving!).

PDDL also allows your ideas to be more _flexible._ If any aspect of the simulator were to change right now, your algorithm would break! The taught logic and implementation specifics may need to be significant reworked. Similarly, the _rules of the game_ could change slightly as well. Your control of high-level reasoning allows you to adapt to these changes easier.

Finally, if using methods like Q-learning, higher action resolution (more specific actions) in PDDL allows your agent to learn substantially better and faster. Q-learning is essentially a single-neuron, and can only learn to do so much. It is far easier to learn how to do simple tasks well (e.g., escape home or eat food) when the tasks are separate, and the agent is not expected to learn a fully generalisable policy to optimise _all aspects_ of "attacking". This also makes it far easier to choose weights for relevant rewards/penalties. I.e., spending more time on the high-level is very _likely to improve your overall performance_.