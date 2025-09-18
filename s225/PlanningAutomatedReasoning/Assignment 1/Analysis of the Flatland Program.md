***Origins of Flatland Challenge***
https://onlinelibrary.wiley.com/doi/abs/10.1002/net.20199

#### rail_env.py

- a **rail generator**, that generates new rail networks on each reset,
    
- a **line generator**, that generates start and end points for each agent on reset,
    
- an **observation builder**, that provides a suitable observation vector to the agents.


# On Understanding Source Code










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
