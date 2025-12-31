 
$$
\text{vectors v and w are orthogonal only if  } v*w=0
$$


| Start | 1   | -2  |
| ----- | --- | --- |
| 1     | 1   | 1   |
| -3    | 1   | 1   |
assuming no extra ore, [(1,0), (1,1), (2,1), (2,2)] is the optimal route with cost - profit = 

| 0   | 0   | 0   |
| --- | --- | --- |
| 0   | 0   | 0   |
| 0   | 5   | 0   |
include the ore map, [(0,1), (1,1), (2,1), (2,2)] is the optimal route cost - profit = 5


## Input Vars
1. the dimensions of grid
2. length of the mine path
3. budget
4. cost of each cell
5. reward for each cell
6. start position
## Decision Vars
1. total profit
2. (x, y) coordinates of mining path
	1. which cost cells will be included
	2. which rewards will be included
## Constraints
1. cannot pass through a cell that is less than 1
2. path must be equal to length given by dataset
3. 
## Desiderta
1. For a 2D array at cell (n, m)
	1. scan all adjacent cells
		1. if adjacent cell < 1, add to cost
			1. continue
		2. else, scan next cell
2. scan cost map and ore map
	1. Note that the grid of of equal dimensions. If bot can scan both maps **and put them in a set** we may be able to maximize the objective function.
		1. ```
		2. 


