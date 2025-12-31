
Below is a summary of the different search algorithms as required in the worksheet:

| Framework                                      | Strategy         | Status           |                  |                  |                   |
| ---------------------------------------------- | ---------------- | ---------------- | ---------------- | ---------------- | ----------------- |
| graph                                          | uniform          | Success          |                  |                  |                   |
|                                                |                  |                  |                  |                  |                   |
|                                                | Cost             | Depth            | Nodes(exp)       | Nodes(gen)       | Runtime           |
|                                                |                  |                  |                  |                  |                   |
| **Control**                                    | 7.7              | 7.7              | 106.066666666667 | 129.566666666667 | 0.368173333333333 |
| **Time-Stamp-Diagonals**                       | 4.94444444444445 | 4.94444444444445 | 111.222222222222 | 178.722222222222 | 0.786111111111111 |
| **Time-Stamp-No-Diagonals**                    | 4.75             | 4.75             | 99.125           | 162.6875         | 0.69555625        |
| **Time-Stamp-No-Diagonals-Heuristic-Metric-1** | 5.47619047619048 | 5.47619047619048 | 157.619047619048 | 238.285714285714 | 1.1469            |
## Search vs Search+Time Comparison

There was a differentce between the performance of the algorithms in the sense that node expansion within a grid while using time-indexing was surprisingly faster. This is likely due to the expander not having to undergo repeats, and "waiting" granting more chances to find optimal routes by reducing the non-expanded search space. 

Below are some graphical examples  between the different search methods:

### Search+Time w/ Diagonals
![[Pasted image 20250901224548.png]]

### Search+Time w/o Diagonals
![[Pasted image 20250901224910.png]]

### Search+Time w/o Diagonals on Manhattan Distance
![[Pasted image 20250901225235.png]]

By calculating the derivative for the different algorithmic functions, we can see that *the search algorithm using a time index under the heuristic of manhattan distance is the most optimal.*