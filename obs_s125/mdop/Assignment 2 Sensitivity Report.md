
# Make an exploration about how much each group of constraints affects the runtime of the model

## Constraint 1
This constraint has  large impact on the objective since it maintains consistency between the vaccines recorded in vs[g] and the actual schedule of s[g,w]. This allows the model to get the vaccine usage only when the correct number of doses are scheduled, affecting runtime.
## Constraint 2
This constraint just restricts overlapping vaccine treatment plans. While it does affect the objective, it has more of an indirect effect.

## Constraint 3
This constraint, as to establish controls in the experiment, sets one specific control per group and the dummy vaccine. Like constraint 2, its impact is more indirect, but still reduces the sample space the solver needs to work with.

## Constraint 4
If only at most vm groups can get vaccinated, this impact is larger as it sets a limiter on the number of groups. If we had, for exmaple, 1,000,000 groups, the objective's total cost would be painfully high, potentially resulting in a runtime of O(n^# of groups).
## Constraint 5
One of the most impactful constraints, this one sets a minimum and maximum limit upper bound of treatments for each group. Much like constraint 4, if no limiter were to be imposed, we could have theoretically:
$$
Group \ n: S....V.V.R...n: V1..Vn \ \text{where n could be a very large number}
$$
## Constraint 6
This constraint simply models a preparation period before vaccinations. This could be due to sterilization purposes, or some other reason.

## Constraint 7
This constraint sets a restriction on the test procedure, and does not have any real impact on the objective.

## Constraint 8
This constraint has a noticeable effect the objective since it sets a restriction on the treatment done on the first and last spots.
## Constraint 9 (C1)
If the cost of each treatment plan must be different, then there cannot be scenarios where the most expensive treatment plans are repeated. Thus, even if the most expensive treatment plan is $10,000, the objective can never be something like 

plan_cost = [a,b,c,d] where a,b,c,d=10,000 = 40,000

While this may not necessarily have a noticeable affect on the runtime, it can have an impact on the objective function.

## Constraint 10 (C2)
If the most expensive plan is at most twice the cheapest, then there will be an invariable effect on the objective, but little effect on the runtime. 

# Compare running the full model with the variant model

|               | rift1.dzn                                        | rift2.dzn                                                   | rift3.dzn                                         | rift4.dzn                                            | rift5.dzn                                        | rift6.dzn                                        | rift7.dzn                                           | rift8.dzn                                           |
| ------------- | ------------------------------------------------ | ----------------------------------------------------------- | ------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------ | --------------------------------------------------- | --------------------------------------------------- |
| constraint 1  | _objective = 36;<br><br>Finished in 891msec.     | <br>_objective = 346;<br><br>----------                     | _objective = 168;<br><br>Finished in 2s 859msec.  | _objective = 36;<br><br><br>Finished in 1s 66msec.\| | _objective = 70;<br><br>Finished in 4s 529msec.  | _objective = 168;<br><br>Finished in 2s 908msec. | _objective = 30;<br><br><br>Finished in 2s 816msec. | _objective = 36;<br><br><br>Finished in 1s 454msec. |
| constraint 2  | _objective = 74;<br><br>Finished in 5s 149msec.  | _objective = 232;<br><br>Finished in 14s 294msec.           | _objective = 168;<br><br>Finished in 1s 14msec.   | _objective = 74;<br><br>Finished in 2s 315msec.      | _objective = 90;<br><br>Finished in 2s 764msec.  | _objective = 168;<br><br>Finished in 778msec.    | _objective = 72;<br><br>Finished in 2s 869msec.     | _objective = 74;<br><br>Finished in 2s 938msec.     |
| constraint 3  | _objective = 100;<br><br>Finished in 1s 716msec. | _objective = 346;<br><br>Finished in 3m 59s.                | _objective = 168;<br><br>Finished in 2s 196msec.  | _objective = 100;<br><br>Finished in 2s 589msec.     | _objective = 130;<br><br>Finished in 5s 310msec. | _objective = 168;<br><br>Finished in 1s 160msec. | _objective = 99;<br><br>Finished in 2s 726msec.     | _objective = 100;<br><br>Finished in 1s 608msec.    |
| constraint 4  | _objective = 100;<br><br>Finished in 3s 28msec.  | _objective = 346;<br><br>----------                         | _objective = 168;<br><br>Finished in 3s 488msec.  | _objective = 100;<br><br>Finished in 2s 668msec.     | _objective = 130;<br><br>Finished in 4s 612msec. | _objective = 168;<br><br>Finished in 991msec.    | _objective = 99;<br><br>Finished in 4s 34msec.      | _objective = 100;<br><br>Finished in 3s 768msec.    |
| constraint 5  | _objective = 78;<br><br>Finished in 1s 966msec.  | _objective = 256;<br><br>Stopped.<br>Finished in 1m 6s.     | _objective = 168;<br><br>Finished in 933msec.     | _objective = 84;<br><br>Finished in 2s 502msec.      | _objective = 168;<br><br>Finished in 888msec.    | _objective = 50;<br><br>Finished in 5s 981msec.  | _objective = 78;<br><br>Finished in 2s 50msec.      | _objective = 76;<br><br>Finished in 747msec         |
| constraint 6  | _objective = 100;<br><br>Finished in 2s 221msec. | _objective = 346;<br><br>Finished in 3m 18s.                | _objective = 168;<br><br>Finished in 2s 337msec.  | _objective = 100;<br><br>Finished in 3s 326msec.     | _objective = 130;<br><br>Finished in 4s 605msec. | _objective = 168;<br><br>Finished in 969msec.    | _objective = 95;<br><br>Finished in 4s 487msec.     | _objective = 100;<br><br>Finished in 3s 390msec.    |
| constraint 7  | _objective = 100;<br><br>Finished in 2s 270msec. | _objective = 346;<br>Stopped.<br><br>Finished in 1m 4s      | _objective = 168;<br><br>Finished in 2s 4msec     | _objective = 100;<br><br>Finished in 3s 993msec      | _objective = 130;<br><br>Finished in 3s 320msec  | _objective = 168;<br><br>Finished in 1s 110msec  | _objective = 99;<br><br>Finished in 2s 231msec      | _objective = 100;<br><br>Finished in 2s 988msec     |
| constraint 8  | _objective = 100;<br><br>Finished in 2s 42msec.  | _objective = 346;<br>Stopped.<br><br>Finished in 1m 15s.    | _objective = 168;<br><br>Finished in 1s 252msec.  | _objective = 100;<br><br>Finished in 3s 148msec      | _objective = 130;<br><br>Finished in 4s 183msec  | _objective = 168;<br><br>Finished in 1s 334msec  | _objective = 99;<br><br>Finished in 3s 337msec      | _objective = 100;<br><br>Finished in 2s 441msec.    |
| Constraint 9  | _objective = 90;<br><br>Finished in 2s 666msec   | _objective = 340;<br><br>Stopped.<br><br>Finished in 1m 29s | _objective = 146;<br><br>Finished in 39s 252msec. | _objective = 90;<br><br>Finished in 3s 65msec        | _objective = 123;<br><br>Finished in 2s 920msec  | _objective = 148;<br><br>Finished in 35s 426msec | _objective = 90;<br><br>Finished in 2s 243msec      | _objective = 90;<br><br>Finished in 1s 801msec.     |
| constraint 10 | _objective = 92;<br><br>Finished in 397msec.     | _objective = 326;<br><br>Finished in 16s 816msec            | _objective = 134;<br><br>Finished in 52s 718msec  | _objective = 92;<br><br>Finished in 659msec.         | _objective = 119;<br><br>Finished in 539msec     | _objective = 134;<br><br>Finished in 23s 43msec  | _objective = 97;<br><br>Finished in 1s 45msec.      | _objective = 92;<br><br>Finished in 603msec         |

## Analysis

	From the table above, we can see that the best overall runtime is from constraint 1 with objective scores of 36 from rift1.dzn and 30 from rift7.dzn, and run times of 0.89 and 2.8, respectively.

# For each data file, give the top 3 constraints that help raise the objective function

For each data file, I have a corresponding line chart that maps the run times and objective values with each constraint removed. I give the top three most contributing constraints to the objective function in descending order, and explain why.
### rift1.dzn

*Top 3*
1. A1, A10, A9

*Justification*
If we were to remove A1, then we can see that the objective is greatly impacted. Thus, the constraint imposes the most limitations of how the treatment plans are ordered with more loose scheduling, and thus the costs. A10 constraints the cost spread, and so without it, the solver can identify overly cheap treatment plans. Removing A9 limits cost variation, similar to A10.

![[Pasted image 20250514164623.png|400]]


### rift2.dzn

*Top 3*
1. A2, A5, A10

*Justification*
With A2 removed, there is a significant dip in the objective value. With each vaccine **not** being different, treatment plans can be reused. A5 imposes a min max upper bound, and thus requires there to be a threshold of what the plans can be (perhaps to avoid exploitation by giving groups the cheapest plans possible without considering any negative impacts.) A10 does similar work as A5.

![[Pasted image 20250514162938.png|400]]

### rift3.dzn

*Top 3*
1. A10, A9, A5

*Justification*
If we remove the fairness constraint from A10, then we see the solver finding potentially exploitative plans. A9 constrains differences in cost and flxibility, and A5 sets a less impactful limiter.

![[Pasted image 20250514163001.png|400]]

### rift4.dzn

*Top 3*
1. A1, A2, A5

Removing A1 results in considerable freedom of vaccine costs since vaccine treatment plans can be administered any number of times regardless of the vaccination. If we remove A2, then identical vaccinations across groups can significantly reduce or raise costs. A5's removal of it's min/max constraint results in cheaper plans to an extent.

![[Pasted image 20250514163841.png|400]]

### rift5.dzn

*Top 3*
1. A1, A2, A10

*Justification*
A1 has the most impact because of its restriction of tremendously cheap and redundant plans.  A2 ensures different plans and cost diversity, and A10 eliminates overly cheap outliers.

![[Pasted image 20250514163855.png|400]]


### rift6.dzn

*Top 3*
1. A5, A9, A10

*Justification*
In this case, A5 is by far the most restrictive as it prevents extremely low cost schedules, which would lead to non-optimal solutions. A10 imposes a global fairness constraint, and so removing it would have the adverse effect. With A9 removed, there can be 0 cost treatments, which simply doesn't make sense.

![[Pasted image 20250514163912.png|400]]

### rift7.dzn

*Top 3*
1. A1, A2, A5

*Justification*
similar to the reasons of rift1.dzn and rift4.dzn, these constraints help balance the costs of realistic schedules.


![[Pasted image 20250514163924.png|400]]

### rift8.dzn

*Top 3*
1. A1, A2, A5

As seen previously, without these constraints, there can be duplicate unrealistic plans with extremely cheap costs through A1 and A2. Without A5's min/max constraint, the solver does not rule unrealistically cheap or expensive cost outliers.

![[Pasted image 20250514163940.png|400]]


# Explain what are the most important constraints of the model overall, and justify your answer.

I believe constraints A1, A2 and A5 are the most overall impactful constraints because they do a series of things:

1. they set bounds on what kinds of plans can be administered so as to avoid numerically odd structured  plans
2.  From a combinatorial perspective, they do the most work in restricting the sample space from which the solver can sample.
3. Socially, these constraints enforce a diversity of treatments, inadvertently minimizing sampling bias, which is a necessary condition for conducting a vaccination experiment of any kind that adheres to the best practices in modern science.
