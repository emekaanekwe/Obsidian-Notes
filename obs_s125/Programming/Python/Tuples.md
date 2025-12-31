A type of data structure that is an immutable sequence of values of type n where n can be any type for python

```Python

row = 5
col = 3
agent_pos = (row, col) #tuple

```

### Purpose
They are used to _define a fixed object_ that has specific properties. An agent's position as two points is always fixed, it cannot be one value, nor can it be something like a string

### Properties
1. sequentially ordered (they always keep the order in which things were inserted with a FIFO rule)
2. zero-base index
3. nestable (can be placed inside other DS)
4. support iteration

### Operations

accessing elements:
```Python

index = 1
agent_pos[1] # gets 5
```
 (Also allows for slicing)

copying
```Python
from copy import copy

copy(agent_pos)
```

repeating
```Python

agent_pos*2 # (5, 3, 5, 3)
```
(repeating is ordered)

#### Sorting Operations

reverse
```Python

reversed(agent_pos) # (3, 5)
```
(can also use slicing)

sort ascending
```Python

sorted(agent_pos) # (3, 5)
```