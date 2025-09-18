### Call $n_1,n_2$ functions  in class $C$ where data $d$ is passed from $n_1,n_2$

	simply use a constructor that will "carry" the data from function to function
```python

class A:
	def __init__(x: int)
		self.x = x
	def one(self):
		return self.x +1
	def two(self):
		return self.x+2
```