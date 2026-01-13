# Using Calculus
## Applying Chain Rule for Training

$$f'(x)=\Delta f(x)=\lim_h->0 \frac{f(x+h)-f(x)}{h}$$

#### Coding with Pytorch
```python

# we have to declar the function params
x = torch.tensor(([1,1,-1]), dtype=torch.float32)
W = torch.rand(3,4)
b = torch.rand(1,4)

# forward prop
h = torch.nn.Sigmoid()(hbar) 
```