# 2024

## q6
  
```python

class create_ResNet (nn.Module): def __init__(self): super().__init__()
self.layers = nn.ModuleList([
ABCDE
nn.LazyConv2d(64, kernel_size=7, stride=2, padding=3),
nn.LazyBatchNorm2d(),
nn.ReLU(),
nn.MaxPool2d(kernel_size=3, stride=2, padding=1), ResnetBlock(64, 2, first_block=True),
ResnetBlock(128, 2),
ResnetBlock(256, 2),
nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(1),
Fnn.LazyLinear(10),
#nn.Softmax(dim=-1)
])
def forward(self, x):
for, layer in enumerate(self.layers):
X = layer (X)
return X
```

## 16
```python

class MyCNN (nn.Module):
def __init__(self):
super (MyCNN, self).__init__()
self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1) self.pool = nn.MaxPool2d (kernel_size=2, stride=2, padding=0) self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) self.dropout = nn. Dropout (0.25)
self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
self.flatten = nn. Flatten()
self.fc1 = nn.Linear(128, 20)
def forward(self, x):
h1
torch.relu(self.conv1(x))
h2 = self.pool (h1)
h3 torch.relu(self.conv2 (h2))
=
# Applied dropout
h3= self.dropout (h3)
h4= self.pool (h3)
h5 torch.relu (self.conv3 (h4))
=
h6= self.avg_pool (h5)
h7 = self.flatten (h6) h8= self.fc1(h7)
return h8
myCNN = MyCNN()
x = torch.randn(128, 3, 64, 64)
y = myCNN (x)
```
## 17

**Inputs**
x0 = "I" --> U --> h0 --> W --> h1
x1 = "like" --> U --> h1 --> V --> y-hat

$$V =\begin{bmatrix} 
1&1 \\
0&1 \\
1&0\end{bmatrix}$$
$$U =\begin{bmatrix} 
1&1&0 \\
0&1&1\end{bmatrix}$$
$$W =\begin{bmatrix} 
1&1&0 \\
0&1&1 \\
1&0&1\end{bmatrix}$$

**Build Up Vocabulary**
1. Like (index 1)
2. Love (index 2)
3. Bad (index 3)
4. Fantastic (index 4)
5. I (index 5)
6. Recommend (index 6)

**Embedding Matrix**: $E[6\times 2]$

| e   | 1    | 0.0  |
| --- | ---- | ---- |
| e2  | -1   | 1    |
| e3  | 1    | 3.3  |
| e4  | -1   | 1.3  |
| e5  | -1   | 1.0  |
| e6  | -1.7 | -1.3 |

## 20

```python
class DCGAN (GAN):
def
_init__(self, data_loader=None, batch_size=32, epochs=10, optimizer=None, noise_dim=30, device='cpu'): super (DCGAN, self).__init__(data_loader, batch_size, epochs, optimizer, noise_dim, device)
def build(self):
# Generator: Upscales noise to images
self.generator = nn.Sequential(
nn.Linear(self.noise_dim, 7 * 7 * 128), # Upscale noise
nn.BatchNorm1d (7 * 7 * 128),
nn.ReLU(inplace=True),
nn.Unflatten (1, (128, 7, 7)),
nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
nn.BatchNorm2d(64),
nn.ReLU(inplace=True),
nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1), nn.Tanh() # Output between -1 and 1
).to(self.device)
# Discriminator: Classifies images as real or fake
self.discriminator = nn. Sequential(
nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1), # Downsample
nn.LeakyReLU(0.2, inplace=True),
nn.Dropout (0.4),
nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # Further downsample
nn.LeakyReLU(0.2, inplace=True),
nn.Dropout (0.4),
nn.Flatten(),
nn. Linear(128 * 7 * 7, 1), # Single output nn.Sigmoid() # Probability output
).to(self.device)
```

## 21
			y1   y2    eos
			|    |
h1 -> c=h_T_x -> q0 -> q1 -> ... q_T_y
|            |                    |          |                |
x1       x_T_x          bos    y1            y_T_y