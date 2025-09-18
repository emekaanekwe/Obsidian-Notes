
# Step 1. Feed-Forward Propogation
when doing forward prop -> need to know the dimension of data.
![[Pasted image 20250806141702.png]]
where the grey dot is, it dhows the dimension of the layer. Note that 32x20 
Note: each dot represents the hidden val of the of n row of the data matrix

logits stay at the output layer. here, you interpret the 4 data point.
	the 26 vals represent the ???

apply the softmax of 

"B = softmax of logits 2" ???

can work out the loss of the 1st, 2nd data points where the P (with red line)

# Step 2: Epochs and Optimizing

use the first minibatch for the optimizer, and run it through an epoch

use batch loss to update the params


# Step 3 Declaring Model ???

**make sure to load your model and data to the same processor (CPU or GPU)**

for optimizer, provide 2 params, updated model, and the learning rate.

![[Pasted image 20250806143204.png]]
*Note: learning decay comes after *

Train model using the training loss

first we have train set and train on a learning rate alpha, then we go to compre it to vlaidation set, and 

the testing set is used to evaluate the model's performance on unseen data, providing an unbiased estimate of how well the model will perform in real-world scenarios.


