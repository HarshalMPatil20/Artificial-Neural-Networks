# (3 samples) Input layer (4) -> Layer 1 (3) -> layer 2 (3)

import numpy as np

# lists of sample Inputs 
# Input layer neurons = 4
# number of sample inputs = 3

inputs=[[1,2,3,2.5],
        [2.0,5.0,-1.0,2.0],
        [-1.5,2.7,3.3,-0.8]]

# number of Current layer neurons (here 3) =  number of Sub-lists 
# number of Input layer neurons (here 4) =  number of Sub-list elements

weights=[[0.2,0.8,-0.5,1.0],
         [0.5,-0.91,0.26,-0.5],
         [-0.26,-0.27,0.17,0.87]]

biases=[2,3,0.5]

# number of Current layer neurons (here 3) 
# number of Input layer neurons (here 3) 

weights2=[[0.1,-0.14,0.5],
          [-0.5,0.12,-0.33],
          [-0.44,0.73,-0.13]]

biases2=[-1,2,-0.5]

#! For Batch processing the inputs, we take transpose of input array for matrix multiplication to match common parameter. M1(m*n) x M2(n*o) = M1*M2(m*o)

layer1_outputs = np.dot(inputs,np.array(weights).T) + biases

# Output of layer 1 is passed as a input parameter to layer 2
layer2_outputs = np.dot(layer1_outputs,np.array(weights2).T) + biases2

print("layer 1 output:")
print(layer1_outputs)
print("layer 2 output:")
print(layer2_outputs)