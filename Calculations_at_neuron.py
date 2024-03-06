import numpy as np

# Inputs From previous layer
inputs=[1,2,3]

# Connecting Edge weightage 
weights=[0.3,0.2,0.5]

# Unique Signature of Every neuron
Bias=3

#output= Dot Product of inputs and weights + Bias

# Manual Product
output = inputs[0]*weights[0]+inputs[1]*weights[1]+inputs[2]*weights[2]+Bias

print(output)

# Using Numpy a Dot Product