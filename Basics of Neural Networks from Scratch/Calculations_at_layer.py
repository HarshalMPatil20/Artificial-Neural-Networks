# now we will create a layer consist of 3 neuron will input of 4 neuron from previous layer

# Previous 4 neuron output respectively
inputs=[1,2,3,4]

# Edge weights from previous 4 neuron to 3 neurons, each sub list a one neuron out of 3
weights=[[0.2,0.8,-0.5,1.0],
         [0.5,-0.91,0.26,-0.5],
         [-0.26,-0.27,0.17,0.87]]

# Bias of each current layer neuron
biases=[2,3,0.5]

# Current Layer Output
Layer_output_1=[]
Layer_output_2=[]


## Calculations of all neurons --(Manual)
for neuron_weights, neuron_bias in zip(weights,biases):

    # output of current Neuron
    neuron_output = 0

    for neuron_input, weight in zip(inputs,neuron_weights):
        # Dot product of Inputs and Respective weights 
        neuron_output += neuron_input*weight 

    # Addition of respective biases
    neuron_output += neuron_bias
    Layer_output_1.append(neuron_output)

print(Layer_output_1)

## Calculations of all neurons --(Numpy Dot Product)
import numpy as np
Layer_output_2 = np.dot(weights, inputs) + biases

#! Imp :- np.dot(inputs,weights)- this expression will give error as weight is 2D array

print(Layer_output_2)