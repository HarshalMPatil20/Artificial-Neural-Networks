# now we will create a layer consist of 3 neuron will input of 4 neuron from previous layer

# Previous 4 neuron output respectively
input=[1,2,3,4]

# Edge weights from previous 4 neuron to 3 neurons, each sub list a one neuron out of 3
weights=[[0.2,0.8,-0.5,1.0],
         [0.5,-0.91,0.26,-0.5],
         [-0.26,-0.27,0.17,0.87]]

# Bias of each current layer neuron
biases=[2,3,0.5]

# Current Layer Output
Layer_output=[]

# Calculations of all neurons
for neuron_weights, neuron_bias in zip(weights,biases):

    # output of current Neuron
    neuron_output = 0

    for neuron_input, weight in zip(input,neuron_weights):
        # Dot product of Inputs and Respective weights 
        neuron_output += neuron_input*weight 

    # Addition of respective biases
    neuron_output += neuron_bias
    Layer_output.append(neuron_output)

print(Layer_output)