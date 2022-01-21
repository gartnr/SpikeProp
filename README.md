# SpikePropSNN
My take on the SpikeProp supervised learning algorithm for Spiking Neural Networks in python due to Bohte et al. (Error-backpropagation in temporally encoded networks of
spiking neurons (2002))

The SNN class is defined in SNNetwork.py. The network consists of three types of neurons, that are defined in Neuron.py. SpikingNeuron objects have connections to previous layer neurons defined in the class Connection in Connections.py.

Some examples of preprocessing data and initialising and training the network are given in main.py.

The algorithm is probably not perfectly optimized. You can write suggestions for improvements to my email gartnr@gmail.com.

For a better understanding of the algorithm go through the article by Bohte et al. I will upload a step-by-step explanation of the algorithm with reference to the uploaded code sometime in the future.
