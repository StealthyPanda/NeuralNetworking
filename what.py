from NeuralNetwork import *

felix = NeuralNetwork()
felix.initialise(3, 2, 3)
felix = felix.matrix
felix.display()
#print(type(felix))

felix.Multiply(-1)
felix.display()