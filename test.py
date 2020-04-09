import matplotlib.pyplot as plotter
from NeuralNetwork import *

testNN = NeuralNetwork("testNN")
testNN.initialise(3, 2, [3, 3, 2])
testNN.display()

newtrainer = Trainer(testNN)
newtrainer.getgradientvector(testNN, 1).display()