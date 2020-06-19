from NeuralNetwork import *

jarvis = NeuralNetwork()
jarvis.extract("jarvis")


tony = Trainer(jarvis)
tony.train(1)