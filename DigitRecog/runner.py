from NeuralNetwork import *

jarvis = NeuralNetwork("Jarvis")
jarvis.extract("Jarvis")


tony = Trainer(jarvis)
tony.train(1)