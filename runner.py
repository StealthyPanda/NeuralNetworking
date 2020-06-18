from NeuralNetwork import *

jarvisiona = NeuralNetwork("Jarvisiona")

jarvisiona.extract("jarvisiona")

obadiah = Trainer(jarvisiona)

print(obadiah.getcostfunction(jarvisiona, 1))