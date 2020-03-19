from NeuralNetwork import *


jarvis = NeuralNetwork("jarvis")
jarvis.initialise(5, 2, 3)
jarvis.save()
jarvis.display()

friday = NeuralNetwork("friday")
friday.extract("jarvis")
friday.display()