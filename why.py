from NeuralNetwork import *

t = NeuralNetwork('t')
t.initialise(2, 2, [2, 2])

trainer = Trainer(t)
t = trainer.trainvectorially(1, 100)

t.save()
