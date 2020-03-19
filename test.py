from NeuralNetwork import *

nn = NeuralNetwork()
nn.extract("why")

trainer = EvolutionaryTrainer(nn)

nn = trainer.train(1, 15, 10)

nn.save()
#nn.display()

#nn.save()