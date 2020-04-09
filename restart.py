from NeuralNetwork import *

nn = NeuralNetwork("lmao")
nn.extract("lmao")
nn.display()


trainer = Trainer(nn)
pp = trainer.trainvectorially(1, 100)
nn.display()
pp.display()


#trainer = EvolutionaryTrainer(nn)
#nn = trainer.train(1, 20, 20)
#nn.save()

#print(nn.matrix.get2dmatrixtypethree())

