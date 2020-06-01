from NeuralNetwork import *

learner = NeuralNetwork("Learner")

learner.extract("Learner")

#exit()

trainer = Trainer(learner)


learner = trainer.trainvectorially(6)


learner.save()