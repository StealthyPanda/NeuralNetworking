from NeuralNetwork import *

just = NeuralNetwork("Just")

just.extract("just")

trainer = Trainer(just)
just = trainer.biotrain(1)

just.save()