from NeuralNetwork import *

testmod = NeuralNetwork("testmod")
testmod.initialise(4, 2, [2, 3, 3, 2])
testmod.display()

testtrainer = Trainer(testmod)

testmod = testtrainer.trainbc(1)

testmod.save()