import matplotlib.pyplot as plotter
from NeuralNetwork import *

#plotter.plot([1, 2, 3], [1, 4, 9], 'ro')
#plotter.show()

friday = NeuralNetwork("friday")
friday.extract("friday")

trainer = EvolutionaryTrainer(friday)
print(trainer.calculateaccuracy(5)[0])
friday = trainer.train(5, 10, 15, 3)
print(trainer.calculateaccuracy(5)[0])
friday.save()


"""trainer = EvolutionaryTrainer(friday)
a = trainer.calculateaccuracy(4)
rights = a[1]
wrongs = a[2]

rightx = []
righty = []
wrongx = []
wrongy = []

for each in range(len(rights)):
	rightx.append(rights[each][0])
	righty.append(rights[each][1])
for each in range(len(wrongs)):
	wrongx.append(wrongs[each][0])
	wrongy.append(wrongs[each][1])

plotter.plot(rightx, righty, 'go')
plotter.plot(wrongx, wrongy, 'ro')
plotter.show()"""
