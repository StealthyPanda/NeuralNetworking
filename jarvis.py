from NeuralNetwork import *

jarvis = NeuralNetwork("Jarvis")
jarvis.extract("Jarvis")
jarvis.reset()

n= 0
trainer = EvolutionaryTrainer(jarvis)

while True:
	n+=1
	jarvis = trainer.train(1, 1, 10)
	cost = trainer.getcostfunction(jarvis, 1)
	if cost <= 0.2: break

print("Final cost function value: " , trainer.getcostfunction(jarvis, 1))
print(n)


jarvis.reset()
jarvis.save()

