from NeuralNetwork import *

friday = NeuralNetwork("Friday")
friday.extract("Friday")
#friday.reset()

n= 0

while True:
	n+=1
	trainer = Trainer(friday)
	friday = trainer.trainvectorially(1, 1)
	cost = trainer.getcostfunction(friday, 1)
	#wait = input()
	if cost <= 0.2: break

print("Final cost function value: " , trainer.getcostfunction(friday, 1))

print(n)

#friday.reset()
friday.save()