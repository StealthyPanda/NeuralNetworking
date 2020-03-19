from NeuralNetwork import *
from random import random
from random import seed
from random import randint
seed(44)
jarvis = NeuralNetwork("Jarvis")

jarvis.initialise(4, 2, 3)


jarvis.setval(randint(0, 3), randint(0, 1), random() * 10 * randint(-2, 0), randint(-1, 1))
jarvis.setval(randint(0, 3), randint(0, 1), random() * 10 * randint(-2, 0), randint(-1, 1))
jarvis.setval(randint(0, 3), randint(0, 1), random() * 10 * randint(-2, 0), randint(-1, 1))
jarvis.setval(randint(0, 3), randint(0, 1), random() * 10 * randint(-2, 0), randint(-1, 1))
jarvis.setval(randint(0, 3), randint(0, 1), random() * 10 * randint(-2, 0), randint(-1, 1))
jarvis.setval(randint(0, 3), randint(0, 1), random() * 10 * randint(-2, 0), randint(-1, 1))
jarvis.setval(randint(0, 3), randint(0, 1), random() * 10 * randint(-2, 0), randint(-1, 1))
jarvis.setval(randint(0, 3), randint(0, 1), random() * 10 * randint(-2, 0), randint(-1, 1))
jarvis.setval(randint(0, 3), randint(0, 1), random() * 10 * randint(-2, 0), randint(-1, 1))
jarvis.setval(randint(0, 3), randint(0, 1), random() * 10 * randint(-2, 0), randint(-1, 1))

jarvis.display()


jarvis.save()

print()