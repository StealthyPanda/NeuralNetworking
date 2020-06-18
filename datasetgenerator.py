from random import seed
from random import random
from random import randint


noofsets = 10
filename = 2
ninputlayer = 2
inputrange = 4



def conditionaloutput(inputset):
	outputset = []

	
	if ((((inputset[0] - 2.5) ** 2) + ((inputset[1] - 1.5) ** 2)) <= 1): outputset = [1, 0]
	else: outputset = [0, 1]

	return outputset


with open(str(filename) + '.txt', 'w') as file:
	file.write("")


with open(str(filename) + '.txt', 'a') as file:
	for each in range(noofsets):
		seed()
		inputset = []
		outputset = []
		for x in range(ninputlayer):
			i = random() * inputrange
			inputset.append(i)
		#print(inputset)
		outputset = conditionaloutput(inputset)
		for x in range(len(inputset)):
			inputset[x] = str(inputset[x])
		for x in range(len(outputset)):
			outputset[x] = str(outputset[x])
		file.write(" ".join(inputset))
		file.write("\n")
		file.write(" ".join(outputset))
		file.write("\n")