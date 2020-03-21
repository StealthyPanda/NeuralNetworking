from random import seed
from random import random
from random import randint


noofsets = 500
filename = 5
ninputlayer = 2
inputrange = 6



def conditionaloutput(inputset):
	outputset = []

	
	if (inputset[0] <= (inputset[1] ** 2)) and (inputset[1] < (inputset[0] ** 2)): outputset = [1, 0]
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