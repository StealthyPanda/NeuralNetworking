from scipy.stats import logistic


class NeuralNetwork(object):
	def __init__(self, name):
		print("New neural network created")
		self.name = name

	#displays the weights and biases of neural network
	def display(self):
		print(self.name + ": ")
		self.matrix.display()


	"""
	saves the NN matrix in a text file in the following format:
	a cell is flanked by |; so |*le cell contents*|
	each layer is on a different line. so:
	layer1 -> cell1|cell2|cell3|cell4
	layer2 -> cell1|cell2|cell3
	layer3 -> cell1|cell2|cell3|cell4
	layer4 -> cell1|cell2

	inside each cell, weights and biases are written separated by commas. so:
	|w1,w2,w3,w4,w5 ... wn, bias| (last element is the bias of the cell)
	"""
	def save(self):
		listoflayers = []
		for each in range(self.matrix.nlayers):
			currentlayer = []
			for i in range(len(self.matrix.layers[each])):
				thestringedcell = []
				for x in range(len(self.matrix.layers[each][i][0])):
					thestringedcell.append(str(self.matrix.layers[each][i][0][x]))
				thestringedcell.append(str(self.matrix.layers[each][i][1]))
				#print(thestringedcell)
				cellstring = ",".join(thestringedcell)
				currentlayer.append(cellstring)
			listoflayers.append("|".join(currentlayer))
		finaloutput = "\n".join(listoflayers)
		with open(self.name + '.txt', 'w') as file:
			file.write(finaloutput)



	def setval(self, layer, cell, value, weight = -1):
		self.matrix.setval(layer, cell, value, weight)

	#does some basic initialising things
	#at end of this function, the matrix is filled with all weights as 1 and
	#all biases as 0
	def initialise(self, nlayers, ninputlayer, ncells):
		self.matrix = FlexiMatrix(nlayers, ninputlayer)
		self.matrix.setcellsineachlayer(ncells)
		self.matrix.initiate()

	def runcycle(self, inputs):
		if len(inputs) != self.matrix.ninputlayer : return "Length of inputs dont match"
		self.inputlayer = inputs
		self.bufferlayer = [0 for i in range(len(self.matrix.layers[0]))]
		for each in range(len(self.bufferlayer)):
			bias = self.matrix.getval(0, each)

			weightedsum = 0

			for i in range(len(self.inputlayer)):
				weightedsum += self.inputlayer[i] * self.matrix.getval(0, each, i)


			self.bufferlayer[each] = logistic.cdf(bias + weightedsum)
		print(self.bufferlayer)
		for eachlayer in range(1, self.matrix.nlayers):
			newbuffer = [0 for i in range(len(self.matrix.layers[eachlayer]))]
			for each in range(len(newbuffer)):
				bias = self.matrix.getval(eachlayer, each)
				weightedsum = 0
				for i in range(len(self.bufferlayer)):
					weightedsum += self.bufferlayer[i] * self.matrix.getval(eachlayer, each, i)
				newbuffer[each] = logistic.cdf(bias + weightedsum)
			self.bufferlayer = newbuffer
			print(self.bufferlayer)
		return self.bufferlayer	


"""
trains the neural network but rnadomly causing variations in the network
and reproducing the best performing one
"""
class EvolutionaryTrainer(object):
	def __init__(self):
		pass

	def Train(self, cycles):
		if cycles >= 4:
			print("Too many cycles")
			return
		self.ncycles = 10**cycles



class FlexiMatrix(object):

	#in init itself no. of layers is set
	def __init__(self, nlayers = 1, ninputlayer = 1):
		self.nlayers = nlayers
		self.ninputlayer = ninputlayer
		self.layers = [[] for i in range(self.nlayers)]
	"""prints out the matrix in the folllowing format:
		->*el layer 1's cells*
		->*el layer 2's cells*
		->*el layer 3's cells* and so on
		"""
	def display(self):
		print("\n")
		for each in range(len(self.layers)):
			print("layer " + str(each + 1) + "" + "->", self.layers[each], "\n")
		#print("\n\n")

	#cells are of the format [[w1, w2, ... wn], bias]
	#inital value of all weights is 1 and all biases is 0
	def initiate(self):
		for each in range(self.nlayers):
			for i in range(len(self.layers[each])):
				if each == 0: self.layers[each][i] = [[1 for i in range(self.ninputlayer)], 0]
				else: self.layers[each][i] = [[1 for i in range(len(self.layers[each - 1]))], 0]

	#after setting no. of layerss, set no. of cells in each layer (which may or may not be same) 
	def setcellsineachlayer(self, ncells):
		try:
			for each in range(len(self.layers)):
				self.layers[each] = [[] for i in range(ncells)]
			return

		except Exception as e:
			if len(ncells) != self.nlayers:
				print("No. of layers and No. of cells not equal")
				return
			else:
				for each in range(len(ncells)):
					self.layers[each] = [[] for i in range(ncells[each])]
			return

	"""gets specific val from layer no. layer, cell no. cell
	and if weight = -1(by default) returns the bias. if any other integer
	is given, that weight is returned
	"""
	def getval(self, layer, cell, weight = -1):

		if weight == -1: return self.layers[layer][cell][1]
		else: return self.layers[layer][cell][0][weight]

	#same logic as above
	def setval(self, layer, cell, value, weight = -1):

		if weight == -1: self.layers[layer][cell][1] = value
		else: self.layers[layer][cell][0][weight] = value