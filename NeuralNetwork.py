from scipy.stats import logistic
from datetime import datetime
from random import seed
from random import random
from random import randint
from random import choice
from Matrices import Matrix
import threading
import math
import copy
ok = True



fileerrormsg = "\nERROR -> Unsupported file type or no file with given name found!\n         Neural Network must be of extension .nn or .txt\n"
uninitmodelerrormsg = "\nERROR -> Model has not been initialised yet!\n"
ninputerrormsg = "Length of inputs dont match"


delta = 10 ** -3



if __name__ == '__main__':
	print("This file is just full of classes. Use another file to run them.")

#kinda a gimmick really wont use it again 0/10
def Log(val):
	stringval = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + str(val)
	with open('Log.txt', 'a') as file:
		file.write('\n')
		file.write(stringval)
		file.write('\n')


class NeuralNetwork(object):
	def __init__(self, name = '', function = ''):
		print("New neural network created")
		self.name = name
		self.function = function

	def __repr__(self):
		try:
			return self.matrix.__repr__()
		except:
			print(uninitmodelerrormsg)
			Log(uninitmodelerrormsg)

	#displays the weights and biases of neural network
	def display(self):
		print(self.name + ": ")
		if self.function: print("Function: " + self.function)
		self.matrix.display()


	#returns the no of all the values in the NeuralNetwork
	def getsize(self):
		return self.matrix.getsize()

	#same as save but in .txt format; not recommended fam
	def savetxt(self):
		firstline = '<' + self.name + '>'
		if self.function: firstline +='<' + self.function + '>'
		#firstline += '\n'
		listoflayers = [firstline]
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
		with open(self.name.lower() + '.txt', 'w') as file:
			file.write(finaloutput)

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
	#saves the neural network in a "proprietary" .nn extension file
	def save(self):
		firstline = '<' + self.name + '>'
		if self.function: firstline +='<' + self.function + '>'
		#firstline += '\n'
		listoflayers = [firstline]
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
		byteout = finaloutput.encode("utf-8")
		with open(self.name.lower() + '.nn', 'wb') as file:
			file.write(byteout)

	#extracts model from saved file
	def extracttxt(self, modelname):
		finallayers = []
		filename = modelname.lower() + '.txt'
		with open(filename, 'r') as file:
			rawtext = file.read()
		finallayers = rawtext.split('\n')
		name = finallayers[0].split('><')[0]
		self.name = name[1:len(name)]
		self.function = finallayers[0].split('><')[1][0:-1]
		finallayers = finallayers[1:len(finallayers)]
		for each in range(len(finallayers)):
			finallayers[each] = finallayers[each].split('|')
		for each in range(len(finallayers)):
			for i in range(len(finallayers[each])):
				celllist = finallayers[each][i].split(',')
				for z in range(len(celllist)):
					celllist[z] = float(celllist[z])
				finallayers[each][i] = [celllist[0:-1], celllist[-1]]
		self.matrix = FlexiMatrix(len(finallayers), len(finallayers[0][0][0]))
		self.matrix.layers = finallayers

	#extracts model from savebin-ed "proprietary" file
	def extractbin(self, modelname):
		finallayers = []
		filename = modelname.lower() + '.nn'
		with open(filename, 'rb') as file:
			rawtext = file.read().decode()
		finallayers = rawtext.split('\n')
		name = finallayers[0].split('><')[0]
		self.name = name[1:len(name)]
		self.function = finallayers[0].split('><')[1][0:-1]
		finallayers = finallayers[1:len(finallayers)]
		for each in range(len(finallayers)):
			finallayers[each] = finallayers[each].split('|')
		for each in range(len(finallayers)):
			for i in range(len(finallayers[each])):
				celllist = finallayers[each][i].split(',')
				for z in range(len(celllist)):
					celllist[z] = float(celllist[z])
				finallayers[each][i] = [celllist[0:-1], celllist[-1]]
		self.matrix = FlexiMatrix(len(finallayers), len(finallayers[0][0][0]))
		self.matrix.layers = finallayers

	#general function for extracting any file type
	def extract(self, modelname):
		global fileerrormsg
		try:
			self.extractbin(modelname)
			return
		except:
			try:
				self.extracttxt(modelname)
				return
			except:
				print(fileerrormsg)
				Log(fileerrormsg)
				return



	def setval(self, layer, cell, value, weight = -1):
		self.matrix.setval(layer, cell, value, weight)

	#does some basic initialising things
	#at end of this function, the matrix is filled with all weights as 1 and
	#all biases as 0
	def initialise(self, nlayers, ninputlayer, ncells):
		self.matrix = FlexiMatrix(nlayers, ninputlayer)
		self.matrix.setcellsineachlayer(ncells)
		self.matrix.initiate()

	#returns the outputlayer for a given input layer
	def runcycle(self, inputs):
		if len(inputs) != self.matrix.ninputlayer : return ninputerrormsg
		self.inputlayer = inputs
		self.bufferlayer = [0 for i in range(len(self.matrix.layers[0]))]
		for each in range(len(self.bufferlayer)):
			bias = self.matrix.getval(0, each)

			weightedsum = 0

			for i in range(len(self.inputlayer)):
				weightedsum += self.inputlayer[i] * self.matrix.getval(0, each, i)


			self.bufferlayer[each] = logistic.cdf(bias + weightedsum)
		#print(self.bufferlayer)
		for eachlayer in range(1, self.matrix.nlayers):
			newbuffer = [0 for i in range(len(self.matrix.layers[eachlayer]))]
			for each in range(len(newbuffer)):
				bias = self.matrix.getval(eachlayer, each)
				weightedsum = 0
				for i in range(len(self.bufferlayer)):
					weightedsum += self.bufferlayer[i] * self.matrix.getval(eachlayer, each, i)
				newbuffer[each] = logistic.cdf(bias + weightedsum)
			self.bufferlayer = newbuffer
			#print(self.bufferlayer)
		return self.bufferlayer	

	#returns a list like [0, 0, ... 1 ..., 0 ,0], where 1 is the
	#cell in final(output layer) that is the brightest. this is for
	#use istead of an out put like [0.843984, 0.0238238,... ,0.0832832]
	#or whatever.
	def modeloutput(self, inputs):
		g = 0
		gi = 0
		o = self.runcycle(inputs)
		if o == ninputerrormsg: return
		#print(o)
		for i in range(len(o)):
			if o[i] > g: 
				g = o[i]
				gi = i
		ol = [0 for x in o]
		ol[gi] = 1
		#print(ol)
		return ol

	#resets the whole neural network
	def reset(self):
		for each in range(len(self.matrix.layers)):
			for i in range(len(self.matrix.layers[each])):
				self.matrix.layers[each][i][1] = 1
				for x in range(len(self.matrix.layers[each][i][0])):
					self.matrix.layers[each][i][0][x] = 1





"""
trains the neural network but randomly causing variations in the network
and reproducing the best performing one
"""
class EvolutionaryTrainer(object):
	def __init__(self, model):
		self.model = model
		self.genno = 0
		self.datadone = 0

	#calcultates accuracy of the model in percentage of data
	#it gets right
	def calculateaccuracy(self, testdata):
		inputset = []
		outputset = []
		rawfile = []
		rights = []
		accuracy = 0
		wrongs = []
		with open(str(testdata) + '.txt', 'r') as file:
			rawfile = file.read().split('\n')
		rawfile.remove("")
		for x in range(0, len(rawfile), 2):
			inputset.append(rawfile[x])
		for x in range(1, len(rawfile), 2):
			outputset.append(rawfile[x])
		outputset.append(rawfile[-1])
		for x in range(len(inputset)):
			inputset[x] = inputset[x].split(" ")
		for x in range(len(outputset)):
			outputset[x] = outputset[x].split(" ")
		for each in range(len(inputset)):
			for i in range(len(inputset[each])):
				try:
					inputset[each][i] = float(inputset[each][i])
				except:
					inputset[each].remove(inputset[each][i])
		for each in range(len(outputset)):
			for i in range(len(outputset[each])):
				try:
					outputset[each][i] = float(outputset[each][i])
				except:
					outputset[each].remove(outputset[each][i])
					pass
		for each in range(len(inputset)):
			#print(inputset)
			prediction = self.model.modeloutput(inputset[each])
			if prediction == outputset[each]:
				rights.append(inputset[each])
			else:
				wrongs.append(inputset[each])
		accuracy = (len(rights)/len(inputset)) * 100
		#print(rights)
		return [accuracy, rights, wrongs]




	#returns a list of NN models with random mutations. rate is the no.
	#of models produced per generation
	#mutation is how much variation in each value is produced
	def reproducemodel(self, model, rate, mutation):
		gen = []
		for x in range(rate):
			buffermodel = copy.deepcopy(model)
			for each in range(len(buffermodel.matrix.layers)):
				seed()
				for i in range(len(buffermodel.matrix.layers[each])):
					for x in range(len(buffermodel.matrix.layers[each][i][0])):
						buffermodel.matrix.layers[each][i][0][x] += (choice([-1, 1]) * mutation * random())
					buffermodel.matrix.layers[each][i][1] += (choice([-1, 1]) * mutation * random())
			gen.append(buffermodel)
		return gen

	def getcostfunction(self, model, dataset):
		cost = 0
		inputset = []
		outputset = []
		rawfile = []
		with open(str(dataset) + '.txt', 'r') as file:
			rawfile = file.read().split('\n')
		rawfile.remove("")
		for x in range(0, len(rawfile), 2):
			inputset.append(rawfile[x])
		for x in range(1, len(rawfile), 2):
			outputset.append(rawfile[x])
		outputset.append(rawfile[-1])
		for x in range(len(inputset)):
			inputset[x] = inputset[x].split(" ")
		for x in range(len(outputset)):
			outputset[x] = outputset[x].split(" ")
		for each in range(len(inputset)):
			for i in range(len(inputset[each])):
				try:
					inputset[each][i] = float(inputset[each][i])
				except:
					inputset[each].remove(inputset[each][i])
		for each in range(len(outputset)):
			for i in range(len(outputset[each])):
				try:
					outputset[each][i] = float(outputset[each][i])
				except:
					outputset[each].remove(outputset[each][i])
					pass

		for each in range(len(inputset)):
			costsum = 0
			modelout = model.runcycle(inputset[each])
			self.datadone += 1
			realout = outputset[each]
			#print(len(modelout))
			#print(len(realout))
			for x in range(len(modelout)):
				#print(modelout[x])
				#print(realout[x])
				try:
					costsum += (( modelout[x] - realout[x] ) ** 2)
				except:
					#print("something wrong here:")
					#print(type(modelout[x]), modelout[x])
					#print(type(realout[x]), realout[x])
					pass
			cost += costsum
		cost = (cost / len(inputset))
		#self.datadone = 0
		#print(len(rawfile))
		return cost



	def updater(self):
		global ok
		prevgen = 0
		prevdata = 0
		while ok:
			if self.genno % 10 == 0 and self.genno != prevgen:
				print("Generation no: " + str(self.genno))
				prevgen = self.genno
			if self.datadone % 10 == 0 and self.datadone != prevdata:
				print("Datasets done: " + str(self.datadone))
				prevdata = self.datadone


	#trains the given model for given rate and mutations, defined 
	#above. this is repeated for cycles no. of times (generations)
	def biotrain(self, dataset, cycles = 10, rate = 10, mutation = 1):
		global ok
		self.genno = 0
		self.datadone = 0
		#updater stuff
		t = threading.Thread(target = self.updater, args = ())
		t.start()
		#ok done
		bestmodel = copy.deepcopy(self.model)
		bestcost = self.getcostfunction(bestmodel, dataset)
		bestcost1 = copy.deepcopy(bestcost)
		for omega in range(cycles):
			self.genno += 1
			buffergen = self.reproducemodel(bestmodel, rate, mutation)
			for each in range(len(buffergen)):
				newcost = self.getcostfunction(buffergen[each], dataset)
				if newcost < bestcost:
					bestmodel = copy.deepcopy(buffergen[each])
					bestcost = newcost
		print("Cost at the start: " + str(bestcost1))
		print("Cost at end of training: " + str(bestcost))
		ok = False
		self.model = copy.deepcopy(bestmodel)
		return bestmodel


class FlexiMatrix(object):

	#in init itself no. of layers is set
	def __init__(self, nlayers = 1, ninputlayer = 1):
		self.nlayers = nlayers
		self.ninputlayer = ninputlayer
		self.layers = [[] for i in range(self.nlayers)]


	def __repr__(self):
		retter = ""
		retter += ("\n")
		for each in range(len(self.layers)):
			retter += ("layer " + str(each + 1) + "" + "-> " + str(self.layers[each]) + "\n")
		return retter


	#converts diagonal Matrix to FlexiMatrix(c)
	#note: must be a diagonal matrix
	def parse(self, matrixtoparse):
		counter = 0
		for each in range(len(self.layers)):
			for i in range(len(self.layers[each])):
				for x in range(len(self.layers[each][i][0])):
					#print(matrixtoparse.rs[counter])
					self.setval(each, i, matrixtoparse.rows[counter][counter], x)
					counter += 1
				self.setval(each, i, matrixtoparse.rows[counter][counter])
				counter += 1



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
	#inital value of all weights is 1 and all biases are also 1
	def initiate(self):
		for each in range(self.nlayers):
			for i in range(len(self.layers[each])):
				if each == 0: self.layers[each][i] = [[1 for i in range(self.ninputlayer)], 1]
				else: self.layers[each][i] = [[1 for i in range(len(self.layers[each - 1]))], 1]

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


	#returns no. of all the values present in the FlexiMatrix(c)
	def getsize(self):
		size = 0
		for each in range(len(self.layers)):
			for i in range(len(self.layers[each])):
				for x in range(len(self.layers[each][i][0])):
					size += 1
				size += 1
		return size


	#returns the FlexiMatrix(c) in a simple 2d matrix of the form:
	"""the matrix is a square matrix with all non initialised values as 1. hopefully it works
	[[w1, w2, ..., b1, w1, w2, .... b2 ... bn],
	 [w1, w2, ..., b1, w1, w2, .... b2 ... bn],
	 .
	 .
	 .
	 [w1, w2, ..., b1, w1, w2, .... b2 ... bn],
	 [1, 1, 1, ..., 1]
	]
	this one simply pushes the fleximatrix values into simple matrix,
	then fills rest of the values with 1
	"""
	def get2dmatrixtypeone(self, appendingval = 1):
		size = int(math.ceil((self.getsize())**0.5))
		finalreturninglayers = Matrix(size, size)
		for each in range(len(self.layers)):
			for i in range(len(self.layers[each])):
				for x in range(len(self.layers[each][i][0])):
					finalreturninglayers.push(self.layers[each][i][0][x])
				finalreturninglayers.push(self.layers[each][i][1])
		while True:
			try:
				finalreturninglayers.push(appendingval)
			except:
				break
		return finalreturninglayers




	#returns the FlexiMatrix(c) in a simple 2d matrix of the form:
	"""the matrix is a square matrix with all non initialised values as 1. hopefully it works
	[[w1, w2, ..., b1, w1, w2, .... b2 ... bn, 1, 1, ... 1],
	 [w1, w2, ..., b1, w1, w2, .... b2 ... bn, 1, 1, ... 1],
	 .
	 .
	 .
	 [w1, w2, ..., b1, w1, w2, .... b2 ... bn, 1, 1, ... 1]
	]
	this one simply makes each layer of length size by adding 1s at the end
	"""
	def get2dmatrixtypetwo(self, appendingval = 1):
		size = 0
		buffmat = []
		for each in range(len(self.layers)):
			bufflayer = []
			for i in range(len(self.layers[each])):
				for x in range(len(self.layers[each][i][0])):
					bufflayer.append(self.layers[each][i][0][x])
				bufflayer.append(self.layers[each][i][1])
			if len(bufflayer) > size:
				size = len(bufflayer)
			buffmat.append(bufflayer)
		finalreturninglayers = Matrix(size, size, appendingval)
		#print(finalreturninglayers)
		for each in range(len(buffmat)):
			for i in range(len(buffmat[each])):
				finalreturninglayers.Set(each+1, i+1, buffmat[each][i])

		return finalreturninglayers


	"""returns a 2d matrix where the diagonal is all the values of the FlexiMatrix(c)
	also this is probably gonna be huge. bruh i dont think a single model can be this big"""

	def get2dmatrixtypethree(self):
		size = self.getsize()
		#print(size)
		finalreturninglayers = Matrix(size, size)
		counter = 1
		for each in range(len(self.layers)):
			for i in range(len(self.layers[each])):
				for x in range(len(self.layers[each][i][0])):
					finalreturninglayers.Set(counter, counter, self.layers[each][i][0][x])
					counter += 1
				finalreturninglayers.Set(counter, counter, self.layers[each][i][1])
				counter += 1

		return finalreturninglayers



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

	def Multiply(self, value):
		for each in range(len(self.layers)):
			for i in range(len(self.layers[each])):
				self.layers[each][i][1] *= value
				for x in range(len(self.layers[each][i][0])):
					self.layers[each][i][0][x] *= value

	def Add(mat1, mat2):
		summat = copy.deepcopy(mat1)
		for each in range(len(summat.layers)):
			for i in range(len(summat.layers[each])):
				summat.layers[each][i][1] = mat1.layers[each][i][1] + mat2.layers[each][i][1]
				for z in range(len(summat.layers[each][i][0])):
					summat.layers[each][i][0][z] = mat1.layers[each][i][0][z] + mat2.layers[each][i][0][z]
		return summat

	#returns a FlexiMatrix(c) whose each value is inverse of each corresponding value of this matrix
	#@deprecated
	def getinverse(self):
		inverse = copy.deepcopy(self)
		for each in range(len(inverse.layers)):
			for i in range(len(inverse.layers[each])):
				for x in range(len(inverse.layers[each][i])):
					inverse.layers[each][i][1] = (1 / inverse.layers[each][i][1])
					for z in range(len(inverse.layers[each][i][0])):
						inverse.layers[each][i][0][z] = (1 / inverse.layers[each][i][0][z])
		return inverse
	#returns magnitude of entire matrix
	def getmagnitude(self):
		mag = 0
		for each in range(len(self.layers)):
			for i in range(len(self.layers[each])):
				for x in range(len(self.layers[each][i])):
					mag += (self.layers[each][i][1] ** 2)
					for z in range(len(self.layers[each][i][0])):
						mag += (self.layers[each][i][0][z] ** 2)
		return mag










"""Trainer that uses gradient vector"""
class Trainer(EvolutionaryTrainer):
	
	
	def getgradientvector(self, model, dataset, n = 2):
		cofxplusepsilondx = 0
		cofx = 0
		epsilon = 0
		difference = 0
		gradient = model.matrix.get2dmatrixtypethree()
		unitfleximatrix = copy.deepcopy(model.matrix)
		

		epsilon = (10 ** (-n))

		buffmodel = copy.deepcopy(model)
		cofx = self.getcostfunction(buffmodel, dataset)
		#print('reached till here')
		unitfleximatrix.Multiply(epsilon)
		buffmodel.matrix = FlexiMatrix.Add(buffmodel.matrix, unitfleximatrix)
		cofxplusepsilondx = self.getcostfunction(buffmodel, dataset)
		#print('and also here')

		difference = cofxplusepsilondx - cofx

		gradient = gradient.ScalarMultiply(difference)
		gradient = gradient.ScalarMultiply(10**n)
			#print('le end is nigh')

		

			
		return gradient

	#the new one that gradients by cell
	def getgradient(self, model, dataset):
		global delta
		index = 0
		modeltotrain = copy.deepcopy(model)
		grad = copy.deepcopy(model.matrix)
		for each in range(len(modeltotrain.matrix.layers)):
			for cell in range(len(modeltotrain.matrix.layers[each])):
				for val in range(len(modeltotrain.matrix.layers[each][cell][0])):
					buffmodel = copy.deepcopy(modeltotrain)
					cofx = self.getcostfunction(buffmodel, dataset)
					deltax = delta * buffmodel.matrix.layers[each][cell][0][val]
					buffmodel.matrix.layers[each][cell][0][val] += deltax
					cofxplusdeltax = self.getcostfunction(buffmodel, dataset)
					del buffmodel
					dobydo = (cofxplusdeltax - cofx)/deltax
					grad.layers[each][cell][0][val] = dobydo
					#print(index)
					index += 1
				buffmodel = copy.deepcopy(modeltotrain)
				cofx = self.getcostfunction(buffmodel, dataset)
				deltax = delta * buffmodel.matrix.layers[each][cell][1]
				buffmodel.matrix.layers[each][cell][1] += deltax
				cofxplusdeltax = self.getcostfunction(buffmodel, dataset)
				del buffmodel
				dobydo = (cofxplusdeltax - cofx)/deltax
				grad.layers[each][cell][1] = dobydo
				print(index)
				index += 1
		del modeltotrain
		grad.Multiply(-1)
		return grad

	def trainbc(self, dataset, generations = 1):

		global ok
		self.genno = 0
		self.datadone = 0
		#updater stuff
		t = threading.Thread(target = self.updater, args = ())
		t.start()

		buff = self.model
		init = self.getcostfunction(buff, dataset)
		for everysingletime in range(generations):
			grad = self.getgradient(buff, dataset)
			buff.matrix = FlexiMatrix.Add(buff.matrix, grad)
			
		post = self.getcostfunction(buff, dataset)

		print("Cost before training: ", init)
		print("Cost after training: ", post)
		ok = False
		return buff


	def trainvectorially(self, dataset, cycles = 10):

		global ok
		self.genno = 0
		self.datadone = 0
		#updater stuff
		t = threading.Thread(target = self.updater, args = ())
		t.start()



		buffmodel = copy.deepcopy(self.model)
		precost = self.getcostfunction(buffmodel, dataset)
		for i in range(cycles):
			gradient = self.getgradientvector(buffmodel, dataset)
			flexigradient = copy.deepcopy(buffmodel.matrix)
			flexigradient.parse(gradient)
			#del gradient
			#flexigradient.display()
			flexigradient.Multiply(-1)
			#flexigradient.display()
			buffmodel.matrix = FlexiMatrix.Add(buffmodel.matrix, flexigradient)
		postcost = self.getcostfunction(buffmodel, dataset)
		print("Cost before training: ", precost)
		print("Cost after training: ", postcost)
		ok = False
		return buffmodel