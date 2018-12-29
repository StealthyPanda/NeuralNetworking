from math import exp

def process(webipa, value):
	weight = webipa[0]
	bias = webipa[1]
	return (value*weight) + bias

def out(config, values):
	finval = 0
	for each in range(len(config)):
		finval += process(config[each], values[each])
	finval = float(1/(1+exp(-finval)))
	return finval

class NeuralNet(object):

	def filllayers(self):
		with open('configs_of_layers.txt', 'r') as file:
			for layer in file:
				if layer[0] == '#': continue
				layer = layer.split('\n')[0]
				parsedlayer = []
				cells = layer.split('|')
				for each in cells:
					cell = []
					webipas = each.split(',')
					for i in webipas:
						pair = i.split(' ')
						ip = [int(pair[0]), int(pair[1])]
						cell.append(ip)
					parsedlayer.append(cell)
				self.layers.append(parsedlayer)

	
	#    				|one cell's thing
	#					|
	#					v
	#one layer -->[[weight-bias pairs], [weight-bias pairs], ...]
	def __init__(self):
		
		self.layers = []
		self.pairs = []
		with open('testvals.txt', 'r') as file:
			for each in file:
				self.pairs.append(each.split('\n')[0].split(' '))
		self.filllayers()


	def displayconfig(self):
		for each in range(len(self.layers)):
			print('Layer '+str(each)+':')
			print(self.layers[each])
			print()

	def testwith(self, testval):
		cell11 = out(self.layer1[0], [testval])
		cell21 = out(self.layer2[0], [cell11])
		cell22 = out(self.layer2[1], [cell11])
		cell23 = out(self.layer2[2], [cell11])
		cell24 = out(self.layer2[3], [cell11])
		cell31 = out(self.layer3[0], [cell21, cell22, cell23, cell24])
		if cell31 > 0.5:
			return True
		else:
			return False