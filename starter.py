from NeuralNet import *

n = NeuralNet()
n.displayconfig()

print()

#print('It\'s output is: ')
#print(out(config, values))
if n.testwith(4): print('Old')
else: print('Young')