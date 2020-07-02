import sys
with open("train-labels.idx1-ubyte", 'rb') as file:
	for each in range(10):
		print(int.from_bytes(file.read(1), byteorder = sys.byteorder, signed = True))