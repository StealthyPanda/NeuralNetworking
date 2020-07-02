with open('train-labels.idx1-ubyte', 'rb') as file:
	print(int.from_bytes(file.read(4), byteorder = "big"))
	print(int.from_bytes(file.read(4), byteorder = "big"))
	print(int.from_bytes(file.read(1), byteorder = "big"))
	print(int.from_bytes(file.read(1), byteorder = "big"))
	print(int.from_bytes(file.read(1), byteorder = "big"))
	print(int.from_bytes(file.read(1), byteorder = "big"))
	print(int.from_bytes(file.read(1), byteorder = "big"))
	print(int.from_bytes(file.read(1), byteorder = "big"))
