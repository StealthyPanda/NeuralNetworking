#This is a datasetgenerator
howmanyimgs = 100
filename = 3
with open("train-labels.idx1-ubyte", 'rb') as labels:
	labels.read(8)
	with open("train-images.idx3-ubyte", 'rb') as file:
		file.read(16)
		for everysingletime in range(howmanyimgs):
			lepixels = []
			for each in range(28):
				for i in range(28):
					lepixels.append(str(int.from_bytes(file.read(1), byteorder = "big")))
			label = ["0" for i in range(10)]
			label[int.from_bytes(labels.read(1), byteorder = "big")] = "1"
			with open(str(filename) + '.txt', 'a') as output:
				output.write(" ".join(lepixels).strip())
				output.write('\n' + " ".join(label).strip() + '\n')