from NeuralNetwork import *
import cv2

"""buffimg = cv2.imread("9/9.png", 1)


print(buffimg[0, 0][0])"""


for each in range(10):
	outputcells = ["0" for i in range(10)]
	outputcells[each] = "1"
	outputcells = " ".join(outputcells)
	outputcells = outputcells.strip()

	buffimg = cv2.imread(str(each) + "/" + str(each) + ".png", 1)

	inputcells = ""

	for i in range(28):
		for x in range(28):
			#if len(buffimg[i, x]) != 3: print(len(buffimg[i, x]))
			for z in range(3):
				inputcells += (str(buffimg[i, x][z]) + " ")

	inputcells = inputcells.strip()



	with open("data.txt", 'a') as file:
		file.write(inputcells)
		file.write("\n")
		file.write(outputcells)
		file.write("\n")