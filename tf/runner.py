import sys
import copy
import cv2
buff = cv2.imread("template.png", 0)
with open("train-labels.idx1-ubyte", 'rb') as labels:
	labels.read(8)
	with open("train-images.idx3-ubyte", 'rb') as file:
		print(int.from_bytes(file.read(4), byteorder = "big"))
		print(int.from_bytes(file.read(4), byteorder = "big"))
		print(int.from_bytes(file.read(4), byteorder = "big"))
		print(int.from_bytes(file.read(4), byteorder = "big"))
		while True:
			img = copy.deepcopy(buff)
			for each in range(28):
				for i in range(28):
					val = int.from_bytes(file.read(1), byteorder = "big")
					#print(val)
					img[each, i] = val
					#resized = cv2.resize(img, (500, 500))
			img = cv2.resize(img, (200, 200))
			cv2.imshow(str(int.from_bytes(labels.read(1), byteorder = "big")), img)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
#print(buff.shape)