ten = []
with open('ten.png', 'rb') as file:
	ten = bytearray(file.read())
nine = []
with open('nine.png', 'rb') as file:
	nine = bytearray(file.read())
common = []
for each in range(len(ten)):
	if ten[each] == nine[each]: common.append(ten[each])
print(len(ten))
print(len(nine))
print("commons = " + str(len(common)))