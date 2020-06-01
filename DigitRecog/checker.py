bufflist = []

with open("1.txt", 'r') as file:
	for each in file:
		bufflist.append(each.split(' '))

print(len(bufflist[2]))