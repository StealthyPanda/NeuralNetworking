with open('1.txt', 'r') as file:
	print(len(file.read().split('\n')[0].split(' ')))