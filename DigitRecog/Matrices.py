class Matrix(object):
	
	



	def __init__(self, r, cols, initval = 0):
		self.rows = [[initval for i in range(cols)] for x in range(r)]
		self.rs = r
		self.cl = cols
		self.nextr = 1
		self.nextc = 1
		self.pextr = 1
		self.pextc = 1

	#nice
	def __repr__(self):
		ret = ""
		highest = 0

		for each in self.rows:
			for i in each:
				if len(str(i)) > highest: highest = len(str(i))

		for each in self.rows:
			for i in each:
				ret += "".join([" " for x in range(highest - len(str(i)))]) + str(i) + "  "
			ret += "\n"

		return ret


	def GetIdentity(order):
		ident = Matrix(order, order)
		for i in range(1, order + 1):
			ident.Set(i, i, 1)
		return ident

	

	def GetSubMatrix(mat, row, column):

		copy = Matrix(mat.rs, mat.cl)

		for each in range(mat.rs):
			for i in range(mat.cl):
				if each + 1 == row or i + 1 == column: copy.Set(each + 1, i + 1, 'x')
				else: copy.Set(each + 1, i + 1, mat.Get(each + 1, i + 1))


		subbed = Matrix(mat.rs - 1, mat.cl - 1)

		counterrow = 1
		countercolumn = 1

		for each in copy.rows:
			for i in each:
				if i != 'x':
					subbed.Set(counterrow, countercolumn, i)
					countercolumn += 1
					if countercolumn > subbed.cl:
						countercolumn = 1
						counterrow += 1



		return subbed



	def GetOrder(self):
		return str(self.rs) + 'x' + str(self.cl)

	def Get(self, row, column):
		return self.rows[row - 1][column - 1]

	def Set(self, row, column, value):
		self.rows[row - 1][column - 1] = value

	def SetRow(self, row, index):
		index -= 1
		if len(row) != len(self.rows[index]): return "Row size doesn't match"
		for each in range(len(self.rows[index])):
			self.rows[index][each] = row[each]

	def SetColumn(self, column, index):
		index -= 1
		if len(column) != len(self.rows): return "Column size doesn't match"
		for each in range(0, len(self.rows)):
			self.rows[each][index] = column[each]

	def Add(mat1, mat2):
		if mat1.GetOrder() != mat2.GetOrder(): return "Can't be added"

		Resultant = Matrix(int(mat1.GetOrder().split('x')[0]), int(mat1.GetOrder().split('x')[1]))
		for each in range(mat1.rs):
			for i in range(mat1.cl):
				row = each + 1
				column = i + 1
				Resultant.Set(row, column, mat1.Get(row, column) + mat2.Get(row, column))

		return Resultant

	def ScalarMultiply(mat, const = 1):
		result = Matrix(mat.rs, mat.cl)

		for each in range(mat.rs):
			for i in range(mat.cl):
				result.Set(each + 1, i + 1, mat.Get(each + 1, i + 1) * const)

		return result

	def CrossMultiply(mat1, mat2):
		rowso1 = int(mat1.GetOrder().split('x')[0])
		colso1 = int(mat1.GetOrder().split('x')[1])
		rowso2 = int(mat2.GetOrder().split('x')[0])
		colso2 = int(mat2.GetOrder().split('x')[1])


		if colso1 != rowso2: return "Multiplication impossible"
		Resultant = Matrix(rowso1, colso2)

		

		for eachrow in range(rowso1):
			for eachcolumn in range(colso2):
				value = 0
				for eachelem in range(colso1):
					relement = mat1.rows[eachrow][eachelem]
					celement = mat2.rows[eachelem][eachcolumn]
					

					value += relement * celement

				
				Resultant.Set(eachrow + 1, eachcolumn + 1, value)

			
		return Resultant

	def Multiply(mat1, mat2, cross = True):
		if cross: return Matrix.CrossMultiply(mat1, mat2)
		return Matrix.ScalarMultiply(mat1, mat2)

	def GetTranspose(self):
		trans = Matrix(self.cl, self.rs)

		for each in range(self.cl):
			for i in range(self.rs):
				trans.Set(i + 1, each + 1, self.Get(each + 1, i + 1))

		return trans

	def ToPower(mat, power):
		if mat.GetOrder().split('x')[0] != mat.GetOrder().split('x')[1]: return "Not a square matrix; Can't be multiplied with itself"
		initial = Matrix.Multiply(mat, Matrix.GetIdentity(int(mat.GetOrder().split('x')[0])))

		for each in range(power - 1):
			initial = Matrix.Multiply(initial, mat)

		return initial

	def GetDeterminant(self):

		if self.rs != self.cl: return 'Not possible to find determinant'

		if self.rs == 1:
			return self.rows[0][0]


		det = 0

		for each in range(self.cl):
			element = self.rows[0][each]

			row = 1
			column = each + 1

			sign = -1

			if (row + column) % 2 == 0: sign = 1

			subdet = Matrix.GetDeterminant(Matrix.GetSubMatrix(self, row, column))

			partialdet = sign * element * subdet

			det += partialdet


		return det

	def GetInverse(mat):

		if mat.rs != mat.cl: return 'Not possible to find inverse'
		#print("gbefore")

		scale = 1/Matrix.GetDeterminant(mat)
		#print("got to here too")
		minormatrix = Matrix(mat.rs, mat.cl)

		for each in range(1, mat.rs + 1):
			for i in range(1, mat.cl + 1):
				minormatrix.Set(each, i, Matrix.GetDeterminant(Matrix.GetSubMatrix(mat, each, i)))
		#print("and now here")

		cofactormatrix = Matrix(mat.rs, mat.cl)

		for each in range(1, 1 + minormatrix.rs):
			for i in range(1, 1 + minormatrix.cl):
				sign = -1
				if (each + i) % 2 == 0: sign = 1
				cofactormatrix.Set(each, i, sign * minormatrix.Get(each, i))

		#print("aaaaaaaaand here too")

		adjointmatrix = cofactormatrix.GetTranspose()

		inversematrix = Matrix.Multiply(adjointmatrix, scale, False)
		#print("finally")


		return inversematrix


	def push(self, value):
		#print(self.nextr, self.nextc)
		if self.nextr > self.rs:
			raise IndexError
		self.Set(self.nextr, self.nextc, value)
		self.nextc += 1
		if self.nextc > self.cl:
			self.nextc = 1
			self.nextr += 1

	
		