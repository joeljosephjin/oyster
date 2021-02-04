# a=A()

class A:
	def __init__(self):
		self.a=5
		self.b=6
		# a=A()
		# print(vars(a))
	def an(self):
		a=A()
		print(vars(a))
a = A()
# print(vars(a))
# f=A()
a.an()