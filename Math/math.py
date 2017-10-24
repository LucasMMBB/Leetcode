"""
: How to swap two numbers without using a temporary variable
"""
def swap(a, b):
	'''
	:type a: int
	:type b: int
	:rtype tuple
	'''
	# method: Arithmetic Operators
	a = a + b
	b = a - b
	a = a - b
	return (a, b)

def swap(a, b):
	# method: Arithmetic Operators
	a = a * b
	b = a / b
	a = a / b
	return (a, b)

def swap(a, b):
	a = a ^ b
	b = a ^ b
	a = a ^ b
	return (a, b)

print swap(1,2)