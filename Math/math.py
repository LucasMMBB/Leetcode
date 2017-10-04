"""
: How to swap two numbers without using a temporary variable
"""
def swap(a, b):
	'''
	:type a: int
	:type b: int
	:rtype tuple
	'''
	a = a + b
	b = a - b
	a = a - b
	return (a,b)

print swap(1,2)