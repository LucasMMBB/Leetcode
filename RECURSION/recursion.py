# RECURSION TUTORIAL
# Author: MAOXU LIU

def fact(n):
	if n <= 1:
		return 1
	else:
		return n*fact(n - 1)


#----------- TEST CODE -----------
print fact(3)