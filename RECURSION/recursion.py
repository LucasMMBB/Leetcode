# RECURSION TUTORIAL
# Author: MAOXU LIU


## ---------- Basic Example --------
def fact(n):
	if n <= 1:
		return 1
	else:
		return n*fact(n - 1)


## Stack Overflow error occurs in recursion
def fact_of(n):
	if n == 100:
		return 1
	else:
		return n * fact_of(n - 1)

## direct and indirect recursion
def directFun():
	directFun()

def indirectFun_1():
	indirectFun_2()

def indirectFun_2():
	indirectFun_1()
#----------- TEST CODE -----------
print fact(3)