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

## tailed and non-tailed recursion
# A recursive function is tail recursive when recursive call is the last thing executed by the function.
# tail recursion is better than non-tail
def printnum(n):
    if n < 0:
        print "fuck you" + str(n)
        return
    print " " + str(n)

    # the last executed statement is recursive call
    printnum(n-1)
#----------- TEST CODE -----------
print fact(3)