'''
: Leetcode Desgin questions
'''
# 225. Implement Stack using Queues
class MyStack(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.queue = [] # use queue as a stack
        
    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: void
        """
        self.queue.append(x)
        if len(self.queue) > 1:
        	for i in range(len(self.queue) - 1):
        		temp = self.queue.pop(0)
        		self.queue.append(temp)       

    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        if len(self.queue) > 0:
        	return self.queue.pop(0)
        
    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        if len(self.queue) > 0:
        	return self.queue[0]

    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        if not self.queue:
        	return True
        else:
        	return False