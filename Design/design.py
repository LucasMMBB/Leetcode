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
# 379. Design Phone Directory
class PhoneDirectory(object):

    def __init__(self, maxNumbers):
        """
        Initialize your data structure here
        @param maxNumbers - The maximum numbers that can be stored in the phone directory.
        :type maxNumbers: int
        """
        self.PhoneDirectory = set(range(maxNumbers))

    def get(self):
        """
        Provide a number which is not assigned to anyone.
        @return - Return an available number. Return -1 if none is available.
        :rtype: int
        """
        return self.PhoneDirectory.pop() if self.PhoneDirectory else -1
        

    def check(self, number):
        """
        Check if a number is available or not.
        :type number: int
        :rtype: bool
        """
        return number in self.PhoneDirectory

    def release(self, number):
        """
        Recycle or release a number.
        :type number: int
        :rtype: void
        """
        self.PhoneDirectory.add(number)


# 346. Moving Average from Data Stream
class MovingAverage(object):

    def __init__(self, size):
        """
        Initialize your data structure here.
        :type size: int
        """
        self.queue = [] # queue
        self.size = size

    def next(self, val):
        """
        :type val: int
        :rtype: float
        """
        self.queue.append(val)
        
        if len(self.queue) < self.size:
            return float(sum(self.queue)) / len(self.queue)
        else:
            while len(self.queue) > self.size:
                self.queue.pop(0)
            return float(sum(self.queue)) / self.size


'''
: method 2: using deque from collections
: time: O(1)
'''
from collections import deque
class MovingAverage_deque(object):
    def __init__(self, size):
        self.queue = deque(maxlen = size)

    def next(self, val):
        self.queue.append(val)
        return float(sum(self.queue)) / len(self.queue)



# 251. Flatten 2D Vector
class Vector2D(object):

    def __init__(self, vec2d):
        """
        Initialize your data structure here.
        :type vec2d: List[List[int]]
        """
        self.data = []
        for row in vec2d:
            self.data += row
        

    def next(self):
        """
        :rtype: int
        """
        return self.data.pop(0)

    def hasNext(self):
        """
        :rtype: bool
        """
        if len(self.data) == 0 or not self.data:
            return False
        else:
            return True