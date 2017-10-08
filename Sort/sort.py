# SORT METHODS
## --------- Selection Sort ---------
# find the smallest one for each round. Two subarrays: sorted, non sorted
class Sort(object):
	def __init__(self, nums):
		self.arr = nums

	def selectionSort(self):
		# time: Best, average and worst is O(n^2)
		# space: O(1)
		for i in range(len(self.arr)):
			pivot = self.arr[i]
			for j in range(i+1, len(self.arr)):
				if self.arr[j] < self.arr[i]:
					self.arr[j], self.arr[i] = self.arr[i], self.arr[j]
			#print self.arr
		return self.arr

	def bubbleSort(self):
		start, end = 0, len(self.arr)
		while start < end:
			for s in range(end - 1):
				if self.arr[s] > self.arr[s+1]:
					# swap them
					self.arr[s], self.arr[s+1] = self.arr[s+1], self.arr[s]
			end -= 1

		return self.arr

# Sample question for Selection Sort
# Question: Given an array of strings, sort the array using Selection Sort by word length.
class Solution(object):
	"""docstring for Solution"""
	def __init__(self, arg):
		self.arg = arg

	def sortSelect(self, str):
		arr = str.split()
		for i in range(len(arr)):
			# pivot = self.arr[i]
			for j in range(i+1, len(arr)):
				if arr[j] < arr[i]:
					arr[j], arr[i] = arr[i], arr[j]
		return arr
		
# 347. Top K Frequent Elements
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        # method hash table and sort
        from heapq import *
        # method hash table and sort
        hq = []
        hashmap = {}
        for num in nums:
            if num in hashmap:
                hashmap[num] += 1

            else:
                hashmap[num] = 1
        for key in hashmap:
            heappush(hq, (-hashmap[key], key))

        return [heappop(hq)[1] for i in range(k)]

    def topKFrequent(self, nums, k):
    	
