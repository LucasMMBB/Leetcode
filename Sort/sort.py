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

	def insertSort(self):
		for i in range(1, len(self.arr)):
			num = self.arr[i]

			j = i - 1
			while j >= 0 and num < self.arr[j]:
				arr[j+1] = arr[j]
				j -= 1
			arr[j+1] = num

	def heapSort(self):
		n = len(self.arr)

		# To heapify subtree rooted at index i
		# n is the size of heap
		def heapify(arr, n, i):
			largest = i
			l = 2 * i + 1 # left
			r = 2 * i + 2 # right

			if l < n and arr[i] < arr[l]:
				largest = l

			if r < n and arr[largest] < arr[r]:
				largest = r

			if largest != i:
				arr[i], arr[largest] = arr[largest], arr[largest] # swap
				heapify(arr, n, largest)
		# Build a maxheap
		for i in range(n, -1, -1):
			heapify(arr, n, i)

		# Remove element one by one
		for i in range(n-1, 0, -1):
			self.arr[i], self.arr[0] = self.arr[0], self.arr[i]
			heapify(arr, i, 0)


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
    	
