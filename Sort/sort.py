# SORT METHODS
## --------- Selection Sort ---------
class Sort(object):
	def __init__(self, nums):
		self.arr = nums

	def selectionSort(self):
		# time: O(n^2), space: 
		for i in range(len(self.arr)):
			pivot = self.arr[i]
			for j in range(i+1, len(self.arr)):
				if self.arr[j] < pivot:
					self.arr[j], self.arr[i] = self.arr[i], self.arr[j]

		return self.arr

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
    	
