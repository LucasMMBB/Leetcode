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
    	
