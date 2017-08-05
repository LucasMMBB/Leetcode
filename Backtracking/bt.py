'''
Leetcode
Backtracking questions
'''
# 39. Combination Sum
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        candidates.sort()
        res = []
        temp = []
        self.helper(res, temp, candidates, target, 0)
        return res
    def helper(self, res, tempList, nums, remain, start):
    	if remain < 0:
    		return
    	if remain == 0:
    		res.append([x for x in tempList])
    		return
    	for i in range(start, len(nums)):
    		tempList.append(nums[start])
    		self.helper(res, tempList, nums, remain - nums[i], i)
    		tempList.pop()

# 40. Combination Sum ||
class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        candidates.sort()
        res = []
        temp = []
        self.helper(res, temp, candidates, target, 0)
        return res
    def helper(self, res, tempList, nums, remain, start):
    	if remain < 0:
    		return
    	if remain == 0:
    		res.append([x for x in tempList])
    		return
    	for i in range(start, len(nums)):
    		if i != start and nums[i] == nums[i - 1]:
    			continue
    		tempList.append(nums[i])
    		self.helper(res, tempList, nums, remain - nums[i], i + 1)
    		tempList.pop()

# 77. Combinations
# method: backtracking
class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        res = []
        line = []
        if k > n:
        	return res
        nums = [i + 1 for i in range(n)]
        self.helper(nums, k, res, line)
        return res

    def helper(self, nums, k, res, line):
    	if len(line) == k:
    		res.append([x for x in line])
    		return
    	for i in range(len(nums)):
    		line.append(nums[i])
    		self.helper(nums[i+1:], k, res, line)
    		line.pop()
