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