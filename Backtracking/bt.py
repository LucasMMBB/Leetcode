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


# 46. Permutations
# method: backtracking(recursive)
# time: O(n!); space: O(n)
class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        if len(nums) <= 1:
        	return [nums]

        res = []
        for i, x in enumerate(nums):
        	for elem in self.permute(nums[:i] + nums[i+1:]):
        		res.append([x] + elem)

       	return res

    def permute_m2(self, nums):
    	# method: iterative
    	if len(nums) <= 1:
    		return [nums]
    	res = [[]]
    	for num in nums:
    		temp = []
    		for line in res:
    			for i in range(len(line) + 1):
    				temp.append(line[:i] + [num] + line[i:])
    		res = temp
    	return res

# 47. Permutations ||
class Solution(object):
    def permuteUnique(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        :description: To handle duplication, just avoid inserting a number
        :before any of its duplicate
        """
        # to be done
    def permuteUnique(self, nums):
    	if len(nums) <= 1:
        	return [nums]
        nums.sort()
        res = []
        for i in range(len(nums)):
        	if i == 0 or nums[i-1] != nums[i]:
        		for elem in self.permuteUnique(nums[:i] + nums[i + 1:]):
        			res.append([nums[i]] + elem)
       	
       	return res

# 78. Subsets
class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        res = []
        line = []
        self.helper(nums, line, res)
        return res
    def helper(self, nums, line, res):
    	res.append([x for x in line])
    	for i, x in enumerate(nums):
    		line.append(x)
    		self.helper(nums[i+1:], line, res)
    		line.pop()

    def subsets_m2(self, nums):
    	# method: iterative
    	res = [[]]
    	nums.sort()
    	for num in nums:
    		temp = [line + [num] for line in res]
    		res += temp
    	return res

# 89. Gray Code
class Solution(object):
    def grayCode(self, n):
        """
        :type n: int
        :rtype: List[int]
        """
        res = [0]
        for i in range(n):
        	res += [x + pow(2, i) for x in reversed(res)]

        return res

# 36. Valid Sudoku
class Solution(object):
    
   	def row_valid(self, board):
   		for row in board:
   			if not self.unit_valid(row):
   				return False
   		return True

   	def col_valid(self, board):
   		for col in zip(*board):
   			if not self.unit_valid(col):
   				return False
   		return True

   	def squ_valid(self, board):
   		for i in (0, 3, 6):
   			for j in (0, 3, 6):
   				tmp = [board[x][y] for x in range(i, i + 3) for y in range(j, j + 3)]
   				if not self.unit_valid(tmp):
   					return False
   		return True

   	def unit_valid(self, unit):
   		tmp = []
   		for i in unit:
   			if i != '.':
   				tmp.append(i)
   		return len(set(tmp)) == len(tmp)

   	def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        return (self.row_valid(board) and
                self.col_valid(board) and
                self.squ_valid(board))

#------------ TBD -----------


# 22. Generate Parentheses
class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """

# 37. Sudoku Solver
class Solution(object):
    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """


# 79. Word Search
class Solution(object):
    def exist(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """

# GeeksForGeeks Questions------------
# Subset Sum
class Solution(object):
    def subsum(self, arr, sum):

