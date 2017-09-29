# -------- BFS ---------
# -------- DFS ---------
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        # recursion
        # time: O(n), space: O(1)
        if not root:
        	return []
      	res = []
      	def dfs(node, string):
      		if not node.left and not node.right:
      			res.append(string + str(node.val))
      		if node.left:
      			dfs(node.left, string + str(node.val) + '->')
      		if node.right:
      			dfs(node.right, string + str(node.val) + '->')
      	dfs(root, "")
        return res

    def binaryTreePaths(self, root):
    	# iterative
    	# time: O(n), O(1)
    	if not root:
    		return []
    	res = []
    	stack = [(root, "")]
    	while stack:
    		node, string = stack.pop()
    		if not node.left and not node.right:
    			res.append(string + str(node.val))
    		if node.right:
    			stack.append((node.right, string + str(node.val) + '->'))
    		if node.left:
    			stack.append((node.left, string + str(node.val) + '->'))
    	return res

    def binaryTreePaths(self, root):
    	if not root:
    		return []
    	res = []
    	queue = collections.deque([(root, "")])
    	while queue:
    		node, string = queue.popleft()
    		if not node.left and not node.right:
    			res.append(string + str(node.val))
    		if node.left:
    			queue.append((node.left, string + str(node.val) + '->'))
    		if node.right:
    			queue.append((node.right, string + str(node.val) + '->'))
    	return res


# Follow up
# return format [[1,2,5], [1,3]]
class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        if not root:
        	return []
        res = []
        self.dfs(root, res, [])
        return res
    def dfs(self, node, res, path):
    	if not node.left and not node.right:
    		path.append(node.val)
    		res.append(path)
    	if node.left:
    		self.dfs(node.left, res, path + [node.val])
    	if node.right:
    		self.dfs(node.right, res, path + [node.val])

    def binaryTreePaths_dfs(self, root):
    	if not root:
    		return []
    	stack = [(root, [])]
    	res = []
    	while stack:
    		node, path = stack.pop()
    		if not node.left and not node.right:
    			path.append(node.val)
    			res.append(path)
    		if node.right:
    			stack.append((node.right, path + [node.val]))
    		if node.left:
    			stack.append((node.left, path + [node.val]))
    	return res

    def binaryTreePaths_bfs(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
    	if not root:
    		return []
    	res = []
    	queue = collections.deque([(root, [])])
        while queue:
			node, path = queue.popleft()
			if not node.left and not node.right:
				path.append(node.val)
				res.append(path)
			if node.left:
				queue.append((node.left, path+[node.val]))
			if node.right:
				queue.append((node.right, path+[node.val]))
        return res


# 113. Path Sum ||
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        # method: recursion
        # time: O(n), space: O(1)
        if not root:
        	return []

        result = []
        self.dfs(root, sum, [], result)
        return result

    def dfs(self, node, sum, path, res):
    	if not node.left and not node.right and node.val == sum:
    		path.append(node.val)
    		res.append(path)
    	if node.left:
    		self.dfs(node.left, sum-node.val, path+[node.val], res)
    	if node.right:
    		self.dfs(node.right, sum-node.val, path+[node.val], res)

   	def pathSum_dfs(self, root, sum):
   		# method: dfs + stack
   		if not root:
   			return []
   		res = []
   		stack = [(root, [], sum)]
   		while stack:
   			node, path, sm = stack.pop()
   			if not node.left and not node.right and sm == node.val:
   				path.append(node.val)
   				res.append(path)
   			if node.right:
   				stack.append((node.right, path+[node.val], sm - node.val))
   			if node.left:
   				stack.append((node.left, path+[node.val], sm - node.val))
   		return res

   	def pathSum_bfs(self, root, sum):
   		if not root:
   			return  []
   		res = []
   		queue = collections.deque([(root, [], sum)])
   		while queue:
   			node, path, sm = queue.popleft()
   			if not node.left and not node.right and sm == node.val:
   				path.append(node.val)
   				res.append(path)
   			if node.left:
   				queue.append((node.left, path + [node.val], sm - node.val))
   			if node.right:
   				queue.append((node.right, path + [node.val], sm - node.val))
   		return res
