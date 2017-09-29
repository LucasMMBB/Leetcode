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
      	string = ""
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
    		node, string = queue.pop(0)
    		if not node.left and not node.right:
    			res.append(string + str(node.val))
    		if node.left:
    			queue.append((node.left, string + str(node.val) + '->'))
    		if node.right:
    			queue.append((node.right, string + str(node.val) + '->'))
    	return res