class TreeNode(object):
	"""docstring for TreeNode"""
	def __init__(self, arg, left=None, right=None):
		self.val = arg
		self.left = left
		self.right = right

left = TreeNode(1)
left.left = TreeNode(2)
left.left.left = TreeNode(3)
right = TreeNode(10)
root = TreeNode(0, left, right)


queue = [root]
level = 1

while queue:
	size = len(queue)
	for i in range(len(queue)):
		node = queue.pop(0)

		if node.left:
			queue.append(node.left)

		if node.right:
			queue.append(node.right)

		if node.left == None and node.right == None:
			return level
	level += 1		