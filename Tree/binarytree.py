'''
: Here are the iterative methods for binary tree
: preorder, inorder and postorder traverse
'''
def preorderTraverse(root):
	# method: BFS
	res = []
	stack = [root]
	while stack:
		node = stack.pop()
		if node:
			res.append(node.val)
			if node.right: stack.append(node.right)
			if node.left: stack.append(node.left)
	return res

def inorderTraverse(root):
	# method: DFS
	res = []
	stack = []
	cur = root
	while stack or cur != None:
		if cur:
			stack.append(cur)
			cur = cur.left
		else:
			node = stack.pop()
			res.append(node.val)
			cur = node.right
	return res

def postorderTraverse(root):
	# method: DFS
	res = []
	stack = []
	cur = root
	while stack or cur != None:
		if cur:
			stack.append(node)
			cur = cur.left
		else:
			cur = stack.pop()
			
