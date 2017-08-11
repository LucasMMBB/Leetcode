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

def preorderTraverse_2(root):
	# method: DFS
	stack, res = [], []
	cur = root
	while stack or cur != None:
		if cur:
			stack.add(cur)
			res.add(cur.val)
			cur = cur.left
		else:
			node = stack.pop()
			cur = node.right
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
			stack.append(cur)
			res.insert(0, cur.val)
			cur = cur.right
		else:
			node = stack.pop()
			cur = node.left
	return res
'''
:Binary Search Tree
:Defination Highlights:
:	- the left subtree of a node contains only nodes with keys less than node's key
:	- the right subttree contains only nodes with keys more than node's key
: 	- the left and right each much also be a binary search tree.
: 	- There mush be no duplicate nodes
'''

# A utility class that represents an individual node in a BST
class Node:
	def __init__(self, key):
		self.left = None
		self.right = None
		self.val = key

# Search a key: method: recursion
def search(root, key):
	if root is None or root.val == key:
		return root

	if root.val < key:
		return search(root.right, key)

	if root.val > key:
		return search(root.left, key)


# Insertion of a key
def insert(root, node):
	if root is None:
		root = node
	else:
		if root.val < node.val:
			if root.right is None:
				root.right = node
			else:
				insert(root.right, node)
		else:
			if root.left is None:
				root.left = node
			else:
				insert(root.left, node)

# Deletion of a key
def delete(root, node):