class Tree:
	def __init__(self, val, left = None, right = None):
		self.val = val
		self.left = left
		self.right = right

	def __str__(self):
		return str(self.val)
'''
	def traverse(self):
		"""
		:rtype: List[int]
		"""
		if self.left is None and self.right is None:
			return
		
		elif self.left is not None and self.right is not None:
			return self.left.traverse() + self.right.traverse() + self.val

		elif self.left is not None and self.right is None:
			return self.left.traverse + self.val

		else:
			return self.right.traverse + self.val
'''

#-------------- Some useful methods ---------------#
def total(tree):
	# print preoder
	# Sequence: root left right
	if tree is  None: return

	print tree.val
	total(tree.left)
	total(tree.right)

def inorder(tree):
	# print inorder
	# Sequence: left root right
	if tree is None: return

	inorder(tree.left)
	print tree.val
	inorder(tree.right)

def postorder(tree):
	# print postorder
	# Sequence: left, right, root
	if tree:
		postorder(tree.left)
		postorder(tree.right)
		print tree.val


#-------------- test part --------------------------#
test = Tree(1, Tree(2, Tree(4), Tree(5)), Tree(3))

inorder(test)
