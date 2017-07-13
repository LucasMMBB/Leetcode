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
	# preoder
	ans = []
	if tree is  None: return
	ans.append(tree.val)
	print tree.val
	total(tree.left)
	total(tree.right)

def inorder(tree):
#-------------- test part --------------------------#
#t = Tree(5, Tree(2, Tree(1), Tree(3)), Tree(6, Tree(5), Tree(7)))
test = Tree(2, Tree(1, Tree(10), Tree(100)), Tree(3, Tree(30), Tree(300)))

a = total(test)
print a
print type(a)