# Tree
class Tree:
	def __init__(self, val, left = None, right = None):
		self.val = val
		self.left = left
		self.right = right
	def size(self):

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


# Trie
class TrieNode(object):

	def __init__(self):
		self.R = 26
		self.links = self.R * [None]
		self.isEnd = False

	def put(ch, node):
		self.links[ch - 'a'] = node
	
	def containsKey(ch):
		return self.links[ch - 'a'] != None

	def get(ch):
		return self.links[ch - 'a']

	def setEnd():
		self.isEnd = True

	def isEnd():
		return self.isEnd

class TrieNode:
	def __init__(self):
		self.children = [None]*26
		self.isEnd = False

class Trie(object):
	"""docstring for Trie"""
	def __init__(self):
		self.root = self.getNode()

	def getNode(self):
		return TrieNode()

	def _charToIndex(self, ch):
		# private helper function
		# converts key current character into index
		# use only 'a' through 'z' and lower case
		return ord(ch) - ord('a')

	def insert(self, key):
		# if not present, inserts key into trie
		# if the key is prefix of trie node
		# just marks leaf node
		node = self.root
		for i in key:
			index = self._charToIndex(i)

			# if current character is not present
			if not node.children[index]:
				node.children[index] = self.getNode()

			node = node.children[index]

		# mark last node as leaf
		node.isEnd = True

	def search(self, key):
		# search key in the trie
		# Returns true if key presents in trie, else false

		node = self.root
		for i in key:
			index = self._charToIndex(i)
			if not node.children[index]:
				return False
			node = node.children[index]

		return node != None and node.isEnd

	def startsWith(self, prefix):
		node = self.root

		for i in prefix:
			index = self._charToIndex(i)
			if not node.children[index]:
				return False
			node = node.children[index]

		return True and node != None

#-------------- test part --------------------------#
test = Tree(1, Tree(2, Tree(4), Tree(5)), Tree(3))

print test.size()
