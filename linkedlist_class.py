# Define a node for linkedlist
class Node(object):
    def __init__(self, data = None, next_node = None):
        self.val = data
        self.next = next_node

# Definition for singly-linked list.
class LinkedList(object):
	def __init__(self):
		self.head = None

	def printlist(self):
		tem = self.head
		while tem:
			print tem.val
			tem = tem.next

	def size(self):
		#cur = self.head
		count = 0
		while cur:
			count += 1
			cur = cur.next
		return count

# some useful methods
def print_list(node):
    while node:
        print node.val,
        node = node.next

def size(node):
	count = 0
	while node:
		node = node.next
		count += 1
	return count

def reverse(node):
	# reverse a linked list
	cur = head
	nex = None
	pre = None

	while cur:
		nex = cur.next
		cur.next = pre
		pre = cur
		cur = nex

	return pre
a = Node(0)
a.next = Node(1)
a.next.next = Node(2)

p = a.next
print(size(a))
print(size(p))