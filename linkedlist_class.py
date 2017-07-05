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
		cur = self.head
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

llist = LinkedList()
llist.head = Node(1)
second = Node(2)
third = Node(3)

llist.head.next = second
second.next = third
listsize = llist.size()
print(listsize)