# Define a node for linkedlist
class Node(object):
    def __init__(self, data = None, next_node = None):
        self.val = data
        self.next = next_node

   	def get_data(self):
   		return self.val

	def get_next(self):
		return self.next

	def set_next(self, new_next):
		self.next = new_next

# Definition for singly-linked list.
def print_list(node):
    while node:
        print node.val,
        node = node.next

a = Node(0)
a.set_next(Node(2))
print_list(a)