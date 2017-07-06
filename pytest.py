# test file for leetcode questions

# Definition for singly-linked list.
class Node(object):

    def __init__(self, x):
        self.val = x
        self.next = None

    def print_list(self):
	    node = self
	    while node:
	        print node.val
	        node = node.next
    
    def reverselist(self):
		cur = self
		nex = None
		pre = None

		while cur:
			nex = cur.next
			cur.next = pre
			pre = cur
			cur = nex

		return pre
    
    def getMid(self): 
		# get middle number
		# 1 2 3 4 5 -> 3; 1 2 3 4 5 6 -> 3
		slow = self
		fast = self.next

		while fast and fast.next:
			slow = slow.next
			fast = fast.next.next

		print slow.val + 1

b = a = Node(0)
for i in range(1,6):
	a.next = Node(i)
	a = a.next

#a.print_list()
#b.print_list()
b.getMid()