# test file for leetcode questions

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

# define an input linked list
dummy = ListNode(0)
cur = dummy
for i in range(1, 10):
	cur.next = ListNode(i)
	cur = cur.next


# print dummy nodes val
cur = dummy
while cur:
	print cur.val
	cur = cur.next