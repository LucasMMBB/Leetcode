"""
/*\
|*| Leetcode answers in Python
|*| Author: Maoxu
|*| Description: share my answers to whoever needs
|*| Create date: April 3rd, 2017
\*/
"""
class Solution(object):
	# 28. Implement strStr
    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        m = len(haystack)
        n = len(needle)

        for i in range(m+1):
        	for j in range(n+1):

        		if j == n:
        			return i

        		if i + j == m:
        			return -1

        		if haystack[i+j] != needle[j]:
        			break

    # 88. Merge Sorted Array
    def merge(self, nums1, m, nums2, n):
        """
        :type nums1: List[int]
        :type m: int
        :type nums2: List[int]
        :type n: int
        :rtype: void Do not return anything, modify nums1 in-place instead.
        """
        i = m - 1;
        j = n - 1;
        k = m + n - 1;
        while k >= 0 and i >= 0 and j >= 0:
        	if nums1[i] > nums2[j]:
        		nums1[k] = nums1[i]
        		i -= 1
        	else:
        		nums1[k] = nums2[j]
        		j -= 1

        	k -= 1

        if j >= 0:
        	nums1[:j+1] = nums2[:j+1]

    # SMALLEST DIFFERENCE
    def smallestDifference(self, A, B):
    	A.sort()
    	B.sort()

    	i = j = 0
    	diff = 2147483647

    	while i < len(A) and j < len(B):
    		if A[i] > B[j]:
    			diff = min(A[i] - B[i], diff)
    			j += 1
    		else:
    			diff = min(B[j] - A[i], diff)
    			i += 1

    	return diff

# 86. Partition List

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        dummy1, dummy2 = ListNode(0), ListNode(0)
        cur1, cur2 = dummy1, dummy2

        while head:
        	if head.val < x:
        		cur1.next = head
        		cur1 = cur1.next
        	else:
        		cur2.next = head
        		cur2 = cur2.next

        	head = head.next

        cur2.next = None
        cur1.next = dummy2.next

        return dummy1.next

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def rotateRight(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if not head or not head.next or k==0:
        	return head

        size = self.getSize(head)
        k %= size

        dummy = ListNode(0)
        dummy.next = head

        p = head

        for i in range(k):
        	p = p.next

        for i in range(size - k - 1):
        	p = p.next
        	head = head.next

        p.next = dummy.next
        dummy.next = head.next
        head.next = None

        return dummy.next

    def getSize(self, head):
    	i = 0
    	while head:
    		head = head.next
    		i += 1
    	return i


# 234. Palindrome Linked List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def isPalindrome(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        if head == None:
        	return True

        mid = self.getMid(head)

        right = mid.next
        mid.next = None

        return self.compare(head, self.rotate(right))

     def getMid(self, head):
     	slow = head
     	fast = head.next

     	while fast and fast.next:
     		slow = slow.next
     		fast = fast.next.next

     	return slow


     def rotate(self, head):
     	pre = None
     	while head:
     		temp = head.next
     		head.next = pre
     		pre = head
     		head = temp

     	return pre

     def compare(sefl, h1, h2):
     	while h1 and h2:
     		if h1.val != h2.val:
     			return False

     		h1 = h1.next
     		h2 = h2.next

     	return True

# 92. Reverse Linked List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseBetween(self, head, m, n):
    	dummy = head
    	left, right = m, n
    	