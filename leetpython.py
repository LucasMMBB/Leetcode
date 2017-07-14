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
# 19. Remove nth node from end of list
class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        dummy = cur = head
        for i in range(n):
            head = head.next
        
        if head is None:
            dummy = dummy.next
            return dummy
        while head.next:
            cur = cur.next
            head = head.next

        cur.next = cur.next.next
        return dummy

# 237.Delete Node in a Linked List
class Solution(object):
    def deleteNode(self, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        node.val = node.next.val
        node.next = node.next.next

# 203. Remove Linked List Elements
class Solution(object):
    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        dummy = pre = ListNode(0)
        pre.next = cur = head
        if cur is None:
            return None
        
        if cur.val is val and cur.next is None:
            return None

        while cur:
            if cur.val == val:
                pre.next = cur.next
            else:
                pre = cur

            cur = cur.next
        return dummy.next

# 92. Reverse Linked List ||
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseBetween(self, head, m, n):
    	dummy = cur = head
    	left, right = m, n

# 2. Add Two Numbers
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        dummy = cur = ListNode(0)
        carry = 0

        while l1 or l2:
            a = l1.val if l1 else 0
            b = l2.val if l2 else 0

            sm = a + b + carry
            cur.next = ListNode(sm%10)
            cur = cur.next
            carry = sm/10
            
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next


        if carry > 0:
            cur.next = ListNode(1)

        return dummy.next


# 445. Add Two Numbers ||
class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        l1 = self.rotate(l1)
        l2 = self.rotate(l2)
        dummy = cur = ListNode(0)
        carry = 0

        while l1 or l2:
            a = l1.val if l1 else 0
            b = l2.val if l2 else 0

            sm = a + b + carry
            cur.next = ListNode(sm%10)
            cur = cur.next
            carry = sm/10
            
            if l1:
                l1 = l1.next
            if l2:
                l2 = l2.next


        if carry > 0:
            cur.next = ListNode(1)

        return self.rotate(dummy.next)
    def rotate(self, head):
        # reverse(rotate) the linked list
        pre = None
        while head:
            temp = head.next
            head.next = pre
            pre = head
            head = temp

        return pre


# 147. INSERTION SORT LIST
# Method: insertion sort list
# Time: O(n^2)
# Space: O(1)
class Solution(object):
    def insertionSortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head is None:
            return head

        dummy = ListNode(0)
        dummy.next = head
        while head and head.next:
            if head.val <= head.next.val:
                head = head.next
            else:
                cur = dummy
                while cur.next.val <= head.next.val:
                    cur = cur.next
                tmp = head.next
                head.next = tmp.next
                tmp.next = cur.next
                cur.next = tmp
        return dummy.next



# 148. SORT LIST
# Method: merge sort

# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class Solution(object):
    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """

# 141. Linked List Cycle
class Solution(object):
    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        slow = fast = head

        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

            if id(slow) == id(fast):
                return True

        return False

# 142. Linked List Cycle ||
class Solution(object):
    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        slow = fast = head

        while fast is not None and fast.next is not None:
            slow = slow.next
            fast = fast.next.next

            if slow == fast:
                slow = head
                while slow != fast:
                    slow = slow.next
                    fast = fast.next
                return slow
        return None

# 274. H-Index
class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """

# 275. H-Index ||
class Solution(object):
    def hIndex(self, citations):
        """
        :type citations: List[int]
        :rtype: int
        """

# 230 Kth Smallest Element in a BST
# Method one: DC + Binary Search
"""
: Definition for a binary tree node.
: class TreeNode(object):
:     def __init__(self, x):
:         self.val = x
:         self.left = None
:         self.right = None
"""
class Solution(object):
    def kthSmallest(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: int
        """
        n = self.count(root.left)
        if n + 1 == k:
            return root.val
        elif n + 1 < k:
            return self.kthSmallest(root.right, k - n -1)
        else:
            return self.kthSmallest(root.left, k)

    def count(self, root):
        # count total node number of the root
        if root == None:
            return 0

        return self.count(root.left) + self.count(root.right) + 1

# 162. Find Peak Element
class Solution(object):
    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """