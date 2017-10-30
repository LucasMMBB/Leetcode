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
        l,r = 0, len(nums) - 1

        while l <= r:
            mid = l + (r - l) / 2

            if(mid == 0 or nums[mid] > nums[mid - 1]) and (mid==len(nums)-1 or nums[mid]>nums[mid + 1]):
                return mid
            elif mid > 0 and nums[mid] < nums[mid - 1]:
                r = mid -1
            else:
                l = mid + 1
        return -1

# 448. Find All Numbers Disappeared in an Array
class Solution(object):
    def findDisappearedNumbers(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        ans = []
        for i in range(len(nums)):
            index = abs(nums[i]) - 1
            if(nums[index] > 0):
                nums[index] = -nums[index]

        for i in range(len(nums)):
            if nums[i] > 0:
                ans.append(i + 1)
        return ans

# 442. Find All Duplicates in an Array
class Solution(object):
    def findDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        res = []
        for i in range( len(nums) ):
            index = abs(nums[i]) - 1
            if nums[index] > 0:
                nums[index] = -nums[index]
            else:
                res.append(index + 1)
        return res

# 628. Maximum Product of Three Numbers
class Solution(object):
    def maximumProduct(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        nums.sort()
        a = nums[-1] * nums[-2] * nums[-3]
        b = nums[0] * nums[1] * nums[-1]
        return max(a,b)


# 35. Search Insert Position
class Solution(object):
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        l,r = 0, len(nums)
        while l < r:
            mid = l + (l - r) / 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                l = mid + 1
            else:
                r = mid - 1
        return l        


# 217. Contains Duplicate
class Solution(object):
    def containsDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        pyset = set()
        for i in nums:
            if i not in pyset:
                pyset.add(i)
            else:
                return True
        return False


# 219. Contains Duplicate ||
class Solution(object):
    def containsNearbyDuplicate(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: bool
        """
        tem = {}
        for i in range(len(nums)):
            if tem.has_key(nums[i]) and i - tem[nums[i]] <= k:
                return True
            tem[nums[i]] = i
        return False

# 220. Contains Duplicate |||
class Solution(object):
    def containsNearbyAlmostDuplicate(self, nums, k, t):
        """
        :type nums: List[int]
        :type k: int
        :type t: int
        :rtype: bool
        """
        if k < 1 or t < 0:
            return False

        dic = collections.OrderedDict()

        for i in range(len(nums)):
            key = nums[i] / max(1, t)

            for m in (key, key - 1, key + 1):
                if m in dic and abs(nums[i] - dic[m]) <= t:
                    return True
            dic[key] = nums[i]

            if i >= k:
                dic.popitem(last = False)

        return False
# 104. Max Depth of Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        :method: recursion
        """
        if root is None:
            return 0

        if root.left is not None or root.right is not None:
            return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
        else:
            return 1

# 111. Minimum Depth of Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def minDepth(self, root):
        """
        :method: Breadth First Search
        :type root: TreeNode
        :rtype: int
        """
        queue = [root]
        level = 1

        while queue:
            for i in range(len(queue)):
                node = queue.pop(0)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

                if not node.left and not node.right:
                    return level
            level += 1
        return level
    def minDepth_m2(self, root):
        '''
        :method: DFS
        :time: O(n), Space: O(logn)
        '''
        if not root:
            return 0

        if not root.left and not root.right:
            return 1
        elif root.left == None:
            return self.minDepth_m2(root.right) + 1
        elif root.right == None:
            return self.minDepth_m2(root.left) + 1
        else:
            return min(self.minDepth_m2(root.left), self.minDepth_m2(root.right)) + 1

  
# 102.Binary Tree Level Order Traversal
class Solution(object):
    def levelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        res = []
        queue = [root]

        while queue:
            size = len(queue)
            lres = []
            for i in range(size):
                node = queue.pop(0)
                lres.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(lres)
        return res


# 637. Average of Levels in Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def averageOfLevels(self, root):
        """
        :type root: TreeNode
        :rtype: List[float]
        """
        queue = [root]
        res = []

        while queue:
            size = len(queue)
            sm = 0
            for i in range(size):
                node = queue.pop(0)
                sm += node.val
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            avg = sm / range(size)
            res.append(avg)
        return res

# 107. Binary Tree Level Order Traversal ||
class Solution(object):
    def levelOrderBottom(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        res = []
        queue = [root]

        while queue:
            size = len(queue)
            lres = []
            for i in range(size):
                node = queue.pop(0)
                lres.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.insert(0, lres)
        return res


# 103. Binary Tree Zigzag Level Order Traversal
class Solution(object):
    def zigzagLevelOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        if not root:
            return []
        res = []
        queue = [root]
        tag = 1
        while queue:
            size = len(queue)
            lres = []
            for i in range(size):
                node = queue.pop(0)
                if tag == 1:
                    lres.append(node.val)
                else:
                    lres.insert(0, node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(lres)
            tag = -tag
        return res

# 199. Binary Tree Right Side View
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def rightSideView(self, root):
        """
        :method: BFS
        :type root: TreeNode
        :rtype: List[int]
        """
        if not root:
            return []
        res = []
        queue = [root]

        while queue:
            size = len(queue)
            lres = []
            for i in range(size):
                node = queue.pop(0)
                lres.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            res.append(lres.pop())
        return res

    def rightSideView_2(self, root):
        #method: DFS
        res = []
        self.helper(res, root, 0)
        return res

    def helper(self, res, root, level):
        if not root:
            return

        if level == len(res):
            res.append(root.val)
        
        self.helper(res, root.right, level + 1)
        self.helper(res, root.left, leve + 1)

# 116. Populating Next Right Pointers in Each Node
# Definition for binary tree with next pointer.
# class TreeLinkNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
#         self.next = None

class Solution:
    # @param root, a tree link node
    # @return nothing
    def connect(self, root):
        if root is None:
            return

        queue = [root]
        level = 0

        while queue:
            size = len(queue)
            for i in range(size):
                node = queue.pop(0)
                if i == size - 1:
                    node.next = None
                else:
                    node.next = queue[0]
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
        return

    def connect(self, root):
        # method: lnked list in each level
        if root is None:
            return
        queue = [root]
        level = 0

        while queue:
            size = len(queue)
            last = None
            for i in range(size):
                node = queue.pop(0)
                if last:
                    last.next = node
                last = node
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            last.next = None
        return

# 117. Populating Next Right Pointers in Each Node ||
class Solution:
    def connect(self, root):
        # method: lnked list in each level
        if root is None:
            return
        queue = [root]
        level = 0

        while queue:
            size = len(queue)
            last = None
            for i in range(size):
                node = queue.pop(0)
                if last:
                    last.next = node
                last = node
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            last.next = None
        return


# 136. Single Number
class Solution(object):
    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        st = Set()
        for num in nums:
            if num in st:
                st.remove(num)
            else:
                st.add(num)
        return st.pop()

# 100. Same Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isSameTree(self, p, q):
        """
        :method: recursion
        :type p: TreeNode
        :type q: TreeNode
        :rtype: bool
        """
        if p == None and q == None:
            return True
        elif p and q and p.val == q.val:
            return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
        else:
            return False    

    def isSameTree(self, p, q):
        # method: no recursion
        if p == None and q == None:
            return True

        stack = [(p, q)]
        while stack:
            lnode, rnode = stack.pop(0)

            if lnode == None and rnode == None:
                continue
            elif lnode and rnode and lnode.val == rnode.val:
                stack.append((lnode.left, rnode.left))
                stack.append((lnode.right, rnode.right))
            else:
                return False
        return True


# 101. Symmetric Tree
class Solution(object):
    def isSymmetric(self, root):
        """
        :method: no recursion
        :type root: TreeNode
        :rtype: bool
        """
        if root == None:
            return True

        left = root.left
        right = root.right
        if left == None and right == None:
            return True
        stack = [[left, right]]
        while stack:
            lnode, rnode = stack.pop(0)

            if lnode == None and rnode == None:
                continue
            elif lnode and rnode and lnode.val == rnode.val:
                stack.append([lnode.left, rnode.right])
                stack.append([lnode.right, rnode.left])
            else:
                return False

        return True

    def isSymmetric(self, root):
        if root == None:
            return True

    def helper(self, left, right):
        if left == None and right == None:
            return True
        elif left and right and left.val == right.val:
            return self.helper(left.left, right.right) and self.helper(left.right, right.left)
        else:
            return False

# 110. Balanced Binary Tree
class Solution(object):
    def isBalanced(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        return self.height(root) != -1
    def height(self, root):
        if root == None:
            return 0
        l = self.height(root.left)
        r = self.height(root.right)

        if l == -1 or r == -1 or abs(l - r) > 1:
            return -1
        return max(l, r) + 1

# 108. Convert Sorted Array to Binary Search Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution(object):
    def sortedArrayToBST(self, nums):
        """
        :type nums: List[int]
        :rtype: TreeNode
        """
        if len(nums) == 0:
            return None
        mid = (0 + len(nums) - 1) / 2
        node = TreeNode(nums[mid])
        node.left = self.sortedArrayToBST(nums[0:mid])
        node.right = self.sortedArrayToBST(nums[mid + 1 : len(nums)])
        return node   


# 109. Convert Sorted List to Binary Search Tree
class Solution(object):
    def sortedListToBST(self, head):
        """
        :type head: ListNode
        :rtype: TreeNode
        """
        if head == None:
            return None
        rt = self.midElem(head)
        midValue = rt[0]
        node = TreeNode(midValue)
        node.left = self.sortedListToBST(rt[1])
        node.right = self.sortedListToBST(rt[2])
        return node

    def midElem(self, head):
        if head == None:
            return None
        prev = None
        slow = head
        fast = head.next
        while fast and fast.next:
            prev = slow
            slow = slow.next
            fast = fast.next.next

        if prev:
            prev.next = None
            left = head
        else:
            left = None
            
        right = slow.next
        return [slow.val, left, right]


# 112. Path Sum
class Solution(object):
    def hasPathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: bool
        """
        if root is None:
            return False

        if root.left == None and root.right == None and sum - root.val == 0:
            return True;

        return hasPathSum(root.left, sum - root.val) or hasPathSum(root.right, sum - root.val)


# 113. Path Sum ||
class Solution(object):
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: List[List[int]]
        """
        res = []
        path = []
        self.helper(root, sum, result, path)
        return result

    def helper(self, root, sum, res, path):
        if root is None:
            return

        path.append(root.val)
        if root.left is None and root.right is None:
            if sum == root.val:    
                newlist = p
                res.append(newlist)
            return

        if root.left is not None:
            self.helper(root.left, sum - root.val, res, p)
            p.pop(len(p) - 1)

        if root.right is not None:
            self.helper(root.right, sum - root.val, res, p)
            p.pop(len(p) - 1)

    def pathSum_m2(self, root, sum):
        # method: recursion with python
        if root is None:
            return []
        res = []
        self.dfs(root, sum, [], res)

    def dfs(self, root, sum, ls, res):
        if root is None:
            return
        if root.left is None and root.right is None and sum == root.val:
            ls.append(root.val)
            res.append(ls)
        if root.left:
            self.dfs(root.left, sum - root.val, ls + [root.val], res)
        if root.right:
            self.dfs(root.right, sum - root.val, ls + [root.val], res)

    def pathSum_m3(self, root, sum):
        if not root:
            return
        if not root.left and not root.right and sum == root.val:
            return [[root.val]]
        tmp = self.pathSum_m3(root.left, sum - root.val) + self.pathSum_m3(root.right, sum - root.val)
        return [[root.val] + i for i in tmp]


# 170. Two Sum ||| - Data structure design
class TwoSum(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.hashtable = dict()
        

    def add(self, number):
        """
        Add the number to an internal data structure..
        :type number: int
        :rtype: void
        """
        if number in self.hashtable:
            self.hashtable[number] += 1
        else:
            self.hashtable[number] = 1


    def find(self, value):
        """
        Find if there exists any pair of numbers which sum is equal to the value.
        :type value: int
        :rtype: bool
        """
        for key in self.hashtable:
            if key * 2 == value:
                if self.hashtable[key] >= 2:
                    return True
            else:
                if value - key in self.hashtable:
                    return True
        return False

# 105. Construct Binary Tree from Preorder and Inorder
class Solution:
    def buildTree(self, preorder, inorder):
        if len(inorder) > 0:
            mid = inorder.index(preorder.pop(0))
            root = TreeNode(inorder[mid])
            root.left = self.buildTree(preorder, inorder[:mid])
            root.right = self.buildTree(preorder, inorder[mid + 1:])
            return root


# 106. Construct Binary Tree from Inorder and Postorder
class Solution(object):
    def buildTree(self, inorder, postorder):
        """
        :type inorder: List[int]
        :type postorder: List[int]
        :rtype: TreeNode
        """
        if len(postorder) > 0:
            mid = inorder.index(postorder.pop())
            root = TreeNode(inorder[mid])
            root.left = self.buildTree(inorder[:mid], postorder)
            root.right = self.buildTree(inorder[mid + 1 :], postorder)
            return root
        return


# 545. Boundary of Binary Tree
class Solution(object):
    def isLeaf(self, root):
        '''
        :rtype: bool
        '''
        if not root:
            return False
        return root.left == None and root.right == None

    def addLeaves(self, res, root):
        if self.isLeaf(root):
            res.append(root.val)
        else:
            if root.left:
                self.addLeaves(res, root.left)
            if root.right:
                self.addLeaves(res, root.right)

    def boundaryOfBinaryTree(self, root):
        res = []
        if not root:
            return res

        if not self.isLeaf(root):
            res.append(root.val)

        t = root.left
        while t:
            if not self.isLeaf(t):
                res.append(t.val)

            if t.left:
                t = t.left
            else:
                t = t.right
        self.addLeaves(res, root)
        t = root.right
        stack = []
        while t:
            if not self.isLeaf(t)
                stack.insert(0, t.val)
            if t.right:
                t = t.right
            else:
                t = t.left
        res += stack
        return res


# 202. Happy Number
class Solution(object):
    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        set = {1}
        while n not in set:
            set.add(n)
            sum = 0
            while n > 0:
                sum += (n%10)**2
                n = n / 10
            if sum == 1:
                return True
            n = sum
        return n == 1

    def isHappy(self, n):
        set = {1}
        while n not in set:
            set.add(n)
            n = sum(int(x)**2 for x in str(n))
        return n == 1


# 258. Add Digits
class Solution(object):
    def addDigits(self, num):
        """
        :type num: int
        :rtype: int
        """
        if num is None:
            return None
        numStr = str(num)
        num = 0
        while len(numStr) != 1:
            for i in range(len(numStr)):
                num += int(numStr[i])
            numStr = str(num)

        return num


# 263. Ugly Number
class Solution(object):
    def isUgly(self, num):
        """
        :type num: int
        :rtype: bool
        """
        if num <= 0: return False
        while num % 2 == 0:
            num /= 2

        while num % 3 == 0:
            num /= 3

        while num % 5 == 0:
            num /= 5

        return num == 1


# 264. Ugly Number ||
class Solution(object):
    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        ugly = [1]
        i2 = i3 = i5 = 0
        for i in range(n - 1):
            next_i2, next_i3, next_i5 = ugly[i2]*2, ugly[i3]*3, ugly[i5]*5
            next_ugly_no = min(next_i2, next_i3, next_i5)
            if next_ugly_no == next_i2:
                i2 += 1

            if next_ugly_no == next_i3:
                i3 += 1

            if next_ugly_no == next_i5:
                i5 += 1

            ugly.append(next_ugly_no)

        return ugly[-1]

# 242. Valid Anagram
class Solution(object):
    def isAnagram(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        if len(s) != len(t): return False
        if len(l) == 0: return True
        set = {}
        for i in range(len(s)):
            k = s[i]
            if k not in set:
                set[k] = 1
            else:
                set[k] += 1

        # check and pop
        for i in range(len(t)):
            k = t[i]
            if k not in set:
                return False
            else:
                if set[k] == 1:
                    set.pop(k)
                else:
                    set[k] -= 1
        return set == {}

    def isAnagram_m2(self, s, t):
        return sorted(s) == sorted(t)
    
    def isAnagram_m3(self, s, t):
        if len(s) != len(t): return False
        if len(t) == 0: return True
        res = [0 for i in range(26)]
        for i in range(len(s)):
            res[ord(s[i]) - ord('a')] += 1
            res[ord(t[i]) - ord('a')] -= 1
        for i in range(26):
            if res[i] != 0:
                return False
        return True

# 49. Group Angarams
class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        dic = {}
        res = []
        for word in strs:
            st = str(sorted(word))
            if st in dic:
                dic[st].append(word)
            else:
                dic[st] = [word]
        for key in dic:
            if(len(dic[key]) >= 1):
                res.append(dic[key])

        return res

# 249. Group Shifted Strings
class Solution(object):
    def groupStrings(self, strings):
        """
        :type strings: List[str]
        :rtype: List[List[str]]
        """
        dic = {}
        res = []
        for i in range(len(strings)):
            key = self.helper(strings[i])
            if key not in dic:
                dic[key] = [strings[i]]
            else:
                dic[key].append(strings[i])
        for key in dic:
            res.append(dic[key])
        return res

    def helper(self, str):
        if len(str) == 0:
            return ""
        if str[0] == 'a':
            return str
        else:
            newstr = ''
            for i in range(len(str)):
                tmp = ord(str[i]) - ord(str[0]) + ord('a')
                if tmp < 97:
                    tmp += 26
                newstr += chr(tmp)
                
            return newstr

    def isShift(self, s, t):
        # method: 
        if len(s) != len(t) or len(s) == 0 or len(t) == 0:
            return False
        for i in range(len(s - 1)):
            if ord(s[i]) - ord(t[i]) != ord(s[i+1]) - ord(t[i+1]):
                return False
        return True 

# 349. Intersection of Two Arrays
class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
# 204. Count Primes
class Solution(object):
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n < 2: return 0 # 1 is not a prime

        res = [True] * n

        for p in range(2, int(n ** 5) + 1):
            if A[i]:
                for j in range(p ** 2, n, p):
                    A[j] = False

        return sum(res) - 2

# 349. Intersection of Two Arrays
class Solution(object):
    def intersection(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        if len(nums1) == 0 or len(nums2) == 0 or nums1 == None or nums2 == None:
            return []
        res = {}
        set = set(nums1)
        for num in nums2:
            if num in set:
                res.add(num)
        return list(res)

# 350. Intersection of Two Arrays ||
class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        # method 1: hashset
        # time: O(n), Space: O(n)
        if len(nums1) == 0 or len(nums2) == 0 or nums1 == None or nums2 == None:
            return []
        dic = {}
        res = []
        for num in nums1:
            if num in dic:
                dic[num] += 1
            else:
                dic[num] = 1
        for num in nums2:
            if num in dic and dic[num] > 0:
                res.append(num)
                dic[num] -= 1
        return res


# 94. Binary Tree Inorder Traversal
class Solution(object):
    def __init__(self):
        self.res = []
    def inorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root:
            self.inorderTraversal(self.left)
            self.res.append(root.val)
            self.inorderTraversal(self.right)
        return self.res

    def inorderTraversal(self, root):
        # method2: iterative
        stack = []
        res = []
        stack.append((root,False))

        while stack:
            node, flag = stack.pop()

            if node:
                if flag:
                    res.append(node.val)
                else:
                    stack.append((node.right, False))
                    stack.append((node, True))
                    stack.append((node.left, False))

        return res

# 144. Binary Tree Preorder Traversal
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def __init__(self):
        self.res = [] # a global value(property) to store results
    def preorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        if root:
            self.res.append(root.val)
            self.preorderTraversal(root.left)
            self.preorderTraversal(root.right)

        return self.res

    def preorderTraversal_2(self, root):
        # method 2: iteratively
        res = []
        stack = [(root, False)]
        while stack:
            node, flag = stack.pop()
            if node:
                if flag:
                    res.append(node.val)
                else:
                    stack.append((node.right, False))
                    stack.append((node.left, False))
                    stack.append((node, True))

        return res
    def preorderTraversal_3(self, root):
        # method 3: another recursive
        res = []
        if root:
            res.append(root.val)
            res += self.preorderTraversal_3(root.left)
            res += self.preorderTraversal_3(root.right)
        return res



# 145. Binary Postorder Traversal
class Solution(object):
    def postorderTraversal(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        stack = [(root, False)]
        while stack:
            node, flag = stack.pop()
            if node:
                if flag:
                    res.append(node.val)
                else:
                    stack.append((node, True))
                    stack.append((node.right, False))
                    stack.append((node.left, False))
        return res

# 155. Min Stack
class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        curMin = self.getMin()
        if curMin == None or x < curMin:
            curMin = x
        self.stack.append((x, curMin))

    def pop(self):
        """
        :rtype: void
        """
        self.stack.pop()
        

    def top(self):
        """
        :rtype: int
        """
        if len(self.stack) == 0:
            return None
        else:
            return self.list[-1][0]
        

    def getMin(self):
        """
        :rtype: int
        """
        if len(self.stack) != 0:
            return self.stack[-1][1]
        else:
            return None
# method 2
class MinStack(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.min = sys.maxint

    def push(self, x):
        """
        :type x: int
        :rtype: void
        """
        if x <= self.min:
            self.stack.append(self.min)
            self.min = x
        self.stack.append(x)


    def pop(self):
        """
        :rtype: void
        """
        if self.stack.pop() == self.min:
            self.min = self.stack.pop()

        

    def top(self):
        """
        :rtype: int
        """
        return self.stack[-1]

        

    def getMin(self):
        """
        :rtype: int
        """
        return self.min

# Your MinStack object will be instantiated and called as such:
# obj = MinStack()
# obj.push(x)
# obj.pop()
# param_3 = obj.top()
# param_4 = obj.getMin()


# 48. Rotate Image
class Solution(object):
    def rotate(self, nums):
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        nums.reverse()
        length = len(nums)
        for i in range(length):
            for j in range(i):
                nums[i][j], nums[j][i] = nums[j][i], nums[i][j]
                
    def rotate_left(self, nums):
        nums.reverse()
        length = len(nums)
        for i in range(length):
            for j in range(length - i):
                nums[i][j], nums[length - 1 - j][length - 1 - i] = nums[length - 1 - j][length - 1 - i], nums[i][j]


# 17. Letter Combinations of a Phone Number
# method: backtracking
class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        if not digits:
            return []
        dic = {
            '1': "",
            '2': ['a', 'b', 'c'],
            '3': ['d', 'e', 'f'],
            '4': ['g', 'h', 'i'],
            '5': ['j', 'k', 'l'],
            '6': ['m', 'n', 'o'],
            '7': ['p', 'q', 'r', 's'],
            '8': ['t', 'u', 'v'],
            '9': ['w', 'x', 'y', 'z']
        }
        res = [""]
        for digit in digits:
            lst = dic[digit]
            temp = []
            for char in lst:
                for str in res:
                    temp.append(str + char)
            res = temp
        return res       

# 39. Combination Sum
# method: Backtracking
class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """

        candidates.sort()
        res = []
        line = []
        self.helper(candidates, target, res, line)
        return res

    def helper(self, candidates, target, res, line):
        if target == 0:
            res.append([x for x in line])
            return

        for i, x in enumerate(candidates):
            if x <= target:
                line.append(x)
                self.helper(candidates[i:], target - x, res, line)
                line.pop()
    
    def combinationSum_m2(self, candidates, target):
        res = []
        candidates.sort()
        self.backtrack(res, [], candidates, target, 0)
        return res
    def backtrack(self, res, tempList, nums, target, start):
        if target < 0:
            return
        elif target == 0:
            res.append([x for x in tempList])
        else: # target > 0
            for i in range(start, len(nums)):
                tempList.append(nums[i])
                self.backtrack(res, tempList, nums, target - nums[i], i)
                tempList.pop()

# 40. Combination Sum 2
class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        candidates.sort()
        res = []
        temp = []
        self.helper(res, temp, candidates, target, 0)
        return res
    def helper(self, res, tempList, nums, remain, start):
        if remain < 0:
            return
        if remain == 0:
            res.append([x for x in tempList])
            return
        for i in range(start, len(nums)):
            if i != start and nums[i] == nums[i - 1]:
                continue
            tempList.append(nums[i])
            self.helper(res, tempList, nums, remain - nums[i], i + 1)
            tempList.pop()

# 77. Combinations
# method: iterative
class Solution(object):
    def combine(self, n, k):
        """
        :type n: int
        :type k: int
        :rtype: List[List[int]]
        """
        combs = [[]]
        for _ in range(k):
            combs = [[i] + c for c in combs for i in range(1, c[0] if c else n+1)]
        return combs
        

# LRU Cache
# method: hashtable + double linkedlist
class Node(object):
    """docstring for Node"""
    def __init__(self, key, val):
        self.key = key
        self.val = val
        self.prev = None
        self.next = None
        

class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        # define double linked list
        self.capacity = capacity
        self.dic = dict()
        self.head = Node(0,0)
        self.tail = Node(0,0)
        self.head.next = self.tail
        self.tail.prev = self.head


    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key in self.dic:
            n = self.dic[key]
            self._remove(n)
            self._add(n)
            return n.val        
        return -1

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rty
        """
        if key in self.dic:
            self._remove(self.dic[key])
        node = Node(key, value)
        self._add(node)
        self.dic[key] = node
        if len(self.dic) > self.capacity:
            n = self.head.next # node
            self._remove(n)
            self.dic.pop(n.key)
    #---- private methods ----
    def _remove(self, node):
        if node is None:
            return
        p = node.prev
        n = node.next
        p.next = n
        n.prev = p

    def _add(self, node):
        if node is None:
            return
        p = self.tail.prev
        p.next = node
        self.tail.prev = node
        node.prev = p
        node.next = self.tail   


# 232. Implement Queue using Stacks
class MyQueue(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.stack_eq = [] # enqueue
        self.stack_dq = [] # dequeue
        

    def push(self, x):
        """
        Push element x to the back of queue.
        :type x: int
        :rtype: void
        """
        self.stack_eq.append(x)
        

    def pop(self):
        """
        Removes the element from in front of queue and returns that element.
        :rtype: int
        """
        if len(self.stack_dq)>0:
            return self.stack_dq.pop()
        else:
            while self.stack_eq:
                temp = self.stack_eq.pop()
                self.stack_dq.append(temp)
            return self.stack_dq.pop()

    def peek(self):
        """
        Get the front element.
        :rtype: int
        """
        if len(self.stack_dq)>0:
            return self.stack_dq[-1]
        else:
            while self.stack_eq:
                temp = self.stack_eq.pop()
                self.stack_dq.append(temp)
            return self.stack_dq[-1]
        

    def empty(self):
        """
        Returns whether the queue is empty.
        :rtype: bool
        """
        if len(self.stack_dq) == 0 and len(self.stack_eq) == 0:
            return True
        else:
            return False

# 225. Implement Stack using Queues
# using a single queue
# push O(n)
# pop O(1)
class MyStack(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.queue = [] # use queue as a stack
        
    def push(self, x):
        """
        Push element x onto stack.
        :type x: int
        :rtype: void
        """
        self.queue.append(x)
        if len(self.queue) > 1:
        	for i in range(len(self.queue) - 1):
        		temp = self.queue.pop(0)
        		self.queue.append(temp)       

    def pop(self):
        """
        Removes the element on top of the stack and returns that element.
        :rtype: int
        """
        if len(self.queue) > 0:
        	return self.queue.pop(0)
        
    def top(self):
        """
        Get the top element.
        :rtype: int
        """
        if len(self.queue) > 0:
        	return self.queue[0]

    def empty(self):
        """
        Returns whether the stack is empty.
        :rtype: bool
        """
        if not self.queue:
        	return True
        else:
        	return False


# 173. Binary Search Tree Iterator
# Definition for a  binary tree node
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class BSTIterator(object):
    def __init__(self, root):
        """
        :type root: TreeNode
        """
        self.stack = []
        self.helper(root)

    def hasNext(self):
        """
        :rtype: bool
        """
        if self.stack:
        	return True
        else:
        	return False

    def next(self):
        """
        :rtype: int
        """
        node = self.stack.pop()
        self.helper(node.right)
        return node.val


    def helper(self, root):
    	# put nodes in stack so that we can use next to pop next smallest key
    	while root:
    		self.stack.append(root)
    		root = root.left

# 200. Number of Islands
class Solution(object):
    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        m = len(grid)
        if m == 0:
        	return 0
        n = len(grid[0])

        res = 0
        queue = []

        for i in range(m):
        	for j in range(n):
        		if grid[i][j] == '1':
        			queue.append([i, j])
        			grid[i][j] = '0'
        			while queue:
        				[a, b] = queue.pop(0)
        				
        				# neighbors
        				if a + 1 < m and grid[a + 1][b] == '1':
        					queue.append([a + 1, b])
        					grid[a + 1][b] = '0'

        				if b + 1 < n and grid[a][b + 1] == '1':
        					queue.append([a, b + 1])
        					grid[a][b + 1] = '0'

        				if a - 1 >= 0 and grid[a - 1][b] == '1':
        					queue.append([a - 1, b])
        					grid[a - 1][b] == '0'

        				if b - 1 >= 0 and grid[a][b - 1] == '1':
        					queue.append([a, b - 1])
        					grid[a][b - 1] = '0'

        			res += 1
       	return res

    def numIslands_m2(self, grid):
    	# simple way
	    def sink(i, j):
	        if 0 <= i < len(grid) and 0 <= j < len(grid[i]) and grid[i][j] == '1':
	            grid[i][j] = '0'
	            map(sink, (i+1, i-1, i, i), (j, j, j+1, j-1))
	            return 1
	        return 0
	    return sum(sink(i, j) for i in range(len(grid)) for j in range(len(grid[i])))

# 279. Perfect Numbers
class Solution(object):
    def numSquares(self, n):
        """
        :type n: int
        :rtype: int
        """
        queue = [(0, 0)]
        visited = [False]*(n + 1)

        while queue:
        	cur, step = queue.pop(0)

        	a = 1
        	while cur + a*a <= n:
        		if cur + a * a == n:
        			return step + 1

        		if visited[cur + a * a] == False:
        			queue.append((cur + a * a, step + 1))
        			visited[cur + a * a] = True

        		a += 1
        return 0

# 127. Word Ladder
from collections import deque
class Solution(object):
    def ladderLength(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: int
        """
        if endWord not in wordList:
            return 0
        queue = deque()
        queue.append((beginWord,1))
        wordList = set(wordList)
        while queue:
        	word, length = queue.popleft()
        	for i in range(len(word)):
        		for c in [chr(x) for x in range(ord('a'), ord('z') + 1)]:
        			newWord = word[:i] + c + word[i+1:]
        			if newWord == endWord:
        				return length + 1

        			if newWord in wordList:
        				queue.append((newWord, length + 1))
        				wordList.remove(newWord)
       	return 0

    def ladderLength_m2(self, beginWord, endWord, wordList):
    	wordList = set(wordList) # save time using hashset
        if endWord not in wordList:
        	return 0
        wordList.append(endWord)
        queue = collections.deque([[beginWord, 1]])
        while queue:
            word, length = queue.popleft()
            if word == endWord:
                return length
            for i in range(len(word)):
                for c in 'abcdefghijklmnopqrstuvwxyz':
                    next_word = word[:i] + c + word[i+1:]
                    if next_word in wordList:
                        wordList.remove(next_word)
                        queue.append([next_word, length + 1])
        return 0

    def ladderLength_m3(self, beginWord, endWord, wordList):
        
        def construct_dict(word_list):
            d = {}
            for word in word_list:
                for i in range(len(word)):
                    s = word[:i] + "_" + word[i+1:]
                    d[s] = d.get(s, []) + [word]
            return d
            
        def bfs_words(begin, end, dict_words):
            queue, visited = deque([(begin, 1)]), set()
            while queue:
                word, steps = queue.popleft()
                if word not in visited:
                    visited.add(word)
                    if word == end:
                        return steps
                    for i in range(len(word)):
                        s = word[:i] + "_" + word[i+1:]
                        neigh_words = dict_words.get(s, [])
                        for neigh in neigh_words:
                            if neigh not in visited:
                                queue.append((neigh, steps + 1))
            return 0
        
        wordList = set(wordList) # saving time using hashset
        d = construct_dict(wordList | set([beginWord, endWord]))
        return bfs_words(beginWord, endWord, d)

# 344. Reverse String
class Solution(object):
	# method: [begin: end: step]. step is -1
    def reverseString(self, s):
        """
        :type s: str
        :rtype: str
        """
        return s[::-1]

    def reverseString(self, s):
    	# method: two pointer
    	if len(s) == 0 or s is None:
    		return s
    	left, right = 0, len(s) - 1
    	res = list(s)
    	while left < right:
    		res[left], res[right] = res[right], res[left]
    		left += 1
    		right -= 1
    	return "".join(res)
    
    def reverseString(self, s):
        l = len(s)
        if l < 2:
            return s
        return self.reverseString(s[l/2:]) + self.reverseString(s[:l/2])

# 541. Reverse String ||
class Solution(object):
    def reverseStr(self, s, k):
        """
        :type s: str
        :type k: int
        :rtype: str
        """
        if len(s) < k:
        	return s[::-1]
        s = list(s)
        for i in xrange(0, len(s), 2*k):
        	temp = s[i : i + k]
        	temp.reverse()
        	s[i : i + k] = temp
       	return "".join(s)


# 151. Reverse Words in a String
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        slow = 0
        res = ""

        for fast in range(len(s) + 1):
        	if fast == len(s) or s[fast] == ' ':
        		if slow != fast:
        			res += s[slow : fast] + ' '
        		slow = fast + 1
        return res.strip()


# 186. Reverse Words in a String ||
class Solution:
    # @param s, a list of 1 length strings, e.g., s = ['h','e','l','l','o']
    # @return nothing
    def reverseWords(self, s):
        self.reverseStr(s, 0, len(s))
        slow = 0
        for fast in range(len(s) + 1):
            if fast == len(s) or s[fast] == ' ':
                self.reverseStr(s, slow, fast)
                slow = fast + 1

    def reverseStr(self, s, slow, fast):
        if len(s) < 2:
            return s
        fast -= 1
        while slow < fast:
            tmp = s[slow]
            s[slow] = s[fast]
            s[fast] = tmp
            slow += 1
            fast -= 1

    def reverseWords_m2(self, s):
    	# using reverse() and reversed()
    	s.reverse()

    	slow = 0
    	for fast in range(len(s)):
    		if s[fast] == ' ':
    			s[slow : fast] = reversed(s[slow : fast])
    			slow = fast + 1
    	s[slow:fast + 1] = reversed(s[slow:fast + 1])

# 557. Reverse Words in a String |||
class Solution(object):
    def reverseWords(self, s):
        """
        :type s: str
        :rtype: str
        """
        s = list(s)
      	l = 0
      	for r in range(len(s) + 1):
      		if r == len(s) or s[r] == ' ':
      			if l != r:
      				a[l:r] = reversed(a[l:r])
      				l = r + 1
        return "".join(s)

    def reverseWords(self, s):
    	s = list(s)
    	l = 0
    	for r in range(len(s) + 1):
    		if r == len(s) or s[r] == ' ':
    			if l != r:
    				self.reverseStr(s, l, r)
    				l = r + 1
    	return "".join(s)
    	
    def reverseStr(self, s, left, right):
        if len(s) < 2:
            return s
        right -= 1
        while left < right:
            tmp = s[left]
            s[left] = s[right]
            s[right] = tmp
            left += 1
            right -= 1

# Copy List of Random Pointer
# Definition for singly-linked list with a random pointer.
# class RandomListNode(object):
#     def __init__(self, x):
#         self.label = x
#         self.next = None
#         self.random = None

class Solution(object):
    def copyRandomList(self, head):
        """
        :time: O(n), space: O(n)
        :type head: RandomListNode
        :rtype: RandomListNode
        :method: hash table
        """
        # deep copy: copy everything
        dic = {}
        m = n = head

        while m:
        	dic[m] = RandomListNode(m.label)
        	m = m.next

        while n:
        	dic[n].next = dic.get(n.next)
        	dic[n].random = dic.get(n.random)
        	n = n.next

        return dic.get(head)


class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        left = [0] * len(nums)
        right = [0] * len(nums)
        res = []
        templeft = tempright = 1
        for i in range(len(nums)):
        	if i > 0:
        		templeft = templeft * nums[i - 1]
        		left[i] = templeft
        		tempright = tempright * nums[len(nums) - i]
        		right[len(nums) - 1 - i]  = tempright
        	elif i == 0:
        		left[i] = templeft
        		right[len(nums)-1-i] = tempright

        for i in range(len(nums)):
        	res.append(left[i] * right[i])

        return res

    def productExceptSelf_m2(self, nums):
    	# smart way
    	# time: O(n), auxiliry sapce: O(1), space: O(n)
    	prod = [1] * len(nums)

    	temp = 1

    	for i in range(len(nums)):
    		prod[i] = temp
    		temp *= nums[i]

    	temp = 1
    	for i in range(len(nums) - 1, -1, -1):
    		prod[i] *= temp
    		temp *= nums[i]

    	return prod

# 38. Count and Say
class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        oldStr = '1'

        for i in range(n-1):
            tmp = ''
            count = 1
            for j in range(len(oldStr)):
                if j < len(oldStr) - 1 and oldStr[j] == oldStr[j+1]:
                    count += 1
                else:
                    tmp += str(count) + oldStr[j]
                    count = 1
            oldStr = tmp

        return oldStr

# 271. Encode and Decode Strings
class Codec:
    def encode(self, strs):
        """Encodes a list of strings to a single string.
        
        :type strs: List[str]
        :rtype: str
        """
        return ''.join([s.replace('|','||') + ' | ' for s in strs])
        

    def decode(self, s):
        """Decodes a single string to a list of strings.
        
        :type s: str
        :rtype: List[str]
        """
        return [str.replace('||','|') for str in s.split(' | ')[:-1]]


# 297. Serialize and Deserialized Binary Tree
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Codec:

    def serialize(self, root):
        """Encodes a tree to a single string.
        :bfs
        :type root: TreeNode
        :rtype: str
        """
        if root is None:
            return ""

        queue = [root]
        res = ""
        while queue:
            node = queue.pop(0)
            if node is None:
                res += "n "
            else:
                res += str(node.val) + " "
                queue.append(node.left)
                queue.append(node.right)
        return res

        

    def deserialize(self, data):
        """Decodes your encoded data to tree.
        :bfs
        :type data: str
        :rtype: TreeNode
        """
        if data == "":
            return None
        values = data.split(" ")[:-1]
        root = TreeNode(int(values[0]))
        queue = [root]
        for i in range(1, len(values), 2):
            node = queue.pop(0)
            if values[i] != "n":
                left = TreeNode(int(values[i]))
                node.left = left
                queue.append(left)

            i += 1
            if values[i] != "n":
                right = TreeNode(int(values[i]))
                node.right = right
                queue.append(right)

        return root


    def serialize(self, root):
        vals = []
        self.treeToStr(root, vals)
        return ' '.join(vals)

    def treeToStr(self, root, vals):
        if root:
            vals.append(str(root.val))
            self.treeToStr(root.left, vals)
            self.treeToStr(root.right, vals)
        else:
            vals.append('#')

    def deserialize(self, data):
        def doit():
            val = next(vals)
            if val == '#':
                return None
            node = TreeNode(int(val))
            node.left = doit()
            node.right = doit()
            return node

        vals = iter(data.split())
        return doit()

    def serialize(self, data):
        # method: DFS
    def serialize(self, data):
        # method: DFS

# 295. Find Median from Data Stream
class MedianFinder(object):

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.data = []

    def addNum(self, num):
        """
        :type num: int
        :rtype: void
        """
        self.data.append(num)
        

    def findMedian(self):
        """
        :rtype: float
        """
        if not self.data:
            return
        self.data.sort()
        length = len(self.data)
        if length % 2 == 0:
            return float(self.data[length/2] + self.data[length/2 - 1])/2
        else:
            return float(self.data[length/2])     

class MedianFinder(object):
    # refactor
    def __init__(self):
        """
        initialize your data structure here.
        """
        self.small = []
        self.large = []

    def addNum(self, num):
        """
        :type num: int
        :rtype: void
        """
        heappush(self.large, num)
        heappush(self.small, -heappop(self.large))
        
        if len(self.large) < len(self.small):
            heappush(self.large, -heappop(self.small))

    def findMedian(self):
        """
        :rtype: float
        """
        if len(self.small) == len(self.large):
            return (self.large[0] + self.small[-1]) / 2.0
        return float(self.large[0])

# 21. Merge Two Sorted Lists
class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        # method: iterative, time: O(n), space: O(1)
        cur = head = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        if not l1 and not l2:
            return head.next

        if not l1:
            cur.next = l2
        if not l2:
            cur.next = l1
        return head.next

    def mergeTwoLists(self, l1, l2):
        # method: recursive, time: O(n), space: O(1)
        if not l1:
            return l2
        if not l2:
            return l1

        if l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2

# 23. Merge k Sorted Lists
class Solution(object):
    def mergeKLists(self, lists):
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        if len(lists) == 0:
            return None
        res = lists[0]
        for i in range(1, len(lists)):
            if i == 1:
                prev = lists[0]
            else:
                prev = res
            res = self.mergeTwoLists(prev, lists[i])

        return
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        cur = head = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                cur.next = l1
                l1 = l1.next
            else:
                cur.next = l2
                l2 = l2.next
            cur = cur.next
        if not l1 and not l2:
            return head.next

        if not l1:
            cur.next = l2
        if not l2:
            cur.next = l1
        return head.next        

    def mergeKLists_m2(self, lists):
        # method: Priority Queue
        from Queue import PriorityQueue
        pq = PriorityQueue()
        cur = dummy = ListNode(0)
        for node in lists:
            if node:
                pq.put((node.val, node))

        while not pq.empty():
            cur.next = pq.get()[1]
            cur = cur.next
            if cur.next:
                pq.put((cur.next.val, cur.next))
        return dummy.next

    def mergeKLists_m3(self, lists):
        # method: heap queue
        from heapq import *
        hq = []
        cur = dummy = ListNode(0)
        for node in lists:
            if node:
                heappush(hq, (node.val, node))
        while len(hq) > 0:
            cur.next = heappop(hq)[1]
            cur = cur.next
            if cur.next:
                heappush(hq, (cur.next.val, cur.next))

        return dummy.next

# 347. Top K Frequent Elements
class Solution(object):
    def topKFrequent(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """
        # method: heapq, hash table
        # time: O(N + klog(k))
        from heapq import *
        dic = {}
        hq = []
        res = []
        for num in nums:
            if num in dic:
                dic[num] += 1
            else:
                dic[num] = 1

        for key in dic:
            heappush(hq, (-dic[key], key))

        for i in range(k):
            res.append(heappop(hq)[1])

        return res     



#160. Intersection of Two Linked List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """
        sizeA = self.getSize(headA)
        sizeB = self.getSize(headB)

        length = min(sizeA, sizeB)

        while sizeA > length:
            headA = headA.next
            sizeA -= 1

        while sizeB > length:
            headB = headB.next
            sizeB -= 1

        res = None

        while headA and headB:
            if headA.val != headB.val:
                if res:
                    res = None
            else:
                if not res:
                    res = headA
            headA = headA.next
            headB = headB.next
        
        return res

    def getInsectionNode(self, headA, headB):
        # clean up the code
        m = self.getSize(headA)
        n = self.getSize(headB)

        if m < n:
            return self.getIntersectionNode(headB, headA)

        for i in range(m - n):
            headA = headA.next

        while headA:
            if headA.val == headB.val:
                return headA
            headA, headB = headA.next, headB.next
        return None

    def getSize(self, head):
        size = 0
        while head:
            head = head.next
            size += 1

        return size
        
# 599. Minimum Index Sum of Two Lists
class Solution(object):
    def findRestaurant(self, list1, list2):
        """
        :type list1: List[str]
        :type list2: List[str]
        :rtype: List[str]
        """
        dic2 = {}
        isum = len(list1) + len(list2) - 2
        res = []

        for j in range(len(list2)):
            dic2[list2[j]] = j # j is index

        for i in range(len(list1)):
            if list1[i] in dic2 and (i + dic2[list1[i]]) < isum:
                isum = i + dic2[list1[i]]
        
        # isum is least list index sum
        for i in range(len(i)):
            if list1[i] in dic2 and i + dic2[list1[i]] == isum:
                res.append(list1[i])
        return res

# 525. Contiguous Array
class Solution(object):
    """docstring for Solution"""
    def findMaxLength(self, nums):
        dic = {}
        dic[0] = -1 # (count, index)
        maxlen = count = 0
        for i in range(len(nums)):
            count = count + ( 1 if nums[i] == 1 else -1 )
            if count in dic:
                maxlen = max(maxlen, i - dic.get(count))
            else:
                dic[count] = i
        return maxlen


# 325. Maximum Size Subarray Sum Equals K
class Solution(object):
    def maxSubArrayLen(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        dic = {}
        maxlen = sum = 0
        for i in range(len(nums)):
            sum += nums[i]
            if sum == k:
                maxlen = max(maxlen, i + 1)
            
            if sum - k in dic:
                maxlen = max(maxlen, i - dic.get(sum - k))
            
            if sum not in dic:
                dic[sum] = i
        return maxlen


# 209. Minimum Size Subarray Sum
class Solution(object):
    def minSubArrayLen(self, s, nums):
        """
        :type s: int
        :type nums: List[int]
        :rtype: int
        """
        # time: O(n), space: O(1)
        # method: two pointers
        l, r, sum, length = 0, 0, 0, len(nums)
        minlen = len(nums) + 1
        while l < len(nums) and r < len(nums):
            sum += nums[r]
            if sum >= s:
                minlen = min(minlen, r - l + 1)
                l += 1
                while l <= r:
                    sum -= nums[l - 1]
                    if sum < s:
                        break
                    else:
                        minlen = min(minlen, r - l + 1)
                        l += 1
            
            r += 1
        
        return minlen if minlen < len(nums) + 1 else 0


# 303. Range Sum Query - Immutable
class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        self.res = nums
        for i in range(1, len(nums)):
            self.res[i] += self.res[i - 1]

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """
        # time: O(1)
        return self.res[j] - (self.res[i - 1] if i > 0 else 0)


# 304. Range Sum Query 2D - Immutable
class NumMatrix(object):
    def __init__(self, matrix):
        """
        :type matrix: List[List[int]]
        """
        if len(matrix) == 0 || len(matrix[0]) == 0:
            return
        self.res = matrix
        m = len(matrix)
        n = len(matrix[0])
        for i in range(1, m):
            self.res[0][i] += self.res[0][i - 1]
        for j in range(1, n):
            self.res[j][0] += self.res[j - 1][0]
        for i in range(1, m):
            for j in range(1, n):
                self.res[i][j] += self.res[i-1][j] + self.res[i][j-1] - self.res[i-1][j-1]
        


    def sumRegion(self, row1, col1, row2, col2):
        """
        :type row1: int
        :type col1: int
        :type row2: int
        :type col2: int
        :rtype: int
        """
        if row1 == 0 and col1 == 0:
            return self.res[row2][col2]

        if row1 == 0 and col1 != 0:
            return self.res[row2][col2] - self.res[row2][col1 - 1]

        if row1 != 0 and col1 == 0:
            return self.res[row2][col2] - self.res[row1 - 1][col2]

        if row1 != 0 and col1 != 0:
            return self.res[row2][col2] - self.res[row1 - 1][col2] - self.res[row2][col1 - 1] + self.res[row1 - 1][col1 - 1]

# clean up the previous code
class NumMatrix(object):
    def __init__(self, matrix):
        if matrix is None or not matrix:
            return
        n, m = len(matrix), len(matrix[0])
        self.sums = [ [0 for j in xrange(m+1)] for i in xrange(n+1) ]
        for i in xrange(1, n+1):
            for j in xrange(1, m+1):
                self.sums[i][j] = matrix[i-1][j-1] + self.sums[i][j-1] + self.sums[i-1][j] - self.sums[i-1][j-1]
    

    def sumRegion(self, row1, col1, row2, col2):
        row1, col1, row2, col2 = row1+1, col1+1, row2+1, col2+1
        return self.sums[row2][col2] - self.sums[row2][col1-1] - self.sums[row1-1][col2] + self.sums[row1-1][col1-1]



# 67. add binary
class Solution(object):
    def addBinary(self, a, b):
        """
        :type a: str
        :type b: str
        :rtype: str
        """
        # method: recursive
        if len(a) == 0:
            return b

        if len(b) == 0:
            return a

        if a[-1] == '0' and b[-1] == '0':
            return self.addBinary(a[:-1], b[:-1]) + '0'
        elif a[-1] == '1' and b[-1] == '1':
            return self.addBinary(self.addBinary(a[:-1], b[:-1]), '1') + '0'
        else:
            return self.addBinary(a[:-1], b[:-1]) + '1'

    def addBinary(self, a, b):
        m = len(a)
        n = len(b)

        maxlen = max(m, n)
        carry = 0
        res = ''
        for i in range(maxlen):
            x = int(a[m - 1 - i]) if i < m else 0
            y = int(b[m - 1 - i]) if i < n else 0

            tmp = (x + y + carry) % 2
            res = str(tmp) + res
            carry = (x + y + carry) / 2

        if carry > 0:
            res = '1' + res
        return res

# 43. Multiply Strings
class Solution(object):
    def multiply(self, num1, num2):
        """
        :type num1: str
        :type num2: str
        :rtype: str
        """
        return self.str2int(num1) * self.str2int(num2)
    
    def str2int(self, strnum):
        if len(strnum) == 0:
            return 0
        if len(strnum) < 2:
            return int(strnum)
        res = int(strnum[0])
        for i in range(1, len(strnum)):
            res = res*10 + int(strnum[i])
        return res

# 66. Plus One
class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        carry = 1
        i = len(digits) - 1
        while carry > 0 and i >= 0:
            sum = digits[i] + (carry if carry == 1 else 0)
            carry = sum / 10
            digits[i] = sum % 10
            i -= 1
        if carry == 1:
            digits.insert(0, 1)
        return digits

# 369. Plus One Linked List
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def plusOne(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        head = self.reverse(head)
        cur = head
        carry = 1
        while cur is not None and carry > 0:
            tmp = carry
            tmp += cur.val
            cur.val = tmp % 10
            carry = tmp / 10
            cur = cur.next

        head = self.reverse(head)
        if carry > 0:
            dummy = ListNode(1)
            dummy.next = head
            return dummy
        return head


    def reverse(self, head):
        if head.next is None:
            return head
        prev = None
        cur = head
        while cur is not None:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        return prev


# 83. Remove Duplicates from Sorted List
# time: O(n), space: O(1)
class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if not head:
            return head
        if not head.next:
            return head
        cur = head
        nxt = cur.next
        
        while nxt:
            if nxt.val == cur.val:
                nxt = nxt.next
                cur.next = nxt
            else:
                cur = nxt
                nxt = nxt.next
        
        return head

    def deleteDuplicates(self, head):
        cur = head
        while cur:
            # remove duplicates
            while cur.next and cur.next.val == cur.val:
                cur.next = cur.next.next
            cur = cur.next
        return head

# 82. Remove Duplicates from Sorted List ||
class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        dummy = pre = ListNode(0)
        cur = head
        pre.next = head

        while cur and cur.next:
            if cur.val == cur.next.val:
                while cur.next and cur.val == cur.next.val:
                    cur = cur.next
                cur = cur.next
                pre.next = cur
            else:
                cur = cur.next
                pre = pre.next
                
        return dummy.next  

# 206. Reverse Linked List
class Solution(object):
    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        # method: iterative
        if head is None:
            return head
        if head.next is None:
            return head
        prev = None
        cur = head
        while cur:
            nxt = cur.next
            cur.next = prev
            prev = cur
            cur = nxt
        return prev

    def reverseList(self, head):
        # method: recursive
        return self._reverse(head)
    def _reverse(self, node, prev=None):
        if not Node:
            return prev
        n = node.next
        node.next = prev
        return  self._reverse(n, node)

# 92. Reverse Linked List ||
# Definition for singly-linked list.
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution(object):
    def reverseBetween(self, head, m, n):
        # time: O(n), space: O(1)
        if m == n:
            return head

        dummy = ListNode(0)
        dummy.next = head
        pre = dummy
        
        for i in range(m - 1):
            pre = pre.next

        # reverse the [m, n] nodes
        reverse = None
        cur = pre.next
        for i in range(n - m + 1):
            nxt = cur.next
            cur.next = reverse
            reverse = cur
            cur = nxt

        pre.next.next = cur
        pre.next = reverse

        return dummy.next

# 355. Design Twitter
# -----------------test case procedure----------------
# Your Twitter object will be instantiated and called as such:
# obj = Twitter()
# obj.postTweet(userId,tweetId)
# param_2 = obj.getNewsFeed(userId)
# obj.follow(followerId,followeeId)
# obj.unfollow(followerId,followeeId)
class Twitter(object):
    '''
    : assume tweetId 5 is posted after tweetId 3
    : time: O(1), space: O(n)
    : data structure:
    :    tweetDB: {userId, tweetIds} % {int, set}
    :    followDB: {userId, followIds} % {int, set}
    '''
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.tweetDB = {}
        self.followDB = {}

    def postTweet(self, userId, tweetId):
        """
        Compose a new tweet.
        :type userId: int
        :type tweetId: int
        :rtype: void
        """
        if userId in self.tweetDB:
            self.tweetDB[userId].add(tweetId)
        else:
            self.tweetDB[userId] = set([tweetId])
            

    def getNewsFeed(self, userId):
        """
        Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
        :type userId: int
        :rtype: List[int]
        """
        news = self.tweetDB.get(userId)
        if news is None:
            news = set()
        if userId in self.followDB and self.followDB.get(userId) is not None:
            for user in self.followDB.get(userId):
                tmp = self.tweetDB.get(user)
                if tmp is not None:
                    news = news | tmp
        res = list(news)
        res.reverse()
        if len(res) < 10:
            return res
        else:
            return res[0:10]

    def follow(self, followerId, followeeId):
        """
        Follower follows a followee. If the operation is invalid, it should be a no-op.
        :type followerId: int
        :type followeeId: int
        :rtype: void
        """
        if followerId not in self.followDB:
            self.followDB[followerId] = set([followeeId])
        else:
            self.followDB[followerId].add(followeeId)
        
        
        

    def unfollow(self, followerId, followeeId):
        """
        Follower unfollows a followee. If the operation is invalid, it should be a no-op.
        :type followerId: int
        :type followeeId: int
        :rtype: void
        """
        if followerId not in self.followDB:
            return
        if followeeId in self.followDB.get(followerId):
            self.followDB[followerId].discard(followeeId)

# Method 2: since method 1 can't pass all test cases
# data structure:
from heapq import *
class Twitter(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.timer = 0
        self.tweetDB = {}
        self.followDB = {}

    def postTweet(self, userId, tweetId):
        """
        Compose a new tweet.
        :type userId: int
        :type tweetId: int
        :rtype: void
        """
        self.timer += 1
        if userId in self.tweetDB:
            self.tweetDB[userId].append((-self.timer, tweetId))
        else:
            self.tweetDB[userId] = [(-self.timer, tweetId)]
        

    def getNewsFeed(self, userId):
        """
        Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
        :type userId: int
        :rtype: List[int]
        """
        tweets = []
        tweets += self.tweetDB.get(userId, [])

        if userId in self.followDB:
            for user in self.followDB.get(userId, set()):
                tmp = self.tweetDB.get(user, [])
                tweets += tmp       

        heapify(tweets)
        res = []
        while len(res) < 10 and tweets:
            res.append(heappop(tweets)[1])

        return res

    def follow(self, followerId, followeeId):
        """
        Follower follows a followee. If the operation is invalid, it should be a no-op.
        :type followerId: int
        :type followeeId: int
        :rtype: void
        """
        if followerId == followeeId:
            return
        if followerId not in self.followDB:
            self.followDB[followerId] = set([followeeId])
        else:
            self.followDB[followerId].add(followeeId)
        
        
        

    def unfollow(self, followerId, followeeId):
        """
        Follower unfollows a followee. If the operation is invalid, it should be a no-op.
        :type followerId: int
        :type followeeId: int
        :rtype: void
        """
        if followerId in self.followDB:
            # do nothing if there is no such followeeId
            self.followDB[followerId].discard(followeeId)



# 20. Valid Parentheses
class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        dic = {
            ')':'(',
            ']':'[',
            '}':'{'
        }
        for char in s:
            if char in dic.values():
                stack.append(char)
            else:
                if char in dic:
                    if not stack or dic[char] != stack.pop():
                        return False
                else:
                    return False
        return stack == []
# 22. Generate Parentheses
class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        
               
# 301. Remove invalid parentheses
class Solution(object):
    def removeInvalidParentheses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        def isvalid(s):
        	count = 0
        	for c in s:
        		if c == 'c':
        			count += 1
        		elif c == ')':
        			count -= 1
        			if count < 0:
        				return False
       	level = {s}
       	while True:
       		valid = filter(isvalid, level)
       		if valid:
       			return valid
       		level = {s[:i] + s[i+1:] for s in level for i in range(len(s))}


# 91. Decode Ways
class Solution(object):
    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        n = len(s)
        if not s or n == 0:
        	return 0

        dp = [0] * (n + 1)
        dp[0] = 1

        if int(s[0]) != 0:
        	dp[1] = 1

        for i in range(2, n + 1):
        	if int(s[i - 1]) != 0:
        		dp[i] += dp[i - 1]

        	two = int(s[i-2] + s[i-1])
        	if two >= 10 and two <= 26:
        		dp[i] += dp[i - 2]

        return dp[n]

    def numDecodings2(self, s):
    	n = len(s)
    	if not s or n == 0:
    		return 0

    	p = 1 # store the number of two digits

    	if int(s[0]) != 0:
    		q = 1

    	for i in range(2, n + 1):
    		tmp = q
    		q = 0
    		if int(s[i - 1]) != 0:
    			q += tmp

    		two = int(s[i-2] + s[i-1])
    		if two >= 10 and two <= 26:
    			q += p

    		p = tmp

    	return q

# 379. Design Phone Directory
class PhoneDirectory(object):

    def __init__(self, maxNumbers):
        """
        Initialize your data structure here
        @param maxNumbers - The maximum numbers that can be stored in the phone directory.
        :type maxNumbers: int
        """
        self.PhoneDirectory = set(range(maxNumbers))

    def get(self):
        """
        Provide a number which is not assigned to anyone.
        @return - Return an available number. Return -1 if none is available.
        :rtype: int
        """
        return self.PhoneDirectory.pop() if self.PhoneDirectory else -1
        

    def check(self, number):
        """
        Check if a number is available or not.
        :type number: int
        :rtype: bool
        """
        return number in self.PhoneDirectory

    def release(self, number):
        """
        Recycle or release a number.
        :type number: int
        :rtype: void
        """
        self.PhoneDirectory.add(number)

# 346. Moving Average from Data Stream
class MovingAverage(object):

    def __init__(self, size):
        """
        Initialize your data structure here.
        :type size: int
        """
        self.queue = [] # queue
        self.size = size

    def next(self, val):
        """
        :type val: int
        :rtype: float
        """
        self.queue.append(val)
        
        if len(self.queue) < self.size:
            return float(sum(self.queue)) / len(self.queue)
        else:
            while len(self.queue) > self.size:
                self.queue.pop(0)
            return float(sum(self.queue)) / self.size


'''
: method 2: using deque from collections
: time: O(1)
'''
from collections import deque
class MovingAverage_deque(object):
	def __init__(self, size):
		self.queue = deque(maxlen = size)

	def next(self, val):
		self.queue.append(val)
		return float(sum(self.queue)) / len(self.queue)



# 251. Flatten 2D Vector
class Vector2D(object):

    def __init__(self, vec2d):
        """
        Initialize your data structure here.
        :type vec2d: List[List[int]]
        """
        self.data = []
        for row in vec2d:
            self.data += row
        

    def next(self):
        """
        :rtype: int
        """
        return self.data.pop(0)

    def hasNext(self):
        """
        :rtype: bool
        """
        if len(self.data) == 0 or not self.data:
            return False
        else:
            return True


# Your Vector2D object will be instantiated and called as such:
# i, v = Vector2D(vec2d), []
# while i.hasNext(): v.append(i.next())


# 284. Peeking Iterator
# Below is the interface for Iterator, which is already defined for you.
#
# class Iterator(object):
#     def __init__(self, nums):
#         """
#         Initializes an iterator object to the beginning of a list.
#         :type nums: List[int]
#         """
#
#     def hasNext(self):
#         """
#         Returns true if the iteration has more elements.
#         :rtype: bool
#         """
#
#     def next(self):
#         """
#         Returns the next element in the iteration.
#         :rtype: int
#         """

class PeekingIterator(object):
    def __init__(self, iterator):
        """
        Initialize your data structure here.
        :type iterator: Iterator
        """
        self.iter = iterator
        self.right = self.iter.next() if self.iter.hasNext() else None
        

    def peek(self):
        """
        Returns the next element in the iteration without advancing the iterator.
        :rtype: int
        """
        return self.right
        

    def next(self):
        """
        :rtype: int
        """
        tmp = self.right
        self.right = self.iter.next() if self.iter.hasNext() else None
        return tmp
        

    def hasNext(self):
        """
        :rtype: bool
        """
        return self.right is not None
        

# Your PeekingIterator object will be instantiated and called as such:
# iter = PeekingIterator(Iterator(nums))
# while iter.hasNext():
#     val = iter.peek()   # Get the next element but not advance the iterator.
#     iter.next()         # Should return the same value as [val].

# 281. Zigzag Iterator
from collections import deque
class ZigzagIterator(object):

    def __init__(self, v1, v2):
        """
        Initialize your data structure here.
        :type v1: List[int]
        :type v2: List[int]
        """
        self.v1 = deque(v1)
        self.v2 = deque(v2)
        self.flag = 0
        
    def next(self):
        """
        :rtype: int
        """
        if len(self.v1) * len(self.v2) != 0:
            if self.flag == 0:
                self.flag = 1
                return self.v1.popleft()
            else:
                self.flag = 0
                return self.v2.popleft()
        
        if len(self.v1) != 0:
            return self.v1.popleft()
        if len(self.v2) != 0:
            return self.v2.popleft()
        

    def hasNext(self):
        """
        :rtype: bool
        """
        return len(self.v1) + len(self.v2) != 0

class ZigzagIterator_queue(object):
	# method 2: using queue by python list
	# time: O(k), space: O(n)
	def __init__(self, v1, v2):
		self.queue = [val for val in (v1, v2) if val]

	def next(self):
		tmp = self.queue.pop(0)
		x = tmp.pop(0)
		if tmp: self.queue.append(tmp)
		return x

	def hasNext(self):
		if self.queue: return True
		return False

    

# 341. Flatten Nested List Iterator
# """
# This is the interface that allows for creating nested lists.
# You should not implement it, or speculate about its implementation
# """
#class NestedInteger(object):
#    def isInteger(self):
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        :rtype bool
#        """
#
#    def getInteger(self):
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        :rtype int
#        """
#
#    def getList(self):
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        :rtype List[NestedInteger]
#        """

class NestedIterator(object):

    def __init__(self, nestedList):
        """
        Initialize your data structure here.
        :type nestedList: List[NestedInteger]
        """
        self.stack = nestedList[::-1]

    def next(self):
        """
        :rtype: int
        """
        return self.stack.pop().getInteger() 

    def hasNext(self):
        """
        :rtype: bool
        """
        while self.stack:
        	top = self.stack[-1]
        	if top.isInteger():
        		return True
        	self.stack = self.stack[:-1] + top.getList()[::-1]
        return False
        		

# Your NestedIterator object will be instantiated and called as such:
# i, v = NestedIterator(nestedList), []
# while i.hasNext(): v.append(i.next())


# 535. Encode and Decode TinyURL
class Codec:
	def __init__(self):
		self.urls = []

	def encode(self, longUrl):
        """Encodes a URL to a shortened URL.
        
        :type longUrl: str
        :rtype: str
        """
        self.urls.append(longUrl)
        return 'http://tinyurl.com' + str(len(self.urls) - 1)

    def decode(self, shortUrl):
        """Decodes a shortened URL to its original URL.
        
        :type shortUrl: str
        :rtype: str
        """
        return self.urls[int(shortUrl.split('/')[-1])]
        

# Your Codec object will be instantiated and called as such:
# codec = Codec()
# codec.decode(codec.encode(url))

class Codec:
	alphabet = string.ascii_letters + '0123456789'
	def __init__(self):
		self.url2code = {} # {url: code}
		self.code2url = {} # {code: url}

	def encode(self, longUrl):
		while longUrl not in self.url2code:
			code = ''.join(random.choice(Codec.alphabet) for _ in range(6))
			if code not in self.code2url:
				self.url2code[longUrl] = code
				self.code2url[code] = longUrl
		return 'http://tinyurl.com/' + self.url2code[longUrl]

	def decode(self, shortUrl):
		return self.code2url[shortUrl[-6:]]

# 621. Task Scheduler
class Solution(object):
    def leastInterval(self, tasks, n):
        """
        :type tasks: List[str]
        :type n: int
        :rtype: int
        """
        task_counts = collections.Counter(tasks).values()
        M = max(task_counts)
        Mct = task_counts.count(M)
        return max(len(tasks), (M - 1) * (N + 1) + Mct)

# 252. Meeting Rooms
# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def canAttendMeetings(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: bool
        """
        # time: sort - O(nlogn), O(n) to go through the array
        # space: O(1): no additional space is allowed
        intervals.sort(key = lambda x : x.start)
        
        for i in range(len(intervals) - 1):
            if intervals[i].end > intervals[i + 1].start:
                return False
        return True
    
    def canAttendMeetings_m2(self, intervals):
        starts = []
        ends = []
        for i in intervals:
            starts.append(i.start)
            ends.append(i.end)
        starts.sort()
        ends.sort()
        for i in range(len(intervals) - 1):
            if not (starts[i] <= ends[i] and starts[i + 1] >= ends[i]):
                return False
        return True   

# 253. Meeting Rooms ||
# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def minMeetingRooms(self, intervals):
        starts = []
        ends = []
        for i in intervals:
            starts.append(i.start)
            ends.append(i.end)

        starts.sort()
        ends.sort()
        s = e = 0
        numRooms = available = 0
        while s < len(starts):
            if starts[s] < ends[e]:
                if available == 0:
                    numRooms += 1
                else:
                    available -= 1

                s += 1
            else:
                available += 1
                e += 1

        return numRooms

# 56. Merge Intervals
# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        # time: O(nlogn) + O(n)
        # space: O(n)            
        out = []
        intervals.sort(key = lambda x : x.start)
        for i in intervals:
            if out and i.start <= out[-1].end:
                out[-1].end = max(out[-1].end, i.end)
            else:
                out.append(i)
        return out

    def merge_m2(self, intervals):
        res, starts, ends = [], [], []
        for i in intervals:
            starts.append(i.start)
            ends.append(i.end)

        starts.sort()
        ends.sort()

        i = 0
        while i < len(intervals):
            s = starts[i]
            while i < len(intervals) - 1 and starts[i + 1] <= ends[i]:
                i += 1
            e = ends[i]
            it = Interval(s, e)
            res.append(it)
            i += 1
        return res

# 57. Insert Interval
# Definition for an interval.
# class Interval(object):
#     def __init__(self, s=0, e=0):
#         self.start = s
#         self.end = e

class Solution(object):
    def insert(self, intervals, newInterval):
        """
        :type intervals: List[Interval]
        :type newInterval: Interval
        :rtype: List[Interval]
        """

# 13. Roman to Integer
# The trick is that the last letter is always added. Except the last one,
# if one letter is less than its latter one, this letter is substracted
class Solution(object):
    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        # time: O(n)
        # space: O(1)
        romanNum = {
            'M': 1000,
            'D': 500,
            'C': 100,
            'L': 50,
            'X': 10,
            'V': 5,
            'I': 1
        }
        num = 0
        for i in range(len(s) - 1):
            if romanNum[s[i]] < romanNum[s[i + 1]]:
                num -= romanNum[s[i]]
            else:
                num += romanNum[s[i]]

        return num + romanNum[s[-1]]

# 12. Integer to Roman
class Solution(object):
    def intToRoman(self, num):
        """
        :type num: int
        :rtype: str
        """
        thousand = {
            1: 'M',
            2: 'MM',
            3: 'MMM'
        }

        hundred = {
            1: 'C',
            2: 'CC',
            3: 'CCC',
            4: 'CD',
            5: 'D',
            6: 'DC',
            7: 'DCC',
            8: 'DCCC',
            9: 'CM'
        }

        ten = {
            1: 'X',
            2: 'XX',
            3: 'XXX',
            4: 'XL',
            5: 'L',
            6: 'LX',
            7: 'LXX',
            8: 'LXXX',
            9: 'XC'
        }

        one = {
            1: 'I',
            2: 'II',
            3: 'III',
            4: 'IV',
            5: 'V',
            6: 'VI',
            7: 'VII',
            8: 'VIII',
            9: 'IX'
        }

        rm = ''
        for i in [1000, 100, 10, 1]:
            a = num / i
            b = num % i

            if a != 0:
                if i == 1000:
                    rm += thousand[a]
                elif i == 100:
                    rm += hundred[a]
                elif i == 10:
                    rm += ten[a]
                elif i == 1:
                    rm += one[a]
                if b == 0:
                    return rm
            num = b
        return rm


     def intToRoman_m2(self, num):
        matrix = {
            1000:{
                1: 'M',
                2: 'MM',
                3: 'MMM'
            },
            100:{
                1: 'C',
                2: 'CC',
                3: 'CCC',
                4: 'CD',
                5: 'D',
                6: 'DC',
                7: 'DCC',
                8: 'DCCC',
                9: 'CM'
            },
            10:{
                1: 'X',
                2: 'XX',
                3: 'XXX',
                4: 'XL',
                5: 'L',
                6: 'LX',
                7: 'LXX',
                8: 'LXXX',
                9: 'XC'
            },
            1:{
                1: 'I',
                2: 'II',
                3: 'III',
                4: 'IV',
                5: 'V',
                6: 'VI',
                7: 'VII',
                8: 'VIII',
                9: 'IX'
            }
        } # matrix ends
        rm = ''
        for i in [1000, 100, 10, 1]:
            a = num / i
            b = num % i

            if a != 0:
                rm += matrix[i][a]

            num = b
        return rm

    def intToRoman_m3(self, num):
        M = ["", "M", "MM", "MMM"];
        C = ["", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"];
        X = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"];
        I = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"];
        return M[num/1000] + C[(num%1000)/100] + X[(num%100)/10] + I[num%10];

# 273.Integer to English Words
class Solution(object):
    def numberToWords(self, num):
        """
        :type num: int
        :rtype: str
        """
        to19 = ['One', 'Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten'
                'Ten', 'Eleven', 'Twelve', 'Thirteen', 'Fourteen', 'Fifteen', 'Sixteen',
                'Seventeen', 'Eighteen', 'Nineteen']
        tens = ['Twenty', 'Thirty', 'Forty', 'Fifty', 'Sixty', 'Seventy', 'Eighty', 'Ninety']

        def words(n):
            if n < 20:
                return to19[n - 1: n]
            if n < 100:
                return [tens[n / 10 - 2]] + words(n % 10)
            if n < 1000:
                return [to19[n / 100 - 1]] + ['Hundred'] + words(n % 100)

            for p, w in enumerate(('Thousand', 'Million', 'Billion'), 1):
                if n < 1000**(p+1):
                    return words(n/1000**p) + [w] + words(n%1000**p)

        return ' '.join(words(num)) or 'Zero'



# 314. Binary Tree Vertical Order Traversal
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def verticalOrder(self, root):
        """
        :type root: TreeNode
        :rtype: List[List[int]]
        """
        cols = collections.defaultdict(list)
        queue = [(root, 0)]

        for node, i in queue:
            if node:
                cols[i].append(node.val)
                queue += (node.left, i - 1), (node.right, i + 1)
        return [cols[i] for i in sorted(cols)]

    def verticalOrder_m2(self, root):
        # time: O(nlogn) 
        # space: O(n)
        queue = [(root, 0)]
        cols = {}

        while queue:
            node, i = queue.pop(0)
            if node:
                if i not in cols:
                    cols[i] = []
                cols[i].append(node.val)
                queue += (node.left, i - 1), (node.right, i + 1)
        return [cols[i] for i in sorted(cols)]

    def verticalOrder_m3(self, root):
        # time: O(n)
        # space: O(n)
        queue = [(root, 0)]
        cols = {}
        mi, mx = 0, 0
        while queue:
            node, i = queue.pop(0)
            if i < mi: mi = i
            if i > mx: mx = i
            if node:
                if i not in cols:
                    cols[i] = []
                cols[i].append(node.val)
                queue += (node.left, i - 1), (node.right, i + 1)
        res = []
        for i in range(mi, mx + 1):
            if i in cols:
                res.append(cols[i])
        return res

# Keyboard Row
class Solution(object):
    def findWords(self, words):
        """
        :type words: List[str]
        :rtype: List[str]
        """
        line1 = set('qwertyuiop')
        line2 = set('asdfghjkl')
        line3 = set('zxcvbnm')

        res = []
        for word in words:
            w = word.lower()
            if set(w).issubset(line1) or set(w).issubset(line2) or set(w).issubset(line3):
                res.append(word)
        return res

# 277. Find the Celebrity
# The knows API is already defined for you.
# @param a, person a
# @param b, person b
# @return a boolean, whether a knows b
# def knows(a, b):

class Solution(object):
    def findCelebrity(self, n):
        """
        :type n: int
        :rtype: int
        """
        # time: O(n**2)
        # space: O(1)
        def NotKnowOthers(p):
            for i in range(n):
                if i != p and knows(p, i):
                    return False
            return True
        def Otherknows(p):
            for i in range(n):
                if i != p and not knows(i, p):
                    return False
            return True
        for person in range(n):
            if NotKnowOthers(person) and Otherknows(person):
                return person
        return -1

    def findCelebrity_m2(self, n):
        candidate = 0
        for i in range(n):
            if knows(candidate, i):
                candidate = i

        for i in range(candidate):
            if knows(candidate, i):
                return -1

        for i in range(n):
            if not knows(i, candidate):
                return -1
        return candidate



# 257. Binary Tree Paths
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def binaryTreePaths(self, root):
        """
        :type root: TreeNode
        :rtype: List[str]
        """
        # recursively
        if not root:
            return []

        res = []
        self.dfs(root, "", res)
        return res

    def dfs(self, root, ls, res):
        if not root.left and not root.right:
            res.append(ls + str(root.val))
        if root.left:
            self.dfs(root.left, ls + str(root.val) + "->", res)
        if root.right:
            self.dfs(root.right, l + str(root.val) + "->", res)       

    def binaryTreePaths(self, root):
        if not root:
            return []

        res, stack = [], [(root, "")]
        while stack:
            node, ls = stack.pop()
            if not node.left and not node.right:
                res.append(ls + str(node.val))
            if node.right:
                stack.append((node.right), ls + str(node.val) + "->")
            if node.left:
                stack.append((node.left), ls + str(node.val) + "->")
        return res

    def binaryTreePaths(self, root):
        if not root:
            return []
        res, queue = [], collections.deque([(root, "")])
        while queue:
            node, ls = queue.popleft()
            if not node.left and not node.right:
                res.append(ls + str(node.val))
            if node.left:
                queue.append((node.left, ls + str(node.val) + "->"))
            if node.right:
                queue.append((node.right, ls + str(node.val) + "->"))
        return res


# 157. Read N Characters Given Read4
# The read4 API is already defined for you.
# @param buf, a list of characters
# @return an integer
# def read4(buf):

class Solution(object):
    def read(self, buf, n):
        """
        :type buf: Destination buffer (List[str])
        :type n: Maximum number of characters to read (int)
        :rtype: The number of characters read (int)
        """
        i = 0
        while i < n: 
            buf4 = ['','','','']
            count = read4(buf4) # Read file into buf4.
            if not count: break # EOF
            count = min(count, n - i)
            buf[i:] = buf4[:count] # Copy from buf4 to buf.
            i += count
        return i

    def read(self, buf, n):
        EOF = False
        total = 0
        tmp = [''] * 4 # tempary buffer

        while not EOF and total< n:
            count = read4(tmp)

            # check if it's the end of the file
            EOF = count < 4

            # get the actual count
            count = min(count, n - total)

            # copy from temp buffer to buf
            for i in range(count):
                buf[total] = tmp[i]
                total += 1

        return total

# 161. One Edit Distance
class Solution(object):
    def isOneEditDistance(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        # time: O(n)
        # space: O(1)
        m, n = len(s), len(t)
        if abs(m - n) > 1:
            return False
        count = 0
        i = 0
        j = 0
        while i < m and j < n:
            if s[i] != t[j]:
                if count == 1:
                    return False
                else:
                    if m > n:
                        i += 1
                    elif m < n:
                        j += 1
                    else:
                        i += 1
                        j += 1
                        
                    count += 1
            else:
                i += 1
                j += 1
        if i < m or j < n:
            count += 1
            
        return count == 1


# 72. Edit Distance( NOT DONE)
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        # method: Dynamic Programming
        # time: O(m*n)
        # space: O(1)
        m, n = len(word1), len(word2)
        # If second string is empty, the only option is to
        # insert all characters of second string into first
        if m == 0:
            return n
        # vice versa
        if n == 0:
            return m

        # If last characters of two strings are same, nothing
        # much to do. Ignore last characters and get count for
        # remaining strings
        if word1[-1] == word2[-1]:
            return self.minDistance(word1[:m-1], word2[:n-1])

        methodInsert = self.minDistance(word1, word2[:n-1])
        methodRemove = self.minDistance(word1[:m-1], word2)
        methodReplace = self.minDistance(word1[:m-1], word2[:n-1])
        res = 1 + min(methodInsert, methodRemove, methodReplace)
        return res

    def minDistance_m2(self, word1, word2):
        m, n = len(word1), len(word2)
        dp = [[0 for x in range(n + 1)] for y in range(m + 1)]
        for i in range(1, m + 1):
            dp[i][0] = i
        for j in range(1, n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1] + 1, min(dp[i][j - 1] + 1, dp[i - 1][j] + 1))
        return dp[m][n]


# 129. Sum Roof to Leaf Numbers
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sumNumbers(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
        	return 0
        res = 0
        stack = [(root, "")]
        while stack:
        	node, path = stack.pop()
        	if not node.left and not node.right:
        		path += str(node.val)
        		res += int(path)
        	if node.left:
        		stack.append((node.left, path+str(node.val)))
        	if node.right:
        		stack.append((node.right, path+ str(node.val)))
        return res


# 437. Path Sum |||
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class Solution(object):
    def pathSum(self, root, sum):
        """
        :type root: TreeNode
        :type sum: int
        :rtype: int
        """

        if root:
        	return self.helper(root, sum) + self.pathSum(root.left, sum) + self.pathSum(root.right, sum)
      	return 0
    def helper(self, node, sum):
    	if not node:
    		return 0
        return int(node.val==sum)+self.helper(node.left, sum-node.val) + self.helper(node.right, sum-node.val)



# 404. Sum of Left Leaves
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def sumOfLeftLeaves(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
        	return 0
        if root.left and not root.left.left and not root.left.right:
        	return root.val + self.sumOfLeftLeaves(root.right)
        return self.sumOfLeftLeaves(root.left) + self.sumOfLeftLeaves(root.right)

# 543. Diameter of Bianry Tree
class Solution(object):
    def diameterOfBinaryTree(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        self.best = 1
        def depth(node):
        	if not node: return 0
        	left, right = depth(node.left), depth(node.right)
        	self.best = max(self.best, left + right + 1)
        	return 1 + max(left, right)

        depth(root)
        return self.best - 1

# 687. Longest Univalue Path
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def longestUnivaluePath(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if not root:
        	return 0
        res = []
        def depth(node):
        	if not node: return 0
        
        depth(node)
        return self.best - 1

# 191. Number of 1 Bits
class Solution(object):
    def hammingWeight(self, n):
        """
        :type n: int
        :rtype: int
        """
        count = 0
        while n != 0:
        	n = n & (n - 1)
        	count += 1
        return count

# 461. Hamming Distance
class Solution(object):
    def hammingDistance(self, x, y):
        """
        :type x: int
        :type y: int
        :rtype: int
        """
        # method: xor 1^0 = 1, 0^1 = 1
        return bin(x^y).count("1")

# 677. Total Hamming Distance
class Solution(object):
    def totalHammingDistance(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        for i in range(len(nums)):
        	
    def hammingDistance(self, x, y):
    	return bin(x^y).count("1")

# 215. Kth Largest Element in an Array
class Solution(object):
    def findKthLargest(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: int
        """
        # method quick select
        # time: O(n), space: O(1)
        pivot = nums[0]
        init = 0
        for i in range(1 : len(nums)):
            if nums[i] > pivot:
                init += 1
                nums[i], nums[init] = nums[init], nums[i]
                
        nums[0], nums[init] = nums[init], nums[0]
        if init + 1 == k:
            return pivot
        elif init + 1 > k:
            return self.findKthLargest(nums[:init], k)
        else:
            return self.findKthLargest(nums[init:],k-init-1)


# 414. Third Maximum Number
class Solution(object):
    def thirdMax(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) < 3:
            return max(nums)
        nums.remove(max(nums))
        nums.remove(max(nums))
        return nums.remove(max(nums))

    def thirdMax_m2(self, nums):
        v = [float('-inf'), float('-inf'), float('-inf')]
        for num in nums:
            if num not in v:
                if num > v[0]:
                    v[0], v[1], v[2] = num, v[0], v[1]
                elif num > v[1]:
                    v[0], v[1], v[2] = v[0], num, v[1]
                elif num > v[2]:
                    v[0], v[1], v[2] = v[0], v[1], num
        return max(nums) if float('-inf') in v else v[2]

# 693. Binary Number with Alternating Bits
class Solution(object):
    def hasAlternatingBits(self, n):
        """
        :type n: int
        :rtype: bool
        """
        prev = n % 2
        n = / 2

        while n > 0:
            cur = n % 2
            if cur == prev:
                return False

            prev = cur
            n = n / 2
        return True     

# 6. ZigZag Conversion
class Solution(object):
    def convert(self, s, numRows):
        """
        :type s: str
        :type numRows: int
        :rtype: str
        """
        if numRows <= 1:
            return s

        res = ''
        n = len(s)
        for i in range(numRows):
            for j in range(i, n, 2*(numRows-1)):
                res += s[j]

                if i > 0 and i < numRows-1 and j + 2*(numRows-1-i)<n:
                    res += s[j+2*(numRows-1-i)]

        return res

# 208. Implement Trie
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
        return ord(ch) - ord('a')

    def insert(self, key):
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

# 648. Replace Words
class Solution(object):
    def replaceWords(self, dict, sentence):
        """
        :type dict: List[str]
        :type sentence: str
        :rtype: str
        """
        # time: O(n*m)
        # space: O(1)
        rootset = set(dict)

        def replace(word):
            for i in range(1, len(word)):
                if word[:i] in rootset:
                    return word[:i]
            return word

        return " ".join(map(replace, sentence.split()))

# 692. Top K Frequent Words
from heapq import *
class Solution(object):
    def topKFrequent(self, words, k):
        """
        :type words: List[str]
        :type k: int
        :rtype: List[str]
        """

        dic = {}
        heapq = []
        res = []

        for word in words:
            if word not in dic:
                dic[word] = 1
            else:
                dic[word] += 1

        for word in dic:
            heappush(heapq, (-dic[word], word))

        for i in range(k):
            res.append(heappop(heapq)[1])

        return res

# 211. Add and Search Word - Data structure design
class TrieNode(object):

    def __init__(self):
        self.children = [None] * 26
        self.isEnd = False

class WordDictionary(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.root = TrieNode()
        

    def addWord(self, word):
        """
        Adds a word into the data structure.
        :type word: str
        :rtype: void
        """

        node = self.root
        for letter in word:
            index = self._charToIndex(letter)
            if not node.children[index]:
                node.children[index] = TrieNode()
            node = node.children[index]
        node.isEnd = True
        

    def search(self, word):
        """
        Returns if the word is in the data structure. A word could contain the dot character '.' to represent any one letter.
        :type word: str
        :rtype: bool
        """
        node = self.root
        for letter in word:
            index =  self._charToIndex(letter)
            if not node.children[index]:
                return False
            node = node.children[index]
            
        return node != None and node.isEnd
        
    def _charToIndex(self, ch):
        return ord(ch) - ord('a')


# Your WordDictionary object will be instantiated and called as such:
# obj = WordDictionary()
# obj.addWord(word)
# param_2 = obj.search(word)


# 292. Nim Game
# you can always win a nim game if the number of stones n in the pile
# is not divisible by 4
class Solution(object):
    def canWinNim(self, n):
        """
        :type n: int
        :rtype: bool
        """

        return n % 4 != 0


# 293. Flip Game
class Solution(object):
    def generatePossibleNextMoves(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        res = []
        
        for i in range(len(s) - 1):
            if s[i] == "+" and s[i + 1] == "+":
                res.append(s[:i] + "--" + s[i+2:])
                
        return res


# 169. Majority Element
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if not nums:
            return []

        count1, count2, candidate1, candidate2 = 0, 0, 0, 1
        for num in nums:
            if num == candidate1:
                count1 += 1
            elif num == candidate2:
                count2 += 2
            elif count1 == 0:
                candidate1, count1 = num, 1
            elif count2 == 0:
                candidate2, count2 = num, 1
            else:
                count1 -= 1
                count2 -= 1

        return [num for num in (candidate1, candidate2) if nums.count(num)>len(nums)/3]


# 229. Majority Element ||
class Solution(object):
    def majorityElement(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        if not nums:
            return []

        count1, count2, candidate1, candidate2 = 0, 0, 0, 1
        for num in nums:
            if num == candidate1:
                count1 += 1
            elif num == candidate2:
                count2 += 2
            elif count1 == 1:
                candidate1, count1 = num, 1
            elif count2 == 2:
                candidate2, count2 = num, 1
            else:
                count1 -= 1
                count2 -= 2

        return [num for num in (candidate1, candidate2) if nums.count(num)>len(nums)/3]




#------------- to do list -------------
# 294. Flip Game ||
class Solution(object):
    def canWin(self, s):
        """
        :type s: str
        :rtype: bool
        """

        
# 8. String to Integer(atoi)
class Solution(object):
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        
        

# 76. Minimum Window Substring
class Solution(object):
    def minWindow(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: str
        """

# 282. Expression Add Operators
class Solution(object):
    def addOperators(self, num, target):
        """
        :type num: str
        :type target: int
        :rtype: List[str]
        """

# 307. Range Sum Query - Mutable
class NumArray(object):

    def __init__(self, nums):
        """
        :type nums: List[int]
        """
        

    def update(self, i, val):
        """
        :type i: int
        :type val: int
        :rtype: void
        """
        

    def sumRange(self, i, j):
        """
        :type i: int
        :type j: int
        :rtype: int
        """

# 126. Word Ladder ||
class Solution(object):
    def findLadders(self, beginWord, endWord, wordList):
        """
        :type beginWord: str
        :type endWord: str
        :type wordList: List[str]
        :rtype: List[List[str]]
        """


# 460. LFU Cache
class LFUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.capacity = capacity
        self.dict = dict()
        self.freq = dict()

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """

        

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: void
        """


# 239. Sliding Window Maximum
class Solution(object):
    def maxSlidingWindow(self, nums, k):
        """
        :type nums: List[int]
        :type k: int
        :rtype: List[int]
        """




# 313. Super Ugly Number
class Solution(object):
    def nthSuperUglyNumber(self, n, primes):
        """
        :type n: int
        :type primes: List[int]
        :rtype: int
        """

    
    def isSuperUgly(self, n, primes):
        for prime in primes:
            while num % prime == 0:
                num /= prime
            if num == 1:
                return True
        return False


# 437. Path Sum |||


# 133. Clone Graph
# Definition for a undirected graph node
# class UndirectedGraphNode:
#     def __init__(self, x):
#         self.label = x
#         self.neighbors = []

class Solution:
    # @param node, a undirected graph node
    # @return a undirected graph node
    def cloneGraph(self, node):


# 222. Count Complete Tree Nodes
class Solution(object):
    def countNodes(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """


# 583. Delete Operation for Two Strings
class Solution(object):
    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """